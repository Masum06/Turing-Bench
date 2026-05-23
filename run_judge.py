"""Run the Turing-Bench judge over a paired-dialogue CSV.

Uses a modular ``model_clients.ModelClient``. Pass ``--model`` to use a hosted
API (provider is inferred from the model name) or ``--provider custom`` to
delegate to ``predict.py``.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import N_THREADS, USE_THREADS
from model_clients import ModelClient, create_model_client


DATASET_PATH = "turing_test_o50_conversations_shuffled.csv"


def load_data(path: str, limit: int | None = None) -> pd.DataFrame:
    print(f"Loading data: {path}")
    df = pd.read_csv(path)

    missing = {"dialogueA", "dialogueB"} - set(df.columns)
    if missing:
        sys.exit(f"Input data is missing required columns: {missing}")

    if limit is not None:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def load_resume_predictions(output_path: Path, expected_rows: int) -> dict[int, str]:
    """Load completed A/B predictions from a previous output CSV."""
    if not output_path.exists():
        return {}

    try:
        previous = pd.read_csv(output_path)
    except Exception as exc:
        print(f"Could not read resume file {output_path}: {exc}")
        return {}

    if len(previous) != expected_rows or "who_is_human" not in previous.columns:
        print("Resume skipped: existing output shape does not match this run.")
        return {}

    resumed: dict[int, str] = {}
    for idx, value in enumerate(previous["who_is_human"]):
        if pd.isna(value):
            continue
        label = str(value).strip().upper()
        if label in {"A", "B"}:
            resumed[idx] = label
    return resumed


def predict_row(row: dict, delay: float, client: ModelClient) -> tuple[int, str, dict | None]:
    idx = row["_idx"]
    try:
        verdict = client.predict(str(row["dialogueA"]), str(row["dialogueB"]))
        if verdict not in ("A", "B"):
            raise ValueError(f"Client returned unexpected label: {verdict!r}")
        error_record = None
        pred = verdict
    except NotImplementedError:
        raise
    except Exception as exc:
        pred = "NA"
        error_record = {
            "row_index": idx,
            "row_id": row.get("id", ""),
            "error": f"{type(exc).__name__}: {exc}",
        }

    if delay > 0:
        time.sleep(delay)

    error_out = None
    if error_record:
        print(f"\nError on row {idx}: {error_record['error']} - defaulting to 'NA'")
        error_out = error_record

    return idx, pred, error_out


def run_single(
    rows: list[dict],
    delay: float,
    client: ModelClient,
    output_path: str,
    df
) -> tuple[list[tuple[int, str]], list[dict]]:
    results: list[tuple[int, str]] = []
    errors: list[dict] = []
    desc = "Running predictions (single-threaded)"

    for row in tqdm(rows, desc=desc):
        idx, pred, err_rec = predict_row(row, delay, client)
        idx, pred, err_rec = predict_row(row, delay, client)
        df.at[idx, "who_is_human"] = pred
        df.to_csv(output_path, index=False)
        if err_rec:
            errors.append(err_rec)
        results.append((idx, pred))

    return results, errors

write_lock = Lock()
def run_threaded(
    rows: list[dict],
    delay: float,
    n_threads: int,
    client: ModelClient,
    output_path: str,
    df
) -> tuple[list[tuple[int, str]], list[dict]]:
    results: dict[int, str] = {}
    errors: list[dict] = []

    print(f"Running predictions with {n_threads} threads...")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(predict_row, row, delay, client) for row in rows]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Running predictions ({n_threads} threads)",
        ):
            idx, pred, err_rec = future.result()
            if err_rec:
                errors.append(err_rec)
            results[idx] = pred
            with write_lock: #error on this line
                df.at[idx, "who_is_human"] = pred
                df.to_csv(output_path, index=False)

    return sorted(results.items()), errors


def run_predictions(
    rows: list[dict],
    delay: float,
    n_threads: int,
    client: ModelClient,
    output_path: str,
    df
) -> tuple[list[tuple[int, str]], list[dict]]:
    try:
        if n_threads > 1:
            return run_threaded(rows, delay, n_threads, client, output_path, df)
        return run_single(rows, delay, client, output_path, df)
    except NotImplementedError:
        sys.exit(
            "\npredict() is not implemented yet.\n"
            "Either pass a hosted --model (provider is inferred) or implement predict() in "
            "predict.py and use --provider custom.\nSee README.md."
        )


def run_benchmark(
    client: ModelClient,
    *,
    input_path: str,
    output_path: str,
    errors_path: str,
    limit: int | None,
    delay: float,
    n_threads: int,
    resume: bool,
) -> None:
    """Load CSV, run ``client.predict`` for each row, write output + optional errors CSV."""
    df = load_data(input_path, limit)
    print(f"Loaded {len(df)} examples.\n")

    rows = [{"_idx": i, **row} for i, row in df.iterrows()]
    resumed = (
        load_resume_predictions(Path(output_path), len(rows)) if resume else {}
    )
    if resumed:
        print(f"Resume: reusing {len(resumed)} existing prediction(s).")

    df["who_is_human"] = [resumed.get(i, "NA") for i in range(len(rows))]
    df.to_csv(output_path, index=False) 

    rows_to_run = [row for row in rows if row["_idx"] not in resumed]
    errors_list: list[dict] = []

    if rows_to_run:
        new_results, errors_list = run_predictions(
            rows_to_run,
            delay,
            n_threads,
            client,
            output_path,
            df
        )
    else:
        new_results = []
        print("All rows already have A/B predictions; nothing to rerun.")

    predictions_by_idx = {**resumed, **dict(new_results)}
    na_indices = [idx for idx, pred in dict(new_results).items() if pred == "NA"]

    if na_indices:
        retry_rows = [row for row in rows_to_run if row["_idx"] in na_indices]
        print(f"\nRetrying {len(retry_rows)} NA row(s) single-threaded...")
        retry_results, retry_errors = run_predictions(retry_rows, delay, 1, client, output_path, df)
        predictions_by_idx.update(dict(retry_results))
        errors_list.extend(retry_errors)

    preds = [predictions_by_idx.get(i, "NA") for i in range(len(rows))]
    df["who_is_human"] = preds
    df.to_csv(output_path, index=False)

    if errors_list:
        pd.DataFrame(errors_list).to_csv(errors_path, index=False)
        print(f"Error details saved to: {errors_path}")

    print(f"\nPredictions saved to: {output_path}")
    print(
        f"  Total: {len(preds)} | A: {preds.count('A')} | "
        f"B: {preds.count('B')} | NA: {preds.count('NA')}"
    )
    if n_threads > 1 and rows_to_run:
        print(f"  Threads used: {n_threads}")
    print(
        "\nNext step: submit your predictions CSV to "
        "https://huggingface.co/spaces/roc-hci/TuringBench-2-Leaderboard"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Turing-Bench runner: uses --model with auto-inferred provider, "
            "or --provider custom for predict.py."
        ),
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "custom", "openai", "anthropic", "bedrock"],
        help=(
            "API to use: auto = infer from --model; custom = predict.predict in predict.py; "
            "or force openai / anthropic / bedrock."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Hosted model id (not used for --provider custom). "
        "With --provider auto, provider is inferred from the model string.",
    )
    parser.add_argument(
        "--aws-region",
        default=None,
        help="AWS region for Bedrock (optional; otherwise SDK/boto3 default chain).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Provider-native reasoning control. GPT-5/o-series accept low, medium, "
            "or high; GPT-4 and lower reject it. Claude accepts Anthropic adaptive "
            "thinking values such as on, off, low, medium, high, max, or xhigh "
            "(model support varies)."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DATASET_PATH,
        help=f"Input CSV path (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv).",
    )
    parser.add_argument(
        "--errors",
        default="errors.csv",
        help="Error CSV path (written only when errors occur).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each prediction.",
    )
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        default=None,
        help=(
            "Worker threads (default: N_THREADS if USE_THREADS else 1)."
        ),
    )
    parser.add_argument(
        "--no-threads",
        action="store_true",
        help="Force single-threaded execution.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N rows for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse A/B from existing --output row-by-row.",
    )

    args = parser.parse_args()

    n_threads = args.threads if args.threads is not None else (N_THREADS if USE_THREADS else 1)
    if args.no_threads:
        n_threads = 1
    if n_threads < 1:
        sys.exit("--threads must be >= 1")

    if args.provider != "custom" and not args.model:
        sys.exit("--model / -m is required unless --provider custom")

    try:
        client = create_model_client(
            args.provider,
            args.model,
            aws_region=args.aws_region,
            reasoning_effort=args.reasoning_effort,
        )
    except ValueError as e:
        sys.exit(str(e))

    run_benchmark(
        client,
        input_path=args.input,
        output_path=args.output,
        errors_path=args.errors,
        limit=args.limit,
        delay=args.delay,
        n_threads=n_threads,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
