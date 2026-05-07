import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from config import USE_THREADS, N_THREADS
from predict import predict
from tinyagent import *

import pandas as pd
from tqdm import tqdm

#  Internals
DATASET_PATH = "turing_test_o50_conversations_shuffled.csv"
HF_SPLIT        = "train"


def load_data() -> pd.DataFrame:
    print(f"Loading data: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    missing = {"dialogueA", "dialogueB"} - set(df.columns)
    if missing:
        sys.exit(f"Input data is missing required columns: {missing}")

    return df


def run_single(rows: list[dict], delay: float) -> list[tuple[int, str]]:
    """Sequential inference with a progress bar."""
    results = []
    for row in tqdm(rows, desc="Running predictions (single-threaded)"):
        try:
            pred = predict(str(row["dialogueA"]), str(row["dialogueB"]))["result"]["verdict"]
            if pred not in ("A", "B"):
                raise ValueError(f"predict() returned {pred!r} — must be 'A' or 'B'")
        except NotImplementedError:
            sys.exit(
                "\n✗ predict() is not implemented yet.\n"
                "  Open this script and fill in the predict() function with your model."
            )
        except Exception as exc:
            print(f"\nError on row {row['_idx']}: {exc} — defaulting to 'NA'")
            pred = "NA"

        results.append((row["_idx"], pred))

        if delay > 0:
            time.sleep(delay)

    return results


def run_threaded(rows: list[dict], delay: float, n_threads: int) -> list[tuple[int, str]]:
    """Parallel inference across n_threads workers."""
    results   = {}
    errors    = 0
    lock      = Lock()
    completed = 0

    print(f"Running predictions with {n_threads} threads...")
    pbar = tqdm(total=len(rows), desc=f"Running predictions ({n_threads} threads)")

    def worker(row: dict) -> tuple[int, str]:
        nonlocal errors, completed
        try:
            pred = predict(str(row["dialogueA"]), str(row["dialogueB"]))["result"]["verdict"]
            if pred not in ("A", "B"):
                raise ValueError(f"predict() returned {pred!r} — must be 'A' or 'B'")
        except NotImplementedError:
            sys.exit(
                "\npredict() is not implemented yet.\n"
                "  Open this script and fill in the predict() function with your model."
            )
        except Exception as exc:
            print(f"\nError on row {row['_idx']}: {type(exc).__name__}: {exc} — defaulting to 'NA'")
            with lock:
                errors += 1
            pred = "NA"

        if delay > 0:
            time.sleep(delay)

        return row["_idx"], pred

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(worker, row): row for row in rows}
        for future in as_completed(futures):
            idx, pred = future.result()
            results[idx] = pred
            pbar.update(1)

    pbar.close()
    return sorted(results.items())  # return in original row order


def main():
    parser = argparse.ArgumentParser(
        description="Turing Test Judge Benchmark — generate predictions with your model."
    )
    parser.add_argument(
        "--output", default="predictions.csv",
        help="Output CSV file path (default: predictions.csv).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to wait between calls (useful for rate-limited APIs, default: 0).",
    )
    args = parser.parse_args()

    df = load_data()
    print(f"Loaded {len(df)} examples.\n")

    # Attach index so threaded results can be re-ordered correctly
    rows = [{"_idx": i, **row} for i, row in df.iterrows()]

    if USE_THREADS:
        ordered = run_threaded(rows, args.delay, N_THREADS)
    else:
        ordered = run_single(rows, args.delay)

    na_indices = {idx for idx, pred in ordered if pred == "NA"}
    if na_indices:
        na_rows = [row for row in rows if row["_idx"] in na_indices]
        print(f"\nRetrying {len(na_rows)} NA row(s) sequentially...")
        retry_results = dict(run_single(na_rows, args.delay))
        ordered = [(idx, retry_results.get(idx, pred)) for idx, pred in ordered]
        still_na = sum(1 for _, pred in ordered if pred == "NA")
        recovered = len(na_indices) - still_na
        print(f"Retry complete — recovered: {recovered}  |  still NA: {still_na}")
    
    preds   = [pred for _, pred in ordered]
    errors  = preds.count("NA")

    df["who_is_human"] = preds
    df.to_csv(args.output, index=False)

    print(f"\n✓ Predictions saved to: {args.output}")
    print(f"  Total : {len(preds)}  |  A: {preds.count('A')}  |  B: {preds.count('B')}  |  NA: {errors}")
    if USE_THREADS:
        print(f"  Threads used: {N_THREADS}")
    if errors:
        print(f"{errors} row(s) errored and defaulted to 'NA'")
    print("\nNext step: submit your predictions CSV to the leaderboard at https://huggingface.co/spaces/roc-hci/TuringBench-2-Leaderboard")


if __name__ == "__main__":
    main()