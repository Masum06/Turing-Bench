import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from config import USE_THREADS, N_THREADS
from predict import predict

import pandas as pd
from tqdm import tqdm

#  Internals
HF_DATASET_PATH = "hf://datasets/roc-hci/Turing-Bench/turing_bench_public_shuffled.csv"
HF_SPLIT        = "train"


def load_json(s: str) -> dict | None:
    import json
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def parse_json(reply: str) -> dict | None:
    if not reply:
        print("Empty reply")
        return None

    reply = reply.strip()
    if reply.startswith("```json"):
        reply = reply[len("```json"):].strip()
        if reply.endswith("```"):
            reply = reply[:-3].strip()

    if not (reply.startswith("{") and reply.endswith("}")):
        print("Not JSON structure")
        return None

    try:
        return load_json(reply)
    except Exception:
        print("Error parsing JSON")
        return None


def load_data(input_path: str | None) -> pd.DataFrame:
    if input_path:
        print(f"Loading data from local file: {input_path}")
        df = pd.read_csv(input_path)
    else:
        print(f"Loading data from HuggingFace: {HF_DATASET_PATH}")
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("datasets package not found. Run: pip install datasets")
        ds = load_dataset("csv", data_files=HF_DATASET_PATH, split=HF_SPLIT)
        df = ds.to_pandas()

    missing = {"dialogueA", "dialogueB"} - set(df.columns)
    if missing:
        sys.exit(f"Input data is missing required columns: {missing}")

    return df


def run_single(rows: list[dict], delay: float) -> list[tuple[int, str]]:
    """Sequential inference with a progress bar."""
    results = []
    for row in tqdm(rows, desc="Running predictions (single-threaded)"):
        try:
            pred = parse_json(
                predict(str(row["dialogueA"]), str(row["dialogueB"]))
            )["result"]["verdict"]
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
            pred = parse_json(
                predict(str(row["dialogueA"]), str(row["dialogueB"]))
            )["result"]["verdict"]
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

    df = load_data(args.input)
    print(f"Loaded {len(df)} examples.\n")

    # Attach index so threaded results can be re-ordered correctly
    rows = [{"_idx": i, **row} for i, row in df.iterrows()]

    if USE_THREADS:
        ordered = run_threaded(rows, args.delay, N_THREADS)
    else:
        ordered = run_single(rows, args.delay)

    preds   = [pred for _, pred in ordered]
    errors  = preds.count("NA")

    out_df = pd.DataFrame({"who_is_human": preds})
    out_df.to_csv(args.output, index=False)

    print(f"\n✓ Predictions saved to: {args.output}")
    print(f"  Total : {len(preds)}  |  A: {preds.count('A')}  |  B: {preds.count('B')}  |  NA: {errors}")
    if USE_THREADS:
        print(f"  Threads used: {N_THREADS}")
    if errors:
        print(f"{errors} row(s) errored and defaulted to 'NA'")
    print("\nNext step: submit your predictions CSV to the leaderboard at https://huggingface.co/spaces/roc-hci/Turing-Bench-Leaderboard")

if __name__ == "__main__":
    main()