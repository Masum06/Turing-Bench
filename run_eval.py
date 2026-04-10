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
HF_DATASET_PATH = "hf://datasets/roc-hci/Turing-Bench/turing_bench_public_shuffled.csv"
HF_SPLIT        = "train"


def load_data() -> pd.DataFrame:
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


def test(input_path: str | None = None, n_rows: int = 3, output: str = "predictions.csv"):
    """
    Smoke-test predict() on the first `n_rows` rows without running the full dataset.
    Prints results to stdout and saves verdicts to `output`.
    """
    df = load_data()
    df = df.head(n_rows)
    print(f"Testing on {len(df)} row(s)...\n")

    results = []

    for i, row in df.iterrows():
        print(f"--- Row {i} ---")
        try:
            parsed = predict(str(row["dialogueA"]), str(row["dialogueB"]))

            if parsed is None:
                print(f"  ✗ parse_json() returned None — raw reply:\n{parsed}\n")
                continue

            verdict = parsed["result"]["verdict"]
            if verdict not in ("A", "B"):
                print(f"  ✗ Bad verdict: {verdict!r} (must be 'A' or 'B')")
            else:
                print(f"  ✓ Verdict: {verdict}")
                results.append(verdict)

        except NotImplementedError:
            print("  ✗ predict() is not implemented yet.")
            return
        except KeyError as exc:
            print(f"  ✗ Missing key in parsed response: {exc}")
            print(f"     Parsed output was: {parsed}")
        except Exception as exc:
            print(f"  ✗ Unexpected error: {exc}")

        print()

    out_df = pd.DataFrame({"who_is_human": results})
    out_df.to_csv(output, index=False)
    print(f"✓ Predictions saved to: {output}")
    print(f"  Total: {len(results)}  |  A: {results.count('A')}  |  B: {results.count('B')}")
    print("\nTest complete.")

if __name__ == "__main__":
    main()