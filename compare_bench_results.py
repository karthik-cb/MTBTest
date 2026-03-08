#!/usr/bin/env python3
"""
Compare latency and speed metrics between two MT-Bench model_answer runs.

Usage:
  python compare_bench_results.py mt_bench/model_answer/gpt-3.5-turbo.jsonl mt_bench/model_answer/gpt-4o-mini.jsonl

Note: gpt-3.5-turbo.jsonl from the repo may not have a "metrics" field (old format).
This script reports what it can: for new runs, it shows metrics; for old runs, it shows counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_answer_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two MT-Bench model_answer JSONL files")
    parser.add_argument("left", type=Path, help="First model answers (e.g. gpt-3.5-turbo.jsonl)")
    parser.add_argument("right", type=Path, help="Second model answers (e.g. gpt-4o-mini.jsonl)")
    args = parser.parse_args()

    left = load_answer_records(args.left)
    right = load_answer_records(args.right)

    if not left or not right:
        print("One or both files are empty.", file=sys.stderr)
        sys.exit(1)

    by_id_left = {r["question_id"]: r for r in left}
    by_id_right = {r["question_id"]: r for r in right}
    common = sorted(set(by_id_left) & set(by_id_right))

    name_left = args.left.stem
    name_right = args.right.stem

    print(f"Comparison: {name_left} vs {name_right}")
    print(f"Questions in both: {len(common)}")
    print()

    has_metrics_left = "metrics" in left[0]
    has_metrics_right = "metrics" in right[0]

    if has_metrics_left and has_metrics_right:
        ttft_left = []
        ttft_right = []
        lat_left = []
        lat_right = []
        tps_left = []
        tps_right = []
        for qid in common:
            L = by_id_left[qid]["metrics"]
            R = by_id_right[qid]["metrics"]
            ttft_left.append(L["time_to_first_token_s"])
            ttft_right.append(R["time_to_first_token_s"])
            lat_left.append(L["total_latency_s"])
            lat_right.append(R["total_latency_s"])
            tps_left.append(L["overall_tokens_per_second"])
            tps_right.append(R["overall_tokens_per_second"])

        n = len(common)
        print("Metric                    | {:20s} | {:20s}".format(name_left[:20], name_right[:20]))
        print("-" * 60)
        print("Avg time to first token   | {:20.4f} | {:20.4f}".format(
            sum(ttft_left) / n, sum(ttft_right) / n))
        print("Avg total latency (s)     | {:20.4f} | {:20.4f}".format(
            sum(lat_left) / n, sum(lat_right) / n))
        print("Avg tokens/sec            | {:20.2f} | {:20.2f}".format(
            sum(tps_left) / n, sum(tps_right) / n))
        tot_tok_left = sum(
            by_id_left[qid]["metrics"]["total_output_tokens_approx"] for qid in common
        )
        tot_tok_right = sum(
            by_id_right[qid]["metrics"]["total_output_tokens_approx"] for qid in common
        )
        print("Total output tokens (app.) | {:20d} | {:20d}".format(tot_tok_left, tot_tok_right))
    else:
        print("Metrics available only for runs from run_mt_bench.py (with 'metrics' field).")
        print(f"{name_left}: metrics={has_metrics_left}, answers={len(left)}")
        print(f"{name_right}: metrics={has_metrics_right}, answers={len(right)}")


if __name__ == "__main__":
    main()
