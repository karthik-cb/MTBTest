#!/usr/bin/env python3
"""
MT-Bench multi-turn benchmark runner.

Runs MT-Bench questions (multi-turn conversations) through an LLM API,
records latency and output speed, and writes answers in the same format
as the existing model_answer JSONL for comparison with gpt-3.5-turbo.

Usage:
  Set OPENAI_API_KEY in .env or environment, then:
  python run_mt_bench.py [--model gpt-4o] [--limit N] [--output-dir mt_bench/model_answer]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai python-dotenv", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

def get_paths(project_root: Path | None = None) -> tuple[Path, Path]:
    project_root = project_root or Path(__file__).resolve().parent
    questions_path = project_root / "mt_bench" / "question.jsonl"
    return project_root, questions_path


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
            if limit is not None and len(questions) >= limit:
                break
    return questions


# ---------------------------------------------------------------------------
# Multi-turn conversation with latency metrics
# ---------------------------------------------------------------------------

def run_multi_turn(
    client: OpenAI,
    model: str,
    turns: list[str],
) -> tuple[list[str], dict]:
    """
    Run a multi-turn conversation. Returns (list of assistant replies, metrics dict).

    Metrics: time_to_first_token_s, total_latency_s, output_tokens (approx), tokens_per_second.
    """
    messages: list[dict] = []
    all_replies: list[str] = []
    metrics_per_turn: list[dict] = []

    for i, user_content in enumerate(turns):
        messages.append({"role": "user", "content": user_content})

        start = time.perf_counter()
        first_token_at: float | None = None
        full_content: list[str] = []
        chunk_count = 0

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            chunk_count += 1
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            content = (delta.content or "") or ""
            if content and first_token_at is None:
                first_token_at = time.perf_counter()
            if content:
                full_content.append(content)

        end = time.perf_counter()
        reply = "".join(full_content)
        all_replies.append(reply)
        messages.append({"role": "assistant", "content": reply})

        # Approximate output tokens (OpenAI ~4 chars per token on average for English)
        approx_tokens = max(1, sum(len(s) for s in full_content) // 4)
        ttft = (first_token_at - start) if first_token_at is not None else (end - start)
        total_latency = end - start
        tps = approx_tokens / total_latency if total_latency > 0 else 0

        metrics_per_turn.append({
            "turn": i + 1,
            "time_to_first_token_s": round(ttft, 4),
            "total_latency_s": round(total_latency, 4),
            "output_tokens_approx": approx_tokens,
            "tokens_per_second": round(tps, 2),
        })

    # Aggregate metrics
    total_latency = sum(m["total_latency_s"] for m in metrics_per_turn)
    total_tokens = sum(m["output_tokens_approx"] for m in metrics_per_turn)
    ttft_first_turn = metrics_per_turn[0]["time_to_first_token_s"] if metrics_per_turn else 0

    metrics = {
        "turns": metrics_per_turn,
        "time_to_first_token_s": round(ttft_first_turn, 4),
        "total_latency_s": round(total_latency, 4),
        "total_output_tokens_approx": total_tokens,
        "overall_tokens_per_second": round(total_tokens / total_latency, 2) if total_latency > 0 else 0,
    }
    return all_replies, metrics


# ---------------------------------------------------------------------------
# Output format (matches mt_bench model_answer schema)
# ---------------------------------------------------------------------------

def write_answer_record(
    out_path: Path,
    question_id: int,
    model_id: str,
    turns: list[str],
    metrics: dict,
) -> None:
    record = {
        "question_id": question_id,
        "answer_id": None,  # optional; could generate if needed
        "model_id": model_id,
        "choices": [{"index": 0, "turns": turns}],
        "tstamp": time.time(),
        "metrics": metrics,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_report(out_path: Path, results: list[dict], model_id: str) -> None:
    if not results:
        return
    total_ttft = sum(r["metrics"]["time_to_first_token_s"] for r in results)
    total_lat = sum(r["metrics"]["total_latency_s"] for r in results)
    total_tok = sum(r["metrics"]["total_output_tokens_approx"] for r in results)
    n = len(results)
    lines = [
        f"# MT-Bench benchmark summary — {model_id}",
        f"Questions run: {n}",
        f"Avg time to first token (s): {total_ttft / n:.4f}",
        f"Avg total latency per question (s): {total_lat / n:.4f}",
        f"Total output tokens (approx): {total_tok}",
        f"Overall throughput (tokens/s): {total_tok / total_lat:.2f}" if total_lat > 0 else "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(filter(None, lines)) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run MT-Bench multi-turn benchmark")
    parser.add_argument(
        "--model",
        default=os.getenv("MT_BENCH_MODEL", "gpt-4o-mini"),
        help="OpenAI model (default: gpt-4o-mini; use gpt-4o for latest)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of questions to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for model_answer JSONL (default: mt_bench/model_answer)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to question.jsonl (default: mt_bench/question.jsonl)",
    )
    args = parser.parse_args()

    project_root, default_questions = get_paths()
    questions_path = args.questions or default_questions
    output_dir = args.output_dir or (project_root / "mt_bench" / "model_answer")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY in .env or environment.", file=sys.stderr)
        sys.exit(1)

    if not questions_path.exists():
        print(f"Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(questions_path, limit=args.limit)
    if not questions:
        print("No questions loaded.", file=sys.stderr)
        sys.exit(1)

    # Output file: one JSONL per model
    model_slug = args.model.replace("/", "-")
    out_jsonl = output_dir / f"{model_slug}.jsonl"
    if out_jsonl.exists():
        # Backup or overwrite: overwrite for a clean run
        out_jsonl.unlink()

    client = OpenAI(api_key=api_key)
    results: list[dict] = []

    print(f"Running MT-Bench: model={args.model}, questions={len(questions)}")
    for i, q in enumerate(questions):
        qid = q["question_id"]
        turns = q["turns"]
        print(f"  [{i+1}/{len(questions)}] question_id={qid} turns={len(turns)}")
        try:
            replies, metrics = run_multi_turn(client, args.model, turns)
            write_answer_record(out_jsonl, qid, args.model, replies, metrics)
            results.append({"question_id": qid, "metrics": metrics})
        except Exception as e:
            print(f"  Error question_id={qid}: {e}", file=sys.stderr)
            raise

    summary_path = output_dir / f"{model_slug}_summary.txt"
    write_summary_report(summary_path, results, args.model)
    print(f"Wrote {out_jsonl} and {summary_path}")


if __name__ == "__main__":
    main()
