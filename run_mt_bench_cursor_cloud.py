#!/usr/bin/env python3
"""
MT-Bench via Cursor Cloud Agents API.

Uses the [Cloud Agents API](https://cursor.com/docs/cloud-agent/api/endpoints) as a proxy:
each question is run as one cloud agent — turn 1 = launch prompt, turn 2+ = follow-ups.
The model used is your Cursor default (or pass --model), which may be Composer if set
as default in Cursor. The API does not expose "Composer" as a separate model ID; use
model "default" to use your account default.

Availability: Cloud Agents API is in Beta (All Plans) — create an API key from
Cursor Dashboard → Integrations: https://cursor.com/dashboard?tab=integrations

Requires:
  CURSOR_API_KEY  — Cloud Agents API key
  CURSOR_AGENT_REPO — GitHub repo URL (e.g. https://github.com/you/ComposerSpeedTest)

Usage:
  python run_mt_bench_cursor_cloud.py [--model default] [--limit N] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import base64
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

import urllib.request
import urllib.error


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
# Cursor Cloud Agents API (Basic Auth)
# ---------------------------------------------------------------------------

API_BASE = "https://api.cursor.com"


def _auth_header(api_key: str) -> str:
    b = base64.b64encode(f"{api_key}:".encode()).decode()
    return f"Basic {b}"


def _req(
    method: str,
    path: str,
    api_key: str,
    body: dict | None = None,
    query: str = "",
) -> dict:
    url = f"{API_BASE}{path}"
    if query:
        url += "?" + query
    headers = {
        "Authorization": _auth_header(api_key),
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("message", err_json.get("error", err_body))
        except Exception:
            msg = err_body or str(e)
        raise RuntimeError(f"API {method} {path}: {e.code} {msg}") from e


def launch_agent(
    api_key: str,
    prompt_text: str,
    repo: str,
    ref: str = "main",
    model: str | None = "default",
) -> str:
    body = {
        "prompt": {"text": prompt_text},
        "source": {"repository": repo, "ref": ref},
        "target": {"autoCreatePr": False},
    }
    if model is not None:
        body["model"] = model
    r = _req("POST", "/v0/agents", api_key, body)
    return r["id"]


def add_followup(api_key: str, agent_id: str, prompt_text: str) -> None:
    _req("POST", f"/v0/agents/{agent_id}/followup", api_key, {"prompt": {"text": prompt_text}})


def get_status(api_key: str, agent_id: str) -> str:
    r = _req("GET", f"/v0/agents/{agent_id}", api_key)
    return r.get("status", "")


def get_conversation(api_key: str, agent_id: str) -> list[dict]:
    r = _req("GET", f"/v0/agents/{agent_id}/conversation", api_key)
    return r.get("messages", [])


# ---------------------------------------------------------------------------
# Run one MT-Bench question via Cloud Agent (multi-turn = launch + follow-ups)
# ---------------------------------------------------------------------------

TERMINAL_STATES = ("FINISHED", "FAILED", "STOPPED")
POLL_INTERVAL = 5
MAX_POLL_ATTEMPTS = 120  # 10 min per run


def run_one_question(
    api_key: str,
    repo: str,
    ref: str,
    model: str | None,
    turns: list[str],
) -> tuple[list[str], dict]:
    """Run one multi-turn question. Returns (list of assistant replies, metrics)."""
    if not turns:
        return [], {"total_latency_s": 0, "notes": "no turns"}

    start = time.perf_counter()
    agent_id = launch_agent(api_key, turns[0], repo, ref, model)

    # Poll until agent finishes first turn
    for _ in range(MAX_POLL_ATTEMPTS):
        status = get_status(api_key, agent_id)
        if status in TERMINAL_STATES:
            break
        time.sleep(POLL_INTERVAL)
    else:
        raise RuntimeError(f"Agent {agent_id} did not finish in time")

    # Remaining turns as follow-ups
    for turn_text in turns[1:]:
        add_followup(api_key, agent_id, turn_text)
        for _ in range(MAX_POLL_ATTEMPTS):
            status = get_status(api_key, agent_id)
            if status in TERMINAL_STATES:
                break
            time.sleep(POLL_INTERVAL)
        else:
            raise RuntimeError(f"Agent {agent_id} did not finish follow-up in time")

    total_latency = time.perf_counter() - start
    messages = get_conversation(api_key, agent_id)
    replies = [m["text"] for m in messages if m.get("type") == "assistant_message"]

    metrics = {
        "total_latency_s": round(total_latency, 4),
        "time_to_first_token_s": None,
        "total_output_tokens_approx": None,
        "overall_tokens_per_second": None,
        "turns": [{"turn": i + 1, "total_latency_s": None} for i in range(len(turns))],
        "notes": "Cloud Agents API; no token-level metrics",
    }
    return replies, metrics


# ---------------------------------------------------------------------------
# Output (same schema as run_mt_bench.py for comparison)
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
        "model_id": model_id,
        "choices": [{"index": 0, "turns": turns}],
        "tstamp": time.time(),
        "metrics": metrics,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MT-Bench via Cursor Cloud Agents API (proxy for Composer/default model)"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CURSOR_AGENT_MODEL", "default"),
        help="Model ID or 'default' for account default (may be Composer)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max questions")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo", default=os.getenv("CURSOR_AGENT_REPO"), help="GitHub repo URL")
    parser.add_argument("--ref", default="main", help="Git ref (branch/tag)")
    args = parser.parse_args()

    api_key = os.getenv("CURSOR_API_KEY")
    repo = args.repo or os.getenv("CURSOR_AGENT_REPO")
    if not api_key:
        print("Set CURSOR_API_KEY (Cloud Agents API key from Cursor Dashboard → Integrations).", file=sys.stderr)
        sys.exit(1)
    if not repo:
        print("Set CURSOR_AGENT_REPO (e.g. https://github.com/you/ComposerSpeedTest).", file=sys.stderr)
        sys.exit(1)

    project_root, default_questions = get_paths()
    questions_path = project_root / "mt_bench" / "question.jsonl"
    output_dir = args.output_dir or (project_root / "mt_bench" / "model_answer")
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path, limit=args.limit)
    if not questions:
        print("No questions loaded.", file=sys.stderr)
        sys.exit(1)

    model_slug = f"cursor-cloud-{args.model.replace('/', '-')}"
    out_jsonl = output_dir / f"{model_slug}.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    print(f"Running MT-Bench via Cloud Agents: model={args.model}, repo={repo}, questions={len(questions)}")
    for i, q in enumerate(questions):
        qid = q["question_id"]
        turns = q["turns"]
        print(f"  [{i+1}/{len(questions)}] question_id={qid} turns={len(turns)}")
        try:
            model_param = None if args.model == "default" else args.model
            replies, metrics = run_one_question(
                api_key, repo, args.ref, model_param, turns
            )
            write_answer_record(out_jsonl, qid, model_slug, replies, metrics)
            time.sleep(2)  # gentle on rate limits
        except Exception as e:
            print(f"  Error question_id={qid}: {e}", file=sys.stderr)
            raise

    print(f"Wrote {out_jsonl}")


if __name__ == "__main__":
    main()
