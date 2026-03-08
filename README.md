# MT-Bench speed benchmark (ComposerSpeedTest)

This project runs the [MT-Bench](https://github.com/mtbench101/mt-bench-101) multi-turn conversation benchmark and measures **latency** and **output speed** so you can compare older models (e.g. GPT-3.5-turbo) with current ones.

## Composer vs Cloud Agents API

- **[Composer](https://cursor.com/blog/composer)** is Cursor’s own fast, frontier agent model used in the IDE. There is **no separate “Composer API”** that accepts a chat message and returns a response; it powers the in-app agent only.
- The **[Cloud Agents API](https://cursor.com/docs/cloud-agent/api/endpoints)** is a **proxy** you can use from code: you launch an agent (with a prompt and a GitHub repo), add follow-up prompts, then fetch the conversation. It’s available in **Beta on all plans** (including Pro). Create an API key from [Cursor Dashboard → Integrations](https://cursor.com/dashboard?tab=integrations). The model used is your account default (or you pass a model ID from `GET /v0/models`); if your Cursor default is set to Composer, using the API with the default model can give you Composer-backed answers. The API does not list “Composer” as a distinct model ID.

## Data

- **`mt_bench/question.jsonl`** — Multi-turn questions (81 items): each line has `question_id`, `category`, and `turns` (list of user messages).
- **`mt_bench/model_answer/gpt-3.5-turbo.jsonl`** — Reference answers from the original benchmark (no latency metrics).

## Option A: OpenAI API (recommended for latency metrics)

The benchmark can use the **OpenAI API** for direct, fast runs with token-level metrics (time to first token, tokens/sec).

### 1. Setup

```bash
cd ComposerSpeedTest
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. API key

Add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Optional: set `MT_BENCH_MODEL` in `.env` (default is `gpt-4o-mini`).

### 3. Run

```bash
# Run all questions with default model (gpt-4o-mini)
python run_mt_bench.py

# Use a specific model (e.g. latest)
python run_mt_bench.py --model gpt-4o

# Run only first 5 questions (for testing)
python run_mt_bench.py --limit 5
```

Outputs:

- **`mt_bench/model_answer/<model>.jsonl`** — One line per question: model replies and **metrics** (time to first token, total latency, approx output tokens, tokens/sec).
- **`mt_bench/model_answer/<model>_summary.txt`** — Summary: avg TTFT, avg latency, total tokens, throughput.

## Metrics recorded

For each question (multi-turn conversation) the script records:

- **Time to first token (s)** — Latency until the first token of the first turn.
- **Total latency (s)** — End-to-end time for all turns.
- **Output tokens (approx)** — Estimated from response length.
- **Tokens per second** — Output throughput.

## Comparing two runs

To compare latency/speed between two model runs (e.g. gpt-3.5-turbo vs a new run):

```bash
python compare_bench_results.py mt_bench/model_answer/gpt-3.5-turbo.jsonl mt_bench/model_answer/gpt-4o-mini.jsonl
```

Note: the included `gpt-3.5-turbo.jsonl` doesn’t have a `metrics` field. The comparison script will show metrics only when both files were produced by `run_mt_bench.py`. To compare quality, open the two JSONL files and compare the `choices[0].turns` text.

## Option B: Cursor Cloud Agents API (proxy for Composer / default model)

If you want answers from **Cursor’s default model** (e.g. Composer if that’s your default), use the Cloud Agents API. Each question is run as one cloud agent: first turn = launch prompt, second turn = follow-up. You need a **GitHub repo** the agent can use (it won’t create a PR).

1. **API key**: [Cursor Dashboard → Integrations](https://cursor.com/dashboard?tab=integrations) → create key (Beta, all plans).
2. **Repo**: Any GitHub repo you have access to (e.g. this project if it’s on GitHub).

```bash
# In .env add:
CURSOR_API_KEY=your_cloud_agents_api_key
CURSOR_AGENT_REPO=https://github.com/your-username/ComposerSpeedTest

# Run (uses account default model; may be Composer)
python run_mt_bench_cursor_cloud.py --limit 3

# Or request a specific model (see GET /v0/models for IDs)
python run_mt_bench_cursor_cloud.py --model claude-4.5-sonnet-thinking --limit 3
```

Output: `mt_bench/model_answer/cursor-cloud-default.jsonl` (or `cursor-cloud-<model>.jsonl`). The API doesn’t return token-level metrics, so only total wall-clock time per question is recorded. Cloud agents are tuned for coding tasks on a repo, so some MT-Bench answers may be more task-oriented.

## Reference

- MT-Bench: [ACL 2024](https://github.com/mtbench101/mt-bench-101) — benchmarking multi-turn dialogue with older models.
- This setup lets you run the same questions through current models and compare both **answer content** and **speed/latency** to see progress over time.
