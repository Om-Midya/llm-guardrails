# Guardrails and Safety Layer for an LLM Product (Project 10)

Date: 2026-10-07
Status: approved by owner. Next step is the implementation plan.

## 1. Goal

Build a composable input and output filtering layer that wraps any LLM endpoint. Prove it works with a hand-written red-team eval set that runs in CI. Ship it as a live public service with a demo banking support bot behind it.

The grader rubric this spec targets:

- Live public URL on Hugging Face Spaces.
- One real domain (banking support) and one hand-written eval set.
- Numbers in the README: catch rate, false-positive rate, p50 and p99 overhead, cost per request.
- CI that blocks a pull request on eval regression, with a screenshot.
- Total spend under 20 USD.

## 2. Non-goals

- A vector database. Retrieval over the FAQ corpus is in-process.
- A full RAG evaluation. Retrieval quality is not graded here.
- Multi-tenant auth, billing, or rate limiting.
- Fine-tuning any model.
- A general-purpose dashboard. The chat UI shows per-request check results. Metrics go to Prometheus text and Langfuse.

## 3. System overview

```
user -> POST /chat -> GuardrailPipeline.run_input -> BankAssist (retrieve + Gemini Flash)
     <-            <- GuardrailPipeline.run_output <-
```

Every request produces one Langfuse trace with one span per check. The active policy file hash is attached to the trace.

The guardrail layer is also exposed standalone as `POST /guard/input` and `POST /guard/output` so any other application can call it without using BankAssist.

## 4. Components

### 4.1 `guardrails/` package

`guardrails/core.py`

- `Action = Literal["allow", "block", "redact"]`
- `Verdict(check: str, action: Action, score: float, reason: str, rewritten_text: str | None, latency_ms: float, shadow: bool)`
- `Context(request_id: str, retrieved_chunks: list[str], schema: type[BaseModel] | None, policy_hash: str)`
- `BaseCheck` abstract class with `name: str`, `stage: Literal["input", "output"]`, `async def check(self, text: str, ctx: Context) -> Verdict`.
- `CHECK_REGISTRY: dict[str, type[BaseCheck]]` populated by a `@register` decorator.

`guardrails/policy.py`

- `CheckPolicy(enabled: bool, mode: Literal["enforce", "shadow"], params: dict)`
- `Policy(version: str, name: str, input: dict[str, CheckPolicy], output: dict[str, CheckPolicy])`
- `load_policy(path) -> tuple[Policy, str]` returns the parsed policy and a sha256 hash of the file bytes.
- Validation: every key under `input` and `output` must exist in `CHECK_REGISTRY` with the matching stage. Unknown check names fail loading.

`guardrails/pipeline.py`

- `GuardrailPipeline(policy: Policy, policy_hash: str)`
- `async run_input(text, ctx) -> PipelineResult`
- `async run_output(text, ctx) -> PipelineResult`
- `PipelineResult(final_text: str, blocked: bool, blocked_by: str | None, verdicts: list[Verdict], overhead_ms: float)`
- Behavior: local checks run concurrently with `asyncio.gather`. LLM-backed checks run after local checks, so a cheap block skips an expensive call. In enforce mode, the first `block` sets `blocked=True` and `final_text` to a fixed refusal message. In shadow mode, a `block` is recorded with `shadow=True` and does not change `final_text`. `redact` verdicts apply `rewritten_text` in sequence, in both modes, because redaction is non-destructive to the request.
- Order within a stage is the order in the policy file.

`guardrails/checks/` one module per check.

Input checks:

| name | method | params |
|------|--------|--------|
| `prompt_injection` | regex pattern list plus `protectai/deberta-v3-base-prompt-injection-v2` on CPU | `threshold` (default 0.8), `use_model` (bool) |
| `jailbreak` | curated regex list (DAN, "ignore previous", role-play, developer mode, hypothetical framing) | `threshold` on count of matched patterns (default 1) |
| `pii_input` | Microsoft Presidio with custom recognizers: Indian PAN, Aadhaar (with Verhoeff check), IFSC, bank account number, card number (Luhn), Indian phone, email | `entities` list, `action` (`redact` default) |
| `topic` | embedding cosine similarity against FAQ corpus centroid plus deny-list regex | `min_similarity` (default 0.35), `deny_patterns` |

Output checks:

| name | method | params |
|------|--------|--------|
| `schema` | Pydantic validation of model JSON output. On failure, one repair call to the app model with the error, then fallback to a safe default object | `max_repairs` (default 1) |
| `hallucination` | Claude Haiku 4.5 judge splits the answer into claims and scores each as supported or not by `ctx.retrieved_chunks`. Score is the fraction unsupported | `max_unsupported_ratio` (default 0.2) |
| `toxicity` | `unitary/toxic-bert` on CPU | `threshold` (default 0.7) |
| `pii_output` | same Presidio engine as `pii_input` | `entities`, `action` |
| `secret_leak` | regex set (AWS keys, Google API keys, GitHub tokens, JWTs, PEM private keys, generic `key=`) plus Shannon entropy on candidate tokens | `entropy_threshold` (default 4.0) |

Model-backed checks (`prompt_injection` model path, `toxicity`, `hallucination`, `topic` embeddings) load lazily on first use and are cached as module singletons.

### 4.2 `bankassist/` package

- `corpus/` about 40 markdown files written by us: savings accounts, cards, loans, KYC, UPI, disputes, fees, branch hours, lost card, NEFT and IMPS limits, FD and RD, nominee rules.
- `retriever.py`: loads corpus, chunks by heading, embeds with `sentence-transformers/all-MiniLM-L6-v2`, returns top 4 chunks by cosine similarity. Index is built at startup and cached to disk.
- `bot.py`: builds the prompt from chunks, calls Gemini 2.5 Flash, asks for JSON `{"answer": str, "sources": list[str], "confidence": float}`.
- `llm.py`: thin async clients for Gemini (app model) and Anthropic (judge), each returning text, input tokens, output tokens, and cost in USD from a price table.

### 4.3 `app/` package

FastAPI application.

| route | purpose |
|-------|---------|
| `GET /` | single-page chat UI |
| `POST /chat` | `{message}` -> guarded BankAssist answer plus verdicts |
| `POST /guard/input` | `{text}` -> `PipelineResult` |
| `POST /guard/output` | `{text, context_chunks?, schema_name?}` -> `PipelineResult` |
| `GET /policy` | active policy, version, hash |
| `GET /metrics` | Prometheus text format |
| `GET /health` | liveness |

Prometheus metrics: `guardrail_check_total{check, action, mode}`, `guardrail_check_latency_seconds{check}` histogram, `guardrail_request_overhead_seconds` histogram, `llm_cost_usd_total{model}` counter.

Langfuse: one trace per `/chat`, spans per check and per LLM call, metadata `policy_version`, `policy_hash`, `shadow_blocks`.

UI: a message box, the answer, and a verdict table with columns check, action, score, latency, and shadow badge. If any shadow verdict is `block`, the UI shows a banner that reads "shadow block".

### 4.4 `policies/`

- `v1.yaml`: all checks enabled, all enforce except `hallucination` in shadow.
- `v2.yaml`: `hallucination` promoted to enforce. Used to show the rollout story.
- The active file is chosen by env var `GUARDRAILS_POLICY` (default `policies/v1.yaml`).

### 4.5 `evals/`

- `redteam_input.jsonl`: about 60 prompts. Fields: `id`, `category` (`injection`, `jailbreak`, `pii`, `offtopic`), `text`, `expected_action`, `expected_check`, `notes`.
- `redteam_output.jsonl`: about 20 model outputs. Categories `hallucination`, `toxicity`, `pii_leak`, `secret_leak`, `bad_json`. Each carries `context_chunks` where relevant.
- `benign.jsonl`: about 50 realistic banking questions that must be allowed, including ones that mention amounts, dates, and partial numbers to pressure the false-positive rate.
- `thresholds.yaml`: per check `min_catch_rate` (default 0.90) and `max_false_positive_rate` (default 0.05). A global `max_catch_rate_drop_vs_main` of 0.02.
- `results/latest.json` written by the eval script, and `results/main.json` committed from the last run on `main` for regression comparison.

### 4.6 `scripts/`

- `run_eval.py`: runs the three eval files through the pipeline. It prints a markdown table per check with catch rate, false-positive rate, p50, p99, and cost. It writes `results/latest.json`. If any threshold fails, it exits non-zero. Flag `--record` records LLM responses to `evals/cassettes/` and `--replay` uses them. CI uses `--replay`.
- `build_index.py`: builds and caches the retriever index.
- `load_test.py`: Locust file hitting `/guard/input` with benign prompts, used once to publish throughput.

### 4.7 `.github/workflows/ci.yml`

Jobs on every pull request and push to `main`:

1. `test`: `ruff check`, `pytest`.
2. `eval`: `python scripts/run_eval.py --replay`, uploads `results/latest.json` as an artifact, posts the table as a PR comment, fails on threshold breach or regression against `results/main.json`.

Model weights are cached with `actions/cache` keyed on the model names.

### 4.8 Deployment

`Dockerfile` based on `python:3.12-slim`, installs CPU torch, pre-downloads the three HF models at build time, runs `uvicorn app.main:app --port 7860`. Hugging Face Space of type Docker. Secrets: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

## 5. Data flow for `POST /chat`

1. Create `request_id`, start Langfuse trace.
2. `run_input(message)`. If blocked, return refusal and verdicts. Redactions replace the message text.
3. `retriever.search(redacted_message)` -> chunks.
4. `bot.answer(redacted_message, chunks)` -> raw JSON text, token counts, cost.
5. `run_output(raw_text, ctx with chunks and schema=BotAnswer)`. `schema` runs first and can repair the JSON. If blocked, return refusal.
6. Return `{answer, sources, verdicts, policy_version, overhead_ms, cost_usd}`.

## 6. Error handling

- A check that raises is recorded as `Verdict(action="allow", score=0, reason="check_error: ...")` and a counter `guardrail_check_errors_total{check}` increments. A broken check never blocks traffic and never crashes the request. This is a deliberate fail-open choice for availability. The README states it and notes that `secret_leak` and `pii_output` are the exceptions and fail closed.
- LLM call failures in BankAssist return HTTP 502 with the input verdicts included.
- Policy load failure at startup aborts startup with a clear message.

## 7. Testing

- `tests/checks/test_<check>.py`: at least 5 positive and 5 negative cases per check, local checks only, no network.
- `tests/test_policy.py`: loads `v1.yaml`, rejects an unknown check, hash is stable.
- `tests/test_pipeline.py`: enforce blocks and shadow passes, redaction applies, error in a check fails open, concurrency does not reorder verdicts.
- `tests/test_api.py`: FastAPI `TestClient` with BankAssist LLM mocked.
- `scripts/run_eval.py --replay` is the integration test and runs in CI.

## 8. Numbers the README must show

- Per check: catch rate, false-positive rate, p50 and p99 latency, all from `results/latest.json`.
- Guardrail overhead p50 and p99 for local checks only and for local plus LLM checks.
- Cost per `/chat` request in USD, split by app model and judge.
- Locust throughput on `/guard/input` on the Space.
- Screenshot of a blocked PR and of a Langfuse trace.

## 9. Budget estimate

- HF models on CPU: 0 USD.
- Gemini 2.5 Flash: eval set and dev chatter, under 1 USD.
- Claude Haiku 4.5 judge: about 100 outputs times about 5 runs, under 2 USD.
- Langfuse Cloud free tier, HF Spaces free tier, GitHub Actions free tier.

## 10. Extensions beyond the brief

Included: standalone `/guard/*` endpoints, Prometheus metrics, Locust load test, two-policy rollout story, cassette-based deterministic CI.

Deferred: Streamlit shadow-mode dashboard, multiple judge models, async background judge scoring on live traffic.
