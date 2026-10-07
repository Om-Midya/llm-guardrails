from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from guardrails.pipeline import PipelineResult

CHECK_TOTAL = Counter("guardrail_check_total", "Verdicts by check", ["check", "action", "mode"])
CHECK_ERRORS = Counter("guardrail_check_errors_total", "Checks that raised", ["check"])
CHECK_LATENCY = Histogram(
    "guardrail_check_latency_seconds", "Per-check latency", ["check"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
OVERHEAD = Histogram(
    "guardrail_request_overhead_seconds", "Stage overhead", ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
LLM_COST = Counter("llm_cost_usd_total", "LLM spend", ["model"])
REQUESTS = Counter("chat_requests_total", "Chat requests", ["outcome"])
CONTENT_TYPE = CONTENT_TYPE_LATEST


def record(stage: str, result: PipelineResult) -> None:
    for v in result.verdicts:
        CHECK_TOTAL.labels(v.check, v.action, "shadow" if v.shadow else "enforce").inc()
        CHECK_LATENCY.labels(v.check).observe(v.latency_ms / 1000)
        if v.reason.startswith("check_error"):
            CHECK_ERRORS.labels(v.check).inc()
    OVERHEAD.labels(stage).observe(result.overhead_ms / 1000)


def render() -> bytes:
    return generate_latest()
