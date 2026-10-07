from prometheus_client import REGISTRY

from app.metrics import record, render
from app.tracing import Tracer
from guardrails.core import Verdict
from guardrails.pipeline import PipelineResult


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_record_counts_actions_and_errors():
    before = {
        "jb": sample("guardrail_check_total", check="jailbreak", action="block", mode="enforce"),
        "topic": sample("guardrail_check_total", check="topic", action="block", mode="shadow"),
        "err": sample("guardrail_check_errors_total", check="pii_input"),
        "over": sample("guardrail_request_overhead_seconds_count", stage="input"),
    }
    res = PipelineResult(
        final_text="x", blocked=True, blocked_by="jailbreak", overhead_ms=3.0,
        shadow_blocks=["topic"],
        verdicts=[
            Verdict(check="jailbreak", action="block", latency_ms=1.0),
            Verdict(check="topic", action="block", shadow=True, latency_ms=2.0),
            Verdict(check="pii_input", action="allow", reason="check_error: boom", latency_ms=0.5),
        ],
    )
    record("input", res)
    jb = sample("guardrail_check_total", check="jailbreak", action="block", mode="enforce")
    topic = sample("guardrail_check_total", check="topic", action="block", mode="shadow")
    assert jb == before["jb"] + 1 and topic == before["topic"] + 1
    assert sample("guardrail_check_errors_total", check="pii_input") == before["err"] + 1
    assert sample("guardrail_request_overhead_seconds_count", stage="input") == before["over"] + 1
    assert b"guardrail_check_total" in render()


def test_tracer_is_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    t = Tracer()
    tr = t.start_trace("chat", input="hi", metadata={})
    with tr.span("check") as s:
        s.update(output="ok")
    tr.end(output="done", metadata={})
