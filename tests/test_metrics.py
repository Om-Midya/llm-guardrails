from app.metrics import record, render
from app.tracing import Tracer
from guardrails.core import Verdict
from guardrails.pipeline import PipelineResult


def test_record_counts_actions_and_errors():
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
    out = render().decode()
    assert 'guardrail_check_total{action="block",check="jailbreak",mode="enforce"} 1.0' in out
    assert 'guardrail_check_total{action="block",check="topic",mode="shadow"} 1.0' in out
    assert 'guardrail_check_errors_total{check="pii_input"} 1.0' in out
    assert 'guardrail_request_overhead_seconds_count{stage="input"} 1.0' in out


def test_tracer_is_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    t = Tracer()
    tr = t.start_trace("chat", input="hi", metadata={})
    with tr.span("check") as s:
        s.update(output="ok")
    tr.end(output="done", metadata={})
