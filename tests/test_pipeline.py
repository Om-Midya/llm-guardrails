import asyncio

import pytest

from guardrails.core import CHECK_REGISTRY, BaseCheck, Context, register
from guardrails.pipeline import REFUSAL, GuardrailPipeline
from guardrails.policy import CheckPolicy, Policy


@pytest.fixture
def checks():
    created = []

    def mk(
        name, stage, action="allow", is_llm=False, fail_closed=False,
        raises=False, rewrite=None, delay=0,
    ):
        class C(BaseCheck):
            async def check(self, text, ctx):
                await asyncio.sleep(delay)
                if raises:
                    raise RuntimeError("boom")
                if action == "block":
                    return self.block(1.0, "blocked")
                if action == "redact":
                    return self.redact(text.replace(rewrite[0], rewrite[1]), 0.5, "pii")
                return self.allow()

        C.name, C.stage, C.is_llm, C.fail_closed = name, stage, is_llm, fail_closed
        register(C)
        created.append(name)
        return C

    yield mk
    for n in created:
        CHECK_REGISTRY.pop(n, None)


def policy(**stages):
    return Policy(
        version="t", name="t",
        **{k: {n: CheckPolicy(**p) for n, p in v.items()} for k, v in stages.items()},
    )


async def test_enforce_blocks_and_shadow_passes(checks):
    checks("a_block", "input", "block")
    checks("b_block", "input", "block")
    p = policy(input={"a_block": {"mode": "shadow"}, "b_block": {"mode": "enforce"}})
    r = await GuardrailPipeline(p, "h").run_input("hi", Context())
    assert r.blocked and r.blocked_by == "b_block" and r.final_text == REFUSAL
    assert r.shadow_blocks == ["a_block"]
    assert [v.check for v in r.verdicts] == ["a_block", "b_block"]
    assert r.verdicts[0].shadow is True and r.verdicts[1].shadow is False


async def test_shadow_only_does_not_block(checks):
    checks("s", "input", "block")
    p = policy(input={"s": {"mode": "shadow"}})
    r = await GuardrailPipeline(p, "h").run_input("hi", Context())
    assert not r.blocked and r.final_text == "hi" and r.shadow_blocks == ["s"]


async def test_redact_rewrites_and_llm_sees_rewritten(checks):
    seen = {}
    checks("red", "input", "redact", rewrite=("1234", "<ACC>"))
    cls = checks("llm", "input", is_llm=True)

    async def spy(self, text, ctx):
        seen["text"] = text
        return self.allow()

    cls.check = spy
    p = policy(input={"red": {}, "llm": {}})
    r = await GuardrailPipeline(p, "h").run_input("acc 1234", Context())
    assert r.final_text == "acc <ACC>" and seen["text"] == "acc <ACC>"


async def test_local_block_skips_llm(checks):
    checks("blk", "input", "block")
    cls = checks("llm", "input", is_llm=True)
    calls = []

    async def spy(self, text, ctx):
        calls.append(1)
        return self.allow()

    cls.check = spy
    p = policy(input={"blk": {}, "llm": {}})
    r = await GuardrailPipeline(p, "h").run_input("x", Context())
    assert r.blocked and calls == [] and [v.check for v in r.verdicts] == ["blk"]


async def test_error_fails_open_unless_fail_closed(checks):
    checks("open", "output", raises=True)
    checks("closed", "output", raises=True, fail_closed=True)
    r = await GuardrailPipeline(policy(output={"open": {}}), "h").run_output("x", Context())
    assert not r.blocked and r.verdicts[0].reason.startswith("check_error")
    r = await GuardrailPipeline(policy(output={"closed": {}}), "h").run_output("x", Context())
    assert r.blocked and r.blocked_by == "closed"


async def test_disabled_check_not_run(checks):
    checks("off", "input", "block")
    p = policy(input={"off": {"enabled": False}})
    r = await GuardrailPipeline(p, "h").run_input("x", Context())
    assert not r.blocked and r.verdicts == []


async def test_local_checks_run_concurrently(checks):
    for n in ("c1", "c2", "c3"):
        checks(n, "input", delay=0.1)
    p = policy(input={"c1": {}, "c2": {}, "c3": {}})
    r = await GuardrailPipeline(p, "h").run_input("x", Context())
    assert r.overhead_ms < 250
    assert [v.check for v in r.verdicts] == ["c1", "c2", "c3"]


async def test_local_checks_rescan_text_rewritten_by_llm_check(checks):
    class Leak(BaseCheck):
        async def check(self, text, ctx):
            return self.block(1.0, "secret") if "AKIA" in text else self.allow()

    Leak.name, Leak.stage = "leak", "output"
    register(Leak)
    cls = checks("fixer", "output", is_llm=True)

    async def rewrite(self, text, ctx):
        return self.redact("repaired AKIA1234", 0.0, "repaired")

    cls.check = rewrite
    try:
        p = policy(output={"leak": {}, "fixer": {}})
        r = await GuardrailPipeline(p, "h").run_output("{bad json", Context())
        assert r.blocked and r.blocked_by == "leak"
    finally:
        CHECK_REGISTRY.pop("leak", None)


async def test_local_ms_is_wall_clock_not_sum(checks):
    for n in ("l1", "l2", "l3"):
        checks(n, "output", delay=0.1)
    p = policy(output={"l1": {}, "l2": {}, "l3": {}})
    r = await GuardrailPipeline(p, "h").run_output("x", Context())
    assert 90 < r.local_ms < 200
    assert sum(v.latency_ms for v in r.verdicts) > 250
