from guardrails.core import CHECK_REGISTRY, BaseCheck, Context, Verdict, normalize, register


def test_register_adds_to_registry():
    @register
    class Dummy(BaseCheck):
        name = "dummy_for_test"
        stage = "input"

        async def check(self, text, ctx):
            return self.allow()

    assert CHECK_REGISTRY["dummy_for_test"] is Dummy
    del CHECK_REGISTRY["dummy_for_test"]


def test_helpers_build_verdicts():
    class C(BaseCheck):
        name = "c"
        stage = "output"

        async def check(self, text, ctx):
            return self.allow()

    c = C(threshold=0.5)
    assert c.params == {"threshold": 0.5}
    assert c.block(0.9, "bad") == Verdict(check="c", action="block", score=0.9, reason="bad")
    r = c.redact("x", 0.5, "pii")
    assert r.action == "redact" and r.rewritten_text == "x"


def test_normalize_strips_zero_width():
    assert normalize("i​gnore   previous") == "ignore previous"


def test_context_defaults():
    ctx = Context()
    assert ctx.retrieved_chunks == [] and ctx.schema is None
