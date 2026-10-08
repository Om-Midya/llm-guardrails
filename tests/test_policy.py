import hashlib
from pathlib import Path

import pytest

from guardrails.core import CHECK_REGISTRY, BaseCheck, register
from guardrails.policy import PolicyError, load_policy


@pytest.fixture
def dummy_checks():
    @register
    class In(BaseCheck):
        name = "dummy_in"
        stage = "input"

        async def check(self, text, ctx):
            return self.allow()

    @register
    class Out(BaseCheck):
        name = "dummy_out"
        stage = "output"

        async def check(self, text, ctx):
            return self.allow()

    yield
    CHECK_REGISTRY.pop("dummy_in")
    CHECK_REGISTRY.pop("dummy_out")


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "p.yaml"
    p.write_text(body)
    return p


def test_loads_and_hashes(tmp_path, dummy_checks):
    p = write(
        tmp_path,
        "version: '1'\nname: t\ninput:\n  dummy_in: {mode: shadow}\noutput:\n  dummy_out: {}\n",
    )
    policy, h = load_policy(p)
    assert policy.input["dummy_in"].mode == "shadow"
    assert policy.output["dummy_out"].enabled is True
    assert h == hashlib.sha256(p.read_bytes()).hexdigest()


def test_unknown_check_rejected(tmp_path, dummy_checks):
    p = write(tmp_path, "version: '1'\nname: t\ninput:\n  nope: {}\noutput: {}\n")
    with pytest.raises(PolicyError, match="unknown check 'nope'"):
        load_policy(p)


def test_wrong_stage_rejected(tmp_path, dummy_checks):
    p = write(tmp_path, "version: '1'\nname: t\ninput:\n  dummy_out: {}\noutput: {}\n")
    with pytest.raises(PolicyError, match="'dummy_out' is an output check"):
        load_policy(p)


def test_shipped_policies_load():
    for name in ("v1", "v2"):
        policy, _ = load_policy(f"policies/{name}.yaml")
        assert set(policy.input) == {
            "prompt_injection", "prompt_injection_model", "jailbreak", "pii_input", "topic",
        }
        assert set(policy.output) == {
            "schema", "hallucination", "toxicity", "pii_output", "secret_leak",
        }
    v1, _ = load_policy("policies/v1.yaml")
    v2, _ = load_policy("policies/v2.yaml")
    assert v1.input["prompt_injection_model"].mode == "shadow"
    assert v1.input["prompt_injection"].mode == "enforce"
    assert v1.output["hallucination"].mode == "shadow"
    assert v2.output["hallucination"].mode == "enforce"
