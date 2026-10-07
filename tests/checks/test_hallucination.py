import json

import pytest

from guardrails.checks import hallucination as h
from guardrails.core import Context
from guardrails.llm import LLMResponse

CHUNKS = [
    "## UPI limit\nThe per transaction UPI limit is 100000 INR and the daily limit is 100000 INR."
]


def fake(claims):
    async def _complete(model, prompt, system=None, json_mode=False, max_tokens=1024):
        return LLMResponse(
            text=json.dumps({"claims": claims}), model=model,
            input_tokens=10, output_tokens=5, cost_usd=0.0,
        )

    return _complete


async def test_supported_allows(monkeypatch):
    monkeypatch.setattr(h, "complete", fake([{"text": "limit is 1 lakh", "supported": True}]))
    v = await h.HallucinationCheck().check(
        '{"answer":"The UPI limit is 1 lakh per day."}', Context(retrieved_chunks=CHUNKS)
    )
    assert v.action == "allow" and v.score == 0.0


async def test_unsupported_blocks(monkeypatch):
    monkeypatch.setattr(h, "complete", fake([
        {"text": "limit is 1 lakh", "supported": True},
        {"text": "limit resets at noon", "supported": False},
    ]))
    v = await h.HallucinationCheck(max_unsupported_ratio=0.2).check(
        "x", Context(retrieved_chunks=CHUNKS)
    )
    assert v.action == "block" and v.score == 0.5


async def test_no_context_allows():
    v = await h.HallucinationCheck().check("anything", Context())
    assert v.action == "allow" and v.reason == "no_context"


async def test_bad_judge_json_raises_so_pipeline_fails_open(monkeypatch):
    async def bad(*a, **k):
        return LLMResponse(text="not json", model="m", input_tokens=1, output_tokens=1, cost_usd=0)

    monkeypatch.setattr(h, "complete", bad)
    with pytest.raises(ValueError):
        await h.HallucinationCheck().check("x", Context(retrieved_chunks=CHUNKS))


def test_extract_answer():
    assert h.extract_answer('{"answer": "hi", "sources": []}') == "hi"
    assert h.extract_answer("plain") == "plain"
