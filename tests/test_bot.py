import json

from bankassist import bot
from bankassist.retriever import Hit
from guardrails.llm import LLMResponse

HITS = [
    Hit(text="UPI\n## Daily limit\nThe daily UPI limit is 100000 INR.", source="upi.md", score=0.8)
]


async def test_answer_sends_context_and_question(monkeypatch):
    seen = {}

    async def fake(model, prompt, system=None, json_mode=False, max_tokens=1024):
        seen.update(model=model, prompt=prompt, system=system, json_mode=json_mode)
        return LLMResponse(
            text=json.dumps({"answer": "1 lakh", "sources": ["upi.md"], "confidence": 0.9}),
            model=model, input_tokens=1, output_tokens=1, cost_usd=0.0,
        )

    monkeypatch.setattr(bot, "complete", fake)
    r = await bot.answer("What is the UPI limit?", HITS)
    assert "100000" in seen["prompt"] and "What is the UPI limit?" in seen["prompt"]
    assert seen["json_mode"] is True and seen["system"] == bot.SYSTEM_PROMPT
    assert bot.BotAnswer.model_validate_json(r.text).sources == ["upi.md"]


async def test_repair_includes_error(monkeypatch):
    seen = {}

    async def fake(model, prompt, system=None, json_mode=False, max_tokens=1024):
        seen["prompt"] = prompt
        return LLMResponse(text="{}", model=model, input_tokens=1, output_tokens=1, cost_usd=0.0)

    monkeypatch.setattr(bot, "complete", fake)
    out = await bot.repair("{bad", "Expecting property name")
    assert out == "{}" and "Expecting property name" in seen["prompt"] and "{bad" in seen["prompt"]


def test_prompt_cites_sources():
    p = bot.build_prompt("q", HITS)
    assert "[upi.md]" in p
