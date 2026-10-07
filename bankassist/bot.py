from __future__ import annotations

from pydantic import BaseModel, Field

from bankassist.retriever import Hit
from guardrails.llm import LLMResponse, app_model, complete


class BotAnswer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


FALLBACK = {
    "answer": (
        "I could not produce a reliable answer. "
        "Please contact Nova Bank customer care on 1800-123-4567."
    ),
    "sources": [],
    "confidence": 0.0,
}

SYSTEM_PROMPT = (
    "You are BankAssist, the support assistant for Nova Bank. Answer only from the CONTEXT. "
    "If the context does not contain the answer, say you do not have that information and point "
    "the customer to 1800-123-4567. Never ask for or repeat OTPs, PINs, passwords, or full card "
    "numbers. Keep answers under 120 words. Reply with JSON only: "
    '{"answer": str, "sources": [file names you used], "confidence": number 0 to 1}'
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{h.source}]\n{h.text}" for h in hits)
    return f"CONTEXT:\n{ctx}\n\nQUESTION:\n{question}"


async def answer(question: str, hits: list[Hit]) -> LLMResponse:
    return await complete(
        app_model(), build_prompt(question, hits), system=SYSTEM_PROMPT,
        json_mode=True, max_tokens=400,
    )


async def repair(bad_text: str, error: str) -> str:
    prompt = (
        "The following text was supposed to be JSON matching "
        '{"answer": str, "sources": [str], "confidence": number} but failed validation.\n'
        f"ERROR: {error}\nTEXT:\n{bad_text}\n\nReturn only the corrected JSON."
    )
    resp = await complete(app_model(), prompt, json_mode=True, max_tokens=400)
    return resp.text
