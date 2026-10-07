from __future__ import annotations

import json

from guardrails.checks.schema import unfence
from guardrails.core import BaseCheck, Context, Verdict, register
from guardrails.llm import complete, judge_model

JUDGE_SYSTEM = (
    "You are a strict fact checker for a bank's support assistant. "
    "Split the ANSWER into atomic factual claims. For each claim decide if it is fully supported "
    "by the CONTEXT. Greetings, apologies, and offers to help are not claims. "
    'Reply with JSON only: {"claims": [{"text": str, "supported": bool}]}'
)


def build_judge_prompt(answer: str, chunks: list[str]) -> str:
    ctx = "\n\n---\n\n".join(chunks)
    return f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"


def extract_answer(text: str) -> str:
    try:
        obj = json.loads(unfence(text))
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            return obj["answer"]
    except ValueError:
        pass
    return text


@register
class HallucinationCheck(BaseCheck):
    name = "hallucination"
    stage = "output"
    is_llm = True

    async def check(self, text: str, ctx: Context) -> Verdict:
        if not ctx.retrieved_chunks:
            return self.allow(0.0, "no_context")
        ratio_max = float(self.params.get("max_unsupported_ratio", 0.2))
        resp = await complete(
            judge_model(),
            build_judge_prompt(extract_answer(text), ctx.retrieved_chunks),
            system=JUDGE_SYSTEM,
            json_mode=True,
            max_tokens=800,
        )
        data = json.loads(unfence(resp.text))
        claims = data.get("claims", [])
        if not claims:
            return self.allow(0.0, "no claims")
        unsupported = [c["text"] for c in claims if not c.get("supported", False)]
        score = len(unsupported) / len(claims)
        reason = f"{len(unsupported)}/{len(claims)} unsupported"
        if unsupported:
            reason += f": {unsupported[0][:80]}"
        return self.block(score, reason) if score > ratio_max else self.allow(score, reason)
