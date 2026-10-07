from __future__ import annotations

import asyncio

from guardrails.core import BaseCheck, Context, Verdict, register
from guardrails.models import MAX_CHARS, toxicity_classifier


@register
class ToxicityCheck(BaseCheck):
    name = "toxicity"
    stage = "output"

    async def check(self, text: str, ctx: Context) -> Verdict:
        threshold = float(self.params.get("threshold", 0.7))
        label, score = await asyncio.to_thread(self._score, text[:MAX_CHARS])
        reason = f"{label} {score:.2f}"
        return self.block(score, reason) if score >= threshold else self.allow(score, reason)

    @staticmethod
    def _score(text: str) -> tuple[str, float]:
        scores = toxicity_classifier()(text)[0]
        top = max(scores, key=lambda s: s["score"])
        return top["label"], float(top["score"])
