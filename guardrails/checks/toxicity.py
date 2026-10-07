from __future__ import annotations

import asyncio

from guardrails.checks.schema import extract_answer
from guardrails.core import BaseCheck, Context, Verdict, normalize, register
from guardrails.models import toxicity_classifier, windows


@register
class ToxicityCheck(BaseCheck):
    name = "toxicity"
    stage = "output"

    async def check(self, text: str, ctx: Context) -> Verdict:
        threshold = float(self.params.get("threshold", 0.7))
        # JSON keys and quotes dilute the classifier; score the answer field when present.
        norm = normalize(extract_answer(text))
        parts = windows(norm)
        results = await asyncio.gather(*(asyncio.to_thread(self._score, w) for w in parts))
        label, score = max(results, key=lambda r: r[1])
        reason = f"{label} {score:.2f} over {len(parts)} window(s)"
        return self.block(score, reason) if score >= threshold else self.allow(score, reason)

    @staticmethod
    def _score(text: str) -> tuple[str, float]:
        scores = toxicity_classifier()(text)[0]
        top = max(scores, key=lambda s: s["score"])
        return top["label"], float(top["score"])
