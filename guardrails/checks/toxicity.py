from __future__ import annotations

import asyncio

from guardrails.core import BaseCheck, Context, Verdict, normalize, register
from guardrails.models import MAX_CHARS, toxicity_classifier

# ponytail: 8 windows = 32k chars, above the 20k /guard/output cap; raise if inputs grow
MAX_WINDOWS = 8


@register
class ToxicityCheck(BaseCheck):
    name = "toxicity"
    stage = "output"

    async def check(self, text: str, ctx: Context) -> Verdict:
        threshold = float(self.params.get("threshold", 0.7))
        norm = normalize(text)
        windows = [norm[i : i + MAX_CHARS] for i in range(0, max(len(norm), 1), MAX_CHARS)]
        windows = windows[:MAX_WINDOWS]
        results = await asyncio.gather(*(asyncio.to_thread(self._score, w) for w in windows))
        label, score = max(results, key=lambda r: r[1])
        reason = f"{label} {score:.2f} over {len(windows)} window(s)"
        return self.block(score, reason) if score >= threshold else self.allow(score, reason)

    @staticmethod
    def _score(text: str) -> tuple[str, float]:
        scores = toxicity_classifier()(text)[0]
        top = max(scores, key=lambda s: s["score"])
        return top["label"], float(top["score"])
