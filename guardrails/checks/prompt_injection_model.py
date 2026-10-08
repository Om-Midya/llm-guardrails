from __future__ import annotations

import asyncio

from guardrails.core import BaseCheck, Context, Verdict, normalize, register
from guardrails.models import injection_classifier, windows


@register
class PromptInjectionModelCheck(BaseCheck):
    """DeBERTa prompt-injection classifier. Split from the regex check so it can run in shadow."""

    name = "prompt_injection_model"
    stage = "input"

    async def check(self, text: str, ctx: Context) -> Verdict:
        threshold = float(self.params.get("threshold", 0.8))
        parts = windows(normalize(text))
        scores = await asyncio.gather(*(asyncio.to_thread(self._model_score, w) for w in parts))
        score = max(scores)
        reason = f"model injection prob {score:.2f} over {len(parts)} window(s)"
        return self.block(score, reason) if score >= threshold else self.allow(score, reason)

    @staticmethod
    def _model_score(text: str) -> float:
        out = injection_classifier()(text)[0]
        if out["label"].upper() == "INJECTION":
            return float(out["score"])
        return 1.0 - float(out["score"])
