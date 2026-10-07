from __future__ import annotations

import asyncio
import re

from guardrails.core import BaseCheck, Context, Verdict, normalize, register
from guardrails.models import MAX_CHARS, injection_classifier

PATTERNS = [
    r"^\s*(system|assistant)\s*:",
    r"\bnew instructions?\s*:",
    r"#{2,}\s*(instruction|system)\b",
    r"\b(translate|summari[sz]e|repeat|print|output|copy) (the )?"
    r"(text|content|everything|conversation) above\b",
    r"<\|im_(start|end)\|>|\[INST\]|<<SYS>>",
    r"\b(send|post|forward|exfiltrate|upload) .{0,60}\b(to|at) https?://",
    r"!\[[^\]]*\]\(https?://[^)]+\)",
    r"\byour (real|actual|true) (task|goal|purpose) is\b",
    r"\b(previous|prior) (rules|instructions) (were|was) (a )?(test|joke|mistake)\b",
    r"\boverride (the )?(system|safety|previous)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PATTERNS]


@register
class PromptInjectionCheck(BaseCheck):
    name = "prompt_injection"
    stage = "input"

    async def check(self, text: str, ctx: Context) -> Verdict:
        threshold = float(self.params.get("threshold", 0.8))
        use_model = bool(self.params.get("use_model", True))
        norm = normalize(text)

        regex_hits = [p.pattern for p in _COMPILED if p.search(norm)]
        score = 1.0 if regex_hits else 0.0
        reason = f"regex hit: {regex_hits[0]}" if regex_hits else "no regex hit"

        if not regex_hits and use_model:
            # Scan every window so an attack placed after the first 4000 chars is not missed.
            windows = [norm[i : i + MAX_CHARS] for i in range(0, max(len(norm), 1), MAX_CHARS)]
            scores = await asyncio.gather(
                *(asyncio.to_thread(self._model_score, w) for w in windows)
            )
            model_score = max(scores)
            score = max(score, model_score)
            reason = f"model injection prob {model_score:.2f} over {len(windows)} window(s)"

        if score >= threshold:
            return self.block(score, reason)
        return self.allow(score, reason)

    @staticmethod
    def _model_score(text: str) -> float:
        out = injection_classifier()(text)[0]
        if out["label"].upper() == "INJECTION":
            return float(out["score"])
        return 1.0 - float(out["score"])
