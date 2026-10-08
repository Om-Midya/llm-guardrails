from __future__ import annotations

import re

from guardrails.core import BaseCheck, Context, Verdict, normalize, register

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
    """Regex half of injection detection. The classifier lives in prompt_injection_model."""

    name = "prompt_injection"
    stage = "input"

    async def check(self, text: str, ctx: Context) -> Verdict:
        norm = normalize(text)
        hits = [p.pattern for p in _COMPILED if p.search(norm)]
        if hits:
            return self.block(1.0, f"regex hit: {hits[0]}")
        return self.allow(0.0, "no regex hit")
