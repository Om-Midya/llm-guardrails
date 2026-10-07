from __future__ import annotations

import math
import re
from collections import Counter

from guardrails.core import BaseCheck, Context, Verdict, register

PATTERNS = {
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "google_api_key": r"\bAIza[0-9A-Za-z\-_]{35}\b",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "stripe_key": r"\b[sr]k_(live|test)_[A-Za-z0-9]{16,}\b",
    "assignment": (
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b\s*(is|[:=])\s*"
        r"['\"]?([A-Za-z0-9_\-!@#$%^&*+=/.]{12,})"
    ),
}
_COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}
_CANDIDATE = re.compile(r"[A-Za-z0-9_\-!@#$%^&*+=/.]{20,}")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _looks_random(tok: str) -> bool:
    classes = sum(
        bool(re.search(p, tok)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]")
    )
    return classes >= 3 and not tok.startswith(("http", "www."))


@register
class SecretLeakCheck(BaseCheck):
    name = "secret_leak"
    stage = "output"
    fail_closed = True

    async def check(self, text: str, ctx: Context) -> Verdict:
        threshold = float(self.params.get("entropy_threshold", 4.0))
        for kind, rx in _COMPILED.items():
            if rx.search(text):
                return self.block(1.0, f"pattern {kind}")
        best = 0.0
        for tok in _CANDIDATE.findall(text):
            if _looks_random(tok):
                best = max(best, shannon_entropy(tok))
        if best >= threshold:
            return self.block(min(1.0, best / 6), f"high entropy token {best:.2f}")
        return self.allow(best / 6 if best else 0.0, "no secret pattern")
