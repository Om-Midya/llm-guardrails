from __future__ import annotations

import re

from guardrails.core import BaseCheck, Context, Verdict, normalize, register

PATTERNS = [
    r"\bignore (all |any )?(previous|prior|above|earlier) (instructions|prompts|rules|guidelines)",
    r"\bdisregard (your|the|all) (guidelines|policy|policies|rules|safety|instructions)",
    r"\bfrom now on you (will|must|should) (ignore|disregard|forget)",
    r"\bforget (all |everything )?(you were told|your (rules|instructions))",
    r"\bDAN\b|\bdo anything now\b",
    r"\bdeveloper mode\b|\bgod mode\b|\bsudo mode\b|\bjailbreak",
    r"\b(disable|turn off|bypass|remove) (your )?(safety|content|ethical) "
    r"(filters?|guidelines|rules)",
    r"\b(you are|act as|pretend (you are|to be)|roleplay as) .{0,60}"
    r"\b(no|without|zero) (rules|restrictions|filters|limits|ethics)",
    r"\byou are now (an? )?(unfiltered|unrestricted|uncensored|evil)",
    r"\bhypothetically.{0,80}\b(bypass|ignore|circumvent)\b.{0,40}"
    r"\b(safety|policy|rules|otp|verification)",
    r"\b(reveal|print|show|output|repeat) (me )?(your|the) "
    r"(system|hidden|initial|original) (prompt|instructions)",
    r"\b(system|hidden) prompt\b",
    r"\brespond only in (base64|rot13|leetspeak|hex)\b",
    r"\bgrandma (exploit|trick)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


@register
class JailbreakCheck(BaseCheck):
    name = "jailbreak"
    stage = "input"

    async def check(self, text: str, ctx: Context) -> Verdict:
        threshold = int(self.params.get("threshold", 1))
        norm = normalize(text)
        hits = [p.pattern for p in _COMPILED if p.search(norm)]
        score = min(1.0, len(hits) / threshold) if threshold > 0 else 1.0
        if len(hits) >= threshold:
            return self.block(score, f"matched {len(hits)} jailbreak pattern(s)")
        reason = "no jailbreak pattern" if not hits else f"{len(hits)} weak hit(s)"
        return self.allow(score, reason)
