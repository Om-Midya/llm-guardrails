from __future__ import annotations

import asyncio
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from guardrails.core import BaseCheck, Context, Verdict, register, strip_invisible


DEFAULT_ENTITIES = [
    "IN_PAN",
    "IN_AADHAAR",
    "IN_IFSC",
    "IN_BANK_ACCOUNT",
    "IN_PHONE",
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
]

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_ok(num: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(num)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(d)]]
    return c == 0


class AadhaarRecognizer(PatternRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_AADHAAR",
            patterns=[Pattern("aadhaar", r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b", 0.6)],
            context=["aadhaar", "aadhar", "uid", "kyc"],
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return verhoeff_ok(pattern_text.replace(" ", "").replace("-", ""))


def _recognizers() -> list[PatternRecognizer]:
    return [
        PatternRecognizer(
            "IN_PAN",
            patterns=[Pattern("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.85)],
            context=["pan", "permanent account"],
        ),
        AadhaarRecognizer(),
        PatternRecognizer(
            "IN_IFSC",
            patterns=[Pattern("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.8)],
            context=["ifsc", "branch"],
        ),
        PatternRecognizer(
            "IN_BANK_ACCOUNT",
            patterns=[Pattern("acct", r"\b\d{11,18}\b", 0.3)],
            context=["account", "a/c", "acct", "account number", "savings", "current"],
        ),
        PatternRecognizer(
            "IN_PHONE",
            patterns=[Pattern("phone", r"(?:\+91[ -]?)?[6-9]\d{4}[ -]?\d{5}\b", 0.6)],
            context=["call", "phone", "mobile", "whatsapp", "contact"],
        ),
    ]


@lru_cache(maxsize=1)
def build_analyzer() -> AnalyzerEngine:
    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()
    engine = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    for r in _recognizers():
        engine.registry.add_recognizer(r)
    return engine


def detect(text: str, entities: list[str], min_score: float) -> list[RecognizerResult]:
    results = build_analyzer().analyze(
        text=text, entities=entities, language="en", score_threshold=min_score
    )
    return sorted(results, key=lambda r: r.start)


def redact_text(text: str, results: list[RecognizerResult]) -> str:
    # Merge overlapping spans so no tail of a second entity survives redaction.
    merged: list[list] = []
    for r in sorted(results, key=lambda r: r.start):
        if merged and r.start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r.end)
        else:
            merged.append([r.start, r.end, r.entity_type])
    out, last = [], 0
    for start, end, entity in merged:
        out.append(text[last:start])
        out.append(f"<{entity}>")
        last = end
    out.append(text[last:])
    return "".join(out)


class _PIIBase(BaseCheck):
    async def check(self, text: str, ctx: Context) -> Verdict:
        entities = list(self.params.get("entities", DEFAULT_ENTITIES))
        action = self.params.get("action", "redact")
        min_score = float(self.params.get("min_score", 0.5))
        clean = strip_invisible(text)
        results = await asyncio.to_thread(detect, clean, entities, min_score)
        if not results:
            return self.allow(0.0, "no pii")
        score = max(r.score for r in results)
        found = sorted({r.entity_type for r in results})
        reason = f"found {', '.join(found)}"
        if action == "block":
            return self.block(score, reason)
        return self.redact(redact_text(clean, results), score, reason)


@register
class PIIInputCheck(_PIIBase):
    name = "pii_input"
    stage = "input"


@register
class PIIOutputCheck(_PIIBase):
    name = "pii_output"
    stage = "output"
    fail_closed = True
