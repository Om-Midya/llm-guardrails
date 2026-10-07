from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from guardrails.core import BaseCheck, Context, Verdict, normalize, register
from guardrails.models import MAX_CHARS, embedder

DEFAULT_DENY = [
    r"\b(stock|share|crypto|coin|bitcoin|nft)s?\b.{0,40}\b(buy|invest|tip|recommend|pick)",
    r"\b(write|compose|generate)\b.{0,30}\b(poem|story|essay|song|code|script|joke)\b",
    r"\b(recipe|weather|forecast|cricket|football|movie|celebrity|horoscope|lottery)\b",
    r"\b(tell me a )?joke\b",
]
# Banking vocabulary that marks a question as in-domain even when it is short or has an ID in it.
DEFAULT_ALLOW = [
    r"\b(pan|aadhaar|aadhar|kyc|upi|imps|neft|rtgs|ifsc|otp|emi|fd|rd|atm|pos)\b",
    r"\b(account|balance|deposit|withdraw|transfer|transaction|statement|passbook)s?\b",
    r"\b(debit|credit) card\b|\bcard (ending|number|pin|block|declin)",
    r"\b(loan|interest|tenure|foreclos|prepay|penalt|charge|fee|refund|dispute|chargeback)",
    r"\b(branch|cheque|locker|nominee|net ?banking|mobile banking|customer care|fraud|phishing)",
    r"\b(savings|current|joint|minor|senior citizen|nri|nre|nro|dormant)\b",
]


@lru_cache(maxsize=4)
def load_reference_chunks(corpus_dir: str) -> list[str]:
    chunks: list[str] = []
    for f in sorted(Path(corpus_dir).glob("*.md")):
        for part in re.split(r"\n(?=## )", f.read_text()):
            part = part.strip()
            if part.startswith("# ") and "\n" in part:
                part = part.split("\n", 1)[1].strip()
            if part and not part.startswith("# "):
                chunks.append(part)
    return chunks


@lru_cache(maxsize=4)
def load_reference_headings(corpus_dir: str) -> list[str]:
    heads: list[str] = []
    for f in sorted(Path(corpus_dir).glob("*.md")):
        heads += [ln[3:].strip() for ln in f.read_text().splitlines() if ln.startswith("## ")]
    return heads


@lru_cache(maxsize=4)
def reference_embeddings(corpus_dir: str) -> np.ndarray:
    # Headings are question-shaped, so a user question scores higher against them than
    # against long policy paragraphs. Both sets are kept; the check takes the max.
    texts = load_reference_chunks(corpus_dir) + load_reference_headings(corpus_dir)
    vecs = embedder().encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


@register
class TopicCheck(BaseCheck):
    name = "topic"
    stage = "input"

    async def check(self, text: str, ctx: Context) -> Verdict:
        corpus_dir = str(self.params.get("corpus_dir", "bankassist/corpus"))
        floor = float(self.params.get("min_similarity", 0.35))
        deny = [
            re.compile(p, re.IGNORECASE) for p in self.params.get("deny_patterns", DEFAULT_DENY)
        ]
        allow = [
            re.compile(p, re.IGNORECASE) for p in self.params.get("allow_patterns", DEFAULT_ALLOW)
        ]
        norm = normalize(text)[:MAX_CHARS]
        for rx in deny:
            if rx.search(norm):
                return self.block(1.0, f"deny pattern {rx.pattern}")
        for rx in allow:
            if rx.search(norm):
                return self.allow(0.0, f"allow keyword {rx.pattern[:30]}")
        sim = await asyncio.to_thread(self._max_similarity, norm, corpus_dir)
        score = 1.0 - sim
        reason = f"max similarity {sim:.2f}"
        return self.block(score, reason) if sim < floor else self.allow(score, reason)

    @staticmethod
    def _max_similarity(text: str, corpus_dir: str) -> float:
        q = embedder().encode([text], normalize_embeddings=True)[0]
        return float(np.max(reference_embeddings(corpus_dir) @ q))
