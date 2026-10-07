from __future__ import annotations

from functools import lru_cache

MAX_CHARS = 4000


@lru_cache(maxsize=1)
def injection_classifier():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="protectai/deberta-v3-base-prompt-injection-v2",
        truncation=True,
        max_length=512,
        device=-1,
    )


@lru_cache(maxsize=1)
def toxicity_classifier():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None,
        truncation=True,
        max_length=512,
        device=-1,
    )


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
