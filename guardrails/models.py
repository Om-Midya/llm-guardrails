from __future__ import annotations

from functools import lru_cache

MAX_CHARS = 4000
# The classifiers read 512 tokens, about 1500 chars. Overlapping windows cover long text.
WINDOW = 1500
OVERLAP = 200
# ponytail: 16 windows = ~21k chars, matches the 20k /guard/output cap; raise with the cap
MAX_WINDOWS = 16


def windows(text: str) -> list[str]:
    step = WINDOW - OVERLAP
    out = [text[i : i + WINDOW] for i in range(0, max(len(text), 1), step)]
    if len(out) > MAX_WINDOWS:
        # Keep the tail: an attacker who pads the front must not push the payload past the cap.
        out = out[: MAX_WINDOWS - 1] + [text[-WINDOW:]]
    return out


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
