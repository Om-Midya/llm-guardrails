from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from guardrails.models import embedder


class Hit(BaseModel):
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self, corpus_dir: str = "bankassist/corpus") -> None:
        self.chunks: list[tuple[str, str]] = []
        for f in sorted(Path(corpus_dir).glob("*.md")):
            body = f.read_text()
            title = body.splitlines()[0].lstrip("# ").strip()
            for part in re.split(r"\n(?=## )", body):
                part = part.strip()
                if part.startswith("## "):
                    self.chunks.append((f.name, f"{title}\n{part}"))
        texts = [t for _, t in self.chunks]
        self.matrix = np.asarray(
            embedder().encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = embedder().encode([query], normalize_embeddings=True)[0]
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:k]
        return [
            Hit(text=self.chunks[i][1], source=self.chunks[i][0], score=float(sims[i]))
            for i in idx
        ]
