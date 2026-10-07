from pathlib import Path

from guardrails.checks.topic import load_reference_chunks

CORPUS = Path("bankassist/corpus")


def test_corpus_shape():
    files = sorted(CORPUS.glob("*.md"))
    assert len(files) >= 40
    for f in files:
        text = f.read_text()
        assert text.startswith("# "), f
        assert text.count("\n## ") >= 3, f


def test_chunks_load():
    chunks = load_reference_chunks(str(CORPUS))
    assert len(chunks) >= 120
    assert all(len(c) > 40 for c in chunks)
