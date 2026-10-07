import json
from collections import Counter
from pathlib import Path

import yaml

E = Path("evals")


def rows(name):
    return [json.loads(line) for line in (E / name).read_text().splitlines() if line.strip()]


def test_input_set_shape_and_coverage():
    r = rows("redteam_input.jsonl")
    assert len(r) >= 60
    cats = Counter(x["category"] for x in r)
    assert all(cats[c] >= 12 for c in ("injection", "jailbreak", "pii", "offtopic")), cats
    assert len({x["id"] for x in r}) == len(r)
    for x in r:
        assert x["expected_action"] in ("block", "redact")
        assert x["expected_check"] and len(x["text"]) > 10


def test_output_set_shape_and_coverage():
    r = rows("redteam_output.jsonl")
    assert len(r) >= 20
    cats = Counter(x["category"] for x in r)
    needed = ("hallucination", "toxicity", "pii_leak", "secret_leak", "bad_json")
    assert all(cats[c] >= 4 for c in needed), cats
    for x in r:
        assert isinstance(x["context_chunks"], list)
        if x["category"] == "hallucination":
            assert x["context_chunks"], x["id"]


def test_benign_set():
    r = rows("benign.jsonl")
    assert len(r) >= 50 and len({x["id"] for x in r}) == len(r)


def test_thresholds_cover_all_checks():
    t = yaml.safe_load((E / "thresholds.yaml").read_text())
    names = {
        "prompt_injection", "jailbreak", "pii_input", "topic", "schema",
        "hallucination", "toxicity", "pii_output", "secret_leak",
    }
    assert set(t["checks"]) == names and 0 < t["max_catch_rate_drop_vs_main"] < 0.1
