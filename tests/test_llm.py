import json

import pytest

from guardrails.llm import Cassette, CassetteMiss, LLMResponse, complete


def test_cassette_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    c = Cassette(p, "record")
    k = c.key("m", None, "hi")
    r = LLMResponse(text="yo", model="m", input_tokens=1, output_tokens=1, cost_usd=0.0)
    c.put(k, r)
    assert Cassette(p, "replay").get(k) == r
    assert json.loads(p.read_text())[k]["text"] == "yo"


async def test_replay_miss_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CASSETTE_MODE", "replay")
    monkeypatch.setenv("LLM_CASSETTE_PATH", str(tmp_path / "empty.json"))
    import guardrails.llm as llm

    llm.cassette.cache_clear()
    with pytest.raises(CassetteMiss):
        await complete("gemini-2.5-flash", "never recorded prompt 123")
    llm.cassette.cache_clear()
