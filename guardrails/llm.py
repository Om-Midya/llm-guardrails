from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

# USD per million tokens (input, output). Checked against provider pricing pages on 2026-10-07.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CassetteMiss(RuntimeError):
    pass


class Cassette:
    def __init__(self, path: str | Path, mode: str) -> None:
        self.path = Path(path)
        self.mode = mode
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text() or "{}")

    @staticmethod
    def key(model: str, system: str | None, prompt: str) -> str:
        return hashlib.sha256(f"{model}\x00{system or ''}\x00{prompt}".encode()).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        raw = self._data.get(key)
        return LLMResponse.model_validate(raw) if raw else None

    def put(self, key: str, resp: LLMResponse) -> None:
        self._data[key] = resp.model_dump()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1, sort_keys=True))


@lru_cache(maxsize=1)
def cassette() -> Cassette:
    return Cassette(
        os.getenv("LLM_CASSETTE_PATH", "evals/cassettes/llm.json"),
        os.getenv("LLM_CASSETTE_MODE", "off"),
    )


def app_model() -> str:
    return os.getenv("APP_MODEL", "gemini-2.5-flash")


def judge_model() -> str:
    return os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")


def cost(model: str, inp: int, out: int) -> float:
    pi, po = PRICES_PER_M.get(model, (0.0, 0.0))
    return (inp * pi + out * po) / 1_000_000


async def _gemini(
    model: str, prompt: str, system: str | None, json_mode: bool, max_tokens: int
) -> LLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=0.2,
        response_mime_type="application/json" if json_mode else None,
    )
    r = await client.aio.models.generate_content(model=model, contents=prompt, config=cfg)
    u = r.usage_metadata
    inp, out = int(u.prompt_token_count or 0), int(u.candidates_token_count or 0)
    return LLMResponse(
        text=r.text or "", model=model, input_tokens=inp, output_tokens=out,
        cost_usd=cost(model, inp, out),
    )


async def _claude(
    model: str, prompt: str, system: str | None, json_mode: bool, max_tokens: int
) -> LLMResponse:
    import anthropic

    client = anthropic.AsyncAnthropic()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    r = await client.messages.create(**kwargs)
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    inp, out = r.usage.input_tokens, r.usage.output_tokens
    return LLMResponse(
        text=text, model=model, input_tokens=inp, output_tokens=out,
        cost_usd=cost(model, inp, out),
    )


async def complete(
    model: str,
    prompt: str,
    system: str | None = None,
    json_mode: bool = False,
    max_tokens: int = 1024,
) -> LLMResponse:
    cas = cassette()
    key = Cassette.key(model, system, prompt)
    if cas.mode in ("replay", "record"):
        hit = cas.get(key)
        if hit:
            return hit
        if cas.mode == "replay":
            raise CassetteMiss(f"no cassette entry for {model}: {prompt[:60]!r}")
    fn = _gemini if model.startswith("gemini") else _claude
    resp = await fn(model, prompt, system, json_mode, max_tokens)
    if cas.mode == "record":
        cas.put(key, resp)
    return resp
