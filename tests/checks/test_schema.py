import json

from pydantic import BaseModel, Field

from guardrails.checks.schema import SchemaCheck
from guardrails.core import Context


class Answer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


GOOD = {"answer": "Limit is 1 lakh.", "sources": ["upi.md"], "confidence": 0.9}


async def test_valid_json_allowed_and_normalized():
    v = await SchemaCheck().check(json.dumps(GOOD), Context(schema=Answer))
    assert v.action in ("allow", "redact")
    if v.action == "redact":
        assert json.loads(v.rewritten_text) == GOOD


async def test_fenced_json_is_unwrapped():
    text = "```json\n" + json.dumps(GOOD) + "\n```"
    v = await SchemaCheck().check(text, Context(schema=Answer))
    assert v.action == "redact" and json.loads(v.rewritten_text) == GOOD


async def test_repair_is_called_once_then_fallback():
    calls = []

    async def repair(bad, err):
        calls.append(err)
        return "still not json"

    fb = {"answer": "Sorry, try again.", "sources": [], "confidence": 0.0}
    ctx = Context(schema=Answer, repair_fn=repair, schema_fallback=fb)
    v = await SchemaCheck(max_repairs=1).check("oops", ctx)
    assert len(calls) == 1
    assert v.action == "redact" and json.loads(v.rewritten_text) == fb and "fallback" in v.reason


async def test_repair_success():
    async def repair(bad, err):
        return json.dumps(GOOD)

    v = await SchemaCheck().check("{bad", Context(schema=Answer, repair_fn=repair))
    assert v.action == "redact" and json.loads(v.rewritten_text) == GOOD


async def test_no_fallback_blocks():
    v = await SchemaCheck(max_repairs=0).check("oops", Context(schema=Answer))
    assert v.action == "block"


async def test_no_schema_allows():
    v = await SchemaCheck().check("free text", Context())
    assert v.action == "allow"
