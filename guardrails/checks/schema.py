from __future__ import annotations

import json
import re

from pydantic import ValidationError

from guardrails.core import BaseCheck, Context, Verdict, register

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def unfence(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text.strip()


def extract_answer(text: str) -> str:
    try:
        obj = json.loads(unfence(text))
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            return obj["answer"]
    except ValueError:
        pass
    return text


@register
class SchemaCheck(BaseCheck):
    name = "schema"
    stage = "output"
    is_llm = True

    async def check(self, text: str, ctx: Context) -> Verdict:
        if ctx.schema is None:
            return self.allow(0.0, "no schema")
        max_repairs = int(self.params.get("max_repairs", 1))
        candidate, error = text, ""
        for attempt in range(max_repairs + 1):
            try:
                obj = ctx.schema.model_validate_json(unfence(candidate))
                canonical = obj.model_dump_json()
                if canonical == text:
                    return self.allow(0.0, "valid")
                reason = "normalized" if attempt == 0 else f"repaired after {attempt}"
                return self.redact(canonical, 0.0, reason)
            except (ValidationError, ValueError) as e:
                error = str(e)[:500]
            if attempt < max_repairs and ctx.repair_fn is not None:
                candidate = await ctx.repair_fn(candidate, error)
            else:
                break
        if ctx.schema_fallback is not None:
            return self.redact(
                json.dumps(ctx.schema_fallback), 1.0, f"schema fallback: {error[:120]}"
            )
        return self.block(1.0, f"invalid output schema: {error[:120]}")
