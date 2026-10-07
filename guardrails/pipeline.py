from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from guardrails.core import CHECK_REGISTRY, BaseCheck, Context, Stage, Verdict
from guardrails.policy import CheckPolicy, Policy

REFUSAL = "I can't help with that request."


class PipelineResult(BaseModel):
    final_text: str
    blocked: bool
    blocked_by: str | None
    verdicts: list[Verdict]
    overhead_ms: float
    shadow_blocks: list[str]


class GuardrailPipeline:
    def __init__(self, policy: Policy, policy_hash: str) -> None:
        self.policy = policy
        self.policy_hash = policy_hash
        self._checks: dict[Stage, list[tuple[BaseCheck, CheckPolicy]]] = {
            "input": self._build(policy.input),
            "output": self._build(policy.output),
        }

    @staticmethod
    def _build(section: dict[str, CheckPolicy]) -> list[tuple[BaseCheck, CheckPolicy]]:
        return [
            (CHECK_REGISTRY[name](**cp.params), cp)
            for name, cp in section.items()
            if cp.enabled
        ]

    async def run_input(self, text: str, ctx: Context) -> PipelineResult:
        return await self._run_stage("input", text, ctx)

    async def run_output(self, text: str, ctx: Context) -> PipelineResult:
        return await self._run_stage("output", text, ctx)

    async def _safe(self, check: BaseCheck, text: str, ctx: Context) -> Verdict:
        t0 = time.perf_counter()
        try:
            v = await check.check(text, ctx)
        except Exception as e:  # noqa: BLE001  a broken check must not crash the request
            reason = f"check_error: {type(e).__name__}: {e}"
            v = check.block(0.0, reason) if check.fail_closed else check.allow(0.0, reason)
        v.latency_ms = (time.perf_counter() - t0) * 1000
        return v

    async def _run_stage(self, stage: Stage, text: str, ctx: Context) -> PipelineResult:
        t0 = time.perf_counter()
        checks = self._checks[stage]
        local = [(c, p) for c, p in checks if not c.is_llm]
        llm = [(c, p) for c, p in checks if c.is_llm]

        current = text
        blocked_by: str | None = None
        shadow_blocks: list[str] = []
        done: dict[str, Verdict] = {}

        def apply(check: BaseCheck, cp: CheckPolicy, v: Verdict) -> None:
            nonlocal current, blocked_by
            v.shadow = cp.mode == "shadow"
            done[check.name] = v
            if v.action == "redact" and v.rewritten_text is not None:
                # ponytail: redactors see the original text and are applied in policy order.
                # Exact with one redactor per stage, which is the shipped policy.
                current = v.rewritten_text
            elif v.action == "block":
                if v.shadow:
                    shadow_blocks.append(check.name)
                elif blocked_by is None:
                    blocked_by = check.name

        results = await asyncio.gather(*(self._safe(c, current, ctx) for c, _ in local))
        for (c, cp), v in zip(local, results, strict=True):
            apply(c, cp, v)

        if blocked_by is None:
            for c, cp in llm:
                before = current
                apply(c, cp, await self._safe(c, current, ctx))
                if current != before:
                    # Text rewritten by an LLM-backed check was never scanned: rescan it.
                    rescans = await asyncio.gather(
                        *(self._safe(lc, current, ctx) for lc, _ in local)
                    )
                    for (lc, lcp), rv in zip(local, rescans, strict=True):
                        apply(lc, lcp, rv)
                if blocked_by is not None:
                    break

        verdicts = [done[c.name] for c, _ in checks if c.name in done]
        return PipelineResult(
            final_text=REFUSAL if blocked_by else current,
            blocked=blocked_by is not None,
            blocked_by=blocked_by,
            verdicts=verdicts,
            overhead_ms=(time.perf_counter() - t0) * 1000,
            shadow_blocks=shadow_blocks,
        )
