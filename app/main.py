from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field, field_validator

from app import metrics
from app.tracing import tracer
from bankassist import bot
from bankassist.retriever import Retriever
from guardrails.core import Context, Verdict
from guardrails.pipeline import GuardrailPipeline, PipelineResult
from guardrails.policy import load_policy

load_dotenv()
STATIC = Path(__file__).parent / "static"
SCHEMAS = {"bot_answer": (bot.BotAnswer, bot.FALLBACK)}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    blocked: bool
    blocked_by: str | None
    input_verdicts: list[Verdict]
    output_verdicts: list[Verdict]
    shadow_blocks: list[str]
    policy_version: str
    policy_hash: str
    overhead_ms: float
    llm_cost_usd: float
    request_id: str


class GuardInputRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class GuardOutputRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    context_chunks: list[str] = Field(default_factory=list)
    schema_name: Literal["bot_answer"] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    policy_path = os.getenv("GUARDRAILS_POLICY", "policies/v1.yaml")
    policy, policy_hash = load_policy(policy_path)
    app.state.policy, app.state.policy_hash = policy, policy_hash
    app.state.pipeline = GuardrailPipeline(policy, policy_hash)
    app.state.retriever = Retriever()
    yield
    tracer().flush()


def _ctx(pipe: GuardrailPipeline, **kw) -> Context:
    return Context(request_id=str(uuid.uuid4()), policy_hash=pipe.policy_hash, **kw)


def public(verdicts: list[Verdict]) -> list[Verdict]:
    # rewritten_text can carry redacted or blocked content; clients only get final_text.
    return [v.model_copy(update={"rewritten_text": None}) for v in verdicts]


def public_result(res: PipelineResult) -> PipelineResult:
    return res.model_copy(update={"verdicts": public(res.verdicts)})


# ponytail: in-memory per-IP sliding window; move to Redis if this ever runs on >1 replica
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
MAX_RATE_BUCKETS = 10_000
RATE_BUCKETS: dict[str, deque[float]] = {}


def client_ip(request: Request) -> str:
    # The rightmost X-Forwarded-For entry is set by the trusted edge proxy; earlier ones
    # are client-controlled and spoofable.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "?"


def rate_limit(request: Request) -> None:
    ip = client_ip(request)
    now = time.monotonic()
    if ip not in RATE_BUCKETS and len(RATE_BUCKETS) >= MAX_RATE_BUCKETS:
        for stale in [k for k, b in RATE_BUCKETS.items() if not b or now - b[-1] > 60]:
            del RATE_BUCKETS[stale]
        if len(RATE_BUCKETS) >= MAX_RATE_BUCKETS:
            RATE_BUCKETS.pop(next(iter(RATE_BUCKETS)))
    bucket = RATE_BUCKETS.setdefault(ip, deque())
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, detail="rate limit exceeded, try again in a minute")
    bucket.append(now)


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Guardrails", lifespan=lifespan)
    limited = [Depends(rate_limit)]

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC / "index.html").read_text()

    @app.get("/metrics")
    async def prom():
        return Response(metrics.render(), media_type=metrics.CONTENT_TYPE)

    @app.get("/policy")
    async def policy(request: Request):
        p = request.app.state.policy
        return {
            "version": p.version,
            "name": p.name,
            "hash": request.app.state.policy_hash,
            "input": {k: v.model_dump() for k, v in p.input.items()},
            "output": {k: v.model_dump() for k, v in p.output.items()},
        }

    @app.post("/guard/input", response_model=PipelineResult, dependencies=limited)
    async def guard_input(req: GuardInputRequest, request: Request):
        pipe: GuardrailPipeline = request.app.state.pipeline
        res = await pipe.run_input(req.text, _ctx(pipe))
        metrics.record("input", res)
        return public_result(res)

    @app.post("/guard/output", response_model=PipelineResult, dependencies=limited)
    async def guard_output(req: GuardOutputRequest, request: Request):
        pipe: GuardrailPipeline = request.app.state.pipeline
        schema, fallback = SCHEMAS.get(req.schema_name or "", (None, None))
        ctx = _ctx(
            pipe, retrieved_chunks=req.context_chunks, schema=schema, schema_fallback=fallback,
            repair_fn=bot.repair if schema else None,
        )
        res = await pipe.run_output(req.text, ctx)
        metrics.record("output", res)
        return public_result(res)

    @app.post("/chat", response_model=ChatResponse, dependencies=limited)
    async def chat(req: ChatRequest, request: Request):
        pipe: GuardrailPipeline = request.app.state.pipeline
        retriever: Retriever = request.app.state.retriever
        ctx = _ctx(pipe)
        trace = tracer().start_trace(
            "chat", input=None,
            metadata={"policy_version": pipe.policy.version, "policy_hash": pipe.policy_hash},
        )

        # Traces only ever see post-redaction text.
        with trace.span("guard_input") as s:
            inp = await pipe.run_input(req.message, ctx)
            s.update(input=inp.final_text, output=public_result(inp).model_dump())
        metrics.record("input", inp)
        if inp.blocked:
            metrics.REQUESTS.labels("blocked_input").inc()
            trace.end(output="blocked", metadata={"blocked_by": inp.blocked_by})
            return ChatResponse(
                answer=inp.final_text, sources=[], blocked=True, blocked_by=inp.blocked_by,
                input_verdicts=public(inp.verdicts), output_verdicts=[],
                shadow_blocks=inp.shadow_blocks,
                policy_version=pipe.policy.version, policy_hash=pipe.policy_hash,
                overhead_ms=inp.overhead_ms, llm_cost_usd=0.0, request_id=ctx.request_id,
            )

        hits = retriever.search(inp.final_text)
        with trace.span("llm", input=inp.final_text) as s:
            try:
                raw = await bot.answer(inp.final_text, hits)
            except Exception as e:  # noqa: BLE001  upstream failure becomes a 502, not a crash
                metrics.REQUESTS.labels("llm_error").inc()
                trace.end(output="llm_error", metadata={"error": str(e)[:200]})
                raise HTTPException(
                    502,
                    detail={
                        "error": "upstream model failed",
                        "input_verdicts": [v.model_dump() for v in inp.verdicts],
                    },
                ) from e
            s.update(
                metadata={
                    "model": raw.model, "input_tokens": raw.input_tokens,
                    "output_tokens": raw.output_tokens, "cost_usd": raw.cost_usd,
                },
            )
        metrics.LLM_COST.labels(raw.model).inc(raw.cost_usd)

        ctx.retrieved_chunks = [h.text for h in hits]
        ctx.schema, ctx.schema_fallback, ctx.repair_fn = bot.BotAnswer, bot.FALLBACK, bot.repair
        with trace.span("guard_output") as s:
            out = await pipe.run_output(raw.text, ctx)
            s.update(output=public_result(out).model_dump())
        metrics.record("output", out)

        if out.blocked:
            answer, sources = out.final_text, []
            metrics.REQUESTS.labels("blocked_output").inc()
        else:
            parsed = bot.BotAnswer.model_validate_json(out.final_text)
            answer, sources = parsed.answer, parsed.sources
            metrics.REQUESTS.labels("ok").inc()
        shadow = inp.shadow_blocks + out.shadow_blocks
        overhead = inp.overhead_ms + out.overhead_ms
        trace.end(
            output=answer,
            metadata={
                "shadow_blocks": shadow, "blocked_by": out.blocked_by, "overhead_ms": overhead,
            },
        )
        return ChatResponse(
            answer=answer, sources=sources, blocked=out.blocked, blocked_by=out.blocked_by,
            input_verdicts=public(inp.verdicts), output_verdicts=public(out.verdicts),
            shadow_blocks=shadow,
            policy_version=pipe.policy.version, policy_hash=pipe.policy_hash,
            overhead_ms=overhead, llm_cost_usd=raw.cost_usd, request_id=ctx.request_id,
        )

    return app


app = create_app()
