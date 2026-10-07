from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

Action = Literal["allow", "block", "redact"]
Stage = Literal["input", "output"]
RepairFn = Callable[[str, str], Awaitable[str]]

_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", _ZERO_WIDTH.sub("", text)).strip()


class Verdict(BaseModel):
    check: str
    action: Action
    score: float = 0.0
    reason: str = ""
    rewritten_text: str | None = None
    latency_ms: float = 0.0
    shadow: bool = False


@dataclass
class Context:
    request_id: str = "local"
    retrieved_chunks: list[str] = field(default_factory=list)
    schema: type[BaseModel] | None = None
    schema_fallback: dict | None = None
    policy_hash: str = ""
    repair_fn: RepairFn | None = None


class BaseCheck(ABC):
    name: ClassVar[str]
    stage: ClassVar[Stage]
    is_llm: ClassVar[bool] = False
    fail_closed: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    async def check(self, text: str, ctx: Context) -> Verdict: ...

    def allow(self, score: float = 0.0, reason: str = "") -> Verdict:
        return Verdict(check=self.name, action="allow", score=score, reason=reason)

    def block(self, score: float, reason: str) -> Verdict:
        return Verdict(check=self.name, action="block", score=score, reason=reason)

    def redact(self, rewritten_text: str, score: float, reason: str) -> Verdict:
        return Verdict(
            check=self.name,
            action="redact",
            score=score,
            reason=reason,
            rewritten_text=rewritten_text,
        )


CHECK_REGISTRY: dict[str, type[BaseCheck]] = {}


def register(cls: type[BaseCheck]) -> type[BaseCheck]:
    CHECK_REGISTRY[cls.name] = cls
    return cls
