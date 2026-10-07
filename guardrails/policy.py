from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class PolicyError(ValueError):
    pass


class CheckPolicy(BaseModel):
    enabled: bool = True
    mode: Literal["enforce", "shadow"] = "enforce"
    params: dict[str, Any] = Field(default_factory=dict)


class Policy(BaseModel):
    version: str
    name: str
    input: dict[str, CheckPolicy] = Field(default_factory=dict)
    output: dict[str, CheckPolicy] = Field(default_factory=dict)


def load_policy(path: str | Path) -> tuple[Policy, str]:
    import guardrails.checks  # noqa: F401  registers all checks
    from guardrails.core import CHECK_REGISTRY

    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw) or {}
    policy = Policy.model_validate(data)
    for stage in ("input", "output"):
        for name in getattr(policy, stage):
            cls = CHECK_REGISTRY.get(name)
            if cls is None:
                raise PolicyError(f"unknown check '{name}' under {stage}")
            if cls.stage != stage:
                raise PolicyError(f"'{name}' is an {cls.stage} check, listed under {stage}")
    return policy, hashlib.sha256(raw).hexdigest()
