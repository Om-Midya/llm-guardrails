from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any


class _NoopSpan:
    def update(self, **kwargs: Any) -> None:
        pass


class Trace:
    def __init__(self, root: Any | None) -> None:
        self._root = root

    @contextmanager
    def span(self, name: str, input: Any = None):
        if self._root is None:
            yield _NoopSpan()
            return
        try:
            s = self._root.start_span(name=name, input=input)
        except Exception:  # noqa: BLE001  tracing must never break a request
            yield _NoopSpan()
            return
        try:
            yield s
        finally:
            try:
                s.end()
            except Exception:  # noqa: BLE001
                pass

    def end(self, output: Any = None, metadata: dict | None = None) -> None:
        if self._root is None:
            return
        try:
            self._root.update_trace(output=output, metadata=metadata or {})
            self._root.end()
        except Exception:  # noqa: BLE001
            pass


class Tracer:
    def __init__(self) -> None:
        self._client = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse

                self._client = Langfuse()
            except Exception:  # noqa: BLE001
                self._client = None

    def start_trace(self, name: str, input: Any, metadata: dict) -> Trace:
        if self._client is None:
            return Trace(None)
        try:
            root = self._client.start_span(name=name, input=input, metadata=metadata)
            return Trace(root)
        except Exception:  # noqa: BLE001
            return Trace(None)

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                pass


@lru_cache(maxsize=1)
def tracer() -> Tracer:
    return Tracer()
