from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

log = logging.getLogger("guardrails.tracing")


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
            s = self._root.start_observation(name=name, as_type="span", input=input)
        except Exception as e:  # noqa: BLE001  tracing must never break a request
            log.warning("langfuse span %s failed: %s", name, e)
            yield _NoopSpan()
            return
        try:
            yield s
        finally:
            try:
                s.end()
            except Exception as e:  # noqa: BLE001
                log.warning("langfuse span end failed: %s", e)

    def end(self, output: Any = None, metadata: dict | None = None) -> None:
        if self._root is None:
            return
        try:
            self._root.update(output=output, metadata=metadata or {})
            self._root.set_trace_io(output=output)
            self._root.end()
        except Exception as e:  # noqa: BLE001
            log.warning("langfuse trace end failed: %s", e)


class Tracer:
    def __init__(self) -> None:
        self._client = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse

                self._client = Langfuse()
            except Exception as e:  # noqa: BLE001
                log.warning("langfuse disabled: %s", e)

    def start_trace(self, name: str, input: Any, metadata: dict) -> Trace:
        if self._client is None:
            return Trace(None)
        try:
            root = self._client.start_observation(
                name=name, as_type="span", input=input, metadata=metadata
            )
            return Trace(root)
        except Exception as e:  # noqa: BLE001
            log.warning("langfuse trace start failed: %s", e)
            return Trace(None)

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:  # noqa: BLE001
                log.warning("langfuse flush failed: %s", e)


@lru_cache(maxsize=1)
def tracer() -> Tracer:
    return Tracer()
