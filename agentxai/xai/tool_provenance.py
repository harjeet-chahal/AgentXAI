"""
Pillar 3 — Tool-Use Provenance.

ToolProvenanceLogger records every ToolUseEvent (tool name, caller, inputs,
outputs, wall-clock duration) as a first-class entity in the store. The
`downstream_impact_score` is left at its default (0.0 = "not yet computed")
until the counterfactual engine runs and calls `attach_impact_score`, which
patches the existing record with the measured score and the counterfactual
run id.

The `@traced_tool(logger, called_by=...)` decorator wraps any callable
(including a LangChain Tool.func) and logs the call automatically.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional

from agentxai.data.schemas import ToolUseEvent
from agentxai.store.trajectory_store import TrajectoryStore


class ToolProvenanceLogger:
    """Record ToolUseEvents and, later, patch in their impact scores."""

    def __init__(self, store: TrajectoryStore, task_id: str) -> None:
        self.store = store
        self.task_id = task_id
        self._events: Dict[str, ToolUseEvent] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_tool_call(
        self,
        tool_name: str,
        called_by: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_ms: float,
    ) -> ToolUseEvent:
        """
        Write a ToolUseEvent with `downstream_impact_score` left at its
        default (0.0 — treated as "not yet computed" until the counterfactual
        engine runs).
        """
        event = ToolUseEvent(
            tool_name=tool_name,
            called_by=called_by,
            inputs={k: _to_jsonable(v) for k, v in (inputs or {}).items()},
            outputs={k: _to_jsonable(v) for k, v in (outputs or {}).items()},
            duration_ms=float(duration_ms),
        )
        self._events[event.tool_call_id] = event
        self.store.save_tool_call(self.task_id, event)
        return event

    def attach_impact_score(
        self,
        tool_call_id: str,
        score: float,
        counterfactual_run_id: str,
    ) -> ToolUseEvent:
        """Patch an existing ToolUseEvent with its counterfactual impact score."""
        event = self._events.get(tool_call_id) or self._load_from_store(tool_call_id)
        if event is None:
            raise KeyError(f"Tool call {tool_call_id!r} not found for task {self.task_id!r}.")

        event.downstream_impact_score = float(score)
        event.counterfactual_run_id = str(counterfactual_run_id)
        self._events[tool_call_id] = event
        self.store.save_tool_call(self.task_id, event)
        return event

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_from_store(self, tool_call_id: str) -> Optional[ToolUseEvent]:
        try:
            record = self.store.get_full_record(self.task_id)
        except KeyError:
            return None
        for tc in record.xai_data.tool_calls:
            if tc.tool_call_id == tool_call_id:
                return tc
        return None


# ---------------------------------------------------------------------------
# @traced_tool decorator factory
# ---------------------------------------------------------------------------

def traced_tool(
    logger: ToolProvenanceLogger,
    called_by: str,
    tool_name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator factory: wrap any callable (typically a LangChain Tool.func)
    so that every invocation produces a ToolUseEvent.

    Usage
    -----
        @traced_tool(logger, called_by="specialist_a")
        def lookup_symptoms(query: str) -> dict: ...

    Raised exceptions are re-raised after the event is logged (with the
    exception type captured in `outputs["error"]`).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = tool_name or getattr(func, "__name__", "") or "tool"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result: Any = None
            error: Optional[BaseException] = None
            try:
                result = func(*args, **kwargs)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                logger.log_tool_call(
                    tool_name=name,
                    called_by=called_by,
                    inputs=_capture_inputs(args, kwargs),
                    outputs=_capture_outputs(result, error),
                    duration_ms=duration_ms,
                )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_inputs(args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize (*args, **kwargs) into a JSON-friendly dict."""
    if len(args) == 1 and not kwargs:
        only = args[0]
        if isinstance(only, dict):
            return {k: _to_jsonable(v) for k, v in only.items()}
        return {"input": _to_jsonable(only)}
    if not args:
        return {k: _to_jsonable(v) for k, v in kwargs.items()}
    return {
        "args": [_to_jsonable(a) for a in args],
        "kwargs": {k: _to_jsonable(v) for k, v in kwargs.items()},
    }


def _capture_outputs(result: Any, error: Optional[BaseException]) -> Dict[str, Any]:
    if error is not None:
        return {"error": f"{type(error).__name__}: {error}"}
    if isinstance(result, dict):
        return {k: _to_jsonable(v) for k, v in result.items()}
    return {"output": _to_jsonable(result)}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)
