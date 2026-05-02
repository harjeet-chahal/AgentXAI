"""
Pillar 1 — Trajectories.

TrajectoryLogger is a LangChain BaseCallbackHandler that wraps a TrajectoryStore
and a current task_id. Every LangChain callback (agent action, tool start/end,
chain start/end) is mapped to a TrajectoryEvent, persisted via the store, and
exposed through current_trajectory().

LangChain is an optional runtime dependency here: when it is importable we
subclass its real BaseCallbackHandler so the logger can be attached to any
Runnable / AgentExecutor; when it is not, we fall back to a trivial stub base
class so the logger remains usable (and unit-testable) without LangChain
installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentxai.data.schemas import TrajectoryEvent
from agentxai.store.trajectory_store import TrajectoryStore

try:  # LangChain >= 0.3 split core out into langchain_core
    from langchain_core.callbacks import BaseCallbackHandler as _LCBase
except ImportError:  # pragma: no cover - legacy import path
    try:
        from langchain.callbacks.base import BaseCallbackHandler as _LCBase
    except ImportError:
        class _LCBase:  # minimal stand-in so the class is still usable
            """Fallback base when LangChain is not installed."""
            pass


class TrajectoryLogger(_LCBase):
    """Observer that records TrajectoryEvents for one task."""

    def __init__(self, store: TrajectoryStore, task_id: str) -> None:
        super().__init__()
        self.store = store
        self.task_id = task_id
        self._events: List[TrajectoryEvent] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_event(
        self,
        agent_id: str,
        event_type: str,
        state_before: Optional[Dict[str, Any]] = None,
        action: str = "",
        action_inputs: Optional[Dict[str, Any]] = None,
        state_after: Optional[Dict[str, Any]] = None,
        outcome: str = "",
    ) -> TrajectoryEvent:
        """Build a TrajectoryEvent, persist it, and return it."""
        event = TrajectoryEvent(
            agent_id=agent_id,
            event_type=event_type,
            state_before=dict(state_before or {}),
            action=action,
            action_inputs=dict(action_inputs or {}),
            state_after=dict(state_after or {}),
            outcome=outcome,
        )
        self.store.save_event(self.task_id, event)
        self._events.append(event)
        return event

    def current_trajectory(self) -> List[TrajectoryEvent]:
        """Return every event logged for the current task, in timestamp order."""
        return sorted(self._events, key=lambda e: e.timestamp)

    # ------------------------------------------------------------------
    # LangChain BaseCallbackHandler hooks
    # ------------------------------------------------------------------

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        agent_id = self._agent_id_from_kwargs(kwargs)
        tool = getattr(action, "tool", "")
        tool_input = getattr(action, "tool_input", {})
        log = getattr(action, "log", "")
        if not isinstance(tool_input, dict):
            tool_input = {"input": tool_input}
        return self.log_event(
            agent_id=agent_id,
            event_type="agent_action",
            action=str(tool),
            action_inputs=tool_input,
            outcome=str(log) if log else "",
        )

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        **kwargs: Any,
    ) -> Any:
        tool_name = (serialized or {}).get("name", "") or (serialized or {}).get("id", "")
        return self.log_event(
            agent_id=self._agent_id_from_kwargs(kwargs),
            event_type="tool_start",
            action=str(tool_name),
            action_inputs={"input": input_str},
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        return self.log_event(
            agent_id=self._agent_id_from_kwargs(kwargs),
            event_type="tool_end",
            state_after={"output": _to_jsonable(output)},
            outcome="success",
        )

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        chain_name = (serialized or {}).get("name", "") or (serialized or {}).get("id", "")
        return self.log_event(
            agent_id=self._agent_id_from_kwargs(kwargs),
            event_type="chain_start",
            action=str(chain_name),
            action_inputs={k: _to_jsonable(v) for k, v in (inputs or {}).items()},
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        return self.log_event(
            agent_id=self._agent_id_from_kwargs(kwargs),
            event_type="chain_end",
            state_after={k: _to_jsonable(v) for k, v in (outputs or {}).items()},
            outcome="success",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_id_from_kwargs(kwargs: Dict[str, Any]) -> str:
        """Best-effort agent id from LangChain callback kwargs (tags/metadata/name)."""
        tags = kwargs.get("tags") or []
        if tags:
            return str(tags[0])
        metadata = kwargs.get("metadata") or {}
        if "agent_id" in metadata:
            return str(metadata["agent_id"])
        return str(kwargs.get("name", "") or "")


def _to_jsonable(value: Any) -> Any:
    """
    Coerce LangChain's message/output objects to something JSON-serializable.

    Handles the common Python container types in-place. For non-trivial
    leaf objects we try, in order:

      1. ``dataclasses.asdict`` — covers every AgentXAI schema dataclass.
      2. Pydantic v2 ``.model_dump()`` and v1 ``.dict()`` — covers
         LangChain's BaseMessage subclasses and the API response models.
      3. ``__json__`` / ``to_dict`` convention — for objects that
         deliberately expose a JSON projection.
      4. Structured fallback ``{"__type__": "<ClassName>", "repr": "..."}``
         — preserves the type name for forensic value instead of
         silently dropping it into an opaque ``str(obj)`` blob.

    The structured fallback was the headline fix here: previous code
    coerced every unknown value with ``str(value)``, so a complex object
    like ``<MyClass object at 0x...>`` was stored verbatim, which is
    indistinguishable from a real string and can't be inspected later.
    """
    import dataclasses

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]

    # 1. Dataclasses (AgentXAI schemas, plus anything user-defined).
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _to_jsonable(dataclasses.asdict(value))
        except Exception:
            pass  # fall through to next strategy

    # 2. Pydantic — v2 first, then v1. LangChain Messages and the
    #    API response models both use one of these.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _to_jsonable(dump())
        except Exception:
            pass
    legacy_dump = getattr(value, "dict", None)
    if callable(legacy_dump):
        try:
            out = legacy_dump()
            if isinstance(out, dict):
                return _to_jsonable(out)
        except Exception:
            pass

    # 3. Convention helpers some objects expose.
    for attr in ("__json__", "to_dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                out = fn()
                # Only honor the result when it's already JSON-shaped.
                if isinstance(out, (dict, list, str, int, float, bool)) or out is None:
                    return _to_jsonable(out)
            except Exception:
                pass

    # 4. Structured fallback. Preserves type name for forensic value;
    #    truncates the repr so a giant object doesn't blow up the JSON
    #    column.
    type_name = f"{type(value).__module__}.{type(value).__name__}"
    raw_repr = repr(value)
    if len(raw_repr) > 240:
        raw_repr = raw_repr[:237] + "..."
    return {"__type__": type_name, "repr": raw_repr}
