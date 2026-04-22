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
    """Coerce LangChain's message/output objects to something JSON-serializable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)
