"""
Pillar 4 — Memory State Changes.

LoggedMemory behaves like a dict; every `__setitem__` and `__getitem__` also
writes a MemoryDiff to the store. MemoryLogger owns one LoggedMemory per
agent_id so callers can ask `memory_logger.for_agent("specialist_a")["k"]`.

The `triggered_by_event_id` field on each MemoryDiff is read from a
ContextVar. The trajectory logger is expected to set it (via `current_event_id.set(event_id)`) immediately before invoking agent code that might touch memory; when unset, diffs carry an empty string, which is still valid for the schema.
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, Iterator

from agentxai.data.schemas import MemoryDiff
from agentxai.store.trajectory_store import TrajectoryStore


current_event_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agentxai_current_event_id", default=""
)
"""ContextVar carrying the event_id that should be attributed as the cause of
the next memory read/write. Set by the trajectory logger around agent calls."""


_MISSING = object()


class LoggedMemory(dict):
    """A dict that emits a MemoryDiff on every read and write."""

    def __init__(
        self,
        agent_id: str,
        store: TrajectoryStore,
        task_id: str,
        initial: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._agent_id = agent_id
        self._store = store
        self._task_id = task_id
        if initial:
            # Seed without logging — diffs are for runtime mutation, not construction.
            super().update(initial)

    # ------------------------------------------------------------------
    # Logged operations
    # ------------------------------------------------------------------

    def __setitem__(self, key: str, value: Any) -> None:
        # dict.get bypasses __getitem__, so this does not recurse or emit a read.
        before = dict.get(self, key, None)
        super().__setitem__(key, value)
        self._emit(
            operation="write",
            key=str(key),
            value_before=before,
            value_after=value,
        )

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)  # raises KeyError → no diff logged
        self._emit(
            operation="read",
            key=str(key),
            value_before=value,
            value_after=value,
        )
        return value

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit(
        self,
        *,
        operation: str,
        key: str,
        value_before: Any,
        value_after: Any,
    ) -> MemoryDiff:
        diff = MemoryDiff(
            agent_id=self._agent_id,
            operation=operation,
            key=key,
            value_before=value_before,
            value_after=value_after,
            triggered_by_event_id=current_event_id.get(""),
        )
        self._store.save_memory_diff(self._task_id, diff)
        return diff


class MemoryLogger:
    """Holds one LoggedMemory per agent_id for a given task."""

    def __init__(self, store: TrajectoryStore, task_id: str) -> None:
        self.store = store
        self.task_id = task_id
        self._memories: Dict[str, LoggedMemory] = {}

    def for_agent(self, agent_id: str) -> LoggedMemory:
        """Return (or lazily create) the LoggedMemory for one agent."""
        mem = self._memories.get(agent_id)
        if mem is None:
            mem = LoggedMemory(agent_id=agent_id, store=self.store, task_id=self.task_id)
            self._memories[agent_id] = mem
        return mem

    def agents(self) -> Iterator[str]:
        return iter(self._memories.keys())
