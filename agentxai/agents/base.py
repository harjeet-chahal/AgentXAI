"""
TracedAgent — base class for every AgentXAI specialist.

Wires together the four cross-cutting loggers (trajectory, plan, memory,
message) so subclasses only need to implement ``run(input_payload)``. The
canonical lifecycle inside a subclass is:

    class SpecialistA(TracedAgent):
        def run(self, input_payload):
            with self.active_plan(["lookup_symptoms", "score_severity"]):
                ...
                self.log_action("lookup_symptoms", {"q": q}, outcome="ok")
                ...
                self.send_message("synthesizer", "finding", {"score": 0.7})
            return {...}

The context manager registers a plan on entry and finalises it on exit (even
on exception); ``log_action`` calls inside the block are auto-attributed to
that plan via ``PlanTracker.record_actual_action`` and also written to the
trajectory log.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from agentxai.data.schemas import TrajectoryEvent
from agentxai.xai.memory_logger import MemoryLogger
from agentxai.xai.message_logger import MessageLogger
from agentxai.xai.plan_tracker import PlanTracker
from agentxai.xai.trajectory_logger import TrajectoryLogger


# ---------------------------------------------------------------------------
# Default LLM factory (Claude via langchain-anthropic)
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "claude-sonnet-4-5"
DEFAULT_LLM_TEMPERATURE = 0.0


def make_default_llm(
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> Any:
    """
    Best-effort: instantiate ``ChatAnthropic(model, temperature=0)`` and return
    it. Returns ``None`` if langchain-anthropic is not installed or the client
    cannot be constructed (e.g. missing ``ANTHROPIC_API_KEY``). Callers that
    must have a working LLM should check for ``None`` and raise.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None
    try:
        return ChatAnthropic(model=model, temperature=temperature)
    except Exception:
        return None


class TracedAgent(ABC):
    """Base class wiring trajectory, plan, memory, and message logging."""

    def __init__(
        self,
        agent_id: str,
        trajectory_logger: TrajectoryLogger,
        plan_tracker: PlanTracker,
        memory_logger: MemoryLogger,
        message_logger: MessageLogger,
        llm: Any = None,
    ) -> None:
        self.agent_id = agent_id
        self.trajectory_logger = trajectory_logger
        self.plan_tracker = plan_tracker
        self.memory_logger = memory_logger
        self.message_logger = message_logger
        self.llm = llm

        # Stack of active plan_ids so nested ``active_plan`` blocks behave
        # predictably; ``log_action`` attributes to the innermost plan.
        self._plan_stack: List[str] = []

    # ------------------------------------------------------------------
    # Plan API
    # ------------------------------------------------------------------

    @property
    def active_plan_id(self) -> Optional[str]:
        """plan_id of the currently-open plan, or None if no plan is active."""
        return self._plan_stack[-1] if self._plan_stack else None

    def emit_plan(self, intended_actions: List[str]) -> str:
        """
        Register a fresh plan with the plan tracker and mark it as the active
        plan for subsequent ``log_action`` calls. Returns the new ``plan_id``.

        Most callers should prefer the ``active_plan`` context manager, which
        also finalises the plan on exit.
        """
        plan = self.plan_tracker.register_plan(self.agent_id, list(intended_actions))
        self._plan_stack.append(plan.plan_id)
        return plan.plan_id

    @contextlib.contextmanager
    def active_plan(self, intended_actions: List[str]) -> Iterator[str]:
        """
        Context manager: register a plan on entry and finalise it on exit
        (even if the body raises). Yields the new ``plan_id``.

        While the block is open, ``log_action`` calls are auto-attributed to
        this plan. Nesting is supported — the innermost plan wins.
        """
        plan_id = self.emit_plan(intended_actions)
        try:
            yield plan_id
        finally:
            try:
                self.plan_tracker.finalize_plan(plan_id)
            finally:
                # Pop our own plan even if the caller did weird things to the
                # stack; only remove the entry we pushed.
                try:
                    self._plan_stack.remove(plan_id)
                except ValueError:
                    pass

    # ------------------------------------------------------------------
    # Action API
    # ------------------------------------------------------------------

    def log_action(
        self,
        action: str,
        inputs: Optional[Dict[str, Any]] = None,
        outcome: str = "",
    ) -> TrajectoryEvent:
        """
        Record one executed action.

        - If a plan is active, append ``action`` to its actual-actions list
          (``PlanTracker.record_actual_action``).
        - Always write a TrajectoryEvent (event_type=``"action"``) via the
          trajectory logger.

        Returns the persisted TrajectoryEvent.
        """
        plan_id = self.active_plan_id
        if plan_id is not None:
            self.plan_tracker.record_actual_action(plan_id, action)

        return self.trajectory_logger.log_event(
            agent_id=self.agent_id,
            event_type="action",
            action=action,
            action_inputs=dict(inputs or {}),
            outcome=outcome,
        )

    # ------------------------------------------------------------------
    # Message API
    # ------------------------------------------------------------------

    def send_message(
        self,
        receiver: str,
        message_type: str,
        content: Dict[str, Any],
    ) -> str:
        """
        Send an inter-agent message (sender = self.agent_id) via the message
        logger and return the new ``message_id``.
        """
        msg = self.message_logger.send(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=dict(content or {}),
        )
        return msg.message_id

    # ------------------------------------------------------------------
    # Memory shortcut
    # ------------------------------------------------------------------

    @property
    def memory(self):
        """Convenience accessor for this agent's LoggedMemory dict."""
        return self.memory_logger.for_agent(self.agent_id)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, input_payload: dict) -> dict:
        """Execute one task end-to-end. Subclasses define the specifics."""
        raise NotImplementedError
