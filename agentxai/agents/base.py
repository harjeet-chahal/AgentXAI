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
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from agentxai.agents._llm_utils import extract_text, parse_json_list
from agentxai.data.schemas import TrajectoryEvent
from agentxai.xai.memory_logger import MemoryLogger, current_event_id
from agentxai.xai.message_logger import MessageLogger
from agentxai.xai.plan_tracker import PlanTracker
from agentxai.xai.trajectory_logger import TrajectoryLogger


# ---------------------------------------------------------------------------
# Default LLM factory (Gemini via langchain-google-genai, direct Google API)
#
# Reads keys from GOOGLE_API_KEYS (comma-separated, plural) or GOOGLE_API_KEY
# (single). Multiple keys → automatic round-robin with 429 fall-over via
# RotatingGeminiLLM. See agentxai/_llm_factory.py for details.
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "gemini-2.5-flash-lite"
DEFAULT_LLM_TEMPERATURE = 0.0


def make_default_llm(
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> Any:
    """Thin wrapper around ``agentxai._llm_factory.build_gemini_llm``."""
    from agentxai._llm_factory import build_gemini_llm
    return build_gemini_llm(model=model, temperature=temperature)


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

        # Rolling snapshot of the agent's memory after the most recent
        # ``log_action`` call. The next action's ``state_before`` is this
        # value; its ``state_after`` replaces it. Populates the otherwise
        # empty state_before / state_after fields on TrajectoryEvent.
        self._last_state_snapshot: Dict[str, Any] = {}

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

    def generate_plan(
        self,
        case_context: str,
        available_actions: List[str],
    ) -> List[str]:
        """
        Ask the LLM to choose a subset (and ordering) of ``available_actions``
        appropriate for ``case_context``. Returns the chosen plan as a list
        of action names, filtered to the available set.

        Falls back to the full ``available_actions`` list when:
          - no LLM is wired,
          - the LLM raises,
          - the response is empty / not valid JSON,
          - or no returned name is a valid action.
        """
        if self.llm is None or not available_actions:
            return list(available_actions)

        prompt = (
            "Given this clinical case, choose which of these actions to "
            "perform and in what order. Return ONLY a JSON array of action "
            "names. Available actions: "
            f"{list(available_actions)}. Case: {case_context}"
        )
        try:
            response = self.llm.invoke(prompt)
        except Exception:
            return list(available_actions)

        parsed = parse_json_list(extract_text(response))
        allowed = set(available_actions)
        plan = [a for a in parsed if a in allowed]
        if not plan:
            return list(available_actions)
        return plan

    @contextlib.contextmanager
    def active_plan(self, intended_actions: List[str]) -> Iterator[str]:
        """
        Context manager: register a plan on entry and finalise it on exit
        (even if the body raises). Yields the new ``plan_id``.

        While the block is open, ``log_action`` calls are auto-attributed to
        this plan. Nesting is supported — the innermost plan wins.

        Also snapshots ``current_event_id`` on entry and restores it on
        exit. Both ``log_action`` and ``traced_action`` push event_ids onto
        that contextvar; without this snapshot they would leak past the
        plan boundary and contaminate any later memory writes (especially
        in shared-process test runners).
        """
        plan_id = self.emit_plan(intended_actions)
        # `.set(.get(...))` creates a token whose `.reset(token)` restores
        # the contextvar to its value immediately before this line — i.e.
        # any internal `current_event_id.set(...)` calls are unwound on exit.
        cev_token = current_event_id.set(current_event_id.get(""))
        try:
            yield plan_id
        finally:
            try:
                current_event_id.reset(cev_token)
            except Exception:
                pass
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
        Record one *already-executed* action that involved no further memory
        writes the caller cares about linking.

        - If a plan is active, append ``action`` to its actual-actions list
          (``PlanTracker.record_actual_action``).
        - Always write a TrajectoryEvent (event_type=``"action"``) via the
          trajectory logger.
        - Push this event's id onto ``current_event_id`` so any *subsequent*
          stray memory write (e.g., a write done between this action and the
          next) attributes to it rather than landing untraced. This is a
          best-effort default — for action blocks that *contain* memory
          writes, prefer ``traced_action(...)`` so the writes link to the
          enclosing event rather than the previous one.

        Returns the persisted TrajectoryEvent.
        """
        plan_id = self.active_plan_id
        if plan_id is not None:
            self.plan_tracker.record_actual_action(plan_id, action)

        state_after = self._snapshot_own_memory()
        event = self.trajectory_logger.log_event(
            agent_id=self.agent_id,
            event_type="action",
            state_before=self._last_state_snapshot,
            action=action,
            action_inputs=dict(inputs or {}),
            state_after=state_after,
            outcome=outcome,
        )
        self._last_state_snapshot = state_after
        # Attribute any straggler writes between actions to the most-recent
        # action — better than dropping them as "outside_traced_action".
        current_event_id.set(event.event_id)
        return event

    @contextlib.contextmanager
    def traced_action(
        self,
        action: str,
        inputs: Optional[Dict[str, Any]] = None,
        outcome: str = "",
    ) -> Iterator[TrajectoryEvent]:
        """
        Run a block of code as a single traced action with proper memory
        attribution.

        Lifecycle:
            1. Snapshot ``state_before`` from current memory.
            2. Persist a provisional TrajectoryEvent (state_after = state_before).
            3. Push the event_id onto ``current_event_id`` so every memory
               write inside the block emits a MemoryDiff linked to this event.
            4. Yield the event.
            5. On exit (success or exception): refresh ``state_after`` to the
               post-block memory snapshot, re-save the event (upsert by id),
               and reset the contextvar.

        Use this for any block that performs memory writes you want linked
        to a specific action — it replaces the older "write first, then
        log_action" pattern that left every diff untraced.
        """
        plan_id = self.active_plan_id
        if plan_id is not None:
            self.plan_tracker.record_actual_action(plan_id, action)

        state_before = self._snapshot_own_memory()
        event = self.trajectory_logger.log_event(
            agent_id=self.agent_id,
            event_type="action",
            state_before=state_before,
            action=action,
            action_inputs=dict(inputs or {}),
            # Provisional state_after; refreshed on exit.
            state_after=state_before,
            outcome=outcome,
        )

        token = current_event_id.set(event.event_id)
        try:
            yield event
        finally:
            try:
                current_event_id.reset(token)
            except Exception:
                # contextvar reset can fail across event loops; best-effort.
                pass
            state_after = self._snapshot_own_memory()
            self._last_state_snapshot = state_after
            if state_after != state_before:
                # Upsert the event with the real post-write state. The store
                # keys events by event_id, so this replaces the provisional
                # row in-place rather than creating a new event.
                event.state_after = state_after
                self.trajectory_logger.store.save_event(
                    self.trajectory_logger.task_id, event,
                )

    def _snapshot_own_memory(self) -> Dict[str, Any]:
        """JSON-safe deep copy of the agent's current memory dict."""
        try:
            return json.loads(json.dumps(dict(self.memory), default=str))
        except (TypeError, ValueError):
            return {k: str(v) for k, v in dict(self.memory).items()}

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
