"""
Pillar 2 — Plans.

PlanTracker captures each agent's intended actions at the start of its run,
records what it actually does, and — on finalize — computes the symmetric
set difference (order-preserving) between the two lists and asks Gemini
(gemini-2.5-flash via langchain-google-genai) to produce one sentence of
reasoning per deviation, grounded in the agent's recent trajectory.

All plans are persisted through TrajectoryStore.save_plan. An in-memory
cache of AgentPlan objects keyed by plan_id is the authoritative live copy
during a run — the store holds snapshots after each mutation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agentxai.data.schemas import AgentPlan, TrajectoryEvent
from agentxai.store.trajectory_store import TrajectoryStore

try:  # Optional runtime dep — tests inject a fake LLM instead.
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]


_DEFAULT_MODEL = "gemini-2.5-flash-lite"


class PlanTracker:
    """Register, update, and finalize AgentPlan records for one task."""

    def __init__(
        self,
        store: TrajectoryStore,
        task_id: str,
        llm: Any = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.model = model
        self._plans: Dict[str, AgentPlan] = {}

        if llm is None and ChatGoogleGenerativeAI is not None:
            try:
                llm = ChatGoogleGenerativeAI(model=model, temperature=0)
            except Exception:
                llm = None
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_plan(self, agent_id: str, intended_actions: List[str]) -> AgentPlan:
        """Create a fresh AgentPlan, persist it, and cache it locally."""
        plan = AgentPlan(
            agent_id=agent_id,
            intended_actions=list(intended_actions),
            actual_actions=[],
            deviations=[],
            deviation_reasons=[],
        )
        self._plans[plan.plan_id] = plan
        self.store.save_plan(self.task_id, plan)
        return plan

    def record_actual_action(self, plan_id: str, action: str) -> AgentPlan:
        """Append an executed action and re-persist the plan."""
        plan = self._get_plan(plan_id)
        plan.actual_actions.append(action)
        self.store.save_plan(self.task_id, plan)
        return plan

    def finalize_plan(self, plan_id: str) -> AgentPlan:
        """
        Compute deviations (intended △ actual, order-preserving) and fill in
        one-sentence LLM explanations for each. Persists the final plan.
        """
        plan = self._get_plan(plan_id)
        plan.deviations = _symmetric_diff(plan.intended_actions, plan.actual_actions)
        plan.deviation_reasons = self._reason_for_deviation(plan)
        self.store.save_plan(self.task_id, plan)
        return plan

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_plan(self, plan_id: str) -> AgentPlan:
        if plan_id not in self._plans:
            raise KeyError(f"Plan {plan_id!r} has not been registered.")
        return self._plans[plan_id]

    def _reason_for_deviation(self, plan: AgentPlan) -> List[str]:
        """One short sentence per deviation, in the same order as plan.deviations."""
        if not plan.deviations:
            return []
        if self.llm is None:
            return ["LLM unavailable; no explanation." for _ in plan.deviations]

        trajectory = self._trajectory_for_agent(plan.agent_id)
        prompt = _build_prompt(plan, trajectory)

        try:
            response = self.llm.invoke(prompt)
            text = _extract_text(response)
            return _parse_reasons(text, expected_count=len(plan.deviations))
        except Exception:
            return ["LLM error; no explanation." for _ in plan.deviations]

    def _trajectory_for_agent(self, agent_id: str) -> List[TrajectoryEvent]:
        try:
            record = self.store.get_full_record(self.task_id)
        except KeyError:
            return []
        return [e for e in record.xai_data.trajectory if e.agent_id == agent_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symmetric_diff(intended: List[str], actual: List[str]) -> List[str]:
    """Order-preserving symmetric difference: missing first, then unexpected."""
    actual_set = set(actual)
    intended_set = set(intended)
    missing = [a for a in intended if a not in actual_set]
    extra = [a for a in actual if a not in intended_set]
    return missing + extra


def _build_prompt(plan: AgentPlan, trajectory: List[TrajectoryEvent]) -> str:
    trajectory_summary = [
        {
            "timestamp": round(e.timestamp, 3),
            "event_type": e.event_type,
            "action": e.action,
            "outcome": e.outcome,
        }
        for e in trajectory[-10:]
    ]
    return (
        "You are analyzing a plan-vs-execution diff for a single agent in a "
        "multi-agent system.\n"
        f"Agent: {plan.agent_id}\n"
        f"Intended actions: {json.dumps(plan.intended_actions)}\n"
        f"Actual actions:   {json.dumps(plan.actual_actions)}\n"
        f"Deviations (in order): {json.dumps(plan.deviations)}\n"
        f"Recent trajectory events for this agent:\n"
        f"{json.dumps(trajectory_summary, indent=2)}\n\n"
        "For each deviation above, in order, write ONE short sentence explaining "
        "why it most likely occurred, grounded in the trajectory. Return ONLY a "
        f"JSON array of exactly {len(plan.deviations)} strings, nothing else."
    )


_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_text(response: Any) -> str:
    """Normalize whatever a chat model returned into a plain string."""
    content = getattr(response, "content", response)
    if isinstance(content, list):  # Anthropic content-block list
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _parse_reasons(text: str, expected_count: int) -> List[str]:
    """Defensive parse of an LLM response into exactly `expected_count` strings."""
    reasons: List[str] = []

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            reasons = [str(x).strip() for x in parsed]
    except Exception:
        pass

    if not reasons:
        match = _ARRAY_RE.search(text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    reasons = [str(x).strip() for x in parsed]
            except Exception:
                pass

    if not reasons:
        # Last resort: treat each non-empty stripped line as one reason.
        reasons = [
            ln.strip(" -*\t•")
            for ln in text.splitlines()
            if ln.strip(" -*\t•")
        ]

    if len(reasons) < expected_count:
        reasons = reasons + ["No explanation."] * (expected_count - len(reasons))
    elif len(reasons) > expected_count:
        reasons = reasons[:expected_count]
    return reasons
