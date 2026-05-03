"""
Orchestrator — top-level router for one MedQA case.

Dynamic routing (replaces the old fixed decompose → A → B → synth pipeline):

1. ``decompose_case`` — log the incoming case.
2. Iterative ``routing_decision`` loop, bounded by ``max_iterations``. Each
   tick the LLM is asked which action to take next given the findings so
   far: ``call_specialist_a``, ``call_specialist_b``, or ``synthesize``.
   Re-calls of the same specialist are allowed and may carry a free-form
   ``feedback_to_specialist`` string that is injected into the specialist's
   input payload so its prompt can react to it.
3. ``handoff_to_synthesizer`` — call Synthesizer.run(payload + specialist ids).

The plan registered for Pillar 2 is itself LLM-generated via
``generate_plan`` over ``_ORCHESTRATOR_PLAN`` (the orchestrator's available
actions), so the intended-actions list is no longer hardcoded.

Specialists are invoked sequentially (never in parallel) — the trajectory
log stays linear and the causal-DAG builder (Pillar 6) sees unambiguous
ordering. The DAG already handles arbitrary trajectories (re-calls, skips)
so no changes there are required for dynamic routing.

Findings flow back to the orchestrator both via the specialists' ``finding``
messages (Pillar 5) and via memory writes that the Synthesizer reads
directly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agentxai.agents._llm_utils import extract_text, parse_json_object
from agentxai.agents.base import TracedAgent


_ORCHESTRATOR_PLAN: List[str] = [
    "decompose_case",
    "routing_decision",
    "route_to_specialist_a",
    "route_to_specialist_b",
    "handoff_to_synthesizer",
]


_ROUTING_PROMPT = (
    "You are routing a multi-agent diagnostic pipeline.\n"
    "Specialists called so far: {specialists_called_so_far}\n"
    "Findings gathered so far:  {summary_of_findings}\n\n"
    "Rules — read carefully:\n"
    "  1. Specialists are how findings get GATHERED. 'synthesize' means "
    "finalize an answer FROM findings. Synthesizing with zero findings is "
    "wrong by definition.\n"
    "  2. If the 'specialists called so far' list is empty, the correct "
    "action is ALWAYS to call a specialist first — never 'synthesize'. "
    "Pick 'call_specialist_a' (symptom / condition analysis) or "
    "'call_specialist_b' (textbook / guideline retrieval) based on what "
    "the case most needs.\n"
    "  3. 'synthesize' is ONLY valid once at least one specialist has run "
    "and produced findings.\n"
    "  4. Re-call a specialist that has already run ONLY when its findings "
    "have a concrete, actionable gap that targeted feedback could fix. "
    "If both specialists have produced reasonable findings, prefer "
    "'synthesize'.\n\n"
    "Choose the next action from:\n"
    "  - 'call_specialist_a' — symptom/condition analysis\n"
    "  - 'call_specialist_b' — textbook/guideline retrieval\n"
    "  - 'synthesize'        — finalize (requires prior findings)\n"
    'Return ONLY JSON: {{"next_action": str, "reason": str, '
    '"feedback_to_specialist": str}}.'
)


_VALID_NEXT_ACTIONS = {"call_specialist_a", "call_specialist_b", "synthesize"}


class Orchestrator(TracedAgent):
    """Receives a case, routes between two specialists, hands off to the synthesizer."""

    def __init__(
        self,
        *,
        agent_id: str = "orchestrator",
        specialist_a: TracedAgent,
        specialist_b: TracedAgent,
        synthesizer: TracedAgent,
        llm: Any = None,
        max_iterations: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id=agent_id, llm=llm, **kwargs)
        self.specialist_a = specialist_a
        self.specialist_b = specialist_b
        self.synthesizer = synthesizer
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, input_payload: dict) -> dict:
        case = (
            input_payload.get("patient_case")
            or input_payload.get("question")
            or ""
        )

        plan = self.generate_plan(case, _ORCHESTRATOR_PLAN)

        findings_so_far: Dict[str, Any] = {}
        specialists_called: List[str] = []
        final: Dict[str, Any] = {}

        with self.active_plan(plan):
            # 1. Decompose / acknowledge the case.
            self.log_action(
                "decompose_case",
                {"case_chars": len(case), "has_options": bool(input_payload.get("options"))},
                outcome="case_received",
            )

            # 2. Iterative routing loop.
            forced_synthesis = False
            for iteration in range(self.max_iterations):
                decision = self._decide_next_action(
                    case, specialists_called, findings_so_far,
                )
                next_action = decision["next_action"]
                reason = decision["reason"]
                feedback = decision["feedback_to_specialist"]

                self.log_action(
                    "routing_decision",
                    {
                        "iteration": iteration,
                        "next_action": next_action,
                        "specialists_called_so_far": list(specialists_called),
                        "feedback_to_specialist": feedback,
                    },
                    outcome=reason or next_action,
                )

                if next_action == "synthesize":
                    break

                if next_action == "call_specialist_a":
                    spec = self.specialist_a
                    action_name = "route_to_specialist_a"
                elif next_action == "call_specialist_b":
                    spec = self.specialist_b
                    action_name = "route_to_specialist_b"
                else:
                    # Defensive: an unknown next_action shouldn't reach here
                    # (validated in _decide_next_action) — bail to synthesize.
                    break

                spec_payload: Dict[str, Any] = dict(input_payload)
                if feedback:
                    spec_payload["feedback_from_orchestrator"] = feedback
                spec_findings = spec.run(spec_payload) or {}
                findings_so_far[spec.agent_id] = spec_findings
                specialists_called.append(spec.agent_id)
                self.log_action(
                    action_name,
                    {
                        "specialist_id": spec.agent_id,
                        "had_feedback":  bool(feedback),
                    },
                    outcome=f"received {len(spec_findings)} keys",
                )
            else:
                # Loop completed without seeing 'synthesize' — guard rail
                # against runaway routing. Force the handoff and surface
                # the reason in the routing log.
                forced_synthesis = True
                self.log_action(
                    "routing_decision",
                    {
                        "iteration": self.max_iterations,
                        "forced":    True,
                    },
                    outcome="max_iterations exhausted; forcing synthesis",
                )

            # 3. Hand off to the Synthesizer with both specialist agent_ids
            #    so it can read their memories directly.
            handoff_payload: Dict[str, Any] = {
                **input_payload,
                "specialist_a_id": self.specialist_a.agent_id,
                "specialist_b_id": self.specialist_b.agent_id,
            }
            final = self.synthesizer.run(handoff_payload) or {}
            outcome = str(final.get("final_diagnosis", "") or "no diagnosis")
            if forced_synthesis:
                outcome = f"{outcome} (forced)"
            self.log_action(
                "handoff_to_synthesizer",
                {"synthesizer_id": self.synthesizer.agent_id},
                outcome=outcome,
            )

        return {
            "final_output":    final,
            "specialist_a_id": self.specialist_a.agent_id,
            "specialist_b_id": self.specialist_b.agent_id,
        }

    # ------------------------------------------------------------------
    # Routing helper
    # ------------------------------------------------------------------

    def _decide_next_action(
        self,
        case: str,
        specialists_called: List[str],
        findings_so_far: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Ask the LLM what to do next. Returns a dict with keys
        ``next_action``, ``reason``, ``feedback_to_specialist``.

        Falls back to a deterministic A → B → synthesize rotation when no
        LLM is wired, the LLM raises, or the response can't be parsed —
        same default-on-failure stance as the specialist gating helpers.
        The fallback also applies if the LLM returns a ``next_action``
        outside the allowed set.
        """
        fallback_sequence = ["call_specialist_a", "call_specialist_b", "synthesize"]
        fallback_action = fallback_sequence[min(len(specialists_called), 2)]
        fallback: Dict[str, str] = {
            "next_action":            fallback_action,
            "reason":                 "fallback (no LLM decision)",
            "feedback_to_specialist": "",
        }

        if self.llm is None:
            return fallback

        # Cap the findings dump so the prompt stays bounded even if a
        # specialist returns a giant retrieved_docs blob.
        try:
            summary = json.dumps(findings_so_far, default=str)[:2000]
        except (TypeError, ValueError):
            summary = str(findings_so_far)[:2000]

        prompt = _ROUTING_PROMPT.format(
            specialists_called_so_far=specialists_called,
            summary_of_findings=summary,
        )
        try:
            response = self.llm.invoke(prompt)
        except Exception:
            return fallback

        parsed = parse_json_object(extract_text(response))
        if not parsed:
            return fallback

        next_action = str(parsed.get("next_action") or "").strip()
        if next_action not in _VALID_NEXT_ACTIONS:
            return fallback

        # Iteration-0 guard: synthesizing before any specialist has run
        # produces an answer with zero findings to draw on, which is
        # wrong by definition. Override to call_specialist_a so the
        # pipeline can gather evidence first.
        if not specialists_called and next_action == "synthesize":
            return {
                "next_action":            "call_specialist_a",
                "reason":                 "iteration 0 guard: must gather findings before synthesis",
                "feedback_to_specialist": "",
            }

        return {
            "next_action":            next_action,
            "reason":                 str(parsed.get("reason") or ""),
            "feedback_to_specialist": str(parsed.get("feedback_to_specialist") or ""),
        }

    # ------------------------------------------------------------------
    # Helper for inspecting collected findings via the message channel.
    # ------------------------------------------------------------------

    def collected_findings(self) -> List[Dict[str, Any]]:
        """
        Return all ``finding`` messages addressed to this orchestrator.

        Not called from `run()` — the orchestrator's own diagnostic flow
        feeds findings forward via memory + the synthesizer's read step.
        This accessor exists so downstream callers (test fixtures, debug
        scripts, future routing variants) can inspect the message-based
        collection path without needing to thread the store through.

        If you find yourself wondering whether to delete this: check
        `tests/test_agents.py::test_orchestrator_routes_to_specialists_and_synthesizer`
        — it asserts the message channel is populated correctly.
        """
        try:
            record = self.message_logger.store.get_full_record(
                self.message_logger.task_id,
            )
        except KeyError:
            return []
        return [
            {
                "sender":  m.sender,
                "type":    m.message_type,
                "content": m.content,
            }
            for m in record.xai_data.messages
            if m.receiver == self.agent_id and m.message_type == "finding"
        ]
