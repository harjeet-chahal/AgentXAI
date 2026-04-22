"""
Orchestrator — top-level router for one MedQA case.

Pipeline (one ``active_plan`` with exactly four intended actions):

1. ``decompose_case``           — log the incoming case
2. ``route_to_specialist_a``    — call SpecialistA.run(payload)
3. ``route_to_specialist_b``    — call SpecialistB.run(payload)
4. ``handoff_to_synthesizer``   — call Synthesizer.run(payload + specialist ids)

Specialists are invoked **sequentially** (not parallel) on purpose so that
the trajectory log stays linear and causal-DAG construction (Pillar 6) sees
unambiguous ordering. Findings flow back to the orchestrator both via the
specialists' ``finding`` messages (Pillar 5) and via memory writes that the
Synthesizer reads directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agentxai.agents.base import TracedAgent


_ORCHESTRATOR_PLAN: List[str] = [
    "decompose_case",
    "route_to_specialist_a",
    "route_to_specialist_b",
    "handoff_to_synthesizer",
]


class Orchestrator(TracedAgent):
    """Receives a case, fans out to two specialists, hands off to the synthesizer."""

    def __init__(
        self,
        *,
        agent_id: str = "orchestrator",
        specialist_a: TracedAgent,
        specialist_b: TracedAgent,
        synthesizer: TracedAgent,
        llm: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id=agent_id, llm=llm, **kwargs)
        self.specialist_a = specialist_a
        self.specialist_b = specialist_b
        self.synthesizer = synthesizer

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, input_payload: dict) -> dict:
        case = (
            input_payload.get("patient_case")
            or input_payload.get("question")
            or ""
        )

        with self.active_plan(_ORCHESTRATOR_PLAN):
            # 1. Decompose / acknowledge the case.
            self.log_action(
                "decompose_case",
                {"case_chars": len(case), "has_options": bool(input_payload.get("options"))},
                outcome="case_received",
            )

            # 2. Route to Specialist A (sequential — keeps trajectory linear).
            findings_a = self.specialist_a.run(dict(input_payload)) or {}
            self.log_action(
                "route_to_specialist_a",
                {"specialist_id": self.specialist_a.agent_id},
                outcome=f"received {len(findings_a)} keys",
            )

            # 3. Route to Specialist B.
            findings_b = self.specialist_b.run(dict(input_payload)) or {}
            self.log_action(
                "route_to_specialist_b",
                {"specialist_id": self.specialist_b.agent_id},
                outcome=f"received {len(findings_b)} keys",
            )

            # 4. Hand off to the Synthesizer with both specialist agent_ids so
            #    it can read their memories directly.
            handoff_payload: Dict[str, Any] = {
                **input_payload,
                "specialist_a_id": self.specialist_a.agent_id,
                "specialist_b_id": self.specialist_b.agent_id,
            }
            final = self.synthesizer.run(handoff_payload) or {}
            self.log_action(
                "handoff_to_synthesizer",
                {"synthesizer_id": self.synthesizer.agent_id},
                outcome=str(final.get("final_diagnosis", "") or "no diagnosis"),
            )

        return {
            "final_output":      final,
            "specialist_a_id":   self.specialist_a.agent_id,
            "specialist_b_id":   self.specialist_b.agent_id,
        }

    # ------------------------------------------------------------------
    # Helper for inspecting collected findings via the message channel.
    # ------------------------------------------------------------------

    def collected_findings(self) -> List[Dict[str, Any]]:
        """
        Return all ``finding`` messages addressed to this orchestrator for the
        current task — provided as an explicit accessor so downstream code
        (or tests) can verify the message-based collection path without
        threading the store through.
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
