"""
AgentXAI data schemas — one dataclass per XAI pillar plus the top-level record.

Pillar 1 — Trajectories  : TrajectoryEvent
Pillar 2 — Plans         : AgentPlan
Pillar 3 — Tool Provenance: ToolUseEvent
Pillar 4 — Memory        : MemoryDiff
Pillar 5 — Communication : AgentMessage
Pillar 6 — Causality     : CausalEdge
Pillar 7 — Accountability: AccountabilityReport
Top-level                 : AgentXAIRecord
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Pillar 1 — Trajectories
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class TrajectoryEvent:
    """Full ordered log of one agent action, state transition, and outcome."""

    event_id: str = field(default_factory=_new_id)
    timestamp: float = field(default_factory=_now)
    agent_id: str = ""
    # "plan" | "action" | "tool_call" | "message" | "memory_write"
    event_type: str = ""
    state_before: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    action_inputs: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryEvent":
        return cls(**d)


# ---------------------------------------------------------------------------
# Pillar 2 — Plans
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class AgentPlan:
    """Intended vs. actual actions per agent, with deviation tracking."""

    plan_id: str = field(default_factory=_new_id)
    agent_id: str = ""
    timestamp: float = field(default_factory=_now)
    intended_actions: List[str] = field(default_factory=list)
    actual_actions: List[str] = field(default_factory=list)   # filled post-execution
    deviations: List[str] = field(default_factory=list)       # symmetric diff
    deviation_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentPlan":
        return cls(**d)


# ---------------------------------------------------------------------------
# Pillar 3 — Tool-Use Provenance
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class ToolUseEvent:
    """A single tool call with its inputs, outputs, and counterfactual impact score."""

    tool_call_id: str = field(default_factory=_new_id)
    tool_name: str = ""
    called_by: str = ""
    timestamp: float = field(default_factory=_now)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    downstream_impact_score: float = 0.0   # 0–1, computed via counterfactual
    counterfactual_run_id: str = ""        # pointer to the re-run record

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolUseEvent":
        return cls(**d)


# ---------------------------------------------------------------------------
# Pillar 4 — Memory State Changes
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class MemoryDiff:
    """One read or write to an agent's persistent memory dict."""

    diff_id: str = field(default_factory=_new_id)
    agent_id: str = ""
    timestamp: float = field(default_factory=_now)
    operation: str = ""        # "read" | "write"
    key: str = ""
    value_before: Any = None
    value_after: Any = None
    triggered_by_event_id: str = ""   # FK → TrajectoryEvent.event_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryDiff":
        return cls(**d)


# ---------------------------------------------------------------------------
# Pillar 5 — Inter-Agent Communication
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class AgentMessage:
    """A directed message between two agents, with behavioral-change tracking."""

    message_id: str = field(default_factory=_new_id)
    sender: str = ""
    receiver: str = ""
    timestamp: float = field(default_factory=_now)
    # "routing" | "finding" | "correction" | "escalation"
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    acted_upon: bool = False
    behavior_change_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentMessage":
        return cls(**d)


# ---------------------------------------------------------------------------
# Pillar 6 — Temporal Causality
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class CausalEdge:
    """One directed edge in the causal DAG with counterfactual-estimated strength."""

    edge_id: str = field(default_factory=_new_id)
    cause_event_id: str = ""     # FK → TrajectoryEvent.event_id
    effect_event_id: str = ""    # FK → TrajectoryEvent.event_id
    causal_strength: float = 0.0  # 0–1, estimated via counterfactual perturbation
    # "direct" | "mediated" | "contributory"
    causal_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalEdge":
        return cls(**d)


# ---------------------------------------------------------------------------
# Pillar 7 — System-Wide Accountability
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class AccountabilityReport:
    """Structured accountability report synthesised from all six other pillars."""

    task_id: str = field(default_factory=_new_id)
    final_outcome: str = ""
    outcome_correct: bool = False
    agent_responsibility_scores: Dict[str, float] = field(default_factory=dict)
    root_cause_event_id: str = ""
    causal_chain: List[str] = field(default_factory=list)   # ordered event_ids
    most_impactful_tool_call_id: str = ""
    critical_memory_diffs: List[str] = field(default_factory=list)   # diff_ids
    most_influential_message_id: str = ""
    plan_deviation_summary: str = ""
    one_line_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AccountabilityReport":
        return cls(**d)


# ---------------------------------------------------------------------------
# Causal graph container (used inside AgentXAIRecord)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class CausalGraph:
    """Container for the full causal DAG (nodes = event_ids, edges = CausalEdge)."""

    nodes: List[str] = field(default_factory=list)   # event_ids
    edges: List[CausalEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalGraph":
        return cls(
            nodes=list(d.get("nodes", [])),
            edges=[CausalEdge.from_dict(e) for e in d.get("edges", [])],
        )


# ---------------------------------------------------------------------------
# XAI data bundle (nested inside AgentXAIRecord)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class XAIData:
    """All XAI artefacts produced for one pipeline run."""

    trajectory: List[TrajectoryEvent] = field(default_factory=list)
    plans: List[AgentPlan] = field(default_factory=list)
    tool_calls: List[ToolUseEvent] = field(default_factory=list)
    memory_diffs: List[MemoryDiff] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    causal_graph: CausalGraph = field(default_factory=CausalGraph)
    accountability_report: Optional[AccountabilityReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": [e.to_dict() for e in self.trajectory],
            "plans": [p.to_dict() for p in self.plans],
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "memory_diffs": [m.to_dict() for m in self.memory_diffs],
            "messages": [msg.to_dict() for msg in self.messages],
            "causal_graph": self.causal_graph.to_dict(),
            "accountability_report": (
                self.accountability_report.to_dict()
                if self.accountability_report is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "XAIData":
        ar_raw = d.get("accountability_report")
        return cls(
            trajectory=[TrajectoryEvent.from_dict(e) for e in d.get("trajectory", [])],
            plans=[AgentPlan.from_dict(p) for p in d.get("plans", [])],
            tool_calls=[ToolUseEvent.from_dict(t) for t in d.get("tool_calls", [])],
            memory_diffs=[MemoryDiff.from_dict(m) for m in d.get("memory_diffs", [])],
            messages=[AgentMessage.from_dict(msg) for msg in d.get("messages", [])],
            causal_graph=CausalGraph.from_dict(d.get("causal_graph", {})),
            accountability_report=(
                AccountabilityReport.from_dict(ar_raw) if ar_raw is not None else None
            ),
        )


# ---------------------------------------------------------------------------
# Top-level — Full Dataset Record
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class AgentXAIRecord:
    """
    Full stored record after the pipeline runs on one MedQA entry.

    Matches the "Full Dataset Record Schema" in the spec exactly.
    """

    task_id: str = field(default_factory=_new_id)
    source: str = "medqa"

    # Raw MedQA input
    input: Dict[str, Any] = field(default_factory=dict)
    # patient_case: str
    # answer_options: Dict[str, str]  (A–D or A–E)

    # Ground-truth labels
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    # correct_answer: str (letter)
    # explanation: str

    # Pipeline output
    system_output: Dict[str, Any] = field(default_factory=dict)
    # final_diagnosis: str
    # confidence: float
    # correct: bool

    # All XAI artefacts
    xai_data: XAIData = field(default_factory=XAIData)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "input": self.input,
            "ground_truth": self.ground_truth,
            "system_output": self.system_output,
            "xai_data": self.xai_data.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentXAIRecord":
        return cls(
            task_id=d.get("task_id", _new_id()),
            source=d.get("source", "medqa"),
            input=d.get("input", {}),
            ground_truth=d.get("ground_truth", {}),
            system_output=d.get("system_output", {}),
            xai_data=XAIData.from_dict(d.get("xai_data", {})),
        )
