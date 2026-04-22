"""
severity_scorer tool — rule-based 0–1 severity score for a list of symptoms.

Hand-curated weight table covering ~35 common presenting symptoms, ranging from
0.10 (mild headache) to 1.00 (cardiac arrest). The score is the mean weight of
recognised symptoms; if 3 or more *severe* symptoms (weight ≥ 0.7) co-occur,
a +0.10 acuity bonus is added. The result is capped at 1.0.

Unrecognised symptoms are silently ignored. An empty / all-unknown input
returns 0.0.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.tools import Tool

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.tool_provenance import ToolProvenanceLogger, traced_tool


# ---------------------------------------------------------------------------
# Severity weight table
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: Dict[str, float] = {
    # mild
    "mild headache":         0.10,
    "runny nose":            0.10,
    "nasal congestion":      0.10,
    "sore throat":           0.15,
    "mild fever":            0.20,
    "headache":              0.20,
    "fatigue":               0.20,
    "cough":                 0.20,
    # moderate
    "nausea":                0.25,
    "joint pain":            0.25,
    "back pain":             0.25,
    "fever":                 0.30,
    "diarrhea":              0.35,
    "vomiting":              0.40,
    "abdominal pain":        0.40,
    "dizziness":             0.40,
    "palpitations":          0.50,
    # serious
    "jaundice":              0.60,
    "dyspnea":               0.65,
    "shortness of breath":   0.65,
    "chest pain":            0.70,
    "chest pressure":        0.70,
    "severe headache":       0.70,
    "confusion":             0.70,
    "hemoptysis":            0.75,
    "melena":                0.75,
    "syncope":               0.75,
    "hematemesis":           0.80,
    "neck stiffness":        0.80,
    # critical
    "altered mental status": 0.85,
    "seizure":               0.85,
    "focal weakness":        0.85,
    "slurred speech":        0.85,
    "cyanosis":              0.85,
    "loss of consciousness": 0.95,
    "unresponsive":          0.95,
    "cardiac arrest":        1.00,
}

_SEVERE_THRESHOLD: float = 0.70
_COOCCURRENCE_BONUS: float = 0.10
_COOCCURRENCE_MIN_COUNT: int = 3


# ---------------------------------------------------------------------------
# Default in-memory provenance logger (overrideable per agent run)
# ---------------------------------------------------------------------------

_DEFAULT_TASK_ID = "default-severity-scorer-task"
_DEFAULT_STORE   = TrajectoryStore(db_url="sqlite:///:memory:")
_DEFAULT_STORE.save_task(AgentXAIRecord(task_id=_DEFAULT_TASK_ID, source="tool"))
_DEFAULT_LOGGER  = ToolProvenanceLogger(store=_DEFAULT_STORE, task_id=_DEFAULT_TASK_ID)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

@traced_tool(_DEFAULT_LOGGER, called_by="default", tool_name="severity_scorer")
def severity_scorer(symptoms: List[str]) -> float:
    """
    Rule-based 0–1 severity score for a collection of symptom phrases.

    Score = mean(weights of recognised symptoms), with a +0.10 bonus when
    three or more weights are ≥ 0.70 (severe co-occurrence). Result capped
    at 1.0. Unrecognised symptoms are ignored. Empty input returns 0.0.
    """
    if not symptoms:
        return 0.0

    weights: List[float] = []
    for s in symptoms:
        if not isinstance(s, str):
            continue
        w = SEVERITY_WEIGHTS.get(s.strip().lower())
        if w is not None:
            weights.append(w)

    if not weights:
        return 0.0

    score = sum(weights) / len(weights)
    severe_count = sum(1 for w in weights if w >= _SEVERE_THRESHOLD)
    if severe_count >= _COOCCURRENCE_MIN_COUNT:
        score += _COOCCURRENCE_BONUS

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# LangChain Tool wrapper
# ---------------------------------------------------------------------------

severity_scorer_tool = Tool(
    name="severity_scorer",
    description=(
        "Compute a 0–1 acuity score from a list of symptom phrases using a "
        "hand-curated weight table; bonus for ≥3 severe symptoms co-occurring."
    ),
    func=severity_scorer,
)
