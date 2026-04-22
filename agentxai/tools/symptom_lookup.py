"""
symptom_lookup tool — returns conditions empirically associated with a symptom.

The lookup table is derived from MedQA US train: every question stem is scanned
for occurrences of curated symptom phrases (~100 common USMLE symptoms); each
match casts a vote for the question's correct-answer text. Likelihood = symptom-
specific share of votes for that condition. Cached as
``data/indices/symptom_table.json`` and rebuilt only when the file is missing.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from langchain_core.tools import Tool

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.tool_provenance import ToolProvenanceLogger, traced_tool


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT       = pathlib.Path(__file__).resolve().parents[2]
_US_TRAIN   = _ROOT / "Med_QA" / "questions" / "US" / "train.jsonl"
_CACHE_FILE = _ROOT / "agentxai" / "data" / "indices" / "symptom_table.json"


# ---------------------------------------------------------------------------
# Curated symptom keyword list (~100 common USMLE-style symptom phrases)
# ---------------------------------------------------------------------------

SYMPTOM_PHRASES: List[str] = [
    "fever", "chills", "fatigue", "weight loss", "weight gain", "night sweats",
    "headache", "dizziness", "vertigo", "syncope", "seizure",
    "loss of consciousness", "confusion", "memory loss", "hallucinations",
    "depression", "anxiety", "insomnia",
    "chest pain", "dyspnea", "shortness of breath", "palpitations", "orthopnea",
    "cough", "hemoptysis", "wheezing", "cyanosis",
    "tachycardia", "bradycardia", "hypertension", "hypotension",
    "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation",
    "hematemesis", "melena", "hematochezia", "jaundice", "ascites",
    "dysphagia", "heartburn", "anorexia",
    "polyuria", "oliguria", "hematuria", "dysuria",
    "urinary incontinence", "urinary retention",
    "flank pain", "back pain", "joint pain",
    "muscle weakness", "paralysis", "tremor",
    "numbness", "tingling", "paresthesia", "ataxia",
    "rash", "pruritus", "itching", "erythema", "petechiae", "purpura",
    "lymphadenopathy", "edema", "swelling", "pallor",
    "blurred vision", "diplopia", "photophobia",
    "tinnitus", "hearing loss", "sore throat", "nasal congestion", "epistaxis",
    "neck stiffness", "dysarthria", "dysphasia", "aphasia", "amnesia",
    "polydipsia", "polyphagia",
    "amenorrhea", "dysmenorrhea", "menorrhagia", "vaginal bleeding",
    "erectile dysfunction", "galactorrhea", "gynecomastia", "infertility",
    "clubbing", "hepatomegaly", "splenomegaly", "abdominal distension",
    "anemia", "bruising", "bleeding",
]

# Compile once: phrase → word-boundary regex (case-insensitive)
_PHRASE_PATTERNS: Dict[str, re.Pattern] = {
    phrase: re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
    for phrase in SYMPTOM_PHRASES
}


# ---------------------------------------------------------------------------
# Table build / load
# ---------------------------------------------------------------------------

def _build_table() -> Dict[str, Dict[str, int]]:
    """Scan US train; return {symptom_phrase: {condition_text: count}}."""
    if not _US_TRAIN.exists():
        raise FileNotFoundError(
            f"MedQA US train not found at {_US_TRAIN}. "
            "Cannot build symptom_lookup table."
        )

    table: Dict[str, Counter] = defaultdict(Counter)
    with _US_TRAIN.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stem    = rec.get("question", "")
            answer  = (rec.get("answer") or "").strip().rstrip(".").strip()
            if not stem or not answer:
                continue
            for phrase, pattern in _PHRASE_PATTERNS.items():
                if pattern.search(stem):
                    table[phrase][answer] += 1

    return {phrase: dict(counter) for phrase, counter in table.items()}


def _load_table() -> Dict[str, Dict[str, int]]:
    """Load cached lookup table, building and caching it on first call."""
    if _CACHE_FILE.exists():
        with _CACHE_FILE.open(encoding="utf-8") as fh:
            return json.load(fh)

    table = _build_table()
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False, indent=2)
    return table


_TABLE: Dict[str, Dict[str, int]] = _load_table()


# ---------------------------------------------------------------------------
# Default in-memory provenance logger (overrideable per agent run)
# ---------------------------------------------------------------------------

_DEFAULT_TASK_ID = "default-symptom-lookup-task"
_DEFAULT_STORE   = TrajectoryStore(db_url="sqlite:///:memory:")
_DEFAULT_STORE.save_task(AgentXAIRecord(task_id=_DEFAULT_TASK_ID, source="tool"))
_DEFAULT_LOGGER  = ToolProvenanceLogger(store=_DEFAULT_STORE, task_id=_DEFAULT_TASK_ID)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

@traced_tool(_DEFAULT_LOGGER, called_by="default", tool_name="symptom_lookup")
def symptom_lookup(symptom: str) -> dict:
    """
    Return MedQA-derived conditions associated with *symptom*.

    Parameters
    ----------
    symptom : free-text symptom phrase (e.g. "chest pain", "fever").

    Returns
    -------
    {
        "related_conditions": [(condition_text, likelihood_float), ...],
        "source": "medqa_derived",
    }
    Likelihood is the symptom-specific share of correct-answer votes. Empty
    list if the phrase is not in the curated keyword set.
    """
    key = (symptom or "").strip().lower()
    counts = _TABLE.get(key, {})
    total = sum(counts.values())
    if total == 0:
        return {"related_conditions": [], "source": "medqa_derived"}

    related: List[Tuple[str, float]] = [
        (cond, round(c / total, 4))
        for cond, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    return {"related_conditions": related, "source": "medqa_derived"}


# ---------------------------------------------------------------------------
# LangChain Tool wrapper
# ---------------------------------------------------------------------------

symptom_lookup_tool = Tool(
    name="symptom_lookup",
    description=(
        "Return a list of (condition, likelihood) pairs empirically associated "
        "with the given symptom phrase, derived from MedQA US train answers."
    ),
    func=symptom_lookup,
)
