# Dataset

## MedQA

The project uses the **MedQA** corpus: multiple-choice medical board exam questions in three languages (English USMLE, Simplified Chinese, Traditional Chinese) plus an accompanying textbook retrieval corpus.

Only the **US split** (USMLE-style, English, 5 options) is used for training and evaluation.

### File layout

```
med_qa/
├── questions/
│   ├── US/
│   │   ├── train.jsonl       (~10 373 questions, 10.3 MB)
│   │   ├── dev.jsonl         (~1 272 questions)
│   │   ├── test.jsonl        (~1 273 questions)
│   │   └── US_qbank.jsonl    (train+dev+test combined, 14.1 MB)
│   ├── Mainland/             (Simplified Chinese, 5 options)
│   └── Taiwan/               (Traditional Chinese, 4 options)
└── textbooks/
    ├── en/                   (18 English medical textbooks, plain UTF-8 .txt)
    │   ├── InternalMed_Harrison.txt          (~22 MB)
    │   ├── Pathology_Robbins.txt
    │   ├── Pharmacology_Katzung.txt
    │   ├── Psichiatry_DSM-5.txt              (upstream typo — preserve as-is)
    │   ├── Obstentrics_Williams.txt          (upstream typo — preserve as-is)
    │   └── ...
    ├── zh_paragraph/         (Chinese textbooks, one paragraph per line)
    └── zh_sentence/          (Chinese textbooks, one sentence per line)
```

### Raw MedQA record schema

Every line in `questions/US/*.jsonl` is one JSON object:

```json
{
  "question":   "A 21-year-old sexually active male presents with ...",
  "answer":     "Ceftriaxone",
  "options": {
    "A": "Chloramphenicol",
    "B": "Gentamicin",
    "C": "Ciprofloxacin",
    "D": "Ceftriaxone",
    "E": "Trimethoprim"
  },
  "meta_info":  "step1",
  "answer_idx": "D"
}
```

Field notes:
- `answer` is the full option text (not the letter).
- `answer_idx` is the letter (`"A"`–`"E"`); use this for scoring.
- The US and Mainland sets have 5 options (A–E); Taiwan has 4 (A–D).

### Normalized record schema

`load_medqa_us()` and `load_medqa_us_all()` return normalized records:

```python
{
    "task_id":    "A00001",   # globally unique across train+dev+test
    "question":   "A 21-year-old ...",
    "options":    {"A": "Chloramphenicol", ..., "D": "Ceftriaxone", "E": "Trimethoprim"},
    "answer":     "D",        # correct letter (NOT the text)
    "answer_idx": 3,          # 0-based (A=0 … E=4)
    "meta_info":  "step1",
    "raw":        { ...original record verbatim... }
}
```

### Data splits used by AgentXAI

`make_splits(records, eval_size=1500, review_size=500, seed=42)` partitions the combined corpus (train+dev+test) into three non-overlapping sets:

| Split | Size | Purpose |
|-------|------|---------|
| `demo` | ~10 000 | Trajectory generation and demonstration |
| `eval` | 1 500 | Accuracy evaluation (ground-truth answer used) |
| `review` | 500 | Manual XAI quality rating (plausibility, completeness, specificity, causal coherence) |

The split is deterministic (seed=42) so the same 500 records are always selected for manual review.

---

## AgentXAIRecord

After the pipeline runs, each normalized record is expanded into a fully-hydrated `AgentXAIRecord` stored in SQLite and returned by `TrajectoryStore.get_full_record(task_id)`.

### Top-level schema

```python
@dataclass
class AgentXAIRecord:
    task_id:       str            # UUID assigned at pipeline start
    source:        str            # "medqa"
    input:         dict           # patient_case, options, raw_task_id, meta_info
    ground_truth:  dict           # correct_answer (letter), answer_text, explanation
    system_output: dict           # final_diagnosis, confidence, correct, predicted_letter
    xai_data:      XAIData        # all 7 pillars (see below)
```

### XAIData bundle

```python
@dataclass
class XAIData:
    trajectory:           List[TrajectoryEvent]
    plans:                List[AgentPlan]
    tool_calls:           List[ToolUseEvent]
    memory_diffs:         List[MemoryDiff]
    messages:             List[AgentMessage]
    causal_graph:         CausalGraph          # nodes (event_ids) + edges (CausalEdge)
    accountability_report: Optional[AccountabilityReport]
```

### Example record (pretty-printed)

The following is a representative (abbreviated) `AgentXAIRecord` for a USMLE Step 1 question about gonorrhea treatment. JSON blobs that would be long are truncated with `...`.

```json
{
  "task_id": "3f8a2c1d-9b4e-4f7a-8d2e-1a6b3c5d7e9f",
  "source": "medqa",

  "input": {
    "patient_case": "A 21-year-old sexually active male presents to the clinic with a 3-day history of urethral discharge and dysuria. Gram stain of the discharge reveals gram-negative diplococci. Which of the following is the most appropriate treatment?",
    "options": {
      "A": "Chloramphenicol",
      "B": "Gentamicin",
      "C": "Ciprofloxacin",
      "D": "Ceftriaxone",
      "E": "Trimethoprim"
    },
    "raw_task_id": "A00042",
    "meta_info": "step1"
  },

  "ground_truth": {
    "correct_answer": "D",
    "answer_text": "Ceftriaxone",
    "explanation": "step1"
  },

  "system_output": {
    "final_diagnosis": "Ceftriaxone",
    "confidence": 0.91,
    "differential": ["Ceftriaxone", "Ciprofloxacin"],
    "rationale": "Gram-negative diplococci are characteristic of Neisseria gonorrhoeae. Current CDC guidelines recommend ceftriaxone as first-line therapy ...",
    "predicted_letter": "D",
    "predicted_text": "Ceftriaxone",
    "correct": true
  },

  "xai_data": {

    "trajectory": [
      {
        "event_id": "evt-001",
        "timestamp": 1714000001.12,
        "agent_id": "orchestrator",
        "event_type": "action",
        "action": "route_to_specialists",
        "action_inputs": {"patient_case": "A 21-year-old ..."},
        "state_before": {"phase": "init"},
        "state_after":  {"phase": "specialist_dispatch"},
        "outcome": "dispatched specialist_a and specialist_b"
      },
      {
        "event_id": "evt-002",
        "timestamp": 1714000002.44,
        "agent_id": "specialist_a",
        "event_type": "tool_call",
        "action": "symptom_lookup",
        "action_inputs": {"symptoms": ["urethral discharge", "dysuria", "gram-negative diplococci"]},
        "state_before": {"findings": {}},
        "state_after":  {"findings": {"related_conditions": ["Gonorrhea", "NGU"]}},
        "outcome": "related_conditions=[Gonorrhea, NGU]"
      },
      { "...": "further events omitted for brevity" }
    ],

    "plans": [
      {
        "plan_id": "plan-a-001",
        "agent_id": "specialist_a",
        "timestamp": 1714000002.10,
        "intended_actions": ["symptom_lookup", "severity_scorer"],
        "actual_actions":   ["symptom_lookup", "severity_scorer"],
        "deviations": [],
        "deviation_reasons": []
      },
      {
        "plan_id": "plan-b-001",
        "agent_id": "specialist_b",
        "timestamp": 1714000003.80,
        "intended_actions": ["pubmed_search", "guideline_lookup"],
        "actual_actions":   ["pubmed_search", "guideline_lookup"],
        "deviations": [],
        "deviation_reasons": []
      }
    ],

    "tool_calls": [
      {
        "tool_call_id": "tc-001",
        "tool_name": "symptom_lookup",
        "called_by": "specialist_a",
        "timestamp": 1714000002.44,
        "inputs":  {"symptoms": ["urethral discharge", "dysuria"]},
        "outputs": {"related_conditions": ["Gonorrhea", "Non-gonococcal urethritis"], "source": "faiss"},
        "duration_ms": 84.2,
        "downstream_impact_score": 0.73,
        "counterfactual_run_id": "3f8a2c1d-..."
      },
      {
        "tool_call_id": "tc-002",
        "tool_name": "guideline_lookup",
        "called_by": "specialist_b",
        "timestamp": 1714000004.91,
        "inputs":  {"condition": "Gonorrhea"},
        "outputs": {"match": {"first_line": "Ceftriaxone 500 mg IM × 1", "source": "CDC 2021"}},
        "duration_ms": 61.7,
        "downstream_impact_score": 0.88,
        "counterfactual_run_id": "3f8a2c1d-..."
      }
    ],

    "memory_diffs": [
      {
        "diff_id": "diff-001",
        "agent_id": "specialist_a",
        "timestamp": 1714000002.50,
        "operation": "write",
        "key": "findings",
        "value_before": null,
        "value_after": {"related_conditions": ["Gonorrhea"], "severity": 0.62},
        "triggered_by_event_id": "evt-002"
      }
    ],

    "messages": [
      {
        "message_id": "msg-001",
        "sender": "specialist_a",
        "receiver": "synthesizer",
        "timestamp": 1714000003.60,
        "message_type": "finding",
        "content": {
          "related_conditions": ["Gonorrhea"],
          "severity": 0.62,
          "recommended_workup": "confirm with NAAT"
        },
        "acted_upon": true,
        "behavior_change_description": "Synthesizer weighted Gonorrhea higher after receiving severity=0.62"
      }
    ],

    "causal_graph": {
      "nodes": ["evt-001", "evt-002", "evt-003", "evt-004", "evt-005"],
      "edges": [
        {
          "edge_id": "edge-001",
          "cause_event_id":  "evt-002",
          "effect_event_id": "evt-005",
          "causal_strength": 0.73,
          "causal_type": "direct"
        },
        {
          "edge_id": "edge-002",
          "cause_event_id":  "evt-004",
          "effect_event_id": "evt-005",
          "causal_strength": 0.88,
          "causal_type": "direct"
        }
      ]
    },

    "accountability_report": {
      "task_id": "3f8a2c1d-9b4e-4f7a-8d2e-1a6b3c5d7e9f",
      "final_outcome": "Ceftriaxone",
      "outcome_correct": true,
      "agent_responsibility_scores": {
        "specialist_a": 0.37,
        "specialist_b": 0.51,
        "synthesizer":  0.12
      },
      "root_cause_event_id": "evt-004",
      "causal_chain": ["evt-001", "evt-002", "evt-003", "evt-004", "evt-005"],
      "most_impactful_tool_call_id": "tc-002",
      "critical_memory_diffs": ["diff-001"],
      "most_influential_message_id": "msg-001",
      "plan_deviation_summary": "No deviations detected across all agents.",
      "one_line_explanation": "Specialist B's guideline_lookup retrieving the CDC ceftriaxone recommendation was the decisive event."
    }

  }
}
```

---

## FAISS knowledge base

Built once from the 18 English textbooks:

```bash
python -m agentxai.data.build_knowledge_base
# outputs:
#   agentxai/data/indices/textbooks/index.faiss
#   agentxai/data/indices/textbooks/metadata.jsonl
#   agentxai/data/indices/guidelines.json
```

| Artefact | Content | Used by |
|----------|---------|---------|
| `textbooks/index.faiss` | FAISS IndexFlatIP over ~500-token passages, embedded with `all-MiniLM-L6-v2` | `pubmed_search` tool (SpecialistB) |
| `textbooks/metadata.jsonl` | Parallel chunk metadata: source file, char offsets, passage text | `pubmed_search` |
| `guidelines.json` | Structured stubs for the 50 most-common conditions in US MedQA answers | `guideline_lookup` tool (SpecialistB) |

> **Note:** `guidelines.json` contains synthetic stubs, not real clinical guidelines. Replace with authoritative data before any clinical application.
