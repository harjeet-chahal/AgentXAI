# Architecture

This document maps each component to its source file and explains how a single task flows through the full system.

---

## Component map

```
AgentXAI/
├── run_pipeline.py              ← Pipeline  (glue between agents, loggers, XAI)
├── agentxai/
│   ├── agents/
│   │   ├── base.py              ← TracedAgent base class, make_default_llm()
│   │   ├── orchestrator.py      ← Orchestrator  (routes, coordinates)
│   │   ├── specialist_a.py      ← SpecialistA   (symptom + severity)
│   │   ├── specialist_b.py      ← SpecialistB   (textbook search + guidelines — see §"Tool naming")
│   │   └── synthesizer.py       ← Synthesizer   (final diagnosis)
│   ├── tools/
│   │   ├── symptom_lookup.py    ← @traced_tool, queries FAISS index
│   │   ├── severity_scorer.py   ← @traced_tool, LLM-rated 0–1 score
│   │   ├── pubmed_search.py     ← @traced_tool, FAISS over textbook chunks
│   │   │                          (NOT real PubMed — see §"Tool naming")
│   │   └── guideline_lookup.py  ← @traced_tool, searches guideline stubs
│   ├── xai/
│   │   ├── trajectory_logger.py ← Pillar 1 — log_event(), all state transitions
│   │   ├── plan_tracker.py      ← Pillar 2 — active_plan() context manager
│   │   ├── tool_provenance.py   ← Pillar 3 — @traced_tool wrapper + impact scores
│   │   ├── memory_logger.py     ← Pillar 4 — for_agent(), read/write diffs
│   │   ├── message_logger.py    ← Pillar 5 — send_message(), mark_acted_upon()
│   │   ├── causal_dag.py        ← Pillar 6 — CausalDAGBuilder, temporal edges
│   │   ├── accountability.py    ← Pillar 7 — AccountabilityReportGenerator
│   │   └── counterfactual_engine.py ← Type-1/2/3 perturbations → impact scores
│   ├── data/
│   │   ├── schemas.py           ← All 7 dataclasses + AgentXAIRecord
│   │   ├── load_medqa.py        ← load_medqa_us(), load_medqa_us_all(), make_splits()
│   │   └── build_knowledge_base.py ← FAISS index + guideline stubs builder
│   ├── store/
│   │   └── trajectory_store.py  ← SQLAlchemy ORM, 9 tables, get_full_record()
│   ├── api/
│   │   └── main.py              ← FastAPI: GET /tasks, GET /tasks/{id}, POST /tasks/run
│   ├── ui/
│   │   ├── dashboard.py         ← 7-tab Streamlit dashboard
│   │   └── pages/review.py      ← Manual review tool (500 records, 4 sliders)
│   └── eval/
│       └── evaluate.py          ← 5-metric eval runner (in-package copy)
└── eval/
    ├── evaluate.py              ← CLI entry point
    └── aggregate_manual_reviews.py ← mean±std over manual_reviews table
```

---

## One-task lifecycle

The diagram below traces a single MedQA record from input to fully-hydrated `AgentXAIRecord`.

```
MedQA record (question, options, answer_idx)
        │
        ▼
Pipeline.run_task(record)
  │
  ├─ 1. TrajectoryStore.save_task()   ← writes tasks row (FK anchor)
  │
  ├─ 2. Build loggers per task_id
  │       TrajectoryLogger  ──► trajectory_events table
  │       PlanTracker       ──► agent_plans table
  │       ToolProvenanceLogger ► tool_use_events table
  │       MemoryLogger      ──► memory_diffs table
  │       MessageLogger     ──► agent_messages table
  │
  ├─ 3. Orchestrator.run(payload)
  │       │
  │       ├─ SpecialistA.run()
  │       │     with active_plan(["lookup_symptoms","score_severity"])
  │       │       symptom_lookup(symptoms)   ─► ToolProvenanceLogger
  │       │       severity_scorer(case)      ─► ToolProvenanceLogger
  │       │     send_message("synthesizer", "finding", {...})
  │       │                                 ─► MessageLogger
  │       │
  │       ├─ SpecialistB.run()
  │       │     with active_plan(["pubmed_search","guideline_lookup"])
  │       │       pubmed_search(query)       ─► ToolProvenanceLogger
  │       │       guideline_lookup(cond)     ─► ToolProvenanceLogger
  │       │     send_message("synthesizer", "finding", {...})
  │       │
  │       └─ Synthesizer.run()
  │             reads specialist memories
  │             LLM → final_diagnosis, confidence, rationale
  │             TrajectoryLogger.log_event("action", ...)
  │
  ├─ 4. system_output persisted (TrajectoryStore.save_task update)
  │
  ├─ 5. CounterfactualEngine sweeps
  │       Type-1: zero out each tool → Pipeline.resume_from()
  │               → ΔP(correct) → downstream_impact_score
  │       Type-2: zero out each specialist memory → resume_from()
  │       Type-3: perturb each inter-agent message → resume_from()
  │       ToolProvenanceLogger.attach_impact_score() for each tool
  │
  ├─ 6. CausalDAGBuilder.build(task_id)
  │       Reads trajectory_events ordered by timestamp
  │       Adds edges: producer → consumer for tool calls and messages
  │       Writes causal_edges rows
  │
  └─ 7. AccountabilityReportGenerator.generate(task_id)
          Aggregates all 6 pillars:
            - root_cause_event_id  (highest-impact causal chain entry)
            - agent_responsibility_scores  (normalized impact fractions)
            - causal_chain  (ordered event_ids from root to outcome)
            - one_line_explanation  (LLM-generated, ≤ 30 words)
            - most_impactful_tool_call_id
            - most_influential_message_id
            - plan_deviation_summary
          Writes accountability_reports row

        ▼
TrajectoryStore.get_full_record(task_id)
  → AgentXAIRecord (all 7 XAI artefacts hydrated)
```

---

## Database schema (SQLite, SQLAlchemy ORM)

All tables live in `agentxai/data/agentxai.db`. Every row has a `task_id` FK back to `tasks`.

| Table | Pillar | Key columns |
|-------|--------|-------------|
| `tasks` | — | task_id · source · input_json · ground_truth_json · system_output_json |
| `trajectory_events` | 1 | event_id · agent_id · event_type · action · state_before_json · state_after_json · outcome |
| `agent_plans` | 2 | plan_id · agent_id · intended_actions_json · actual_actions_json · deviations_json |
| `tool_use_events` | 3 | tool_call_id · tool_name · called_by · inputs_json · outputs_json · downstream_impact_score |
| `memory_diffs` | 4 | diff_id · agent_id · operation · key · value_before_json · value_after_json |
| `agent_messages` | 5 | message_id · sender · receiver · message_type · acted_upon · behavior_change_description |
| `causal_edges` | 6 | edge_id · cause_event_id · effect_event_id · causal_strength · causal_type |
| `accountability_reports` | 7 | one_line_explanation · agent_responsibility_scores_json · causal_chain_json · root_cause_event_id |
| `manual_reviews` (legacy) | — | task_id · plausibility · completeness · specificity · causal_coherence · notes · status — kept readable; new writes go to v2 |
| `manual_reviews_v2` | — | review_id · medqa_task_id (UNIQUE) · pipeline_task_id (FK→tasks.task_id, nullable) · 4 ratings · notes · status · reviewed_at |

---

## Manual reviews — schema migration to a linked v2 table

The original `manual_reviews` table was created by raw SQL inside the
Streamlit review page with a single `task_id` column and **no foreign
key to anything**. That column held the **MedQA stable id** (e.g.
`A00042`) — *not* the per-pipeline-run UUID stored in `tasks.task_id`,
so even retrofitting a naive FK would have failed: the two columns live
in different namespaces.

The migration introduces `manual_reviews_v2` as a first-class
SQLAlchemy ORM table managed alongside the rest of the schema:

| Column | Notes |
|---|---|
| `review_id` (PK) | Auto-increment integer. |
| `medqa_task_id` | UNIQUE, NOT NULL. Stable MedQA id (e.g. `A00042`) — what reviewers actually rate. The deterministic 500-record split is keyed by this. |
| `pipeline_task_id` | TEXT NULL with `FOREIGN KEY → tasks.task_id`. Soft link to the most recent pipeline run for this MedQA record. NULL when no run exists yet (review can be saved either way). |
| `plausibility` / `completeness` / `specificity` / `causal_coherence` | INTEGER NULL — same 1-5 ratings as the legacy table. |
| `notes` / `status` / `reviewed_at` | TEXT — same semantics as legacy. |

**Backward compatibility:**

* The legacy `manual_reviews` table is **left in place untouched** and
  remains queryable directly.
* On every `TrajectoryStore` init, `migrate_legacy_manual_reviews()`
  copies legacy rows into v2 (idempotent — v2 wins on conflict).
* `TrajectoryStore.list_manual_reviews(include_legacy=True)` returns a
  unified stream of both tables, marked with `"source": "v2"` or
  `"source": "legacy"`.
* `eval/aggregate_manual_reviews.py` reads via the unified stream so
  all existing reviews continue to count toward the report.

**Orphan prevention:**

* `TrajectoryStore.save_manual_review` validates that
  `pipeline_task_id`, when supplied, exists in `tasks`. Bad ids raise
  `KeyError` rather than landing as orphans.
* The Streamlit review page validates that `medqa_task_id` is in the
  deterministic 500-record split before writing. Typos / mis-pasted ids
  surface as a UI error instead of a silent bad row.
* The page also auto-populates `pipeline_task_id` with the most recent
  pipeline run for the MedQA record (via
  `latest_pipeline_task_id_for(medqa_task_id)`) so newly-saved reviews
  are reliably linked.

---

## Tool naming — `pubmed_search` is a local FAISS textbook search

The tool registered as `pubmed_search` does **not** call the PubMed/NCBI
API. Despite the name, it runs a top-k semantic search over a local
FAISS index built from 18 medical textbooks (Harrison, Robbins, First
Aid, …) using `all-MiniLM-L6-v2` embeddings. See
`agentxai/tools/pubmed_search.py` for the full implementation.

**Why the name is preserved.** The identifier `pubmed_search` is a
stable integration point referenced in three places that we don't want
to churn just to fix a label:

  1. **Stored records** — every historical `tool_use_events` row carries
     `tool_name="pubmed_search"`. Renaming would either invalidate them
     or require a migration.
  2. **Specialist B's plan + log_action calls** — the action is logged
     under `pubmed_search`, which feeds the trajectory log, the causal
     DAG, and the accountability report.
  3. **Tests + mocks** — `pubmed_search_fn` is the constructor parameter
     SpecialistB exposes for dependency-injection (used by the smoke
     test to stub out FAISS on Apple Silicon).

**How the truth is surfaced.** The Streamlit dashboard maps the stored
name to a clarified label via `_TOOL_DISPLAY_OVERRIDES` in
`agentxai/ui/dashboard.py`:

> `pubmed_search` → **pubmed_search (local textbook FAISS)**

The Tool Provenance tab also renders an info caption noting the
display-alias rule when an aliased tool appears in the task. Stored
records, API responses, and call sites continue to use the bare name.

**How to swap in real PubMed later.** Either:

  * Replace the body of `pubmed_search()` in
    `agentxai/tools/pubmed_search.py` with a real PubMed/E-utilities
    client; the function signature `(query: str, k: int) -> List[dict]`
    and the `{doc_id, text, score, source_file}` result shape are the
    contract every downstream consumer expects, OR
  * Inject a different implementation via SpecialistB's
    `pubmed_search_fn` constructor parameter and remove
    `pubmed_search` from `_TOOL_DISPLAY_OVERRIDES` so the alias caption
    no longer fires.

---

## Scoring weights — `XAIScoringConfig`

Every numeric knob the causal DAG, counterfactual engine, and
accountability scorer use lives on a single dataclass:
`agentxai.xai.config.XAIScoringConfig`. The defaults reproduce the
historical (pre-config) behaviour exactly — instantiating
`XAIScoringConfig()` and passing it to the three component constructors
gives identical scores. Older module-level constants (`_DX_WEIGHT`,
`_TEMPORAL_WEIGHT`, `_RESP_WEIGHTS`, `_TIE_EPSILON`, …) are kept as
backward-compat aliases that read from the default config.

| Field | Default | Used by |
|---|---|---|
| `cf_dx_weight` | 0.6 | `_outcome_delta` (counterfactual_engine.py) |
| `cf_conf_weight` | 0.4 | `_outcome_delta` |
| `temporal_edge_weight` | 0.3 | `CausalDAGBuilder._add_temporal_edges` |
| `message_acted_upon_weight` | 1.0 | `CausalDAGBuilder._add_message_edges` (when no cf delta) |
| `message_ignored_weight` | 0.5 | same |
| `root_tool_bonus` | 1.5 | `_root_cause_score` (accountability.py) |
| `root_acted_msg_bonus` | 1.0 | same |
| `root_mem_bonus` | 0.5 | same |
| `root_upstream_discount` | 0.3 | same — max % shaved from latest candidate |
| `responsibility_weights` | `{counterfactual: 0.35, tool_impact: 0.20, message_efficacy: 0.15, memory_used: 0.15, usefulness: 0.10, causal_centrality: 0.05}` | `_combine_signals` |
| `question_type_priors` | one row per type (see `agentxai/data/question_classifier.py`) | `_prior_for` |
| `tie_epsilon` | 0.05 | `_tied_top_agents` (explanation builders) |

### These weights are heuristic — please ablate

These values were calibrated qualitatively against the test fixtures in
`tests/`. There is no held-out scoring rubric, no ground-truth label set,
and no formal calibration study. Any of the following would be a real
research contribution:

* **Counterfactual delta weights** — sweep `cf_dx_weight` ∈ [0.4, 0.8]
  against a labelled outcome-change dataset, report the AUC delta on the
  faithfulness metric (`eval/evaluate.py::_faithfulness`).
* **Causal-DAG edge weights** — does the temporal-vs-message ratio
  matter for downstream accountability scores? A sensitivity sweep would
  surface whether the current 0.3/1.0 split is load-bearing.
* **Responsibility-composite weights** — currently the counterfactual
  signal carries 35%. An ablation that varied each weight in turn while
  holding the others proportional would show which signals are doing
  the work and which are noise.
* **Question-type priors** — direction-only (A < B for screening / drug /
  anatomy questions). The exact 0.5–0.8 multipliers are unstudied; a
  per-question-type accuracy sweep against MedQA could re-tune them.

To ablate any of these, build a custom config and pass it to the three
constructors:

```python
from agentxai.xai.config import XAIScoringConfig
from agentxai.xai.accountability import AccountabilityReportGenerator
from agentxai.xai.causal_dag import CausalDAGBuilder
from agentxai.xai.counterfactual_engine import CounterfactualEngine

cfg = XAIScoringConfig(
    cf_dx_weight=0.8,
    cf_conf_weight=0.2,
    responsibility_weights={
        "counterfactual": 0.50,   # bump cf weight
        "tool_impact":    0.20,
        "message_efficacy": 0.10,
        "memory_used":    0.10,
        "usefulness":     0.05,
        "causal_centrality": 0.05,
    },
)
CausalDAGBuilder(store, config=cfg).build(task_id)
CounterfactualEngine(store, pipeline, task_id, config=cfg)
AccountabilityReportGenerator(store, pipeline=pipe, llm=llm, config=cfg)
```

Existing stored records and tests use the default config and are
unaffected.

---

## Counterfactual engine — three perturbation types

`agentxai/xai/counterfactual_engine.py` implements the protocol required by `Pipeline.resume_from()`.

| Type | Override key | What changes | Suffix re-run |
|------|-------------|--------------|--------------|
| 1 — tool output | `tool_output` | Neutral stub replaces one tool's return value | Affected specialist + Synthesizer |
| 2 — agent memory | `agent_memory` | One specialist's memory dict replaced wholesale | Synthesizer only |
| 3 — message content | `message_content` | Inter-agent message content replaced | Synthesizer only |

Impact score = `|P(correct | original) − P(correct | perturbed)|`, measured over a binary correct/incorrect outcome signal.

---

## Streamlit dashboard — tab mapping

| Tab | Pillar | Render function |
|-----|--------|----------------|
| Trajectory | 1 | `render_trajectory_tab()` — vertical timeline, per-agent color bars |
| Plans | 2 | `render_plans_tab()` — intended vs. actual, deviations highlighted red |
| Tool Provenance | 3 | `render_tool_provenance_tab()` — sortable table, impact ProgressColumn |
| Memory | 4 | `render_memory_tab()` — per-agent write log, before/after diff |
| Communication | 5 | `render_communication_tab()` — pyvis directed graph |
| Causality | 6 | `render_causality_tab()` — pyvis DAG, root cause highlighted red |
| Accountability | 7 | `render_accountability_tab()` — responsibility bar chart, causal chain, explanation banner |
