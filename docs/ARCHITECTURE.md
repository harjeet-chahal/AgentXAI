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
│   │   ├── specialist_b.py      ← SpecialistB   (PubMed + guidelines)
│   │   └── synthesizer.py       ← Synthesizer   (final diagnosis)
│   ├── tools/
│   │   ├── symptom_lookup.py    ← @traced_tool, queries FAISS index
│   │   ├── severity_scorer.py   ← @traced_tool, LLM-rated 0–1 score
│   │   ├── pubmed_search.py     ← @traced_tool, FAISS over textbook chunks
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
| `manual_reviews` | — | task_id · plausibility · completeness · specificity · causal_coherence · notes · status |

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
