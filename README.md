# AgentXAI

**Explainable multi-agent AI for medical triage** — a framework that explains every dimension of agentic behavior simultaneously, demonstrated on USMLE-style clinical reasoning.

> **Thesis.** Existing XAI techniques (SHAP, Grad-CAM, LIME) explain individual model predictions. Modern AI systems are _agentic_: they plan, use tools, communicate, remember, and act across time. AgentXAI builds a framework that explains all of that behavior across **7 orthogonal pillars** — applied to a team of specialized agents that collaborates to diagnose a patient case from the MedQA corpus.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        MedQA Patient Case                           │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Orchestrator      │  routes + coordinates
                    └──┬──────────────────┬─┘
                       │                  │
           ┌───────────▼───┐       ┌──────▼───────────┐
           │  Specialist A  │       │   Specialist B    │
           │ symptom_lookup │       │  pubmed_search    │
           │severity_scorer │       │ guideline_lookup  │
           └───────┬───────┘       └──────┬────────────┘
                   │   findings            │   findings
                   └──────────┬────────────┘
                    ┌─────────▼──────────┐
                    │     Synthesizer     │  final diagnosis
                    └─────────┬──────────┘
                              │
         ┌────────────────────▼──────────────────────┐
         │             XAI Runtime Layer              │
         │  P1 Trajectory   P2 Plans   P3 Tools       │
         │  P4 Memory       P5 Comms   P6 Causality   │
         │                  P7 Accountability         │
         └────────────────────┬──────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌────▼──────────┐
    │  SQLite Store   │ │ FastAPI /api │ │   Streamlit   │
    │ agentxai.db     │ │  :8000      │ │  dashboard    │
    └─────────────────┘ └─────────────┘ └───────────────┘
```

---

## The 7 XAI Pillars

| # | Pillar | What it explains | Key artefact |
|---|--------|-----------------|--------------|
| 1 | **Trajectory** | Ordered log of every agent action and state transition | `TrajectoryEvent` |
| 2 | **Plans** | Intended vs. actual actions; deviation detection | `AgentPlan` |
| 3 | **Tool Provenance** | Which tool was called, by whom, with what counterfactual impact | `ToolUseEvent` |
| 4 | **Memory** | Every write to agent memory with before/after diffs, linked to the trajectory event that produced it (via `traced_action`); per-(owner, key) usage attribution (`read_by`, `used_in_final_answer`, `influence_score`) | `MemoryDiff`, `MemoryUsage` |
| 5 | **Communication** | Inter-agent messages and whether they caused behavior changes | `AgentMessage` |
| 6 | **Causality** | Temporal causal DAG over events, with counterfactual-estimated edge strengths | `CausalEdge` |
| 7 | **Accountability** | **Composite responsibility scores** (counterfactual + tool impact + acted-upon messages + used memory + confidence + causal centrality, weighted by question type), root-cause event with selection rationale, supporting evidence ids, one-line explanation. Tunable via `XAIScoringConfig`. | `AccountabilityReport` |

---

## Faithful accountability scoring

The headline number on each agent isn't a single counterfactual run — it's
a weighted composite of six observable signals. Designed so an agent can't
score high just because the Synthesizer's prompt structurally read its
(empty) memory.

| Signal | Weight | What it measures |
|---|---|---|
| `counterfactual` | **0.35** | Outcome change when the agent's memory is zeroed |
| `tool_impact` | **0.20** | Max `downstream_impact_score` of tools the agent invoked |
| `message_efficacy` | **0.15** | Max counterfactual delta of the agent's outgoing messages (or `acted_upon` heuristic when no cf delta exists) |
| `memory_used` | **0.15** | Fraction of the agent's memory writes that the Synthesizer's rationale actually cites (substring heuristic) |
| `usefulness` | **0.10** | Self-reported `confidence` / `retrieval_confidence` / `severity_score` from the agent's final memory |
| `causal_centrality` | **0.05** | Sum of outgoing edge weight from the agent's events into the terminal-reachable set, max-normalized |

Then multiplied by a per-(question_type, agent) prior — a deterministic
regex classifier (`agentxai/data/question_classifier.py`) labels each
MedQA stem as `diagnosis`, `treatment`, `screening_or_test`,
`pharmacology`, `mechanism`, `risk_factor`, `anatomy`, `prognosis`, or
`unknown`, and Specialist A's share is gently shrunk on questions where
symptom analysis isn't the right tool. Defaults reproduce the
pre-composite behaviour exactly when called with no config.

**Root-cause selection** filters aggregator/routing actions
(`read_specialist_memories`, `route_to_*`, `handoff_to_*`, …) and scores
the remaining ancestors by graph weight + tool/message/memory bonuses,
with a mild upstream factor breaking near-ties. The chosen event is
recorded with a human-readable `root_cause_reason` so reviewers see
*why* it was picked.

**The Synthesizer emits option-level reasoning**: every prediction
carries a `predicted_letter`, `predicted_text`, per-option `verdict`
+ `reason` in `option_analysis`, and `supporting_evidence_ids` from
Specialist B's `top_evidence`. When the LLM forgets the citation list,
a heuristic infers it from rationale ↔ snippet word overlap.

> ⚠ **All weights are heuristic.** Calibrated qualitatively against the
> test fixtures, *not* against held-out ground truth. The
> `XAIScoringConfig` dataclass exposes every knob (cf weights, edge
> weights, root-cause bonuses, responsibility weights, question-type
> priors, tie epsilon) so you can ablate them. See
> `docs/ARCHITECTURE.md` for the full table and tuning guidance.

---

## Project Layout

```
AgentXAI/
├── agentxai/
│   ├── agents/                 # Orchestrator, SpecialistA, SpecialistB, Synthesizer
│   │                           #   (TracedAgent provides traced_action so memory writes
│   │                           #    link to a single trajectory event)
│   ├── api/                    # FastAPI backend (env-driven CORS + optional bearer-token auth)
│   ├── data/
│   │   ├── load_medqa.py       # MedQA loader + deterministic train/eval/review splits
│   │   ├── build_knowledge_base.py   # FAISS index builder (textbooks + guideline stubs)
│   │   ├── schemas.py          # Dataclasses: TrajectoryEvent, MemoryDiff, MemoryUsage,
│   │   │                       #   AccountabilityReport (with new evidence + root_cause_reason
│   │   │                       #   + memory_usage fields), AgentXAIRecord
│   │   └── question_classifier.py    # Regex classifier — 9 question types
│   ├── store/                  # SQLAlchemy ORM (idempotent ALTER TABLE migrations);
│   │                           #   manual_reviews_v2 with FK→tasks.task_id
│   ├── tools/                  # LangChain tools — symptom_lookup, severity_scorer,
│   │                           #   guideline_lookup, pubmed_search ⚠ (local FAISS over
│   │                           #   medical textbooks, NOT the real PubMed API — see
│   │                           #   docs/ARCHITECTURE.md "Tool naming")
│   ├── xai/
│   │   ├── trajectory_logger.py     # Pillar 1 (with structured _to_jsonable fallback)
│   │   ├── plan_tracker.py          # Pillar 2
│   │   ├── tool_provenance.py       # Pillar 3
│   │   ├── memory_logger.py         # Pillar 4 + LoggedMemory (contextvar-linked diffs)
│   │   ├── memory_usage.py          # Heuristic read-vs-used attribution per memory key
│   │   ├── message_logger.py        # Pillar 5 (returns nx.MultiDiGraph)
│   │   ├── causal_dag.py            # Pillar 6
│   │   ├── counterfactual_engine.py # 3 perturbation types
│   │   ├── accountability.py        # Pillar 7 — composite scorer + selectable root cause
│   │   ├── confidence_factors.py    # 5-factor decomposition of headline confidence
│   │   ├── evidence_attribution.py  # Heuristic supporting_evidence_ids inference + ranking
│   │   └── config.py                # XAIScoringConfig — every numeric knob in one place
│   └── ui/
│       ├── dashboard.py             # 7-tab Streamlit dashboard
│       ├── faithfulness_checks.py   # 6 sanity checks rendered on the Accountability tab
│       └── disabled_pages/
│           └── review.py            # Manual XAI quality review (500 records)
│                                    #   — relocated out of pages/ so Streamlit doesn't
│                                    #   show it in the dashboard sidebar; still
│                                    #   runnable directly via `make review`.
├── eval/
│   ├── evaluate.py                  # Five-metric pipeline-rerun eval (accuracy, sufficiency,
│   │                                #   necessity, stability, faithfulness)
│   ├── evaluate_existing.py         # Same metrics over stored snapshots (no LLM re-runs)
│   ├── evaluate_accountability.py   # 4-metric attribution-quality monitor over stored records
│   └── aggregate_manual_reviews.py  # Mean ± std summary over both manual_reviews tables
├── docs/
│   ├── ARCHITECTURE.md      # Pillar-to-file mapping, scoring weights, tool naming, mr v2 migration
│   └── DATASET.md           # MedQA schema, AgentXAIRecord example
├── med_qa/                  # MedQA corpus (JSONL + textbooks)
├── run_pipeline.py          # End-to-end pipeline runner
├── Makefile                 # install / test / index / run-one / run-eval / dashboard / api
├── pyproject.toml
└── requirements.txt
```

---

## Setup

**Requirements:** Python ≥ 3.11, ~4 GB disk (textbook FAISS index), a Google Gemini API key (free tier; get one at https://aistudio.google.com/app/apikey).

```bash
git clone https://github.com/harjeetschahal/AgentXAI
cd AgentXAI

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt   # or: make install

cp .env.example .env
# Edit .env — set GOOGLE_API_KEY (required) and OPENAI_API_KEY (optional)
```

Build the FAISS textbook index (one-time, ~5 min on CPU):

```bash
make index
# or: python -m agentxai.data.build_knowledge_base
```

---

## Running a single task

```bash
# CLI — runs 5 records from the train split, prints accuracy summary
python run_pipeline.py --split train --limit 5

# Programmatic
python - <<'EOF'
from agentxai.data.load_medqa import load_medqa_us
from run_pipeline import Pipeline

record = load_medqa_us("train")[0]
result = Pipeline().run_task(record)

print(result.system_output["final_diagnosis"])
print(result.xai_data.accountability_report.one_line_explanation)
EOF

# or: make run-one
```

---

## Running the evaluation suite

Three runners with different cost/coverage trade-offs:

```bash
# (a) Full re-run — every metric over a fresh pipeline pass (slow, costs LLM tokens).
#     Writes eval/results_<timestamp>.{json,md}
python -m eval.evaluate --limit 1500 --samples-for-stability 100
# or: make run-eval

# (b) Snapshot eval — recomputes the same five metrics over already-stored
#     snapshots without re-running the LLM. Fast, free, weaker stability.
python -m eval.evaluate_existing --limit 100 --samples-for-stability 10

# (c) Attribution-quality monitor — read-only over stored records. No LLM,
#     no perturbations re-executed. Reports four faithfulness-relevant
#     metrics: empty-agent penalty, impact alignment, root-cause validity,
#     faithfulness (top-vs-low cf-delta gap).
python -m eval.evaluate_accountability --limit 100 --out-json acc_eval.json
```

Metrics computed by `eval.evaluate`: task accuracy · XAI sufficiency · XAI necessity · stability (Spearman ρ) · faithfulness.

---

## Launching the dashboard

```bash
# Terminal 1 — FastAPI backend
uvicorn agentxai.api.main:app --reload
# or: make api

# Terminal 2 — Streamlit dashboard
streamlit run agentxai/ui/dashboard.py
# or: make dashboard
```

Open `http://localhost:8501`.  The sidebar lists all processed tasks; select one to explore its 7 XAI tabs.

> The Manual Review tool is hidden from the main dashboard nav. Run it standalone with `make review` (or `streamlit run agentxai/ui/disabled_pages/review.py`) when you want to rate the 500-record review split.

**Case overview panels** (landing page for each task):

* Question-type pill (heuristic classifier label) + correctness pill + confidence
* Answer options with per-option Synthesizer verdict (`✓ correct` / `✗ incorrect` / `~ partial`) and reason
* `supporting_evidence_ids` caption + expandable evidence cards (doc_id · source_file · score · "✓ used in rationale" badge per doc)
* Confidence breakdown card — 5 heuristic factors with progress bars and a "not clinically calibrated" disclaimer
* Memory-usage attribution table — per-(owner, key) with `read_by`, `used_in_final_answer`, `influence_score`

**Accountability tab** additionally shows a **Faithfulness Checks** panel
(green ✓ / yellow ⚠ / red ✗ / gray ⊘) with six sanity assertions:

  1. Most-impactful tool is on the causal path
  2. Most-influential message was acted upon
  3. Top responsible agent has at least one observable signal
  4. Root cause is not an aggregator event
  5. Rationale references retrieved evidence
  6. No high-responsibility agent with empty signals

---

## API security (local-first defaults)

The FastAPI backend is built for **local-first** use — the dashboard
talks to it on localhost, and out of the box it accepts un-authenticated
calls from common loopback origins. Two env-var levers harden it for
anything beyond a laptop without breaking the local dev loop:

| Env var | Default | Purpose |
|---|---|---|
| `AGENTXAI_API_TOKEN` | _unset_ | If set, every `POST /tasks/run` request must carry `Authorization: Bearer $AGENTXAI_API_TOKEN`. Read endpoints stay unauthenticated. |
| `AGENTXAI_CORS_ORIGINS` | _unset_ → localhost-only | Comma-separated allow-list of origins. Defaults to `http://localhost:{8501,8000,3000}` and `http://127.0.0.1:{...}`. |
| `AGENTXAI_ALLOW_CORS_ALL` | `false` | Emergency wildcard. Set to `true` to restore the historical `Access-Control-Allow-Origin: *` behaviour. Use only when you genuinely need to accept any origin (public demos). |
| `AGENTXAI_API_URL` | `http://localhost:8000` | Where the dashboard looks for the API. |

**Local dev — do nothing.** Defaults work; the dashboard hits the API
without sending any token, and CORS only allows your laptop's
localhost ports.

**Exposing the API beyond localhost** — set both:

```bash
# In the API server's environment (or .env):
export AGENTXAI_API_TOKEN="$(openssl rand -hex 32)"
export AGENTXAI_CORS_ORIGINS="https://your-dashboard.example.com"

uvicorn agentxai.api.main:app --host 0.0.0.0 --port 8000
```

Then **same `AGENTXAI_API_TOKEN` value** in the dashboard's environment
so it can write:

```bash
export AGENTXAI_API_URL="https://your-api.example.com"
export AGENTXAI_API_TOKEN="$(... same value as above ...)"
streamlit run agentxai/ui/dashboard.py
```

The dashboard transparently attaches the bearer token to every API
call when the env var is set; reads work either way.

> **Read endpoints are still unauthenticated** even when the token is
> configured. They expose only what's already in the SQLite store —
> any caller who can reach the API can also read the DB file directly.
> If you need to lock down reads, put the API behind an authenticating
> reverse proxy.

---

## Manual quality review

The Streamlit Review page writes ratings to `manual_reviews_v2` — an
ORM-managed table with a soft `FOREIGN KEY → tasks.task_id` and a
unique key on the stable MedQA id. Reviews are saved with the
most-recent pipeline run for that record auto-linked, and bad MedQA
ids are rejected at save time. The legacy `manual_reviews` table
stays readable; rows are auto-migrated into v2 on store init. See
the "Manual reviews" section in `docs/ARCHITECTURE.md` for the
schema-migration rationale.

```bash
# After rating, aggregate results (reads v2 + legacy tables together)
python eval/aggregate_manual_reviews.py \
  --out-md  eval/manual_review_summary.md \
  --out-json eval/manual_review_summary.json
```

---

## Results

> **Placeholder — fill in after running the evaluation suite.**

| Metric | Value |
|--------|-------|
| Task accuracy (1 500 records) | — |
| XAI sufficiency | — |
| XAI necessity | — |
| Stability (mean Spearman ρ) | — |
| Faithfulness | — |
| Manual review — plausibility (mean ± std) | — |
| Manual review — completeness (mean ± std) | — |
| Manual review — specificity (mean ± std) | — |
| Manual review — causal coherence (mean ± std) | — |

---

## Development status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project scaffolding | ✅ Complete |
| 1 | Data loading, schema definitions, FAISS index | ✅ Complete |
| 2 | Agents (Orchestrator, Specialists, Synthesizer) | ✅ Complete |
| 3 | Tools (symptom_lookup, severity_scorer, pubmed_search ⚠ local FAISS, guideline_lookup) | ✅ Complete |
| 4 | XAI runtime layer (all 7 pillars) | ✅ Complete |
| 5 | Counterfactual engine | ✅ Complete |
| 6 | SQLite store + FastAPI backend (env-driven CORS + optional bearer-token auth) | ✅ Complete |
| 7 | Streamlit dashboard (7 tabs + review page + per-case panels) | ✅ Complete |
| 8 | Evaluation suite (5 pipeline-rerun metrics + 5 snapshot metrics + 4 attribution metrics + manual-review aggregation) | ✅ Complete |
| 9 | **Faithfulness pass** — composite responsibility scoring · root-cause selector · memory-usage attribution · event-linked memory diffs · option-level Synthesizer reasoning · `XAIScoringConfig` · dashboard faithfulness checks · `manual_reviews_v2` with FK · evidence attribution · confidence factors · question classifier | ✅ Complete |

**Test suite**: 570 unit/integration tests passing (6 slow integration tests
require `--run-slow` + `GOOGLE_API_KEY`). Run with `make test` or
`pytest tests/`.

---

## License

MIT — see `pyproject.toml`.  
MedQA corpus is distributed under its own license; see `med_qa/` for details.
