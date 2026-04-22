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
| 4 | **Memory** | Every read/write to agent memory with before/after diffs | `MemoryDiff` |
| 5 | **Communication** | Inter-agent messages and whether they caused behavior changes | `AgentMessage` |
| 6 | **Causality** | Temporal causal DAG over events, with counterfactual-estimated edge strengths | `CausalEdge` |
| 7 | **Accountability** | Responsibility scores per agent, root-cause event, one-line explanation | `AccountabilityReport` |

---

## Project Layout

```
AgentXAI/
├── agentxai/
│   ├── agents/              # Orchestrator, SpecialistA, SpecialistB, Synthesizer
│   ├── api/                 # FastAPI REST backend
│   ├── data/                # MedQA loader, FAISS index builder, all schemas
│   ├── eval/                # Evaluation: accuracy, sufficiency, necessity, stability, faithfulness
│   ├── store/               # SQLite persistence via SQLAlchemy ORM
│   ├── tools/               # LangChain tools (symptom_lookup, severity_scorer, pubmed_search, guideline_lookup)
│   ├── ui/
│   │   ├── dashboard.py     # 7-tab Streamlit dashboard
│   │   └── pages/
│   │       └── review.py    # Manual XAI quality review tool (500 records)
│   └── xai/                 # 7 XAI loggers + counterfactual engine
├── eval/
│   ├── evaluate.py          # Five-metric eval runner
│   └── aggregate_manual_reviews.py
├── docs/
│   ├── ARCHITECTURE.md      # Pillar-to-file mapping, task lifecycle flow
│   └── DATASET.md           # MedQA schema, AgentXAIRecord example
├── notebooks/
│   └── demo.ipynb           # End-to-end walkthrough
├── med_qa/                  # MedQA corpus (JSONL + textbooks)
├── run_pipeline.py          # End-to-end pipeline runner
├── Makefile                 # install / test / index / run-one / run-eval / dashboard / api
├── pyproject.toml
└── requirements.txt
```

---

## Setup

**Requirements:** Python ≥ 3.11, ~4 GB disk (textbook FAISS index), a Google Gemini API key (get one at https://aistudio.google.com/app/apikey).

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

```bash
# All five XAI metrics on 1 500 records (writes eval/results_<timestamp>.{json,md})
python -m eval.evaluate --limit 1500 --samples-for-stability 100

# or: make run-eval
```

Metrics computed: task accuracy · XAI sufficiency · XAI necessity · stability (Spearman ρ) · faithfulness.

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

Open `http://localhost:8501`.  The sidebar lists all processed tasks; select one to explore its 7 XAI tabs.  The **Review** page (`/Review` in the left nav) lets you rate 500 records for manual quality assessment.

---

## Manual quality review

```bash
# After rating, aggregate results
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
| 3 | Tools (symptom_lookup, severity_scorer, pubmed_search, guideline_lookup) | ✅ Complete |
| 4 | XAI runtime layer (all 7 pillars) | ✅ Complete |
| 5 | Counterfactual engine | ✅ Complete |
| 6 | SQLite store + FastAPI backend | ✅ Complete |
| 7 | Streamlit dashboard (7 tabs + review page) | ✅ Complete |
| 8 | Evaluation suite (5 metrics + manual review aggregation) | ✅ Complete |

---

## License

MIT — see `pyproject.toml`.  
MedQA corpus is distributed under its own license; see `med_qa/` for details.
