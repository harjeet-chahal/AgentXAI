# AgentXAI Improvement Plan

Phased rewrite addressing the seven evaluated failures: irrelevant SpecialistA outputs, broken severity scoring, zero tool counterfactual impact, generic retrieval, Synthesizer dominance, overconfidence, and explanations that don't audit reasoning quality.

## Constraints (apply to every phase)

- **Repo never broken.** `pytest tests/` must stay green at the end of each phase. Update existing tests when behaviour changes deliberately; add smoke tests when modifying a file that has none.
- **Additive over replacement.** New fields (`system_confidence`, `cited_evidence_ids`, `evidence_matrix`) live alongside old ones — don't repurpose existing keys. Old dashboards, stored records, and existing tests must keep working.
- **Mini-baseline at the end of every phase.** Re-run [`eval/mini_baseline.py`](../eval/mini_baseline.py) on the same 15 fixed tasks and write to `eval/baselines/phase<N>.json`. Compare against the previous phase's JSON to confirm the change moved the metric in the right direction.
- **No silent fallbacks for new code.** Old code uses default-on-failure (returns empty / runs everything). New code should *log* when it falls back so we can see when it's firing.

## Phase summary

| Phase | Goal | Primary files | Est. wall time | Est. cost |
|-------|------|---------------|----------------|-----------|
| 0 | Mini-baseline harness + frozen 15-task sample | `eval/mini_baseline.py` (new) | 30 min build + 10 min run | ~$0.05 |
| 1 | Fix reasoning inputs (severity, symptom_lookup) | `tools/severity_scorer.py`, `tools/symptom_lookup.py` | 2-3 h | trivial |
| 2 | Fix retrieval (per-candidate + reranker + gating) | `agents/specialist_b.py`, optional `data/build_knowledge_base.py` | 4-6 h | trivial |
| 3 | Fix XAI counterfactual signal | `xai/counterfactual_engine.py` | 3-5 h | ~$0.20 |
| 4 | Calibrated `system_confidence` + multi-sample disagreement | `xai/confidence_factors.py`, `agents/synthesizer.py` | 3-4 h | ~$0.30 |
| 5 | Aggregator + evidence-only Synthesizer + critic loop | new `agents/aggregator.py`, `agents/synthesizer.py`, `agents/orchestrator.py` | 6-8 h | ~$0.40 |
| 6 | Faithfulness + entailment + full eval expansion | `eval/evaluate.py`, new `xai/reasoning_audit.py` | 4-6 h | ~$1.00 |

Total estimated cost across all phases: < $5 on Gemini Flash Lite.

## Mini-baseline JSON contract

Every `eval/baselines/phase<N>.json` has the same shape so phases can be diffed:

```json
{
  "phase": "phase1",
  "n_tasks": 15,
  "task_ids": ["A00001", "A00042", ...],
  "wall_time_s": 612.4,
  "n_llm_calls": 247,
  "accuracy": 0.467,
  "mean_llm_confidence": 0.853,
  "mean_system_confidence": null,
  "confidence_gap": null,
  "mean_tool_impact": 0.0,
  "tool_impact_by_tool": {
    "symptom_lookup": 0.0,
    "severity_scorer": 0.0,
    "textbook_search": 0.0,
    "guideline_lookup": 0.0
  },
  "mean_responsibility_share": {
    "specialist_a": 0.51,
    "specialist_b": 0.49
  },
  "faithfulness_check_pass_rate": 0.62,
  "retrieval_relevance_mean": 0.34,
  "notes": "free-form string — log surprises here"
}
```

Fields populated by later phases (e.g. `mean_system_confidence` from phase 4) are `null` in earlier phases — never omitted.

---

## Phase 0 — Mini-baseline harness

### What we're building

A lean, deterministic eval script that runs the existing pipeline on 15 fixed MedQA US tasks and records the metrics above. Reusable across every subsequent phase.

### Concrete changes

1. New file `eval/mini_baseline.py`:
   - CLI: `python -m eval.mini_baseline --phase phase0 [--n 15] [--out eval/baselines/phase0.json]`.
   - Loads MedQA US via `agentxai.data.load_medqa.load_medqa_us_all()` + `make_splits(seed=42)`, takes first `n` tasks from the dev split. Same seed/source as existing eval so the sample is reproducible.
   - Runs `Pipeline.run_task(record)` for each.
   - Aggregates the JSON contract above. Mean over tasks; for `tool_impact_by_tool`, mean over all tool calls of that name.
   - Writes JSON + appends a one-line summary to `eval/baselines/_log.md`.
2. New directory `eval/baselines/` with `.gitkeep`.
3. Run it once → `eval/baselines/phase0.json` is the recorded baseline.

### Acceptance criteria

- [ ] `python -m eval.mini_baseline --phase phase0` completes in < 20 minutes
- [ ] `eval/baselines/phase0.json` exists with all contract fields populated (or `null` where appropriate)
- [ ] `pytest tests/` green
- [ ] Re-running with the same args produces an identical `task_ids` list

### Prompt to start Phase 0

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md. Execute Phase 0 as specified there.

Build eval/mini_baseline.py with the CLI and JSON contract from the plan.
Use load_medqa_us_all() + make_splits(seed=42), dev split, first 15 tasks.
After the script works, run it and verify eval/baselines/phase0.json
matches the contract. Confirm pytest tests/ is green.

Do not modify any other files.
```

---

## Phase 1 — Reasoning inputs

### What we're fixing

`severity_scorer` returns 0 for severe trauma because it's a strict-string lookup against a table that doesn't contain `"hypotension"` or `"hypoxia"`. `symptom_lookup` returns endocrine conditions for trauma cases because raw co-occurrence in MedQA stems is dominated by base-rate confounds.

### Concrete changes

**`tools/severity_scorer.py`:**
1. Expand `SEVERITY_WEIGHTS` to include `hypotension`, `hypoxia`, `tachypnea`, `shock`, `hemorrhage`, `unstable vitals`, `trauma`, `hypoxemia`, `gcs <= 8`, `septic`, `unresponsive`, `pulseless` (use NEWS2 / qSOFA-aligned weights). Cite the rubric in a comment.
2. Add an alias map: `"low blood pressure" → "hypotension"`, `"SOB" → "shortness of breath"`, `"dyspnea" → "shortness of breath"`, `"AMS" → "altered mental status"`, etc. Apply before lookup.
3. Replace strict lookup with rapidfuzz (mirror `guideline_lookup.py`'s pattern, cutoff 85). Log a debug message when a fuzzy match fires so we can audit.
4. Add vital-sign regex parsing: `"BP 78/40"`, `"SpO2 84%"`, `"HR 130"`, `"RR 28"`, `"GCS 6"`. Map to NEWS2 thresholds.
5. Replace `mean(weights)` with `max(weights) + min(0.15, 0.05 * max(0, severe_count - 1))`. Cap at 1.0.
6. Keep the public signature `severity_scorer(symptoms: List[str]) -> float`.

**`tools/symptom_lookup.py`:**
1. In `_build_table`, additionally compute condition-level base rates: `base_rate[cond] = total occurrences of cond / total questions`.
2. Cache as `data/indices/symptom_table_v2.json` with shape `{symptom: {cond: {count: int, lift: float}}}`. Bump the cache file path to v2 so the old cache stays untouched and can be diffed.
3. In `symptom_lookup`, return `(cond, lift)` pairs filtered by `count >= 3 AND lift >= 1.5`, sorted by lift descending. Add a `support` field per entry.
4. Add an alias normalization step (same alias dict as severity_scorer — share via `tools/_clinical_aliases.py`).
5. Update the response shape additively: keep `related_conditions` returning `(cond, score)` for backward compat, add `evidence` field with the per-pair `{count, lift, support}` provenance.

**Tests:**
- Add `tests/test_severity_scorer.py`: trauma case (`["BP 78/40", "SpO2 84%", "GCS 6"]`) returns ≥ 0.85; mild case (`["mild headache", "runny nose"]`) returns ≤ 0.20; alias case (`["low blood pressure", "SOB"]`) returns ≥ 0.70; empty list returns 0.0.
- Add `tests/test_symptom_lookup.py`: known high-specificity symptom returns lift > 2.0 for its top condition; rare phrase returns empty list; alias normalization works.
- Update any existing tests that asserted on the old `mean`-based score or strict lookup.

### Acceptance criteria

- [ ] Trauma test case scores ≥ 0.85 severity (was 0)
- [ ] `symptom_lookup("hypotension")` returns shock-related conditions in top-3 (manually verify)
- [ ] `pytest tests/` green
- [ ] `eval/baselines/phase1.json` shows `accuracy` ≥ phase0 and a non-trivial change in `mean_responsibility_share` (specialists' relative weight should shift)

### Prompt to start Phase 1

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md and AgentXAI/eval/baselines/phase0.json.
Execute Phase 1 as specified.

Touch only:
- agentxai/tools/severity_scorer.py
- agentxai/tools/symptom_lookup.py
- new agentxai/tools/_clinical_aliases.py
- new agentxai/data/indices/symptom_table_v2.json (auto-generated)
- new tests/test_severity_scorer.py
- new tests/test_symptom_lookup.py
- update existing tests only if they assert on the changed contracts

Keep public function signatures stable. Keep `related_conditions` field
shape backward-compatible (add new fields, don't rename old ones).

When done: pytest tests/ must be green, then run
  python -m eval.mini_baseline --phase phase1
and verify the JSON shows the trauma severity acceptance criterion.
```

---

## Phase 2 — Retrieval

### What we're fixing

SpecialistB does one FAISS search using the entire patient case as the query, which retrieves common-presentation passages (COPD, pneumonia) instead of condition-specific evidence. No reranker, no relevance gate.

### Concrete changes

**`agents/specialist_b.py`:**
1. In the `textbook_search` action, replace the single `textbook_search(case)` call with a per-candidate loop: `for cond in candidates[:5]: textbook_search(f"{cond} clinical features", k=4)`. Aggregate by `doc_id`, keep `max_score`. Cap total docs at 12.
2. After aggregation, run a cross-encoder rerank on the top 12 → keep top 5 by rerank score. Use `cross-encoder/ms-marco-MiniLM-L-6-v2` (or `ncbi/MedCPT-Cross-Encoder` if installable). Lazy-load the model the same way `_ensure_model()` works in `textbook_search.py`.
3. Add a relevance gate: if best rerank score < 0.5, set `evidence_status = "no_relevant_docs"`, leave `top_evidence = []`, set `retrieval_confidence = 0.0`. Log this case.
4. Store a new memory key `evidence_matrix`: `{candidate: [{doc_id, score, rerank_score, snippet}, ...]}` so XAI can attribute evidence to specific candidates.
5. Add `evidence_status: "ok" | "no_relevant_docs" | "low_confidence"` to memory.

**`tools/textbook_search.py`:**
- No changes required — keep the function signature. Reranking lives in SpecialistB.

**Optional (gated; do only if Phase 2 metrics still bad):**
- Swap `_MODEL_NAME` in `data/build_knowledge_base.py` to `pritamdeka/S-PubMedBert-MS-MARCO`. Requires rebuilding the FAISS index — expensive (~10-20 min), only do if reranking alone doesn't hit the acceptance criterion.

**Tests:**
- Add `tests/test_specialist_b_retrieval.py`: stub `textbook_search_fn` with hand-crafted documents, assert per-candidate aggregation works; assert relevance gate triggers when scores are below threshold; assert `evidence_matrix` is populated correctly.

### Acceptance criteria

- [ ] `eval/baselines/phase2.json` shows `retrieval_relevance_mean` improved by ≥ 0.05 over phase1
- [ ] On a hand-picked aortic-rupture case (find one in MedQA US dev), retrieved evidence includes at least one passage mentioning aortic dissection / aneurysm
- [ ] `evidence_matrix` is non-empty in the dashboard for at least one task
- [ ] `pytest tests/` green

### Prompt to start Phase 2

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md and the latest baseline file in
AgentXAI/eval/baselines/. Execute Phase 2 as specified.

Touch only:
- agentxai/agents/specialist_b.py
- new tests/test_specialist_b_retrieval.py
- update agentxai/agents/specialist_b.py's docstring memory-keys list

Keep textbook_search.py's function signature unchanged. Add cross-encoder
reranking inline in SpecialistB. The cross-encoder model should lazy-load
the same way the embedding model does in textbook_search.py.

If sentence-transformers' CrossEncoder isn't available, fall back to
scoring with the existing embedding model (cosine of query+doc pairs)
and note this in the file's docstring.

When done: pytest tests/ green, then
  python -m eval.mini_baseline --phase phase2
```

---

## Phase 3 — XAI counterfactual signal

### What we're fixing

Every tool's `downstream_impact_score` = 0 because `_neutral_baseline` returns `[]` / `{}` and the LLM ignores the empty field. The metric is also binary on `final_diagnosis` flip, which is too coarse for a robust LLM.

### Concrete changes

**`xai/counterfactual_engine.py`:**
1. Replace `_neutral_baseline(value)` with a `_misleading_baseline(tool_name, value)` dispatcher:
   - `symptom_lookup` → `{"related_conditions": [("type 2 diabetes", 0.9), ("hypertension", 0.85)], "source": "perturbed"}`
   - `severity_scorer` → `0.1` (force "mild" reading)
   - `textbook_search` → fixed off-topic chunk (e.g. one paragraph from `Pediatrics_Nelson.txt` about routine vaccination) — load once at module init
   - `guideline_lookup` → `{"match": "common cold", "match_score": 0.95, "recommendation": "rest, fluids"}`
   - Default fallback to today's neutral behavior.
2. Add a graded outcome delta. Replace `_outcome_delta`'s binary `dx_changed` with:
   - Run the synthesizer's resume `n_samples=5` times at `temperature=0.3`.
   - Compute `flip_rate = 1 - (count of original_letter / 5)`.
   - Embed both rationales with the existing sentence-transformer; `rationale_distance = 1 - cos_sim`.
   - `delta = 0.5 * flip_rate + 0.3 * rationale_distance + 0.2 * abs(conf_orig - conf_new)`.
3. Add `perturb_placebo()`: shuffle the order of memory keys in the synthesizer's input (no information change). Run once per task. If placebo delta is comparable to real perturbation deltas, the metric is noisy — log a warning into the task's notes.
4. Make `n_samples` and weights live on `XAIScoringConfig` so they're ablatable.

**`xai/config.py`:**
- Add `cf_n_samples: int = 5`, `cf_flip_weight: float = 0.5`, `cf_rationale_weight: float = 0.3`, `cf_conf_weight: float = 0.2` (rename existing `cf_conf_weight` to keep semantics — do this carefully and update all references).

**Mini-baseline:**
- Update `eval/mini_baseline.py` to record `mean_placebo_impact` and `placebo_to_real_ratio` (should be < 0.3 for the metric to be trustworthy).

**Tests:**
- Add `tests/test_misleading_baselines.py`: stub the pipeline so resume_from echoes the override; verify each tool gets the right baseline shape.
- Update `tests/test_counterfactual_engine.py` (if exists) for the new delta formula.

### Acceptance criteria

- [ ] `eval/baselines/phase3.json` shows `mean_tool_impact > 0.10` (was 0)
- [ ] `placebo_to_real_ratio < 0.30` (placebo perturbation has materially less impact than real perturbations)
- [ ] At least one tool per task has `downstream_impact_score > 0`
- [ ] `pytest tests/` green

### Prompt to start Phase 3

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md and the latest baseline. Execute Phase 3.

Touch only:
- agentxai/xai/counterfactual_engine.py
- agentxai/xai/config.py
- eval/mini_baseline.py (add placebo metrics)
- tests/test_misleading_baselines.py (new)
- existing test files only if they assert on the old delta formula

The misleading_baseline dispatcher must keep _neutral_baseline as the
default fallback for unknown tools. The off-topic textbook chunk should
be loaded once at module import (cache it; don't re-read per call).

When done: pytest tests/ green, then
  python -m eval.mini_baseline --phase phase3
Verify mean_tool_impact > 0.10 and placebo_to_real_ratio < 0.30.
```

---

## Phase 4 — Calibrated confidence

### What we're fixing

Synthesizer's `confidence` is the LLM's self-report and systematically overconfident. The five `confidence_factors` are computed but never folded into a calibrated headline.

### Concrete changes

**`xai/confidence_factors.py`:**
1. Add `compute_system_confidence(factors: Dict[str, float]) -> float`:
   - `system_confidence = 0.30 * retrieval_relevance + 0.25 * specialist_agreement + 0.20 * option_match_strength + 0.15 * evidence_coverage - 0.30 * contradiction_penalty`, clamped to `[0, 1]`.
   - Document the weights with rationale; expose them on `XAIScoringConfig` for ablation.

**`agents/synthesizer.py`:**
1. After the main synthesis call, run the synthesizer `n=3` times at `temperature=0.5` (separate from the primary `temperature=0` call, only for disagreement measurement). Compute `disagreement = 1 - max_letter_share`. Cache so we don't re-run for counterfactuals.
2. Add to memory `final_output`:
   - `llm_confidence`: alias of existing `confidence` (rename later if convenient; keep `confidence` populated for backward compat)
   - `system_confidence`: from `compute_system_confidence`
   - `disagreement`: from sampling
   - `confidence_gap`: `llm_confidence - system_confidence`
3. **Do not change the primary prediction** — sampling is for a confidence signal only.

**`xai/config.py`:**
- Add `system_confidence_weights: Dict[str, float]`.

**`eval/mini_baseline.py`:**
- Populate `mean_system_confidence`, `confidence_gap`, and add `ece` (expected calibration error, 10 bins) for both `llm_confidence` and `system_confidence`.

**Tests:**
- Add `tests/test_system_confidence.py`: known-good factor values produce expected score; weights sum check.

### Acceptance criteria

- [ ] `eval/baselines/phase4.json` populates `mean_system_confidence` and `confidence_gap`
- [ ] On the 15-task sample: `ece(system_confidence) < ece(llm_confidence)` — the calibrated number is better calibrated than the self-report
- [ ] `confidence_gap` mean > 0.10 (system confidence is generally lower than LLM self-report — that's the expected direction)
- [ ] `pytest tests/` green; the existing `confidence` field in `final_output` still has its old value

### Prompt to start Phase 4

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md and the latest baseline. Execute Phase 4.

Touch only:
- agentxai/xai/confidence_factors.py (additive: new compute_system_confidence)
- agentxai/agents/synthesizer.py (additive: new fields, do not rename `confidence`)
- agentxai/xai/config.py (add weight field)
- eval/mini_baseline.py (compute ECE for both confidence numbers)
- tests/test_system_confidence.py (new)

CRITICAL: do not change the synthesizer's predicted_letter / confidence /
rationale outputs. Only ADD new fields. The 3-sample disagreement run is
purely for the disagreement signal — keep the original temperature=0
call as the source of truth for the prediction.

When done: pytest tests/ green, then
  python -m eval.mini_baseline --phase phase4
Verify ece(system_confidence) < ece(llm_confidence).
```

---

## Phase 5 — Aggregator + evidence-only Synthesizer

### What we're fixing

Synthesizer dominates because it's given the full case + options. Specialists become decorative. This is the architectural fix that makes the system actually agentic.

### Concrete changes

**New `agents/aggregator.py`:**
- Class `Aggregator(TracedAgent)`. Reads both specialists' memories. Produces a per-option score:
  - `score[option] = w_symptom * symptom_lookup_match + w_evidence * top_evidence_match + w_guideline * guideline_match + w_severity * severity_consistency`
  - Match functions: token-overlap between the option text and the relevant memory field. Concrete; deterministic; no LLM.
- Writes `aggregator.memory["option_scores"]: Dict[letter, float]` and `option_provenance: Dict[letter, List[str]]` (which evidence pieces voted for each option).
- Logs as a normal `TracedAgent` so it shows up in the dashboard.

**`agents/synthesizer.py`:**
1. Drop the `case` from `_SYNTH_PROMPT` — replace with `option_scores` from the aggregator + `top_evidence` snippets. The Synthesizer's job becomes: pick the best option, write a rationale, *cite each major claim with an evidence_id*.
2. Update prompt to require `[claim text] (evidence: doc_id_X)` markup. Parse server-side into `claim_citations: List[{claim, evidence_id}]`. Add to memory.
3. If `option_scores` has a unique top scorer with margin > 0.15, the Synthesizer should select it; the LLM only writes the rationale. Log when LLM overrides this.

**`agents/orchestrator.py`:**
1. Insert aggregator between specialists and synthesizer in the routing sequence.
2. Wire the existing `Critic` agent into the loop: after Synthesizer produces output, run Critic; if `needs_revision=True AND confidence_in_critique > 0.6`, re-call the Orchestrator routing once with `feedback_to_specialist = critic.missing_considerations[0]`.
3. Cap total iterations at `max_iterations` (already exists) — Critic loop counts toward this cap.

**`xai/accountability.py`:**
- Add `aggregator` to `_NON_SPECIALIST_AGENTS` so it doesn't get a responsibility share (it's deterministic; responsibility lives with the specialists whose findings it aggregates).

**Mini-baseline:**
- Add `aggregator_override_rate` (how often the Synthesizer LLM picked something different from the aggregator's top option).
- Add `claim_citation_rate` (fraction of Synthesizer claims that have a parsed evidence_id).

**Tests:**
- `tests/test_aggregator.py`: hand-crafted memories produce expected option scores.
- Update `tests/test_orchestrator.py` (or create) to verify the new routing sequence.

### Acceptance criteria

- [ ] `eval/baselines/phase5.json` shows `accuracy >= phase4 * 0.95` (small regression OK; large regression = revert and rethink)
- [ ] `mean_responsibility_share` for specialists has shifted meaningfully — neither agent sits at exactly 0.5 across the board
- [ ] `claim_citation_rate > 0.6`
- [ ] `aggregator_override_rate < 0.3` (Synthesizer mostly defers to the deterministic scoring)
- [ ] `pytest tests/` green

### Prompt to start Phase 5

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md and the latest baseline. Execute Phase 5.

This is the biggest architectural change in the plan. Touch:
- new agentxai/agents/aggregator.py
- agentxai/agents/synthesizer.py (drop `case`, require citations)
- agentxai/agents/orchestrator.py (insert aggregator, wire critic)
- agentxai/xai/accountability.py (add 'aggregator' to _NON_SPECIALIST_AGENTS)
- eval/mini_baseline.py (new aggregator metrics)
- tests/test_aggregator.py (new)

Critic agent already exists at agentxai/agents/critic.py — wire it into
the orchestrator's loop, don't rebuild it.

The Synthesizer must continue to produce all its existing output fields
(predicted_letter, predicted_text, final_diagnosis, confidence,
rationale, option_analysis, supporting_evidence_ids, plus the Phase 4
additions). New: claim_citations array.

When done: pytest tests/ green, then
  python -m eval.mini_baseline --phase phase5
Confirm accuracy didn't tank (>= 0.95 * phase4) and citation rate > 0.6.
If accuracy dropped > 5%, do not commit — investigate which specialist
the aggregator is mis-weighting.
```

---

## Phase 6 — Evaluation expansion

### What we're fixing

Current eval audits internal consistency, not whether reasoning is correct. No automated rationale-quality signal.

### Concrete changes

**New `xai/reasoning_audit.py`:**
1. `llm_judge_faithfulness(case, ground_truth, rationale, llm) -> Dict[str, float]`: prompts a separate LLM with case + ground truth + rationale, returns 1-5 scores on (factual correctness, relevance, completeness). Average to a [0, 1] normalized score.
2. `nli_entailment_score(rationale, evidence_snippets) -> float`: split rationale into sentences (use `nltk.sent_tokenize` or a simple regex), for each sentence run NLI (`MoritzLaurer/DeBERTa-v3-base-mnli`) against each snippet, return fraction of sentences with at least one entailment.
3. `citation_ablation_faithfulness(task_id, store, pipeline) -> float`: for each `claim_citations` entry, ablate that evidence and re-run synthesizer; fraction of cited evidence whose ablation flipped the answer.

**`eval/evaluate.py`:**
- Wire in the three new metrics. Keep the existing 5 metrics.
- Add `coverage_at_threshold(0.5)` and `selective_accuracy_at_coverage(0.8)` — built from existing `system_confidence`.

**`eval/mini_baseline.py`:**
- Add the three new metrics so each phase's baseline carries them. They're expensive — gate behind `--full` flag, default skip; in mini-baseline only run on 5 of 15 tasks for speed.

**Tests:**
- `tests/test_reasoning_audit.py`: stub the LLM/NLI models; verify the three functions return floats in [0, 1].

### Acceptance criteria

- [ ] All three new metrics populate in `eval/baselines/phase6.json`
- [ ] Run the full `eval/evaluate.py --limit 50` — verify it completes and the new metrics are reasonable (faithfulness > 0.5 on correct-answer tasks, < 0.5 on incorrect-answer tasks would be the ideal signal)
- [ ] `pytest tests/` green
- [ ] The phase 6 baseline file is the headline result for the project write-up

### Prompt to start Phase 6

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md and the latest baseline. Execute Phase 6.

Touch:
- new agentxai/xai/reasoning_audit.py
- eval/evaluate.py (wire in new metrics; keep existing 5)
- eval/mini_baseline.py (add gated full-eval flag)
- tests/test_reasoning_audit.py (new)

The NLI model (DeBERTa-v3-base-mnli) is ~440MB — lazy-load it the same
way the embedding model is loaded in textbook_search.py. If unavailable,
fall back to a simpler entailment heuristic (token overlap > 0.4 between
rationale sentence and snippet) and note this.

When done: pytest tests/ green; run mini_baseline with --full on 5 tasks
and verify all metrics populate; then run eval/evaluate.py --limit 50.
The resulting eval/results_*.json is the project's headline result.
```

---

## Master "do the next phase" prompt

For convenience — a single prompt you can re-paste each session:

```
Read AgentXAI/docs/IMPROVEMENT_PLAN.md. Look at AgentXAI/eval/baselines/
to determine the highest-numbered phase<N>.json that exists. Execute the
NEXT phase after that (phase 0 if nothing exists yet) using the per-phase
prompt and acceptance criteria from the plan.

Constraints from the plan apply to every phase:
- pytest tests/ must stay green
- additive over replacement (new fields beside old, never rename)
- mini-baseline must run successfully and write phase<N+1>.json
- if acceptance criteria fail, do not commit — surface the failure and stop

Confirm with me which phase you're executing before making changes.
```

---

## Cross-phase notes

- **Pinning the seed:** phase 0 freezes the 15-task sample by `make_splits(seed=42)` + first 15. Don't change the seed across phases.
- **When a baseline regresses:** if accuracy drops more than 5% in any phase, *don't commit* — the change broke something. Investigate first.
- **Eval expense vs CI:** mini-baseline is for phase gates, not CI. Don't add it to `pytest`.
- **Backward-compat for stored records:** the SQLite store has historical records with the old field shapes. Every read path must handle missing new fields (default to `None` / `0` / `[]`).
- **What to keep in mind for the write-up:** the most compelling result for a research project is the *delta* between phase 0 and phase 6 baselines on three numbers: `mean_tool_impact` (0 → > 0.1 = central XAI claim), `ece(system_confidence) - ece(llm_confidence)` (calibration improvement), and `faithfulness_check_pass_rate` (overall XAI trustworthiness).
