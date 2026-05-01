# AgentXAI Evaluation Report

**Generated:** 2026-04-27T19:15:56+00:00
**Records evaluated:** 100  **Stability samples:** 10

## 1. Task Performance

**Accuracy:** 48.0%  (48 / 100 correct)

### Confidence Calibration (reliability diagram, 10 bins)

| Bin | Mean conf | Fraction correct | Count |
|-----|-----------|-----------------|-------|
| 0.0–0.1 | 0.05 | 0.000 | 3 |
| 0.1–0.2 | 0.15 | — | 0 |
| 0.2–0.3 | 0.25 | — | 0 |
| 0.3–0.4 | 0.35 | — | 0 |
| 0.4–0.5 | 0.45 | — | 0 |
| 0.5–0.6 | 0.55 | — | 0 |
| 0.6–0.7 | 0.65 | — | 0 |
| 0.7–0.8 | 0.75 | 0.400 | 5 |
| 0.8–0.9 | 0.85 | 0.471 | 34 |
| 0.9–1.0 | 0.95 | 0.517 | 58 |

### Per-Option Confusion Matrix (rows = ground truth, cols = predicted)

| truth \ pred | A | B | C | D | E |
|---|---|---|---|---|---|
| **A** | 13 | 0 | 0 | 1 | 0 |
| **B** | 3 | 8 | 0 | 1 | 1 |
| **C** | 2 | 2 | 9 | 1 | 1 |
| **D** | 1 | 2 | 1 | 10 | 0 |
| **E** | 1 | 1 | 3 | 1 | 8 |

## 2. Sufficiency (XAI)

**Sufficiency score:** 84.9%
- Diagnosis **unchanged** in 84 / 99 tasks when re-run with only the top-1 responsible agent's memory.
- Errors: 0

## 3. Necessity (XAI)

**Necessity score:** 53.5%
- Diagnosis **changed** in 53 / 99 tasks after zeroing out the top-1 agent's memory.
- Errors: 0

## 4. Stability

**Mean Spearman ρ:** 0.2000 ± 0.9798
- Computed on 5 / 10 sampled tasks (5 skipped, 0 errors).

## 5. Faithfulness

**Faithfulness score:** 0.0%
- Outcome **changed** in 0 / 0 tasks after zeroing out the root-cause agent's memory.
- Errors: 0
