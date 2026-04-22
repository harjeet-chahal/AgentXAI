# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository type

This repository contains **data only** — no source code, build system, tests, package manifests, or CI configuration. It is the **MedQA** corpus: multiple-choice medical board exam questions in three languages plus the accompanying textbook corpus used as retrieval context. Any "development" in this repo means inspecting, filtering, converting, or sampling JSONL/text files — typically from outside tooling (the consumer's own scripts) rather than something checked in here.

Do not invent build/test commands. If the user asks to "run tests" or "build", clarify — this repo has nothing to build.

## Top-level layout

```
med_qa/
├── questions/        # MCQ data, partitioned by source country
│   ├── US/           # USMLE-style, English, 5 options
│   ├── Mainland/     # Mainland China medical licensing, Simplified Chinese, 5 options
│   └── Taiwan/       # Taiwan medical licensing, Traditional Chinese, 4 options
└── textbooks/        # Retrieval corpus (plain .txt)
    ├── en/           # 18 English medical textbooks (Harrison, Robbins, First Aid, …)
    ├── zh_paragraph/ # Chinese textbooks, one paragraph per line
    └── zh_sentence/  # Same Chinese textbooks, one sentence per line
```

Each `questions/<country>/` directory has a canonical `train.jsonl` / `dev.jsonl` / `test.jsonl` split plus a combined `*_qbank.jsonl` (the union before splitting). Sizes are large — use streaming (`head`, `wc -l`, line-by-line reads) rather than loading whole files into memory. For reference: US train has ~10k questions, Mainland train ~27k, Taiwan train ~11k; `InternalMed_Harrison.txt` alone is ~22 MB.

## Question JSONL schema

Every `questions/**/*.jsonl` file has one JSON object per line. Core fields are shared across languages:

- `question` — stem text (English, Simplified Chinese, or Traditional Chinese depending on subdir)
- `options` — object keyed `"A".."E"` (US, Mainland) or `"A".."D"` (Taiwan)
- `answer` — the correct option's **text**, not its letter
- `answer_idx` — the correct option's **letter** (use this for scoring)
- `meta_info` — source tag, e.g. `"step1"`, `"step2&3"`, `"卫生法规"`, `"taiwanese_test_Q"`

The US and Mainland sets have 5 options (A–E); Taiwan has 4 (A–D). When writing evaluation code, do not hardcode 5.

### Variants that add or change fields

- `questions/US/4_options/phrases_no_exclude_{split}.jsonl` — US questions reduced to 4 options **and** enriched with a `metamap_phrases` array (MetaMap-extracted clinical phrases from the stem). Note the `answer_idx` is re-lettered for the 4-option set and will differ from the 5-option file for the same question.
- `questions/US/metamap_extracted_phrases/{split}/phrases_{split}.jsonl` — original 5-option US questions with `metamap_phrases` added (no option reduction).
- `questions/Mainland/4_options/{split}.jsonl` — Mainland questions reduced to 4 options (no metamap).
- `questions/Taiwan/metamap/{split}/` — MetaMap-augmented Taiwan questions (the Taiwan set was translated to English to run MetaMap).
- `questions/Taiwan/tw_translated_jsonl/en/*-2en.jsonl` and `.../zh/*-2zh.jsonl` — Traditional↔translated versions of the Taiwan splits.

When a user says "the US test set" without qualification, default to `questions/US/test.jsonl` (the 5-option canonical file). Ask before substituting a `4_options` or `metamap` variant — they are not interchangeable.

## Textbooks

`textbooks/en/*.txt` and `textbooks/zh_*/*.txt` are plain UTF-8. `zh_paragraph` and `zh_sentence` cover the same source books at different granularities — pick based on the retrieval chunk size the downstream task needs. English filenames encode `Subject_AuthorOrEdition.txt` (e.g. `Pathology_Robbins.txt`). Note the upstream typos `Psichiatry_DSM-5.txt` and `Obstentrics_Williams.txt` — preserve them when referencing files.

## Working with this data

- Prefer `Grep` / `Glob` / head-based sampling over reading whole files; the qbank and Harrison textbook will blow past sensible context limits if read end-to-end.
- `.DS_Store` files are macOS Finder metadata — ignore them; never commit new ones.
- If the user wants to produce a derived artifact (filtered subset, reformatted CSV, train/eval script), put it **outside** `med_qa/` so the dataset tree stays pristine unless the user explicitly says otherwise.
