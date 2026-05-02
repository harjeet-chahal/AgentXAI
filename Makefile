.PHONY: install test index run-one run-eval dashboard api review help

# ── configurable ──────────────────────────────────────────────────────────────
PYTHON      ?= python
SPLIT       ?= train
LIMIT       ?= 5
EVAL_LIMIT  ?= 1500
STABILITY_N ?= 100
API_HOST    ?= 127.0.0.1
API_PORT    ?= 8000
# ──────────────────────────────────────────────────────────────────────────────

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:       ## Install all Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:          ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

index:         ## Build the FAISS textbook index and guideline stubs (one-time, ~5 min)
	$(PYTHON) -m agentxai.data.build_knowledge_base

run-one:       ## Run the pipeline on LIMIT records (default 5) from SPLIT (default train)
	$(PYTHON) run_pipeline.py --split $(SPLIT) --limit $(LIMIT)

run-eval:      ## Run all five XAI evaluation metrics (writes eval/results_<ts>.{json,md})
	$(PYTHON) -m eval.evaluate \
	  --limit $(EVAL_LIMIT) \
	  --samples-for-stability $(STABILITY_N)

review-agg:    ## Aggregate manual review ratings → eval/manual_review_summary.{md,json}
	$(PYTHON) eval/aggregate_manual_reviews.py \
	  --out-md  eval/manual_review_summary.md \
	  --out-json eval/manual_review_summary.json

api:           ## Start the FastAPI backend on API_HOST:API_PORT
	uvicorn agentxai.api.main:app \
	  --host $(API_HOST) \
	  --port $(API_PORT) \
	  --reload

dashboard:     ## Launch the Streamlit dashboard (requires api to be running)
	streamlit run agentxai/ui/dashboard.py

review:        ## Open the manual review page directly (hidden from main dashboard nav)
	streamlit run agentxai/ui/disabled_pages/review.py
