"""
End-to-end runner: loads a MedQA record, runs the multi-agent diagnosis
pipeline, and triggers the full XAI layer (counterfactual perturbations,
causal-DAG construction, accountability report).

Usage
-----
Programmatic::

    from run_pipeline import Pipeline
    pipe = Pipeline()
    record = pipe.run_task(medqa_record)   # AgentXAIRecord

CLI::

    python run_pipeline.py --task-id-source medqa --limit 5

Pipeline.resume_from(state_snapshot, overrides) is the protocol method
required by ``CounterfactualEngine``: given a snapshot of the original run's
specialist memories + sent messages, plus an override (Type-1 tool, Type-2
agent memory, or Type-3 message content), it re-runs the minimum necessary
suffix of the pipeline (downstream specialist + Synthesizer for tool
perturbations; just the Synthesizer otherwise) on a scratch in-memory store
and returns the perturbed final output.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_log = logging.getLogger(__name__)

load_dotenv(override=True)

from agentxai.agents.base import make_default_llm
from agentxai.agents.orchestrator import Orchestrator
from agentxai.agents.specialist_a import SpecialistA
from agentxai.agents.specialist_b import SpecialistB
from agentxai.agents.synthesizer import Synthesizer
from agentxai.data.load_medqa import load_medqa_us
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.tools.guideline_lookup import guideline_lookup as _gl_tool
from agentxai.tools.pubmed_search import pubmed_search as _pm_tool
from agentxai.tools.severity_scorer import severity_scorer as _sev_tool
from agentxai.tools.symptom_lookup import symptom_lookup as _sym_tool
from agentxai.xai.accountability import AccountabilityReportGenerator
from agentxai.xai.causal_dag import CausalDAGBuilder
from agentxai.xai.counterfactual_engine import CounterfactualEngine
from agentxai.xai.memory_logger import MemoryLogger
from agentxai.xai.message_logger import MessageLogger
from agentxai.xai.plan_tracker import PlanTracker
from agentxai.xai.tool_provenance import ToolProvenanceLogger, traced_tool
from agentxai.xai.trajectory_logger import TrajectoryLogger


# ---------------------------------------------------------------------------
# Underlying tool callables (peel off the default-logger @traced_tool wrapper
# so we can re-wrap with the per-task ToolProvenanceLogger).
# ---------------------------------------------------------------------------

_RAW_SYMPTOM_LOOKUP   = getattr(_sym_tool, "__wrapped__", _sym_tool)
_RAW_SEVERITY_SCORER  = getattr(_sev_tool, "__wrapped__", _sev_tool)
_RAW_PUBMED_SEARCH    = getattr(_pm_tool,  "__wrapped__", _pm_tool)
_RAW_GUIDELINE_LOOKUP = getattr(_gl_tool,  "__wrapped__", _gl_tool)


# Neutral "no information" replacements per tool, matching each tool's
# expected return shape (so a patched specialist still composes cleanly).
_NEUTRAL_TOOL_RESULT: Dict[str, Callable[..., Any]] = {
    "symptom_lookup":   lambda *a, **kw: {"related_conditions": [], "source": "neutral"},
    "severity_scorer":  lambda *a, **kw: 0.0,
    "pubmed_search":    lambda *a, **kw: [],
    "guideline_lookup": lambda *a, **kw: {"match": None},
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Glue between MedQA records, the four agents, and the seven XAI loggers."""

    def __init__(
        self,
        *,
        db_url: Optional[str] = None,
        llm: Any = None,
        symptom_lookup_fn: Optional[Callable[..., Any]] = None,
        severity_scorer_fn: Optional[Callable[..., Any]] = None,
        pubmed_search_fn: Optional[Callable[..., Any]] = None,
        guideline_lookup_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.db_url = db_url
        # Lazily build the default LLM only if the caller didn't pass one.
        self.llm = llm if llm is not None else make_default_llm()

        self._raw_symptom_lookup   = symptom_lookup_fn   or _RAW_SYMPTOM_LOOKUP
        self._raw_severity_scorer  = severity_scorer_fn  or _RAW_SEVERITY_SCORER
        self._raw_pubmed_search    = pubmed_search_fn    or _RAW_PUBMED_SEARCH
        self._raw_guideline_lookup = guideline_lookup_fn or _RAW_GUIDELINE_LOOKUP

        # Snapshots of completed runs, keyed by task_id (used by resume_from).
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public: run one task end-to-end
    # ------------------------------------------------------------------

    def run_task(self, record: dict) -> AgentXAIRecord:
        """Run the full pipeline on one MedQA record and return the persisted XAI record."""
        task_id = str(uuid.uuid4())
        store = self._open_store()

        # Accept both shapes:
        #   1. Flat MedQA: {question, options, answer_idx, ...}
        #   2. Nested UI:  {input: {patient_case, answer_options},
        #                   ground_truth: {correct_answer, explanation}}
        # The nested shape was previously surfaced by the dashboard's default
        # template; pipelines silently produced empty cases when given it.
        # Flatten it here so submission shape never matters.
        if not record.get("question") and not record.get("patient_case"):
            inp = record.get("input") or {}
            gt = record.get("ground_truth") or {}
            if isinstance(inp, dict) and (
                inp.get("patient_case") or inp.get("answer_options")
            ):
                record = {
                    **record,
                    "question": inp.get("patient_case") or inp.get("question") or "",
                    "options": (
                        inp.get("answer_options")
                        or inp.get("options")
                        or {}
                    ),
                    "answer_idx": (
                        gt.get("correct_answer")
                        or record.get("answer_idx")
                        or ""
                    ),
                    "meta_info": (
                        gt.get("explanation")
                        or record.get("meta_info")
                        or ""
                    ),
                }

        case = record.get("question") or record.get("patient_case", "") or ""
        options = dict(record.get("options") or {})
        correct_letter = _resolve_correct_letter(record)
        ground_truth = {
            "correct_answer": correct_letter,
            "answer_text":    options.get(correct_letter, "") or record.get("answer", ""),
            "explanation":    record.get("meta_info", ""),
        }

        # 1+8. Persist the task row up-front; FK checks elsewhere need a parent row.
        task_record = AgentXAIRecord(
            task_id=task_id,
            source=record.get("source", "medqa"),
            input={
                "patient_case":  case,
                "options":       options,
                "raw_task_id":   record.get("task_id", ""),
                "meta_info":     record.get("meta_info", ""),
            },
            ground_truth=ground_truth,
            system_output={},
        )
        store.save_task(task_record)

        # 2+3. Build all loggers + agents wired to this task_id.
        loggers, agents, tool_logger = self._build_run(store, task_id)

        # 4. Orchestrator → SpecialistA → SpecialistB → Synthesizer.
        agent_payload = {"patient_case": case, "options": options}
        result = agents["orchestrator"].run(agent_payload) or {}
        final = result.get("final_output", {}) or {}

        predicted_letter, predicted_text = _match_option_letter(
            final.get("final_diagnosis", ""), options,
        )
        correct = bool(predicted_letter and predicted_letter == ground_truth["correct_answer"])
        system_output = {
            "final_diagnosis":  final.get("final_diagnosis", ""),
            "confidence":       float(final.get("confidence", 0.0) or 0.0),
            "differential":     list(final.get("differential", []) or []),
            "rationale":        final.get("rationale", ""),
            "predicted_letter": predicted_letter,
            "predicted_text":   predicted_text,
            "correct":          correct,
        }
        task_record.system_output = system_output
        store.save_task(task_record)

        # Build the snapshot used by counterfactual perturbations.
        snapshot = self._build_snapshot(store, task_id, agent_payload, system_output)
        self._snapshots[task_id] = snapshot

        # 5. Counterfactual engine — Type-1, Type-2, Type-3 perturbations.
        engine = CounterfactualEngine(
            store=store,
            pipeline=self,
            task_id=task_id,
            state_snapshot=snapshot,
            original_output=system_output,
        )
        full_record = store.get_full_record(task_id)
        for tc in full_record.xai_data.tool_calls:
            try:
                score = engine.perturb_tool_output(tc.tool_call_id)
                tool_logger.attach_impact_score(
                    tool_call_id=tc.tool_call_id,
                    score=score,
                    counterfactual_run_id=task_id,
                )
            except Exception as exc:
                _log.warning(
                    "tool perturbation failed for tool_call_id=%s (%s): %s",
                    tc.tool_call_id, tc.tool_name, exc,
                )
                continue
        for agent_id in ("specialist_a", "specialist_b"):
            try:
                engine.perturb_agent_output(agent_id)
            except Exception as exc:
                _log.warning(
                    "agent_output perturbation failed for %s: %s", agent_id, exc,
                )
                continue
        for m in full_record.xai_data.messages:
            try:
                changed, description = engine.perturb_message(m.message_id)
                if changed:
                    loggers["message_logger"].mark_acted_upon(m.message_id, description)
            except Exception as exc:
                _log.warning(
                    "message perturbation failed for message_id=%s: %s",
                    m.message_id, exc,
                )
                continue

        # 6. Causal DAG.
        CausalDAGBuilder(store).build(task_id)

        # 7. Accountability report.
        AccountabilityReportGenerator(
            store=store,
            pipeline=self,
            specialist_agents=("specialist_a", "specialist_b"),
            llm=self.llm,
        ).generate(
            task_id=task_id,
            state_snapshot=snapshot,
            original_output=system_output,
        )

        # 9. Re-load and return the fully-hydrated record.
        return store.get_full_record(task_id)

    # ------------------------------------------------------------------
    # Pipeline Protocol — required by CounterfactualEngine
    # ------------------------------------------------------------------

    def resume_from(
        self,
        state_snapshot: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Re-run the minimum suffix of the pipeline against `overrides`.

        Type 1 ("tool_output"):  re-run the downstream specialist with the
                                 affected tool patched to a neutral output,
                                 then re-run the Synthesizer.
        Type 2 ("agent_memory"): replace the affected specialist's memory
                                 wholesale, then re-run the Synthesizer only.
        Type 3 ("message_content"): replace the sender's memory snapshot with
                                 the perturbed message content (findings IS
                                 the memory snapshot), then re-run Synthesizer.

        Always runs against a fresh scratch in-memory store so the original
        task's XAI tables are not polluted by the counterfactual sweep.
        """
        scratch_store = TrajectoryStore(db_url="sqlite:///:memory:")
        scratch_task = "cf-" + uuid.uuid4().hex[:8]
        scratch_store.save_task(
            AgentXAIRecord(task_id=scratch_task, source="counterfactual"),
        )

        loggers, _, _ = self._build_run(scratch_store, scratch_task)

        payload = dict(state_snapshot.get("input_payload") or {})
        snap_mem = state_snapshot.get("agent_memory") or {}
        mem_a = dict(snap_mem.get("specialist_a") or {})
        mem_b = dict(snap_mem.get("specialist_b") or {})

        # ----- Type 1: tool perturbations -----------------------------
        tool_overrides: Dict[str, Any] = overrides.get("tool_output") or {}
        rerun_a, rerun_b = False, False
        patched_a: Dict[str, Callable[..., Any]] = {}
        patched_b: Dict[str, Callable[..., Any]] = {}
        for tool_id, _baseline in tool_overrides.items():
            owner = (state_snapshot.get("tool_owner") or {}).get(tool_id, "")
            name  = (state_snapshot.get("tool_name")  or {}).get(tool_id, "")
            neutral = _NEUTRAL_TOOL_RESULT.get(name)
            if neutral is None:
                continue
            if owner == "specialist_a":
                patched_a[name] = neutral
                rerun_a = True
            elif owner == "specialist_b":
                patched_b[name] = neutral
                rerun_b = True

        if rerun_a:
            mem_a = self._rerun_specialist_a(loggers, payload, patched_a)
        if rerun_b:
            mem_b = self._rerun_specialist_b(loggers, payload, patched_b)

        # ----- Type 2: agent_memory perturbations ---------------------
        for agent_id, override_mem in (overrides.get("agent_memory") or {}).items():
            override_mem = dict(override_mem or {})
            if agent_id == "specialist_a":
                mem_a = override_mem
            elif agent_id == "specialist_b":
                mem_b = override_mem

        # ----- Type 3: message_content perturbations ------------------
        for msg_id, override_content in (overrides.get("message_content") or {}).items():
            sender = (state_snapshot.get("message_senders") or {}).get(msg_id, "")
            override_content = dict(override_content or {})
            if sender == "specialist_a":
                mem_a = override_content
            elif sender == "specialist_b":
                mem_b = override_content

        # Seed the synthesizer's view of specialist memory and re-synthesise.
        scratch_mem_a = loggers["memory_logger"].for_agent("specialist_a")
        scratch_mem_b = loggers["memory_logger"].for_agent("specialist_b")
        dict.clear(scratch_mem_a)
        dict.update(scratch_mem_a, mem_a)
        dict.clear(scratch_mem_b)
        dict.update(scratch_mem_b, mem_b)

        synth = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=self.llm,
            **loggers,
        )
        return synth.run(payload) or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_store(self) -> TrajectoryStore:
        return TrajectoryStore(db_url=self.db_url)

    def _build_run(
        self,
        store: TrajectoryStore,
        task_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], ToolProvenanceLogger]:
        traj = TrajectoryLogger(store, task_id)
        plan = PlanTracker(store, task_id, llm=self.llm)
        mem  = MemoryLogger(store, task_id)
        msg  = MessageLogger(store, task_id)
        tool_logger = ToolProvenanceLogger(store, task_id)

        loggers: Dict[str, Any] = {
            "trajectory_logger": traj,
            "plan_tracker":      plan,
            "memory_logger":     mem,
            "message_logger":    msg,
        }

        sym_fn = _wrap_tool(tool_logger, "specialist_a", "symptom_lookup",   self._raw_symptom_lookup)
        sev_fn = _wrap_tool(tool_logger, "specialist_a", "severity_scorer",  self._raw_severity_scorer)
        pub_fn = _wrap_tool(tool_logger, "specialist_b", "pubmed_search",    self._raw_pubmed_search)
        gl_fn  = _wrap_tool(tool_logger, "specialist_b", "guideline_lookup", self._raw_guideline_lookup)

        spec_a = SpecialistA(
            agent_id="specialist_a",
            symptom_lookup_fn=sym_fn,
            severity_scorer_fn=sev_fn,
            orchestrator_id="orchestrator",
            llm=self.llm,
            **loggers,
        )
        spec_b = SpecialistB(
            agent_id="specialist_b",
            pubmed_search_fn=pub_fn,
            guideline_lookup_fn=gl_fn,
            orchestrator_id="orchestrator",
            llm=self.llm,
            **loggers,
        )
        synth = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=self.llm,
            **loggers,
        )
        orch = Orchestrator(
            agent_id="orchestrator",
            specialist_a=spec_a,
            specialist_b=spec_b,
            synthesizer=synth,
            llm=self.llm,
            **loggers,
        )
        agents = {
            "specialist_a": spec_a,
            "specialist_b": spec_b,
            "synthesizer":  synth,
            "orchestrator": orch,
        }
        return loggers, agents, tool_logger

    def _build_snapshot(
        self,
        store: TrajectoryStore,
        task_id: str,
        payload: Dict[str, Any],
        original_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = store.get_full_record(task_id)

        # Pull each specialist's memory off the live MemoryLogger via the
        # store-side state (we only need the post-run snapshot of writes).
        mem_a = _last_write_state(record.xai_data.memory_diffs, "specialist_a")
        mem_b = _last_write_state(record.xai_data.memory_diffs, "specialist_b")

        tool_owner = {tc.tool_call_id: tc.called_by for tc in record.xai_data.tool_calls}
        tool_name  = {tc.tool_call_id: tc.tool_name for tc in record.xai_data.tool_calls}

        message_senders  = {m.message_id: m.sender        for m in record.xai_data.messages}
        message_contents = {m.message_id: dict(m.content) for m in record.xai_data.messages}

        return {
            "input_payload":    dict(payload),
            "agent_memory":     {"specialist_a": mem_a, "specialist_b": mem_b},
            "tool_owner":       tool_owner,
            "tool_name":        tool_name,
            "message_senders":  message_senders,
            "message_contents": message_contents,
            "original_output":  dict(original_output),
        }

    def _rerun_specialist_a(
        self,
        loggers: Dict[str, Any],
        payload: Dict[str, Any],
        patched_tools: Dict[str, Callable[..., Any]],
    ) -> Dict[str, Any]:
        sym_fn = patched_tools.get("symptom_lookup",  self._raw_symptom_lookup)
        sev_fn = patched_tools.get("severity_scorer", self._raw_severity_scorer)
        spec_a = SpecialistA(
            agent_id="specialist_a",
            symptom_lookup_fn=sym_fn,
            severity_scorer_fn=sev_fn,
            orchestrator_id="orchestrator",
            llm=self.llm,
            **loggers,
        )
        return dict(spec_a.run(payload) or {})

    def _rerun_specialist_b(
        self,
        loggers: Dict[str, Any],
        payload: Dict[str, Any],
        patched_tools: Dict[str, Callable[..., Any]],
    ) -> Dict[str, Any]:
        pub_fn = patched_tools.get("pubmed_search",    self._raw_pubmed_search)
        gl_fn  = patched_tools.get("guideline_lookup", self._raw_guideline_lookup)
        spec_b = SpecialistB(
            agent_id="specialist_b",
            pubmed_search_fn=pub_fn,
            guideline_lookup_fn=gl_fn,
            orchestrator_id="orchestrator",
            llm=self.llm,
            **loggers,
        )
        return dict(spec_b.run(payload) or {})


# ---------------------------------------------------------------------------
# Convenience module-level entry point
# ---------------------------------------------------------------------------

def run_task(record: dict, **pipeline_kwargs: Any) -> AgentXAIRecord:
    """One-shot helper: build a fresh Pipeline and run a single record."""
    return Pipeline(**pipeline_kwargs).run_task(record)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_tool(
    tool_logger: ToolProvenanceLogger,
    called_by: str,
    tool_name: str,
    callable_obj: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap an unwrapped tool callable in a per-task @traced_tool decorator."""
    return traced_tool(tool_logger, called_by=called_by, tool_name=tool_name)(callable_obj)


def _last_write_state(memory_diffs, agent_id: str) -> Dict[str, Any]:
    """Reconstruct an agent's final memory dict from its ordered MemoryDiff log."""
    state: Dict[str, Any] = {}
    for d in memory_diffs:
        if d.agent_id != agent_id or d.operation != "write":
            continue
        state[d.key] = d.value_after
    return state


def _resolve_correct_letter(record: dict) -> str:
    """
    Pull the ground-truth option letter ("A"–"E") out of a record regardless
    of whether it came from the raw MedQA JSONL (answer_idx is a letter
    string), load_medqa_us's normalised output (answer_idx is an int 0–4 and
    answer is the letter), or a UI submission (mixed).
    """
    raw = record.get("answer_idx")
    if isinstance(raw, int):
        return chr(ord("A") + raw) if 0 <= raw < 26 else ""
    if isinstance(raw, str) and raw.strip():
        s = raw.strip()
        if s.isdigit():
            i = int(s)
            return chr(ord("A") + i) if 0 <= i < 26 else ""
        return s.upper()
    answer = str(record.get("answer", "")).strip()
    if len(answer) == 1 and answer.upper().isalpha():
        return answer.upper()
    return ""


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Matches a leading option-letter token: bare ("A"), with delimiter ("A.",
# "A:", "A) text"), or wrapped in brackets ("(A)", "[A]"). The trailing
# delimiter or end-of-string requirement is what stops "Acute" from being
# read as "A" + "cute".
_LEADING_LETTER_RE = re.compile(
    r"^\s*[\(\[]?([A-Za-z])(?:[\)\].:\-\s]|$)"
)


def _match_option_letter(
    diagnosis: str,
    options: Dict[str, str],
) -> Tuple[str, str]:
    """
    Map a free-text diagnosis to the best matching answer-option letter.

    Two-stage strategy:

      1. **Letter-prefix shortcut.** If the diagnosis starts with a bare
         option-letter token — "A", "A.", "A:", "A) Aldosterone excess",
         "(A)", etc. — we accept that letter directly. Without this, a
         synthesizer that returns just the letter (which Gemini sometimes
         does despite prompting for the disease name) gets scored as
         incorrect even when the letter matches the ground truth.

      2. **Token-overlap fallback.** Otherwise pick the option with the
         highest Jaccard score against the diagnosis text. Ties broken by
         earliest letter. Returns ("", "") if no options or zero overlap.
    """
    if not diagnosis or not options:
        return "", ""

    # ----- Stage 1: leading-letter shortcut --------------------------------
    upper_keys = {k.upper(): k for k in options.keys()}
    m = _LEADING_LETTER_RE.match(diagnosis)
    if m:
        letter = m.group(1).upper()
        if letter in upper_keys:
            actual_key = upper_keys[letter]
            return letter, options.get(actual_key, "") or ""

    # ----- Stage 2: token-overlap fallback ---------------------------------
    dx_tokens = set(t.lower() for t in _WORD_RE.findall(diagnosis))
    if not dx_tokens:
        return "", ""

    best_letter = ""
    best_score  = 0.0
    best_text   = ""
    for letter in sorted(options.keys()):
        text = options[letter] or ""
        opt_tokens = set(t.lower() for t in _WORD_RE.findall(text))
        if not opt_tokens:
            continue
        overlap = len(dx_tokens & opt_tokens)
        if overlap == 0:
            continue
        union = len(dx_tokens | opt_tokens)
        score = overlap / union if union else 0.0
        if score > best_score:
            best_score, best_letter, best_text = score, letter, text

    return best_letter, best_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the AgentXAI pipeline on N records and print a summary."
    )
    p.add_argument(
        "--task-id-source",
        choices=["medqa"],
        default="medqa",
        help="Where to source records (only 'medqa' supported today).",
    )
    p.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="train",
        help="Which MedQA US split to draw records from (default: train).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of records to run (default: 5).",
    )
    p.add_argument(
        "--db-url",
        default=None,
        help="Override the SQLAlchemy DB URL (default: project sqlite file).",
    )
    return p


def _print_summary(rows: List[Tuple[str, str, str, bool]]) -> None:
    print(f"{'task_id':40} {'pred':>5} {'truth':>6} {'correct':>8}")
    print("-" * 64)
    for tid, pred, truth, correct in rows:
        mark = "yes" if correct else "no"
        print(f"{tid:40} {pred:>5} {truth:>6} {mark:>8}")
    n = len(rows)
    n_correct = sum(1 for _, _, _, c in rows if c)
    if n:
        print("-" * 64)
        print(f"accuracy: {n_correct}/{n} = {n_correct / n:.1%}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.task_id_source != "medqa":
        print(f"Unsupported source: {args.task_id_source!r}", file=sys.stderr)
        return 2

    records = load_medqa_us(args.split)[: max(0, args.limit)]
    if not records:
        print("No records to process.", file=sys.stderr)
        return 1

    pipeline = Pipeline(db_url=args.db_url)
    rows: List[Tuple[str, str, str, bool]] = []
    for rec in records:
        result = pipeline.run_task(rec)
        rows.append((
            result.task_id,
            result.system_output.get("predicted_letter", "") or "",
            result.ground_truth.get("correct_answer", "") or "",
            bool(result.system_output.get("correct", False)),
        ))

    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
