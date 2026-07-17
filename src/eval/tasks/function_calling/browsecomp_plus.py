from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.concurrent_runner import run_episodes
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.env_config import resolve_judge_max_workers, resolve_judge_model_config, resolve_judge_timeout_s
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.db.eval_service import create_eval_service, init_eval_store
from src.eval.tasks.function_calling.browsecomp import (
    BrowseCompJudgeConfig,
    BrowseCompRecord,
    judge_browsecomp_answers,
)
from src.eval.tasks.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    compute_function_calling_diagnostics,
    compute_function_calling_metrics,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.tasks.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS, normalize_rwkv_text, truncate_text
from src.eval.tasks.function_calling.final_answer import (
    FinalAnswerCall,
    final_answer_tool_schema,
    parse_final_answer_call,
    render_final_answer_call,
)
from src.eval.tasks.function_calling.agent_loop import (
    _agent_loop_candidate_router_config_from_args,
    _agent_loop_candidate_router_mode,
    _should_use_agent_loop_candidate_router,
)
from src.eval.experiments.parallel_candidate_router.router import (
    ParallelCandidateRouterConfig,
    route_parallel_candidate_tool_call,
)
from src.eval.tasks.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.tasks.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES, render_json_function_call
from src.eval.tasks.function_calling.simple_tool_call import decode_simple_tool_call_response
from src.eval.tasks.function_calling.tool_router import (
    ToolRoutingConfig,
    route_tools_for_prompt,
    tool_routing_config_from_args,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_messages_for_long_context, infer_query_from_messages
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage
from src.eval.scheduler.config import DEFAULT_DB_CONFIG

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

OFFICIAL_BROWSECOMP_PLUS_SOURCE = "texttron/BrowseComp-Plus"
DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT = Path("/tmp/rwkv-official-refs/BrowseComp-Plus")
DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS = 1400
DEFAULT_BROWSECOMP_PLUS_CHUNK_OVERLAP = 220
DEFAULT_BROWSECOMP_PLUS_TOP_K = 5
DEFAULT_BROWSECOMP_PLUS_RETRIEVE_DOCS = 50
DEFAULT_BROWSECOMP_PLUS_PROMPT_MAX_CHARS = 8192
DEFAULT_BROWSECOMP_PLUS_MAX_STEPS = 12
# The unified function-calling CLI currently defaults --max-steps to tau's
# 200-turn budget. Treat that inherited value as "not explicitly set" for
# BrowseComp-Plus so the benchmark keeps its own episode budget by default.
_INHERITED_TAU_DEFAULT_MAX_STEPS = 200
BROWSECOMP_PLUS_JUDGE_MODE_ENV = "RWKV_BROWSECOMP_PLUS_JUDGE_MODE"
BROWSECOMP_PLUS_JUDGE_MODES = frozenset({"inline", "defer", "judge"})
BROWSECOMP_PLUS_RETRIEVER_ENV = "RWKV_BROWSECOMP_PLUS_RETRIEVER"
BROWSECOMP_PLUS_RETRIEVER_MODES = frozenset({"record", "bm25", "auto"})


@dataclass(frozen=True, slots=True)
class BrowseCompPlusRecord:
    task_id: str
    query_id: str
    question: str
    answer: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BrowseCompPlusStepResult:
    observation: str
    done: bool = False
    final_answer: str = ""
    details: dict[str, Any] | None = None


BROWSECOMP_PLUS_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "search",
        "description": "Search the fixed BrowseComp-Plus corpus and return relevant evidence chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "Retrieve one BrowseComp-Plus document by docid.",
        "parameters": {
            "type": "object",
            "properties": {"docid": {"type": "string"}},
            "required": ["docid"],
        },
    },
    {
        "name": "get_document_chunks",
        "description": "Read chunked passages from one retrieved document id.",
        "parameters": {
            "type": "object",
            "properties": {
                "docid": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["docid"],
        },
    },
    final_answer_tool_schema(answer_description="The exact BrowseComp-Plus answer with concise evidence when useful."),
)


def browsecomp_plus_source_path(root: str | Path | None = None) -> Path:
    resolved = Path(root).expanduser().resolve() if root else DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT
    return (resolved / "data" / "browsecomp_plus_decrypted.jsonl").resolve()


def browsecomp_plus_index_path(root: str | Path | None = None) -> Path:
    resolved = Path(root).expanduser().resolve() if root else DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT
    return (resolved / "indexes" / "bm25").resolve()


def load_browsecomp_plus_rows_from_decrypted_jsonl(
    path: str | Path,
    *,
    official_root: str | Path | None = None,
    dataset_name: str = "browsecomp_plus",
    include_documents: bool = False,
) -> list[dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    root = Path(official_root).expanduser().resolve() if official_root else DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT
    index_path = browsecomp_plus_index_path(root)
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if not isinstance(item, Mapping):
                continue
            query_id = str(item.get("query_id") or item.get("id") or index)
            query = str(item.get("query") or item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not query:
                continue
            metadata = {
                "source_format": "official_browsecomp_plus",
                "official_source": OFFICIAL_BROWSECOMP_PLUS_SOURCE,
                "browsecomp_plus_official_root": str(root),
                "browsecomp_plus_bm25_index_path": str(index_path),
                "browsecomp_plus_source_path": str(source_path),
                "query_id": query_id,
                "answer": answer,
            }
            if include_documents:
                metadata["browsecomp_plus_documents"] = _official_row_documents(item)
            rows.append(
                {
                    "task_id": f"{dataset_name}__{query_id}",
                    "instruction": query,
                    "question": query,
                    "answer": answer,
                    "tools": [dict(tool) for tool in BROWSECOMP_PLUS_TOOL_SCHEMAS],
                    "metadata": metadata,
                }
            )
    return rows


def load_browsecomp_plus_manifest_records(path: str | Path) -> list[BrowseCompPlusRecord]:
    records: list[BrowseCompPlusRecord] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for line_index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            metadata = dict(payload.get("metadata") or {})
            query_id = str(metadata.get("query_id") or payload.get("query_id") or line_index)
            question = str(payload.get("question") or payload.get("instruction") or metadata.get("query") or "")
            answer = str(payload.get("answer") or metadata.get("answer") or "")
            records.append(
                BrowseCompPlusRecord(
                    task_id=str(payload.get("task_id") or f"browsecomp_plus__{query_id}"),
                    query_id=query_id,
                    question=question,
                    answer=answer,
                    metadata=metadata,
                )
            )
    return records


def _bound_browsecomp_plus_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = normalize_rwkv_text(str(message.get("content") or ""))
        if content:
            normalized.append({"role": role, "content": content})
    if max_chars <= 0:
        return normalized
    total = 0
    kept_reversed: list[dict[str, str]] = []
    for message in reversed(normalized):
        size = len(message["content"])
        if kept_reversed and total + size > max_chars:
            break
        kept_reversed.append(message)
        total += size
    return list(reversed(kept_reversed))


def _render_browsecomp_plus_agent_state(messages: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    if not messages:
        return "", ""
    current = str(messages[-1].get("content") or "")
    trajectory_rows: list[str] = []
    for message in messages[:-1]:
        role = str(message.get("role") or "user").strip().lower()
        content = str(message.get("content") or "")
        if not content:
            continue
        if role == "assistant":
            trajectory_rows.append(f"Assistant action: {content}")
        else:
            trajectory_rows.append(f"Environment: {content}")
    return "\n".join(trajectory_rows), current


class BrowseCompPlusEnv:
    def __init__(self, record: BrowseCompPlusRecord) -> None:
        self.record = record
        self.official_root = Path(
            str(record.metadata.get("browsecomp_plus_official_root") or DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT)
        ).expanduser().resolve()
        self.index_path = Path(
            str(record.metadata.get("browsecomp_plus_bm25_index_path") or browsecomp_plus_index_path(self.official_root))
        ).expanduser().resolve()
        self.source_path = Path(
            str(record.metadata.get("browsecomp_plus_source_path") or browsecomp_plus_source_path(self.official_root))
        ).expanduser().resolve()
        self.retrieved_docids: set[str] = set()
        self.tool_call_counts: dict[str, int] = {}
        self.final_answer = ""
        self.status = "incomplete"

    def initial_user_message(self) -> str:
        return normalize_rwkv_text(
            "\n".join(
                [
                    "You are answering a BrowseComp-Plus deep-research question against a fixed corpus.",
                    "Use search or get_document as needed. When ready, call final_answer.",
                    "Final answer should include the exact answer and concise evidence citations using [docid] when available.",
                    f"Question: {self.record.question}",
                ]
            )
        )

    def step(self, call: Mapping[str, Any]) -> BrowseCompPlusStepResult:
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        arguments = dict(arguments or {})
        self.tool_call_counts[name] = self.tool_call_counts.get(name, 0) + 1
        if name == "search":
            query = str(arguments.get("query") or "").strip() or self.record.question
            chunks = self.search(query, DEFAULT_BROWSECOMP_PLUS_TOP_K)
            return BrowseCompPlusStepResult(
                observation=json.dumps({"chunks": chunks}, ensure_ascii=False, separators=(",", ":")),
                details=self._details(),
            )
        if name == "get_document":
            docid = str(arguments.get("docid") or "").strip()
            document = self._document_by_id(docid)
            if document is None:
                observation = json.dumps({"docid": docid, "error": "not_found"}, ensure_ascii=False, separators=(",", ":"))
            else:
                observation = json.dumps(_document_payload(document), ensure_ascii=False, separators=(",", ":"))
            return BrowseCompPlusStepResult(
                observation=observation,
                details=self._details(),
            )
        if name == "get_document_chunks":
            docid = str(arguments.get("docid") or "").strip()
            query = str(arguments.get("query") or self.record.question)
            chunks = self.document_chunks(docid, query=query, limit=DEFAULT_BROWSECOMP_PLUS_TOP_K)
            return BrowseCompPlusStepResult(
                observation=json.dumps({"docid": docid, "chunks": chunks}, ensure_ascii=False, separators=(",", ":")),
                details=self._details(),
            )
        if name == "final_answer":
            self.final_answer = str(arguments.get("answer") or "").strip()
            self.status = "completed" if self.final_answer else "incomplete"
            return BrowseCompPlusStepResult(
                observation="Final answer recorded.",
                done=True,
                final_answer=self.final_answer,
                details=self._details(),
            )
        return BrowseCompPlusStepResult(
            observation=f"Unknown BrowseComp-Plus tool: {name}",
            done=True,
            details={**self._details(), "fail_reason": "unknown_tool"},
        )

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        expanded_query = _browsecomp_plus_expanded_query(self.record.question, query)
        documents = self._retrieve_documents(expanded_query, k=max(DEFAULT_BROWSECOMP_PLUS_RETRIEVE_DOCS, k))
        return _top_chunks(documents, query=expanded_query, limit=k)

    def document_chunks(self, docid: str, *, query: str, limit: int) -> list[dict[str, Any]]:
        document = self._document_by_id(docid)
        if document is None:
            return [{"docid": docid, "error": "not_found"}]
        return _top_chunks([document], query=query, limit=limit)

    def _retrieve_documents(self, query: str, *, k: int) -> list[dict[str, Any]]:
        record_documents = self._record_documents()
        if record_documents and not self._use_bm25():
            scored = sorted(record_documents, key=lambda item: _document_score(query, item), reverse=True)
            for item in scored[:k]:
                if item.get("docid"):
                    self.retrieved_docids.add(str(item["docid"]))
            return scored[:k]
        use_bm25 = self._use_bm25()
        if self.index_path.exists() and _pyserini_available():
            documents = _search_bm25(self.index_path, query, k)
            for item in documents:
                if item.get("docid"):
                    self.retrieved_docids.add(str(item["docid"]))
            return documents
        if use_bm25:
            return []
        scored = sorted(record_documents, key=lambda item: _document_score(query, item), reverse=True)
        for item in scored[:k]:
            if item.get("docid"):
                self.retrieved_docids.add(str(item["docid"]))
        return scored[:k]

    def _document_by_id(self, docid: str) -> dict[str, Any] | None:
        for document in self._record_documents():
            if str(document.get("docid") or document.get("id") or "") == docid:
                self.retrieved_docids.add(docid)
                return dict(document)
        if self.index_path.exists() and _pyserini_available():
            document = _bm25_document(self.index_path, docid)
            if document is not None:
                self.retrieved_docids.add(docid)
                return document
        return None

    def _record_documents(self) -> list[dict[str, Any]]:
        docs = _list_of_dicts(self.record.metadata.get("browsecomp_plus_documents"))
        if docs:
            return docs
        return _load_documents_for_query(str(self.source_path), self.record.query_id)

    def _details(self) -> dict[str, Any]:
        return {
            "browsecomp_plus_run": {
                "query_id": self.record.query_id,
                "status": self.status,
                "retrieved_docids": sorted(self.retrieved_docids),
                "tool_call_counts": dict(self.tool_call_counts),
                "result": [{"type": "output_text", "output": self.final_answer}] if self.final_answer else [],
            },
            "retriever": "bm25" if self._use_bm25() and self.index_path.exists() and _pyserini_available() else "record_documents",
        }

    def _use_bm25(self) -> bool:
        mode = _browsecomp_plus_retriever_mode()
        if mode == "bm25":
            return True
        if mode == "auto":
            return not bool(self._record_documents())
        return False


def build_browsecomp_plus_prompt(
    messages: Sequence[Mapping[str, Any]],
    *,
    history_max_chars: int,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    rendered_tools = [dict(tool) for tool in (tools or BROWSECOMP_PLUS_TOOL_SCHEMAS)]
    tool_names = {str(tool.get("name") or "") for tool in rendered_tools}
    if tool_names == {"final_answer"}:
        action_guidance = [
            "The search budget is exhausted. Do not call search or get_document again.",
            "Use final_answer now with the best exact answer supported by the gathered evidence.",
        ]
    else:
        action_guidance = [
            "Use search and get_document to gather evidence. Use final_answer only when ready to answer.",
        ]
    bounded_messages = _bound_browsecomp_plus_messages(messages, max_chars=history_max_chars)
    trajectory, current = _render_browsecomp_plus_agent_state(bounded_messages)
    return normalize_rwkv_text(
        "\n".join(
            [
                "You are controlling tools in a BrowseComp-Plus deep-research environment.",
                "Respond with exactly one JSON tool call and no extra text.",
                'Use this shape: {"name":"ToolName","arguments":{"arg":"value"}}',
                *action_guidance,
                'For final_answer, use exactly {"name":"final_answer","arguments":{"answer":"<exact answer>"}}.',
                "Do not use reason, reasoning, explanation, output, or response keys for final_answer.",
                "Available tools:",
                json.dumps(rendered_tools, ensure_ascii=False, indent=2),
                "",
                "Trajectory:",
                trajectory,
                "",
                "Current observation:",
                current,
                "",
                "Assistant: <think>",
                "</think>",
                "```json",
            ]
        )
    )


def build_browsecomp_plus_budgeted_prompt(
    messages: Sequence[Mapping[str, Any]],
    *,
    history_max_chars: int,
    prompt_max_chars: int,
    long_doc_config: LongDocEvidenceConfig,
    tool_routing_config: ToolRoutingConfig | None = None,
    engine: Any | None = None,
    sampling: Any | None = None,
    prompt_seed: int | None = None,
    force_final_answer: bool = False,
) -> tuple[str, dict[str, Any]]:
    query = infer_query_from_messages(messages, skip_longer_than=max(1, int(long_doc_config.min_long_text_chars)))
    compaction = compact_messages_for_long_context(
        messages,
        query=query,
        config=long_doc_config,
        engine=engine,
        sampling=sampling,
        progress_desc="BrowseCompPlus-LongDoc",
        prompt_seed=prompt_seed,
    )
    if force_final_answer:
        selected_tools = [final_answer_tool_schema(answer_description="The exact BrowseComp-Plus answer.")]
        tool_route_trace: dict[str, Any] = {
            "routed": True,
            "reason": "force_final_answer",
            "selected_names": ["final_answer"],
            "total_tool_count": len(BROWSECOMP_PLUS_TOOL_SCHEMAS),
        }
    else:
        tool_route = route_tools_for_prompt(
            BROWSECOMP_PLUS_TOOL_SCHEMAS,
            compaction.messages,
            config=tool_routing_config or ToolRoutingConfig(),
            engine=engine,
            sampling=sampling,
            control_tool_names=("final_answer",),
            progress_desc="BrowseCompPlus-ToolRouter",
            prompt_seed=None if prompt_seed is None else int(prompt_seed) + 1_000_000,
        )
        selected_tools = tool_route.selected_tools
        tool_route_trace = tool_route.trace_payload()
    effective_history_chars = max(0, int(history_max_chars))
    if prompt_max_chars > 0:
        effective_history_chars = min(effective_history_chars, max(0, int(prompt_max_chars) - 2048))
    prompt = build_browsecomp_plus_prompt(
        compaction.messages,
        history_max_chars=effective_history_chars,
        tools=selected_tools,
    )
    if prompt_max_chars > 0 and len(prompt) > prompt_max_chars:
        overage = len(prompt) - int(prompt_max_chars)
        effective_history_chars = max(0, effective_history_chars - overage - 128)
        prompt = build_browsecomp_plus_prompt(
            compaction.messages,
            history_max_chars=effective_history_chars,
            tools=selected_tools,
        )
    trace = {
        "mode": long_doc_config.mode if long_doc_config.enabled else "off",
        "enabled": bool(long_doc_config.enabled),
        "query_chars": len(query),
        "compacted_message_count": int(compaction.compacted_message_count),
        "selected_chunk_ids": {str(key): list(value) for key, value in compaction.selected_chunk_ids.items()},
        "history_max_chars": int(history_max_chars),
        "effective_history_chars": int(effective_history_chars),
        "prompt_chars": len(prompt),
        "output_format": "rwkv_json_function_call",
        "force_final_answer": bool(force_final_answer),
        "tool_route": tool_route_trace,
    }
    return prompt, trace


def _browsecomp_plus_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("judge_reason") or agent_result.get("error") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=str(agent_info.get("final_answer") or ""),
        ref_answer=str(agent_info.get("reference_answer") or ""),
    )


def _browsecomp_plus_long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
    mode = str(getattr(args, "long_doc_mode", "lexical") or "lexical").strip().lower()
    enabled = mode != "off"
    if mode == "off":
        mode = "lexical"
    return LongDocEvidenceConfig(
        enabled=enabled,
        mode=mode,  # type: ignore[arg-type]
        max_chunk_chars=max(1, int(getattr(args, "long_doc_max_chars", 1000) or 1000)),
        overlap_lines=max(0, int(getattr(args, "long_doc_overlap_lines", 3) or 0)),
        min_long_text_chars=max(1, int(getattr(args, "long_doc_min_chars", 6000) or 6000)),
        max_evidence_chunks=max(1, int(getattr(args, "long_doc_max_evidence_chunks", 4) or 4)),
        max_evidence_chars=max(1, int(getattr(args, "long_doc_max_evidence_chars", 6000) or 6000)),
    )


def _run_browsecomp_plus(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_browsecomp_plus_manifest_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        run.dataset_slug,
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]
    if not records:
        raise ValueError("BrowseComp-Plus manifest is empty")
    if not bool(getattr(args, "skip_runtime_preflight", False)):
        preflight_browsecomp_plus_runtime(records)

    plan = _resolve_function_calling_plan(
        run.dataset_slug,
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    if sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    sampling = clamp_function_calling_sampling(sampling, max(1, int(args.decision_max_tokens or 1024)))
    history_max_chars = max(0, int(args.history_max_chars or DEFAULT_HISTORY_MAX_CHARS))
    prompt_max_chars = int(args.prompt_max_chars or DEFAULT_BROWSECOMP_PLUS_PROMPT_MAX_CHARS)
    long_doc_config = _browsecomp_plus_long_doc_config(args)
    tool_routing_config = tool_routing_config_from_args(args)
    candidate_router_mode = _agent_loop_candidate_router_mode(args)
    candidate_router_config = _agent_loop_candidate_router_config_from_args(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, sampling)]),
        long_doc_config=long_doc_config,
        tool_routing_config=tool_routing_config,
        candidate_router_config=candidate_router_config,
        prompt_max_chars=prompt_max_chars,
    )
    sampling_payload["browsecomp_plus_adapter"] = {
        "candidate_router_mode": candidate_router_mode,
        "decision_io": "rwkv_json_or_parallel_candidate",
    }
    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=max(1, int(args.batch_size or 1)))
        prompts = [
            build_browsecomp_plus_budgeted_prompt(
                [{"role": "user", "content": BrowseCompPlusEnv(record).initial_user_message()}],
                history_max_chars=history_max_chars,
                prompt_max_chars=prompt_max_chars,
                long_doc_config=long_doc_config,
                tool_routing_config=tool_routing_config,
                engine=run.engine,
                sampling=sampling,
            )[0]
            for _index, record in repeated
        ]
        run.engine.generate(
            prompts,
            sampling=sampling,
            batch_size=len(prompts),
            progress_desc="BrowseCompPlus-Probe",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("BrowseComp-Plus requires JUDGE_MODEL + JUDGE_API_KEY")
    judge = BrowseCompJudgeConfig(
        api_key=judge_cfg.api_key,
        model=judge_cfg.model_name,
        base_url=judge_cfg.base_url,
        max_workers=resolve_judge_max_workers(getattr(args, "judge_max_workers", None), default=4),
        timeout_s=resolve_judge_timeout_s(default=60.0),
    )
    judge_mode = _resolve_browsecomp_plus_judge_mode(args)

    job_name = _resolve_job_name("function_browsecomp_plus", run_context=run_context)
    if judge_mode == "judge" and getattr(args, "browsecomp_plus_judge_task_id", None):
        return _judge_browsecomp_plus_task_by_id(
            args=args,
            run=run,
            task_id=str(args.browsecomp_plus_judge_task_id),
            judge=judge,
            job_name=job_name,
            plan=plan,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            sampling_payload=sampling_payload,
        )

    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 8),
        run_context=run_context,
        judger_model_name=judge.model,
    )
    if judge_mode == "judge":
        return _judge_existing_browsecomp_plus_run(
            args=args,
            run=run,
            ctx=ctx,
            judge=judge,
            job_name=job_name,
            plan=plan,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            sampling_payload=sampling_payload,
        )

    runtime = ctx.runtime
    writer = ctx.writer
    flush_partial = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_browsecomp_plus_completion_to_eval_payload,
        runner_name="browsecomp_plus",
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=flush_partial,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                sample_workers = max(1, int(getattr(args, "sample_workers", 1) or 1))
                defer_judge = judge_mode == "defer" or sample_workers > 1
                judge_futures = []

                def _run_pending_item(item: tuple[Any, BrowseCompPlusRecord]) -> dict[str, Any]:
                    key, record = item
                    return _run_one_browsecomp_plus_attempt(
                        args=args,
                        run=run,
                        record=record,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        sampling=sampling,
                        sampling_payload=sampling_payload,
                        history_max_chars=history_max_chars,
                        prompt_max_chars=prompt_max_chars,
                        long_doc_config=long_doc_config,
                        tool_routing_config=tool_routing_config,
                        candidate_router_mode=candidate_router_mode,
                        candidate_router_config=candidate_router_config,
                        judge=judge,
                        defer_judge=defer_judge,
                    )

                if judge_mode == "defer":
                    run_episodes(
                        pending,
                        _run_pending_item,
                        max_workers=sample_workers,
                        on_result=writer.enqueue,
                        label="browsecomp_plus episode",
                        collect_results=False,
                    )
                elif defer_judge:
                    judge_workers = max(1, int(judge.max_workers))
                    with ThreadPoolExecutor(max_workers=judge_workers, thread_name_prefix="browsecomp-plus-judge") as judge_executor:
                        def _submit_judge(payload: dict[str, Any]) -> None:
                            judge_futures.append(
                                judge_executor.submit(_judge_and_enqueue_browsecomp_plus_payload, payload, judge, writer.enqueue)
                            )

                        run_episodes(
                            pending,
                            _run_pending_item,
                            max_workers=sample_workers,
                            on_result=_submit_judge,
                            label="browsecomp_plus episode",
                            collect_results=False,
                        )
                        for future in as_completed(judge_futures):
                            future.result()
                else:
                    run_episodes(
                        pending,
                        _run_pending_item,
                        max_workers=sample_workers,
                        on_result=writer.enqueue,
                        label="browsecomp_plus episode",
                        collect_results=False,
                    )
            except Exception:  # noqa: BLE001
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        if judge_mode == "defer":
            completions_payloads = ctx.runtime.complete_attempt_stage(writer, timeout_s=float(args.db_close_timeout_s))
            ctx.runtime.fail_task(error="browsecomp_plus judge deferred")
            print(f"browsecomp_plus deferred judge: {len(completions_payloads)} samples")
            return 0

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_browsecomp_plus_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: _browsecomp_plus_score_payload(
                run=run,
                plan=plan,
                job_name=job_name,
                completions_payloads=completions_payloads,
                metrics=metrics,
                history_max_chars=history_max_chars,
                prompt_max_chars=prompt_max_chars,
                sampling_payload=sampling_payload,
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"browsecomp_plus done: {len(completions_payloads)} samples")
    return 0


def _resolve_browsecomp_plus_judge_mode(args: argparse.Namespace) -> str:
    raw = getattr(args, "browsecomp_plus_judge_mode", None) or os.environ.get(BROWSECOMP_PLUS_JUDGE_MODE_ENV) or "inline"
    mode = str(raw).strip().lower()
    if mode not in BROWSECOMP_PLUS_JUDGE_MODES:
        expected = ", ".join(sorted(BROWSECOMP_PLUS_JUDGE_MODES))
        raise ValueError(f"unsupported BrowseComp-Plus judge mode {raw!r}; expected one of {expected}")
    return mode


def _browsecomp_plus_retriever_mode() -> str:
    raw = os.environ.get(BROWSECOMP_PLUS_RETRIEVER_ENV) or "record"
    mode = str(raw).strip().lower()
    if mode not in BROWSECOMP_PLUS_RETRIEVER_MODES:
        return "record"
    return mode


def _judge_existing_browsecomp_plus_run(
    *,
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    ctx: Any,
    judge: BrowseCompJudgeConfig,
    job_name: str,
    plan: Any,
    history_max_chars: int,
    prompt_max_chars: int,
    sampling_payload: dict[str, Any],
) -> int:
    ctx.writer.close(timeout_s=float(args.db_close_timeout_s))
    return _judge_browsecomp_plus_payloads_for_service(
        service=ctx.service,
        task_id=str(ctx.task_id),
        run=run,
        judge=judge,
        job_name=job_name,
        plan=plan,
        history_max_chars=history_max_chars,
        prompt_max_chars=prompt_max_chars,
        sampling_payload=sampling_payload,
    )


def _judge_browsecomp_plus_task_by_id(
    *,
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    task_id: str,
    judge: BrowseCompJudgeConfig,
    job_name: str,
    plan: Any,
    history_max_chars: int,
    prompt_max_chars: int,
    sampling_payload: dict[str, Any],
) -> int:
    del args
    init_eval_store(DEFAULT_DB_CONFIG)
    service = create_eval_service()
    return _judge_browsecomp_plus_payloads_for_service(
        service=service,
        task_id=task_id,
        run=run,
        judge=judge,
        job_name=job_name,
        plan=plan,
        history_max_chars=history_max_chars,
        prompt_max_chars=prompt_max_chars,
        sampling_payload=sampling_payload,
    )


def _judge_browsecomp_plus_payloads_for_service(
    *,
    service: Any,
    task_id: str,
    run: ResolvedFunctionCallingRun,
    judge: BrowseCompJudgeConfig,
    job_name: str,
    plan: Any,
    history_max_chars: int,
    prompt_max_chars: int,
    sampling_payload: dict[str, Any],
) -> int:
    completions_payloads = service.list_completion_payloads(task_id=task_id, status="Completed")
    expected_attempts = plan_attempt_count(plan, max_pass_k=1)
    if len(completions_payloads) != expected_attempts:
        raise RuntimeError(
            f"BrowseComp-Plus judge mode expected {expected_attempts} completions for task_id={task_id}, "
            f"found {len(completions_payloads)}"
        )
    judge_workers = max(1, int(judge.max_workers))
    with ThreadPoolExecutor(max_workers=judge_workers, thread_name_prefix="browsecomp-plus-judge-existing") as executor:
        futures = [executor.submit(_judge_browsecomp_plus_payload, dict(payload), judge) for payload in completions_payloads]
        judged_payloads = [future.result() for future in as_completed(futures)]
    judged_payloads.sort(
        key=lambda payload: (
            int(payload.get("sample_index", 0) or 0),
            int(payload.get("repeat_index", 0) or 0),
            int(payload.get("pass_index", 0) or 0),
        )
    )
    service.insert_completion_payloads_batch(payloads=judged_payloads, task_id=task_id)
    eval_payloads = [_browsecomp_plus_completion_to_eval_payload(payload) for payload in judged_payloads]
    service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
    metrics = compute_function_calling_metrics(eval_payloads, avg_k=plan.avg_k)
    metrics.update(compute_function_calling_diagnostics(judged_payloads))
    service.record_score_payload(
        payload=_browsecomp_plus_score_payload(
            run=run,
            plan=plan,
            job_name=job_name,
            completions_payloads=judged_payloads,
            metrics=metrics,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            sampling_payload=sampling_payload,
        ),
        task_id=task_id,
    )
    print(f"browsecomp_plus judged: task_id={task_id} samples={len(judged_payloads)} metrics={metrics}")
    return 0


def _browsecomp_plus_score_payload(
    *,
    run: ResolvedFunctionCallingRun,
    plan: Any,
    job_name: str,
    completions_payloads: Sequence[dict[str, object]],
    metrics: dict[str, float],
    history_max_chars: int,
    prompt_max_chars: int,
    sampling_payload: dict[str, Any],
) -> Mapping[str, object]:
    return make_score_payload(
        run.dataset_slug,
        is_cot=False,
        model_name=run.model_name,
        metrics=metrics,
        samples=len(completions_payloads),
        problems=plan.sample_size,
        task=job_name,
        task_details=build_plan_task_details(plan, cot_mode=CoTMode.NO_COT.value),
        extra={
            "cot_mode": CoTMode.NO_COT.value,
            "history_max_chars": history_max_chars,
            "prompt_max_chars": prompt_max_chars,
            "sampling_config": sampling_payload,
        },
    )


def _render_browsecomp_plus_tool_result_user_content(call: Mapping[str, Any], observation: str) -> str:
    tool_name = str(call.get("name") or "tool").strip() or "tool"
    return normalize_rwkv_text(
        "\n".join(
            [
                f"Tool result from {tool_name}.",
                "This is read-only evidence, not the next assistant JSON object.",
                "Do not copy the evidence JSON shape or its chunk fields.",
                "Next assistant message must be exactly one JSON tool call with keys name and arguments.",
                "Valid tool names are search, get_document, and final_answer.",
                'For final_answer, arguments must contain answer, for example {"answer":"<exact answer>"}.',
                "",
                str(observation),
            ]
        )
    )


def _run_one_browsecomp_plus_attempt(
    *,
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    record: BrowseCompPlusRecord,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    sampling: Any,
    sampling_payload: dict[str, Any],
    history_max_chars: int,
    prompt_max_chars: int,
    long_doc_config: LongDocEvidenceConfig,
    tool_routing_config: ToolRoutingConfig,
    judge: BrowseCompJudgeConfig,
    candidate_router_mode: str = "off",
    candidate_router_config: ParallelCandidateRouterConfig | None = None,
    defer_judge: bool = False,
) -> dict[str, Any]:
    env = BrowseCompPlusEnv(record)
    messages: list[dict[str, str]] = [{"role": "user", "content": env.initial_user_message()}]
    stages: list[StageRecord] = []
    trace: list[dict[str, Any]] = []
    fail_reason = ""
    final_answer = ""
    final_answer_call_id = ""
    decoded_final_answer_call: dict[str, Any] = {}
    max_steps = _resolve_browsecomp_plus_max_steps(getattr(args, "max_steps", None))
    for step_index in range(1, max_steps + 1):
        force_final_answer = _should_force_browsecomp_plus_final_answer(
            step_index=step_index,
            max_steps=max_steps,
            trace=trace,
        )
        decision_completion = ""
        decision_stop_reason = ""
        candidate_trace: dict[str, Any] | None = None
        decision_io = "rwkv_json"
        long_doc_trace: dict[str, Any] = {}
        decoded: list[dict[str, Any]] = []
        use_candidate_router = _should_use_browsecomp_plus_candidate_router(
            mode=candidate_router_mode,
            config=candidate_router_config,
            messages=messages,
            force_final_answer=force_final_answer,
        )
        if use_candidate_router and candidate_router_config is not None:
            route = route_parallel_candidate_tool_call(
                tools=BROWSECOMP_PLUS_TOOL_SCHEMAS,
                messages=messages,
                domain_policy=_browsecomp_plus_candidate_policy(),
                domain="browsecomp_plus",
                facts_text=_browsecomp_plus_candidate_facts(record, env),
                engine=run.engine,
                sampling=sampling,
                config=candidate_router_config,
                progress_desc=f"BrowseCompPlus-CandidateRouter sample {sample_index} step {step_index}",
                prompt_seed=sample_repeat_seed(
                    sample_index,
                    repeat_index,
                    pass_index=pass_index,
                    stage=50_000 + step_index,
                ),
            )
            decision_io = "parallel_candidate"
            candidate_trace = route.trace_payload(include_prompts=True)
            decision_completion = route.aggregate_completion
            decision_stop_reason = route.aggregate_finish_reason or ("candidate_router_empty" if route.selected is None else "stop")
            stages.append(
                StageRecord(
                    prompt=route.aggregate_prompt,
                    completion=decision_completion,
                    stop_reason=decision_stop_reason,
                )
            )
            if route.selected is None:
                fail_reason = route.aggregate_error or "candidate router did not select a tool call"
                trace.append(
                    {
                        "step": step_index,
                        "decision_io": decision_io,
                        "candidate_router": candidate_trace,
                        "parse_error": fail_reason,
                    }
                )
                break
            call = {"name": route.selected.name, "arguments": dict(route.selected.arguments)}
            if str(call.get("name") or "").strip() == "final_answer":
                final_answer_call_id = "final_answer"
        else:
            prompt, long_doc_trace = build_browsecomp_plus_budgeted_prompt(
                messages,
                history_max_chars=history_max_chars,
                prompt_max_chars=prompt_max_chars,
                long_doc_config=long_doc_config,
                tool_routing_config=tool_routing_config,
                engine=run.engine,
                sampling=sampling,
                prompt_seed=sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=step_index * 10),
                force_final_answer=force_final_answer,
            )
            output = run.engine.generate(
                [prompt],
                sampling=sampling,
                batch_size=1,
                progress_desc="BrowseCompPlus-Step",
                prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                prompt_seeds=[sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=step_index)],
            )[0]
            decision_completion = output.text
            decision_stop_reason = output.finish_reason
            stages.append(StageRecord(prompt=prompt, completion=decision_completion, stop_reason=decision_stop_reason))
            try:
                if _looks_like_template_leak(decision_completion):
                    raise ValueError("decision stage leaked internal template/control tokens")
                decoded = decode_simple_tool_call_response(decision_completion)
                if not decoded:
                    raise ValueError("model returned no tool call")
                call = decoded[0]
                if str(call.get("name") or "").strip() == "final_answer":
                    final_call = parse_final_answer_call(decision_completion, context_label="browsecomp-plus final answer")
                    call = dict(final_call.call)
                    final_answer_call_id = final_call.call_id
            except Exception as exc:  # noqa: BLE001
                final_call = _recover_browsecomp_plus_final_answer_call(decision_completion)
                if final_call is None:
                    fail_reason = str(exc)
                    trace.append({"step": step_index, "raw": decision_completion, "parse_error": fail_reason})
                    break
                call = dict(final_call.call)
                final_answer_call_id = final_call.call_id
        result = env.step(call)
        messages.append({"role": "assistant", "content": render_json_function_call(call["name"], call["arguments"])})
        messages.append({"role": "user", "content": _render_browsecomp_plus_tool_result_user_content(call, result.observation)})
        trace_entry = {
            "step": step_index,
            "decision_io": decision_io,
            "decision_completion": decision_completion,
            "decoded_call": call,
            "observation": truncate_text(result.observation, 4000),
            "done": result.done,
            "details": result.details or {},
            "long_doc": long_doc_trace,
            "force_final_answer": bool(force_final_answer),
        }
        if candidate_trace is not None:
            trace_entry["candidate_router"] = candidate_trace
        if result.done:
            final_answer = result.final_answer
            if str(call.get("name") or "").strip() == "final_answer":
                decoded_final_answer_call = dict(call)
                if final_answer:
                    trace_entry["sandbox_return"] = render_final_answer_call(final_answer, call_id=final_answer_call_id)
        trace.append(trace_entry)
        if result.done:
            break
    if not final_answer and not fail_reason:
        fail_reason = "browsecomp_plus produced no final answer"

    if final_answer and not defer_judge:
        outcome = judge_browsecomp_answers(
            [(BrowseCompRecord(record.task_id, record.question, record.answer, "en"), final_answer)],
            config=judge,
        )[0]
        is_passed = bool(outcome.is_passed)
        judge_reason = str(outcome.reason)
        if not is_passed:
            fail_reason = judge_reason
    else:
        is_passed = False
        judge_reason = "judge pending" if final_answer else fail_reason

    payload = SampleRecord(
        benchmark_name=run.benchmark_name,
        dataset_split=run.dataset_split,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        stages=stages,
        sampling_config=sampling_payload,
    ).as_payload()
    payload["agent_result"] = {
        "reward": 1.0 if is_passed else 0.0,
        "num_turns": len(trace),
        "cost": 0.0,
        "is_passed": is_passed,
        "error": fail_reason or None,
    }
    payload["agent_info"] = {
        "final_answer": final_answer,
        "final_answer_call": render_final_answer_call(final_answer, call_id=final_answer_call_id) if final_answer else "",
        "decoded_final_answer_call": decoded_final_answer_call,
        "reference_answer": record.answer,
        "judge_reason": judge_reason,
        "fail_reason": fail_reason,
        "judge_pending": bool(final_answer and defer_judge),
        "cot_mode": CoTMode.NO_COT.value,
        "long_context": {
            "prompt_max_chars": int(prompt_max_chars),
            "history_max_chars": int(history_max_chars),
            "long_doc": {
                "enabled": bool(long_doc_config.enabled),
                "mode": long_doc_config.mode,
            },
            "tool_router": {
                "mode": tool_routing_config.mode,
                "max_tools": int(tool_routing_config.max_tools),
            },
        },
    }
    payload["agent_trace"] = trace
    payload["task_id"] = record.task_id
    payload["domain"] = "function_call"
    payload["instruction"] = record.question
    payload["metadata"] = dict(record.metadata)
    payload.update(env._details())
    return payload


def _resolve_browsecomp_plus_max_steps(raw: Any) -> int:
    if raw is None:
        return DEFAULT_BROWSECOMP_PLUS_MAX_STEPS
    max_steps = int(raw)
    if max_steps == _INHERITED_TAU_DEFAULT_MAX_STEPS:
        return DEFAULT_BROWSECOMP_PLUS_MAX_STEPS
    return max(1, max_steps)


def _should_use_browsecomp_plus_candidate_router(
    *,
    mode: str,
    config: ParallelCandidateRouterConfig | None,
    messages: Sequence[Mapping[str, object]],
    force_final_answer: bool,
) -> bool:
    if force_final_answer:
        return False
    raw_message_chars = sum(len(str(message.get("content") or "")) for message in messages)
    return _should_use_agent_loop_candidate_router(
        mode=mode,
        config=config,
        tools=BROWSECOMP_PLUS_TOOL_SCHEMAS,
        messages=messages,
        candidate_context_chars=raw_message_chars,
        raw_message_chars=raw_message_chars,
        long_doc_compacted=False,
    )


def _browsecomp_plus_candidate_policy() -> str:
    return normalize_rwkv_text(
        "BrowseComp-Plus policy: choose exactly one next JSON function call. "
        "Use search to retrieve evidence, get_document or get_document_chunks to inspect a known docid, "
        "and final_answer only when the exact answer is supported or the search budget is nearly exhausted. "
        "Search queries should preserve the important entities and constraints from the original question. "
        "Do not invent docids or answers; use only listed tool names and JSON-object arguments."
    )


def _browsecomp_plus_candidate_facts(record: BrowseCompPlusRecord, env: BrowseCompPlusEnv) -> str:
    payload = {
        "task_id": record.task_id,
        "query_id": record.query_id,
        "question": record.question,
        "retrieved_docids": sorted(env.retrieved_docids),
    }
    return normalize_rwkv_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _should_force_browsecomp_plus_final_answer(
    *,
    step_index: int,
    max_steps: int,
    trace: Sequence[Mapping[str, Any]],
) -> bool:
    if step_index >= max_steps:
        return True
    if step_index < max(4, max_steps - 2):
        return False
    recent = list(trace[-3:])
    if len(recent) < 3:
        return False
    names = [
        str((entry.get("decoded_call") or {}).get("name") or "")
        for entry in recent
        if isinstance(entry.get("decoded_call"), Mapping)
    ]
    if len(names) != 3 or any(name not in {"search", "get_document", "get_document_chunks"} for name in names):
        return False
    retrieved_snapshots = [
        tuple(
            ((entry.get("details") or {}).get("browsecomp_plus_run") or {}).get("retrieved_docids")
            or ()
        )
        for entry in recent
    ]
    return len(set(retrieved_snapshots)) <= 1


def _recover_browsecomp_plus_final_answer_call(response: str) -> FinalAnswerCall | None:
    try:
        return parse_final_answer_call(
            response,
            answer_keys=("answer",),
            context_label="browsecomp-plus final answer",
        )
    except Exception:  # noqa: BLE001
        pass

    value = _load_leading_browsecomp_plus_json_value(response)
    if value is None:
        return None
    if isinstance(value, Mapping):
        name = str(value.get("name") or "").strip()
        if name == "final_answer":
            try:
                return parse_final_answer_call(
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                    answer_keys=("answer",),
                    context_label="browsecomp-plus final answer",
                )
            except Exception:  # noqa: BLE001
                pass
        for answer_key in ("answer", "final_answer", "response", "output", "final"):
            if answer_key not in value:
                continue
            answer = normalize_rwkv_text(str(value.get(answer_key) or ""))
            if answer:
                return FinalAnswerCall(
                    answer=answer,
                    call={"name": "final_answer", "arguments": {"answer": answer}, "id": "final_answer"},
                    call_id="final_answer",
                )
            break
    answer = _extract_browsecomp_plus_answer_string(response)
    if answer:
        return FinalAnswerCall(
            answer=answer,
            call={"name": "final_answer", "arguments": {"answer": answer}, "id": "final_answer"},
            call_id="final_answer",
        )
    return None


def _extract_browsecomp_plus_answer_string(response: str) -> str:
    text = normalize_rwkv_text(response)
    if "final_answer" not in text:
        return ""
    match = re.search(
        r'(?:\\?")answer(?:\\?")\s*:\s*(?:\\?")(?P<answer>(?:\\\\.|\\(?!")|[^"\\])*)',
        text,
        flags=re.DOTALL,
    )
    if match is None:
        return ""
    raw_answer = match.group("answer")
    try:
        answer = json.loads(f'"{raw_answer}"')
    except json.JSONDecodeError:
        answer = raw_answer.replace(r"\"", '"').replace(r"\n", "\n")
    return normalize_rwkv_text(str(answer or ""))


def _load_leading_browsecomp_plus_json_value(response: str) -> Any | None:
    text = normalize_rwkv_text(response).strip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:") :].lstrip()
    text = re.sub(r"(?s)^<think>.*?</think>\s*", "", text).strip()
    if text.startswith("```json"):
        text = text[len("```json") :].lstrip()
    elif text.startswith("```"):
        text = text[len("```") :].lstrip()
    try:
        value, _end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    return value


def _judge_and_enqueue_browsecomp_plus_payload(
    payload: dict[str, Any],
    judge: BrowseCompJudgeConfig,
    enqueue: Any,
) -> None:
    judged = _judge_browsecomp_plus_payload(payload, judge)
    enqueue(judged)


def _judge_browsecomp_plus_payload(payload: dict[str, Any], judge: BrowseCompJudgeConfig) -> dict[str, Any]:
    agent_info = payload.get("agent_info")
    agent_result = payload.get("agent_result")
    if not isinstance(agent_info, dict) or not isinstance(agent_result, dict):
        return payload
    final_answer = str(agent_info.get("final_answer") or "").strip()
    if not final_answer:
        agent_info["judge_pending"] = False
        return payload

    reference_answer = str(agent_info.get("reference_answer") or "")
    task_id = str(payload.get("task_id") or "")
    question = str(payload.get("instruction") or "")
    outcome = judge_browsecomp_answers(
        [(BrowseCompRecord(task_id, question, reference_answer, "en"), final_answer)],
        config=judge,
    )[0]
    is_passed = bool(outcome.is_passed)
    judge_reason = str(outcome.reason)
    fail_reason = "" if is_passed else judge_reason
    agent_result["reward"] = 1.0 if is_passed else 0.0
    agent_result["is_passed"] = is_passed
    agent_result["error"] = fail_reason or None
    agent_info["judge_reason"] = judge_reason
    agent_info["fail_reason"] = fail_reason
    agent_info["judge_pending"] = False
    return payload


def preflight_browsecomp_plus_runtime(
    records: Sequence[BrowseCompPlusRecord],
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    retriever_mode = _browsecomp_plus_retriever_mode()
    records_by_root: dict[str, list[BrowseCompPlusRecord]] = {}
    for record in records:
        raw_root = str(record.metadata.get("browsecomp_plus_official_root") or DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT)
        records_by_root.setdefault(raw_root, []).append(record)
    needs_pyserini = False
    for raw_root, root_records in sorted(records_by_root.items()):
        root = Path(raw_root).expanduser().resolve()
        if not (root / "scripts_evaluation" / "evaluate_run.py").exists():
            errors.append(f"missing_official_evaluator:{root}")
        record_probe = _probe_browsecomp_plus_record_corpus(root_records)
        if record_probe and retriever_mode == "record":
            errors.append(record_probe)
        needs_bm25 = retriever_mode == "bm25" or (retriever_mode == "auto" and record_probe is not None)
        if needs_bm25:
            needs_pyserini = True
            index_path = browsecomp_plus_index_path(root)
            if not index_path.exists() or not any(index_path.glob("segments_*")):
                errors.append(f"missing_bm25_index:{index_path}")
            elif _pyserini_available():
                probe_error = _probe_browsecomp_plus_bm25_corpus(
                    index_path,
                    root_records,
                    qrel_path=root / "topics-qrels" / "qrel_evidence.txt",
                )
                if probe_error:
                    errors.append(probe_error)
    if needs_pyserini and not _pyserini_available():
        errors.append("missing_pyserini: install rwkv-skills[function-calling-official] or BrowseComp-Plus pyserini deps")
    report = {"ok": not errors, "errors": errors}
    if errors and raise_on_error:
        raise RuntimeError("BrowseComp-Plus runtime preflight failed: " + "; ".join(errors))
    return report


def _probe_browsecomp_plus_record_corpus(records: Sequence[BrowseCompPlusRecord]) -> str | None:
    for record in records[: min(5, len(records))]:
        documents = _list_of_dicts(record.metadata.get("browsecomp_plus_documents"))
        if not documents:
            root = Path(
                str(record.metadata.get("browsecomp_plus_official_root") or DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT)
            ).expanduser().resolve()
            source_path = Path(
                str(record.metadata.get("browsecomp_plus_source_path") or browsecomp_plus_source_path(root))
            ).expanduser().resolve()
            documents = _load_documents_for_query(str(source_path), record.query_id)
        if not _has_document_text(documents):
            return f"empty_record_corpus:{record.task_id}"
    return None


def _probe_browsecomp_plus_bm25_corpus(
    index_path: Path,
    records: Sequence[BrowseCompPlusRecord],
    *,
    qrel_path: Path | None = None,
) -> str | None:
    try:
        searcher = _lucene_searcher(str(index_path))
        num_docs = getattr(searcher, "num_docs", None)
        if callable(num_docs):
            num_docs = num_docs()
        if isinstance(num_docs, int) and num_docs <= 0:
            return f"empty_bm25_index:{index_path}"
        probe = next((record for record in records if record.question.strip()), records[0] if records else None)
        if probe is None:
            return "empty_bm25_probe:no_records"
        documents = _search_bm25(index_path, probe.question, 1)
    except Exception as exc:  # noqa: BLE001
        return f"bm25_probe_failed:{index_path}:{type(exc).__name__}:{exc}"
    if not _has_document_text(documents):
        task_id = probe.task_id if probe else "unknown"
        return f"empty_bm25_corpus:{index_path}:{task_id}"
    qrels = _load_browsecomp_plus_qrels(qrel_path) if qrel_path else {}
    if qrels:
        overlap_error = _probe_browsecomp_plus_bm25_qrel_overlap(index_path, records, qrels)
        if overlap_error:
            return overlap_error
    return None


def _has_document_text(documents: Sequence[Mapping[str, Any]]) -> bool:
    return any(_document_text(document).strip() for document in documents)


def _probe_browsecomp_plus_bm25_qrel_overlap(
    index_path: Path,
    records: Sequence[BrowseCompPlusRecord],
    qrels: Mapping[str, set[str]],
) -> str | None:
    checked = 0
    for record in records[: min(10, len(records))]:
        relevant = qrels.get(str(record.query_id)) or set()
        if not relevant:
            continue
        checked += 1
        query = _browsecomp_plus_expanded_query(record.question, record.question)
        retrieved = {str(item.get("docid") or "") for item in _search_bm25(index_path, query, 50)}
        if retrieved & relevant:
            return None
    if checked:
        return f"bm25_qrel_probe_zero_overlap:{index_path}:checked={checked}"
    return None


def _top_chunks(documents: Sequence[Mapping[str, Any]], *, query: str, limit: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        docid = str(document.get("docid") or document.get("id") or "")
        for chunk_index, text in enumerate(_chunk_text(_document_text(document))):
            chunks.append(
                {
                    "docid": docid,
                    "chunk_id": f"{docid}:{chunk_index}",
                    "score": _text_score(query, text),
                    "text": text,
                }
            )
    chunks.sort(key=lambda item: (float(item["score"]), str(item["docid"])), reverse=True)
    return chunks[: max(1, int(limit))]


def _chunk_text(text: str) -> list[str]:
    normalized = normalize_rwkv_text(text)
    if len(normalized) <= DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS:
        return [normalized] if normalized else []
    chunks: list[str] = []
    start = 0
    step = max(1, DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS - DEFAULT_BROWSECOMP_PLUS_CHUNK_OVERLAP)
    while start < len(normalized):
        chunk = normalized[start : start + DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _document_score(query: str, document: Mapping[str, Any]) -> float:
    return _text_score(query, _document_text(document))


def _document_payload(document: Mapping[str, Any]) -> dict[str, str]:
    return {
        "docid": str(document.get("docid") or document.get("id") or ""),
        "text": _document_text(document),
    }


def _text_score(query: str, text: str) -> float:
    tokens = _tokenize(query)
    if not tokens:
        return 0.0
    lowered = text.lower()
    return float(sum(1 for token in tokens if token in lowered))


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _browsecomp_plus_expanded_query(question: str, query: str) -> str:
    question_text = normalize_rwkv_text(question)
    query_text = normalize_rwkv_text(query)
    if not query_text:
        return question_text
    if query_text.lower() in question_text.lower():
        return question_text
    return normalize_rwkv_text(f"{question_text}\n{query_text}")


def _document_text(document: Mapping[str, Any]) -> str:
    for key in ("text", "contents", "content", "snippet", "body"):
        value = document.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(dict(document), ensure_ascii=False, separators=(",", ":"))


def _official_row_documents(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("gold_docs", "evidence_docs", "negative_docs"):
        for document in _list_of_dicts(item.get(key)):
            docid = str(document.get("docid") or document.get("id") or "")
            if docid and docid in seen:
                continue
            if docid:
                seen.add(docid)
            documents.append(document)
    return documents


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _load_browsecomp_plus_qrels(path: Path | None) -> dict[str, set[str]]:
    if path is None or not path.exists():
        return {}
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            query_id, _iteration, docid = parts[:3]
            qrels.setdefault(str(query_id), set()).add(str(docid))
    return qrels


@lru_cache(maxsize=256)
def _load_documents_for_query(source_path: str, query_id: str) -> list[dict[str, Any]]:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, Mapping) and str(item.get("query_id") or item.get("id") or "") == str(query_id):
                return _official_row_documents(item)
    return []


def _pyserini_available() -> bool:
    try:
        from pyserini.search.lucene import LuceneSearcher  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@lru_cache(maxsize=4)
def _lucene_searcher(index_path: str) -> Any:
    from pyserini.search.lucene import LuceneSearcher

    return LuceneSearcher(index_path)


def _search_bm25(index_path: Path, query: str, k: int) -> list[dict[str, Any]]:
    searcher = _lucene_searcher(str(index_path))
    documents: list[dict[str, Any]] = []
    for hit in searcher.search(query, k):
        raw = _bm25_hit_raw(searcher, hit)
        text = _raw_lucene_contents(raw)
        if not text:
            continue
        docid = _raw_lucene_docid(raw) or str(hit.docid)
        documents.append({"docid": docid, "text": text, "score": float(hit.score)})
    return documents


def _bm25_document(index_path: Path, docid: str) -> dict[str, Any] | None:
    searcher = _lucene_searcher(str(index_path))
    document = searcher.doc(docid)
    if document is None:
        return None
    return {"docid": str(docid), "text": _raw_lucene_contents(document.raw())}


def _bm25_hit_raw(searcher: Any, hit: Any) -> Any:
    raw = getattr(hit, "raw", None)
    if raw:
        return raw
    try:
        raw = hit.lucene_document.get("raw")
    except Exception:  # noqa: BLE001
        raw = None
    if raw:
        return raw
    try:
        document = searcher.doc(hit.docid)
    except Exception:  # noqa: BLE001
        return None
    if document is None:
        return None
    try:
        return document.raw()
    except Exception:  # noqa: BLE001
        try:
            return document.contents()
        except Exception:  # noqa: BLE001
            return None


def _raw_lucene_contents(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, Mapping):
        value = parsed.get("contents") or parsed.get("text")
        if isinstance(value, str):
            return value
    return raw


def _raw_lucene_docid(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if isinstance(parsed, Mapping):
        return str(parsed.get("docid") or parsed.get("id") or "").strip()
    return ""


__all__ = [
    "BROWSECOMP_PLUS_TOOL_SCHEMAS",
    "BrowseCompPlusEnv",
    "BrowseCompPlusRecord",
    "OFFICIAL_BROWSECOMP_PLUS_SOURCE",
    "_run_browsecomp_plus",
    "browsecomp_plus_index_path",
    "browsecomp_plus_source_path",
    "build_browsecomp_plus_budgeted_prompt",
    "build_browsecomp_plus_prompt",
    "load_browsecomp_plus_manifest_records",
    "load_browsecomp_plus_rows_from_decrypted_jsonl",
    "preflight_browsecomp_plus_runtime",
]
