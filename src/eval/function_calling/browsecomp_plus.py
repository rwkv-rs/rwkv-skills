from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.env_config import resolve_judge_max_workers, resolve_judge_model_config
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.browsecomp import (
    BrowseCompJudgeConfig,
    BrowseCompRecord,
    judge_browsecomp_answers,
)
from src.eval.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS, normalize_rwkv_text, truncate_text
from src.eval.function_calling.final_answer import (
    final_answer_tool_schema,
    parse_final_answer_call,
    render_final_answer_call,
)
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_rwkv_json_call_prompt,
    render_function_output_user_block,
    render_json_function_call,
)
from src.eval.function_calling.simple_tool_call import decode_simple_tool_call_response
from src.eval.function_calling.tool_router import (
    ToolRoutingConfig,
    route_tools_for_prompt,
    tool_routing_config_from_args,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_messages_for_long_context, infer_query_from_messages
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

OFFICIAL_BROWSECOMP_PLUS_SOURCE = "texttron/BrowseComp-Plus"
DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT = Path("/tmp/rwkv-official-refs/BrowseComp-Plus")
DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS = 1400
DEFAULT_BROWSECOMP_PLUS_CHUNK_OVERLAP = 220
DEFAULT_BROWSECOMP_PLUS_TOP_K = 5
DEFAULT_BROWSECOMP_PLUS_PROMPT_MAX_CHARS = 8192


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
                    "Answer this BrowseComp-Plus deep-research question using the fixed corpus tools.",
                    "Search and read evidence chunks before final_answer.",
                    "Use concise citations like [docid] when evidence is available.",
                    "",
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
        documents = self._retrieve_documents(query, k=max(20, k))
        return _top_chunks(documents, query=query, limit=k)

    def document_chunks(self, docid: str, *, query: str, limit: int) -> list[dict[str, Any]]:
        document = self._document_by_id(docid)
        if document is None:
            return [{"docid": docid, "error": "not_found"}]
        return _top_chunks([document], query=query, limit=limit)

    def _retrieve_documents(self, query: str, *, k: int) -> list[dict[str, Any]]:
        if self.index_path.exists() and _pyserini_available():
            documents = _search_bm25(self.index_path, query, k)
            for item in documents:
                if item.get("docid"):
                    self.retrieved_docids.add(str(item["docid"]))
            return documents
        documents = self._record_documents()
        scored = sorted(documents, key=lambda item: _document_score(query, item), reverse=True)
        for item in scored[:k]:
            if item.get("docid"):
                self.retrieved_docids.add(str(item["docid"]))
        return scored[:k]

    def _document_by_id(self, docid: str) -> dict[str, Any] | None:
        if self.index_path.exists() and _pyserini_available():
            document = _bm25_document(self.index_path, docid)
            if document is not None:
                self.retrieved_docids.add(docid)
                return document
        for document in self._record_documents():
            if str(document.get("docid") or document.get("id") or "") == docid:
                self.retrieved_docids.add(docid)
                return dict(document)
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
            "retriever": "bm25" if self.index_path.exists() and _pyserini_available() else "record_documents",
        }


def build_browsecomp_plus_prompt(
    messages: Sequence[Mapping[str, Any]],
    *,
    history_max_chars: int,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    rendered_tools = [dict(tool) for tool in (tools or BROWSECOMP_PLUS_TOOL_SCHEMAS)]
    system_prompt = normalize_rwkv_text(
        "\n".join(
            [
                "Tools:",
                json.dumps(rendered_tools, ensure_ascii=False, indent=2),
                "Output JSON schema:",
                json.dumps(
                    {
                        "type": "object",
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                            "id": {"type": "string"},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "Return exactly one JSON function call object.",
                "Include id when producing final_answer; use id final_answer.",
                "Use search and get_document_chunks to gather chunked evidence.",
                "Use final_answer only when ready to answer.",
                "Return no prose, no markdown, and no extra text outside the JSON value.",
            ]
        )
    )
    return build_rwkv_json_call_prompt(system_prompt, messages, history_max_chars=history_max_chars)


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
    effective_history_chars = max(0, int(history_max_chars))
    if prompt_max_chars > 0:
        effective_history_chars = min(effective_history_chars, max(0, int(prompt_max_chars) - 2048))
    prompt = build_browsecomp_plus_prompt(
        compaction.messages,
        history_max_chars=effective_history_chars,
        tools=tool_route.selected_tools,
    )
    if prompt_max_chars > 0 and len(prompt) > prompt_max_chars:
        overage = len(prompt) - int(prompt_max_chars)
        effective_history_chars = max(0, effective_history_chars - overage - 128)
        prompt = build_browsecomp_plus_prompt(
            compaction.messages,
            history_max_chars=effective_history_chars,
            tools=tool_route.selected_tools,
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
        "tool_route": tool_route.trace_payload(),
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
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, sampling)]),
        long_doc_config=long_doc_config,
        tool_routing_config=tool_routing_config,
        prompt_max_chars=prompt_max_chars,
    )
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
    )

    job_name = _resolve_job_name("function_browsecomp_plus", run_context=run_context)
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
                for key, record in pending:
                    payload = _run_one_browsecomp_plus_attempt(
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
                        judge=judge,
                    )
                    writer.enqueue(payload)
            except Exception:  # noqa: BLE001
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_browsecomp_plus_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={
                    "cot_mode": CoTMode.COT.value,
                    "history_max_chars": history_max_chars,
                    "prompt_max_chars": prompt_max_chars,
                    "sampling_config": sampling_payload,
                },
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"browsecomp_plus done: {len(completions_payloads)} samples")
    return 0


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
) -> dict[str, Any]:
    env = BrowseCompPlusEnv(record)
    messages: list[dict[str, str]] = [{"role": "user", "content": env.initial_user_message()}]
    stages: list[StageRecord] = []
    trace: list[dict[str, Any]] = []
    fail_reason = ""
    final_answer = ""
    final_answer_call_id = ""
    decoded_final_answer_call: dict[str, Any] = {}
    max_steps = max(1, int(getattr(args, "max_steps", 12) or 12))
    for step_index in range(1, max_steps + 1):
        prompt, long_doc_trace = build_browsecomp_plus_budgeted_prompt(
            messages,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            long_doc_config=long_doc_config,
            tool_routing_config=tool_routing_config,
            engine=run.engine,
            sampling=sampling,
            prompt_seed=sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=step_index * 10),
        )
        output = run.engine.generate(
            [prompt],
            sampling=sampling,
            batch_size=1,
            progress_desc="BrowseCompPlus-Step",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
            prompt_seeds=[sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=step_index)],
        )[0]
        stages.append(StageRecord(prompt=prompt, completion=output.text, stop_reason=output.finish_reason))
        decoded: list[dict[str, Any]] = []
        try:
            if _looks_like_template_leak(output.text):
                raise ValueError("decision stage leaked internal template/control tokens")
            decoded = decode_simple_tool_call_response(output.text)
            if not decoded:
                raise ValueError("model returned no tool call")
            call = decoded[0]
            if str(call.get("name") or "").strip() == "final_answer":
                final_call = parse_final_answer_call(output.text, context_label="browsecomp-plus final answer")
                call = dict(final_call.call)
                final_answer_call_id = final_call.call_id
        except Exception as exc:  # noqa: BLE001
            fail_reason = str(exc)
            trace.append({"step": step_index, "raw": output.text, "parse_error": fail_reason})
            break
        result = env.step(call)
        messages.append({"role": "assistant", "content": render_json_function_call(call["name"], call["arguments"])})
        messages.append({"role": "user", "content": render_function_output_user_block(result.observation)})
        trace_entry = {
            "step": step_index,
            "decision_completion": output.text,
            "decoded_call": call,
            "observation": truncate_text(result.observation, 4000),
            "done": result.done,
            "details": result.details or {},
            "long_doc": long_doc_trace,
        }
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

    if final_answer:
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
        judge_reason = fail_reason

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
        "cot_mode": CoTMode.COT.value,
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


def preflight_browsecomp_plus_runtime(
    records: Sequence[BrowseCompPlusRecord],
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    roots = {
        str(record.metadata.get("browsecomp_plus_official_root") or DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT)
        for record in records
    }
    for raw_root in sorted(roots):
        root = Path(raw_root).expanduser().resolve()
        if not (root / "scripts_evaluation" / "evaluate_run.py").exists():
            errors.append(f"missing_official_evaluator:{root}")
        index_path = browsecomp_plus_index_path(root)
        if not index_path.exists() or not any(index_path.glob("segments_*")):
            errors.append(f"missing_bm25_index:{index_path}")
    if not _pyserini_available():
        errors.append("missing_pyserini: install rwkv-skills[function-calling-official] or BrowseComp-Plus pyserini deps")
    report = {"ok": not errors, "errors": errors}
    if errors and raise_on_error:
        raise RuntimeError("BrowseComp-Plus runtime preflight failed: " + "; ".join(errors))
    return report


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


def _text_score(query: str, text: str) -> float:
    tokens = _tokenize(query)
    if not tokens:
        return 0.0
    lowered = text.lower()
    return float(sum(1 for token in tokens if token in lowered))


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


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
        raw = hit.lucene_document.get("raw")
        documents.append({"docid": str(hit.docid), "text": _raw_lucene_contents(raw), "score": float(hit.score)})
    return documents


def _bm25_document(index_path: Path, docid: str) -> dict[str, Any] | None:
    searcher = _lucene_searcher(str(index_path))
    document = searcher.doc(docid)
    if document is None:
        return None
    return {"docid": str(docid), "text": _raw_lucene_contents(document.raw())}


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
