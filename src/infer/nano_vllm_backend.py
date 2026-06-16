from __future__ import annotations

"""Inference backend backed by the external nano-vLLM RWKV engine."""

import importlib
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from types import ModuleType
from typing import Any, Callable, Literal, Sequence

import torch
from tqdm import tqdm

from .constraints import DecodeConstraint
from .engine import DEFAULT_PREFILL_CHUNK_SIZE
from .sampling import GeneratedTextDelta, GenerationOutput, SamplingConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NANO_VLLM_PATH = _REPO_ROOT / "vendor" / "nano-vllm-rwkv"


@dataclass(slots=True, frozen=True)
class NanoVLLMBackendConfig:
    model_path: str
    model_name: str | None = None
    nano_vllm_path: str | Path = DEFAULT_NANO_VLLM_PATH
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    rwkv_prefill_token_budget: int = 2048
    rwkv_prefill_max_batch_size: int = 128
    rwkv_prefill_chunk_size: int = -1
    rwkv_state_cache_enable: bool = False
    max_state_slots: int = -1
    rwkv_state_cache_safety_reserve_slots: int = 0
    sampling_bucket_temperature_resolution: float = 0.0
    sampling_bucket_top_p_resolution: float = 0.0
    rwkv_quant_int8: bool = False
    rwkv_int8_fp16_lm_head: bool = False
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    def engine_kwargs(self) -> dict[str, object]:
        return {
            "max_num_batched_tokens": int(self.max_num_batched_tokens),
            "max_num_seqs": int(self.max_num_seqs),
            "max_model_len": int(self.max_model_len),
            "rwkv_prefill_token_budget": int(self.rwkv_prefill_token_budget),
            "rwkv_prefill_max_batch_size": int(self.rwkv_prefill_max_batch_size),
            "rwkv_prefill_chunk_size": int(self.rwkv_prefill_chunk_size),
            "rwkv_state_cache_enable": bool(self.rwkv_state_cache_enable),
            "max_state_slots": int(self.max_state_slots),
            "rwkv_state_cache_safety_reserve_slots": int(self.rwkv_state_cache_safety_reserve_slots),
            "sampling_bucket_temperature_resolution": float(self.sampling_bucket_temperature_resolution),
            "sampling_bucket_top_p_resolution": float(self.sampling_bucket_top_p_resolution),
            "rwkv_quant_int8": bool(self.rwkv_quant_int8),
            "rwkv_int8_fp16_lm_head": bool(self.rwkv_int8_fp16_lm_head),
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
            "tensor_parallel_size": int(self.tensor_parallel_size),
            "enforce_eager": bool(self.enforce_eager),
        }


@dataclass(slots=True, frozen=True)
class _NanoVLLMRuntime:
    root: Path
    llm_engine_cls: type
    scheduler_cls: type
    model_runner_cls: type
    sampling_params_cls: type


@dataclass(slots=True)
class _NanoSubmitRequest:
    prompt_index: int
    prompt: str
    sampling: SamplingConfig
    future: Future[GenerationOutput]
    prompt_stop_suffixes: Sequence[str] | None
    on_complete: Callable[[GenerationOutput], None] | None
    on_token: Callable[[int, GeneratedTextDelta], None] | None
    enqueued_at: int


@dataclass(slots=True)
class _NanoScoreRequest:
    prompt: str
    choice_token_texts: tuple[str, ...]
    future: Future[tuple[dict[str, float], str]]
    enqueued_at: int


@dataclass(slots=True)
class _NanoActiveRecord:
    prompt_index: int
    prompt: str
    seq: object
    future: Future[GenerationOutput]
    prompt_stop_suffixes: Sequence[str] | None
    on_complete: Callable[[GenerationOutput], None] | None
    on_token: Callable[[int, GeneratedTextDelta], None] | None
    enqueued_at: int
    admitted_at: int
    prompt_token_count: int
    cache_admission_observed: bool = False


@dataclass(slots=True)
class _NanoActiveScoreRecord:
    prompt: str
    seq: object
    future: Future[tuple[dict[str, float], str]]
    choice_token_texts: tuple[str, ...]
    choice_token_ids: tuple[int, ...]
    enqueued_at: int
    admitted_at: int
    prompt_token_count: int
    cache_admission_observed: bool = False


@dataclass(slots=True)
class NanoVLLMInferenceBackend:
    config: NanoVLLMBackendConfig
    runtime: _NanoVLLMRuntime
    engine: object
    scheduler: object
    model_runner: object
    _inbox: Queue[_NanoSubmitRequest | _NanoScoreRequest | None]
    _stop: threading.Event
    _owner: threading.Thread
    _active_lock: threading.Lock
    _active_records: dict[int, _NanoActiveRecord]
    _active_score_records: dict[int, _NanoActiveScoreRecord]
    _metrics_lock: threading.Lock
    _last_batch_metrics: dict[str, object]
    _metrics_totals: dict[str, int]

    @classmethod
    def from_config(cls, config: NanoVLLMBackendConfig) -> "NanoVLLMInferenceBackend":
        runtime = load_nano_vllm_runtime(config.nano_vllm_path)
        engine = runtime.llm_engine_cls(str(config.model_path), **config.engine_kwargs())
        scheduler = getattr(engine, "scheduler", None)
        model_runner = getattr(engine, "model_runner", None)
        if not isinstance(scheduler, runtime.scheduler_cls):
            raise RuntimeError("nano-vLLM LLMEngine did not construct a Scheduler instance")
        if not isinstance(model_runner, runtime.model_runner_cls):
            raise RuntimeError("nano-vLLM LLMEngine did not construct a ModelRunner instance")
        backend = cls(
            config=config,
            runtime=runtime,
            engine=engine,
            scheduler=scheduler,
            model_runner=model_runner,
            _inbox=Queue(),
            _stop=threading.Event(),
            _owner=threading.Thread(),
            _active_lock=threading.Lock(),
            _active_records={},
            _active_score_records={},
            _metrics_lock=threading.Lock(),
            _last_batch_metrics={},
            _metrics_totals={
                "submitted_requests": 0,
                "completed_requests": 0,
                "error_requests": 0,
                "cancelled_requests": 0,
                "prompt_prefill_tokens": 0,
                "engine_step_tokens": 0,
                "generated_tokens": 0,
                "engine_step_count": 0,
                "state_cache_admissions": 0,
                "state_cache_hits": 0,
                "state_cache_exact_hits": 0,
                "state_cache_misses": 0,
            },
        )
        backend._owner = threading.Thread(
            target=backend._run_owner,
            name="rwkv-nano-vllm-owner",
            daemon=True,
        )
        backend._owner.start()
        return backend

    @property
    def model_name(self) -> str:
        raw_name = self.config.model_name
        if raw_name:
            return str(raw_name)
        return Path(str(self.config.model_path)).stem

    def last_batch_metrics(self) -> dict[str, object]:
        with self._metrics_lock:
            last = dict(self._last_batch_metrics)
            totals = dict(self._metrics_totals)
        with self._active_lock:
            active_records = len(self._active_records)
        scheduler = _scheduler_snapshot(self.scheduler)
        state_cache_admissions = int(totals.get("state_cache_admissions", 0))
        state_cache_hits = int(totals.get("state_cache_hits", 0))
        hit_rate = (
            float(state_cache_hits) / float(state_cache_admissions)
            if state_cache_admissions > 0
            else 0.0
        )
        last.update(
            {
                "engine_inbox": self._inbox.qsize(),
                "active_records": active_records,
                "owner_alive": self._owner.is_alive(),
                "scheduler_snapshot": scheduler,
                "scheduler_waiting": _scheduler_count(self.scheduler, "waiting"),
                "scheduler_running": _scheduler_count(self.scheduler, "running"),
                "submitted_requests": int(totals.get("submitted_requests", 0)),
                "completed_requests": int(totals.get("completed_requests", 0)),
                "error_requests": int(totals.get("error_requests", 0)),
                "cancelled_requests": int(totals.get("cancelled_requests", 0)),
                "prompt_prefill_tokens": int(totals.get("prompt_prefill_tokens", 0)),
                "generated_tokens": int(totals.get("generated_tokens", 0)),
                "state_cache": {
                    "admissions": state_cache_admissions,
                    "hits": state_cache_hits,
                    "exact_hits": int(totals.get("state_cache_exact_hits", 0)),
                    "misses": int(totals.get("state_cache_misses", 0)),
                    "hit_rate": round(hit_rate, 6),
                },
            }
        )
        return last

    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling: SamplingConfig,
        batch_size: int,
        progress_desc: str = "Generating",
        probe_only: bool = False,
        on_complete: Callable[[GenerationOutput], None] | None = None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None = None,
        prompt_stop_suffixes: Sequence[Sequence[str] | None] | None = None,
        constraints: Sequence[DecodeConstraint | None] | None = None,
        constraint_mode: Literal["off", "soft", "strict"] = "off",
        prompt_seeds: Sequence[int | None] | None = None,
        top_logprobs: int = 0,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        show_progress: bool = True,
    ) -> list[GenerationOutput]:
        _ = top_logprobs
        _ = prefill_chunk_size
        _ = batch_size
        if not prompts:
            return []
        if prompt_stop_suffixes is not None and len(prompt_stop_suffixes) != len(prompts):
            raise ValueError("prompt_stop_suffixes length must match prompts length")
        if prompt_seeds is not None:
            if len(prompt_seeds) != len(prompts):
                raise ValueError("prompt_seeds length must match prompts length")
            if any(seed is not None for seed in prompt_seeds):
                raise NotImplementedError("nano-vLLM backend does not support per-prompt seeds")
        if _has_active_constraints(constraints=constraints, constraint_mode=constraint_mode):
            raise NotImplementedError("nano-vLLM backend does not support prompt constraints")

        effective_sampling = sampling.clamp(1) if probe_only else sampling
        progress = tqdm(total=len(prompts), desc=progress_desc, unit=" prompt", disable=not show_progress)
        futures: list[Future[GenerationOutput]] = []

        def _on_submit_complete(output: GenerationOutput) -> None:
            if on_complete is not None and not probe_only:
                on_complete(output)
            progress.update(1)

        try:
            for prompt_index, prompt in enumerate(prompts):
                futures.append(
                    self.submit(
                        str(prompt),
                        sampling=effective_sampling,
                        prompt_index=prompt_index,
                        prompt_stop_suffixes=(
                            None if prompt_stop_suffixes is None else prompt_stop_suffixes[prompt_index]
                        ),
                        prompt_seed=None if prompt_seeds is None else prompt_seeds[prompt_index],
                        constraints=None,
                        constraint_mode=constraint_mode,
                        on_complete=_on_submit_complete,
                        on_token=on_token if not probe_only else None,
                    )
                )
            outputs = [future.result() for future in futures]
        finally:
            progress.close()
        return outputs

    def submit(
        self,
        prompt: str,
        *,
        sampling: SamplingConfig,
        prompt_index: int = 0,
        prompt_stop_suffixes: Sequence[str] | None = None,
        prompt_seed: int | None = None,
        constraints: Sequence[DecodeConstraint | None] | None = None,
        constraint_mode: Literal["off", "soft", "strict"] = "off",
        top_logprobs: int = 0,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        probe_only: bool = False,
        on_complete: Callable[[GenerationOutput], None] | None = None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None = None,
    ) -> Future[GenerationOutput]:
        _ = top_logprobs
        _ = prefill_chunk_size
        if probe_only:
            sampling = sampling.clamp(1)
        if self._stop.is_set():
            raise RuntimeError("nano-vLLM backend is shut down")
        if prompt_seed is not None:
            raise NotImplementedError("nano-vLLM backend does not support per-prompt seeds")
        if _has_active_constraints(constraints=constraints, constraint_mode=constraint_mode):
            raise NotImplementedError("nano-vLLM backend does not support prompt constraints")
        future: Future[GenerationOutput] = Future()
        request = _NanoSubmitRequest(
            prompt_index=int(prompt_index),
            prompt=str(prompt),
            sampling=sampling,
            future=future,
            prompt_stop_suffixes=prompt_stop_suffixes,
            on_complete=on_complete,
            on_token=on_token,
            enqueued_at=time.monotonic_ns(),
        )
        with self._metrics_lock:
            self._metrics_totals["submitted_requests"] += 1
        self._inbox.put(request)
        return future

    def _run_owner(self) -> None:
        shutdown_error = RuntimeError("nano-vLLM backend shut down")
        try:
            while True:
                admitted = self._drain_inbox(block=self._active_count() == 0)
                if self._stop.is_set():
                    self._cancel_pending_inbox(shutdown_error)
                    self._fail_all_active(shutdown_error)
                    return
                if self._active_count() == 0:
                    if not admitted:
                        time.sleep(0.001)
                    continue
                try:
                    self._observe_cache_admissions()
                    step_started = time.perf_counter()
                    raw_outputs, num_tokens = self._step_engine()
                    elapsed_s = max(time.perf_counter() - step_started, 1e-9)
                    self._observe_cache_admissions()
                    self._record_owner_step(num_tokens=num_tokens, elapsed_s=elapsed_s)
                except BaseException as exc:
                    self._fail_all_active(exc)
                    continue
                self._complete_raw_outputs(raw_outputs)
        finally:
            self._cancel_pending_inbox(shutdown_error)
            self._fail_all_active(shutdown_error)
            exit_method = getattr(self.engine, "exit", None)
            if callable(exit_method):
                exit_method()

    def _drain_inbox(self, *, block: bool) -> bool:
        admitted = False
        while True:
            try:
                item = self._inbox.get(timeout=0.05 if block and not admitted else 0.0)
            except Empty:
                return admitted
            if item is None:
                if self._stop.is_set():
                    return admitted
                continue
            if self._stop.is_set():
                self._set_future_exception(item.future, RuntimeError("nano-vLLM backend is shut down"))
                self._record_error_request()
                continue
            try:
                if isinstance(item, _NanoScoreRequest):
                    self._admit_score_request(item)
                else:
                    self._admit_request(item)
                admitted = True
            except BaseException as exc:
                self._set_future_exception(item.future, exc)
                self._record_error_request()
                self._fail_all_active(exc)

    def _admit_request(self, request: _NanoSubmitRequest) -> None:
        sampling_params = self._to_sampling_params(request.sampling)
        seq = self.engine.add_request(request.prompt, sampling_params)
        stop_token_seqs = self._build_stop_token_seqs(
            sampling=request.sampling,
            stop_suffixes=request.prompt_stop_suffixes,
        )
        if stop_token_seqs:
            setattr(seq, "stop_token_seqs", stop_token_seqs)
        record = _NanoActiveRecord(
            prompt_index=request.prompt_index,
            prompt=request.prompt,
            seq=seq,
            future=request.future,
            prompt_stop_suffixes=request.prompt_stop_suffixes,
            on_complete=request.on_complete,
            on_token=request.on_token,
            enqueued_at=request.enqueued_at,
            admitted_at=time.monotonic_ns(),
            prompt_token_count=_prompt_token_count(self.engine, request.prompt),
        )
        seq_id = int(getattr(seq, "seq_id"))
        with self._active_lock:
            self._active_records[seq_id] = record
        with self._metrics_lock:
            self._metrics_totals["prompt_prefill_tokens"] += int(record.prompt_token_count)

    def _admit_score_request(self, request: _NanoScoreRequest) -> None:
        prompt_token_ids = self._encode_text(request.prompt)
        if not prompt_token_ids:
            raise ValueError("choice scoring prompt must tokenize to at least one token")
        choice_token_ids = tuple(self._single_token_id(text) for text in request.choice_token_texts)
        sampling_params = self._to_sampling_params(
            SamplingConfig(
                max_generate_tokens=1,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                alpha_presence=0.0,
                alpha_frequency=0.0,
                alpha_decay=1.0,
                stop_tokens=(),
                no_penalty_token_ids=(),
            )
        )
        seq = self.engine.add_request(request.prompt, sampling_params)
        setattr(seq, "allow_sparse_penalty_state", True)
        record = _NanoActiveScoreRecord(
            prompt=request.prompt,
            seq=seq,
            future=request.future,
            choice_token_texts=request.choice_token_texts,
            choice_token_ids=choice_token_ids,
            enqueued_at=request.enqueued_at,
            admitted_at=time.monotonic_ns(),
            prompt_token_count=len(prompt_token_ids),
        )
        seq_id = int(getattr(seq, "seq_id"))
        with self._active_lock:
            self._active_score_records[seq_id] = record
        with self._metrics_lock:
            self._metrics_totals["prompt_prefill_tokens"] += int(record.prompt_token_count)

    def _step_engine(self) -> tuple[object, object]:
        if self._active_score_count() > 0:
            return self._step_engine_with_scoring()
        return self.engine.step()

    def _step_engine_with_scoring(self) -> tuple[list[tuple[int, object]], int]:
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = (
            sum(_seq_prefill_step_tokens(seq, _scheduler_prefill_chunk_size(self.scheduler)) for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        logits = self.model_runner.call("run_logits", seqs, is_prefill)
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2:
            raise RuntimeError(f"nano-vLLM run_logits returned unexpected shape {tuple(logits.shape)}")

        token_ids: list[int | None] = [None] * len(seqs)
        sample_indices: list[int] = []
        sample_seqs: list[object] = []
        score_results: list[tuple[int, dict[str, float]]] = []

        for index, seq in enumerate(seqs):
            seq_id = int(getattr(seq, "seq_id"))
            score_record = self._get_active_score_record(seq_id)
            if score_record is not None:
                if _can_emit_sequence_token(self.scheduler, seq, is_prefill):
                    score_results.append((seq_id, _score_choices_from_logits(logits[index], score_record)))
                continue
            if not _can_emit_sequence_token(self.scheduler, seq, is_prefill):
                continue
            if getattr(seq, "pending_hidden_finalize", False):
                continue
            sample_indices.append(index)
            sample_seqs.append(seq)

        if sample_indices:
            index_tensor = torch.tensor(sample_indices, dtype=torch.int64, device=logits.device)
            sample_logits = logits.index_select(0, index_tensor) if len(sample_indices) != len(seqs) else logits
            sampled_tokens = self.model_runner.sampler(sample_logits, sample_seqs).tolist()
            for index, token_id in zip(sample_indices, sampled_tokens, strict=True):
                token_ids[index] = int(token_id)

        self.model_runner.call("prepare_postprocess", seqs, token_ids)
        self.scheduler.postprocess(seqs, token_ids)

        for seq_id, scores in score_results:
            self._complete_score_record(seq_id, scores)

        outputs: list[tuple[int, object]] = []
        for seq in seqs:
            seq_id = int(getattr(seq, "seq_id"))
            if self._get_active_record(seq_id) is None or not bool(getattr(seq, "is_finished", False)):
                continue
            outputs.append((seq_id, getattr(seq, "raw_completion_token_ids", [])))
        return outputs, int(num_tokens)

    def _complete_raw_outputs(self, raw_outputs: object) -> None:
        for item in raw_outputs or ():
            raw_seq_id = item[0]
            token_ids = item[1] if len(item) > 1 else []
            seq_id = int(raw_seq_id)
            with self._active_lock:
                record = self._active_records.pop(seq_id, None)
            if record is None:
                continue
            try:
                visible_token_ids = getattr(record.seq, "completion_token_ids", None)
                tokens_source = visible_token_ids if visible_token_ids is not None else token_ids
                tokens = [int(token_id) for token_id in tokens_source]
                text = self._decode_token_ids(tokens)
                text, trimmed = _trim_at_stop_suffix(text, record.prompt_stop_suffixes)
                output = GenerationOutput(
                    prompt_index=record.prompt_index,
                    prompt=record.prompt,
                    token_ids=tokens,
                    text=text,
                    finish_reason=_finish_reason(record.seq, trimmed=trimmed),
                )
                if record.on_token is not None and output.text:
                    _safe_callback(
                        record.on_token,
                        record.prompt_index,
                        GeneratedTextDelta(text=output.text, tokens=list(output.tokens)),
                    )
                if record.on_complete is not None:
                    _safe_callback(record.on_complete, output)
                if not record.future.done():
                    record.future.set_result(output)
                self._record_completed_request(generated_tokens=len(tokens))
            except BaseException as exc:
                self._set_future_exception(record.future, exc)
                self._record_error_request()

    def _observe_cache_admissions(self) -> None:
        waiting_ids = _scheduler_seq_ids(self.scheduler, "waiting")
        observations: list[tuple[int, bool, bool]] = []
        with self._active_lock:
            for records in (self._active_records, self._active_score_records):
                for seq_id, record in records.items():
                    if record.cache_admission_observed or seq_id in waiting_ids:
                        continue
                    cache_hit = getattr(record.seq, "cache_hit_slot", None) is not None
                    exact_hit = bool(getattr(record.seq, "exact_cache_hit", False))
                    record.cache_admission_observed = True
                    observations.append((seq_id, cache_hit or exact_hit, exact_hit))
        if not observations:
            return
        with self._metrics_lock:
            for _seq_id, cache_hit, exact_hit in observations:
                self._metrics_totals["state_cache_admissions"] += 1
                if cache_hit:
                    self._metrics_totals["state_cache_hits"] += 1
                else:
                    self._metrics_totals["state_cache_misses"] += 1
                if exact_hit:
                    self._metrics_totals["state_cache_exact_hits"] += 1

    def _record_owner_step(self, *, num_tokens: object, elapsed_s: float) -> None:
        engine_step_tokens = _safe_int(num_tokens)
        with self._metrics_lock:
            self._metrics_totals["engine_step_count"] += 1
            self._metrics_totals["engine_step_tokens"] += int(engine_step_tokens)
            self._last_batch_metrics = {
                "engine_step_tokens": int(engine_step_tokens),
                "engine_step_count": int(self._metrics_totals["engine_step_count"]),
                "active_records": self._active_count(),
                "engine_inbox": self._inbox.qsize(),
                "elapsed_ms": round(float(elapsed_s) * 1000.0, 4),
                "engine_step_tok_s": round(float(engine_step_tokens) / max(float(elapsed_s), 1e-9), 4),
                "scheduler_snapshot": _scheduler_snapshot(self.scheduler),
            }

    def _record_completed_request(self, *, generated_tokens: int) -> None:
        with self._metrics_lock:
            self._metrics_totals["completed_requests"] += 1
            self._metrics_totals["generated_tokens"] += max(0, int(generated_tokens))

    def _record_error_request(self) -> None:
        with self._metrics_lock:
            self._metrics_totals["error_requests"] += 1

    def _record_cancelled_request(self) -> None:
        with self._metrics_lock:
            self._metrics_totals["cancelled_requests"] += 1

    def _active_count(self) -> int:
        with self._active_lock:
            return len(self._active_records) + len(self._active_score_records)

    def _active_score_count(self) -> int:
        with self._active_lock:
            return len(self._active_score_records)

    def _get_active_record(self, seq_id: int) -> _NanoActiveRecord | None:
        with self._active_lock:
            return self._active_records.get(int(seq_id))

    def _get_active_score_record(self, seq_id: int) -> _NanoActiveScoreRecord | None:
        with self._active_lock:
            return self._active_score_records.get(int(seq_id))

    def _fail_all_active(self, exc: BaseException) -> None:
        with self._active_lock:
            records = list(self._active_records.values())
            score_records = list(self._active_score_records.values())
            self._active_records.clear()
            self._active_score_records.clear()
        for record in records:
            self._set_future_exception(record.future, exc)
            self._record_error_request()
        for record in score_records:
            self._set_future_exception(record.future, exc)
            self._record_error_request()

    def _cancel_pending_inbox(self, exc: BaseException) -> None:
        while True:
            try:
                item = self._inbox.get_nowait()
            except Empty:
                return
            if item is None:
                continue
            self._set_future_exception(item.future, exc)
            self._record_cancelled_request()

    @staticmethod
    def _set_future_exception(future: Future[Any], exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)

    def _encode_text(self, text: str) -> list[int]:
        tokenizer = getattr(self.engine, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise RuntimeError("nano-vLLM engine tokenizer does not expose encode()")
        return [int(token_id) for token_id in encode(str(text))]

    def _single_token_id(self, text: str) -> int:
        token_ids = self._encode_text(str(text))
        if len(token_ids) != 1:
            raise ValueError(f"choice token text {text!r} must encode to exactly one token, got {token_ids}")
        return int(token_ids[0])

    def _to_sampling_params(self, sampling: SamplingConfig) -> object:
        top_k = int(sampling.top_k)
        if top_k <= 0:
            top_k = -1
        return self.runtime.sampling_params_cls(
            temperature=float(sampling.temperature),
            top_k=top_k,
            top_p=float(sampling.top_p),
            presence_penalty=float(sampling.alpha_presence),
            repetition_penalty=float(sampling.alpha_frequency),
            penalty_decay=float(sampling.alpha_decay),
            max_tokens=max(1, int(sampling.max_generate_tokens)),
            ignore_eos=not _contains_eos_token(self.engine, sampling.stop_tokens),
        )

    def _build_stop_token_seqs(
        self,
        *,
        sampling: SamplingConfig,
        stop_suffixes: Sequence[str] | None,
    ) -> tuple[tuple[int, ...], ...]:
        token_seqs: list[tuple[int, ...]] = [(int(token_id),) for token_id in sampling.stop_tokens]
        tokenizer = getattr(self.engine, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        if callable(encode) and stop_suffixes:
            for suffix in stop_suffixes:
                if not suffix:
                    continue
                token_ids = tuple(int(token_id) for token_id in encode(str(suffix)))
                if token_ids:
                    token_seqs.append(token_ids)
        return _dedupe_token_seqs(token_seqs)

    def _decode_token_ids(self, token_ids: Sequence[int]) -> str:
        tokenizer = getattr(self.engine, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise RuntimeError("nano-vLLM engine tokenizer does not expose decode()")
        tokens = [int(token_id) for token_id in token_ids]
        try:
            return str(decode(tokens, utf8_errors="replace"))
        except TypeError:
            pass
        try:
            return str(decode(tokens))
        except UnicodeDecodeError:
            decode_bytes = getattr(tokenizer, "decodeBytes", None)
            if callable(decode_bytes):
                return bytes(decode_bytes(tokens)).decode("utf-8", errors="replace")
            decode_bytes = getattr(tokenizer, "decode_bytes", None)
            if callable(decode_bytes):
                return bytes(decode_bytes(tokens)).decode("utf-8", errors="replace")
            raise

    def score_choice_tokens(
        self,
        *,
        prompt: str,
        choice_token_texts: Sequence[str],
    ) -> tuple[dict[str, float], str]:
        if self._stop.is_set():
            raise RuntimeError("nano-vLLM backend is shut down")
        choices = tuple(str(item) for item in choice_token_texts)
        if not choices:
            raise ValueError("choice_token_texts cannot be empty")
        future: Future[tuple[dict[str, float], str]] = Future()
        request = _NanoScoreRequest(
            prompt=str(prompt),
            choice_token_texts=choices,
            future=future,
            enqueued_at=time.monotonic_ns(),
        )
        with self._metrics_lock:
            self._metrics_totals["submitted_requests"] += 1
        if threading.current_thread() is self._owner:
            self._admit_score_request(request)
            while not future.done():
                step_started = time.perf_counter()
                raw_outputs, num_tokens = self._step_engine()
                elapsed_s = max(time.perf_counter() - step_started, 1e-9)
                self._record_owner_step(num_tokens=num_tokens, elapsed_s=elapsed_s)
                self._complete_raw_outputs(raw_outputs)
        else:
            self._inbox.put(request)
        return future.result()

    def _complete_score_record(self, seq_id: int, scores: dict[str, float]) -> None:
        with self._active_lock:
            record = self._active_score_records.pop(int(seq_id), None)
        if record is None:
            return
        try:
            abort = getattr(self.scheduler, "abort", None)
            if callable(abort):
                abort(int(seq_id))
            best_text = max(record.choice_token_texts, key=lambda item: scores[item])
            if not record.future.done():
                record.future.set_result((scores, best_text))
            self._record_completed_request(generated_tokens=0)
        except BaseException as exc:
            self._set_future_exception(record.future, exc)
            self._record_error_request()

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._inbox.put(None)
        self._owner.join(timeout=10.0)
        if self._owner.is_alive():
            shutdown_error = RuntimeError("nano-vLLM backend shutdown timed out")
            self._cancel_pending_inbox(shutdown_error)
            self._fail_all_active(shutdown_error)
            exit_method = getattr(self.engine, "exit", None)
            if callable(exit_method):
                exit_method()


def load_nano_vllm_runtime(nano_vllm_path: str | Path = DEFAULT_NANO_VLLM_PATH) -> _NanoVLLMRuntime:
    root = Path(nano_vllm_path).expanduser().resolve()
    package_dir = root / "nanovllm"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"nano-vLLM package not found at {package_dir}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    llm_module = importlib.import_module("nanovllm.engine.llm_engine")
    scheduler_module = importlib.import_module("nanovllm.engine.scheduler")
    model_runner_module = importlib.import_module("nanovllm.engine.model_runner")
    sampling_module = importlib.import_module("nanovllm.sampling_params")
    for module in (llm_module, scheduler_module, model_runner_module, sampling_module):
        _verify_module_under_root(module, root)

    return _NanoVLLMRuntime(
        root=root,
        llm_engine_cls=_required_type(llm_module, "LLMEngine"),
        scheduler_cls=_required_type(scheduler_module, "Scheduler"),
        model_runner_cls=_required_type(model_runner_module, "ModelRunner"),
        sampling_params_cls=_required_type(sampling_module, "SamplingParams"),
    )


def _required_type(module: ModuleType, name: str) -> type:
    value = getattr(module, name, None)
    if not isinstance(value, type):
        raise ImportError(f"{module.__name__} does not expose class {name}")
    return value


def _verify_module_under_root(module: ModuleType, root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return
    try:
        Path(str(module_file)).resolve().relative_to(root)
    except ValueError as exc:
        raise ImportError(f"{module.__name__} was imported from {module_file}, not {root}") from exc


def _has_active_constraints(
    *,
    constraints: Sequence[DecodeConstraint | None] | None,
    constraint_mode: str | None,
) -> bool:
    mode = str(constraint_mode or "off").strip().lower()
    if mode == "off" or constraints is None:
        return False
    return any(constraint is not None for constraint in constraints)


def _contains_eos_token(engine: object, stop_tokens: Sequence[int]) -> bool:
    tokenizer = getattr(engine, "tokenizer", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    try:
        eos_token_id = int(eos)
    except (TypeError, ValueError):
        return bool(stop_tokens)
    return any(int(token_id) == eos_token_id for token_id in stop_tokens)


def _safe_int(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except Exception:
        return 0


def _safe_len(value: object) -> int | None:
    if value is None:
        return None
    try:
        return len(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _safe_callback(callback: Callable[..., object], *args: object) -> None:
    try:
        callback(*args)
    except Exception:
        return


def _scheduler_count(scheduler: object, name: str) -> int:
    return _safe_len(getattr(scheduler, name, None)) or 0


def _scheduler_seq_ids(scheduler: object, name: str) -> set[int]:
    values = getattr(scheduler, name, None)
    if values is None:
        return set()
    try:
        return {int(getattr(seq, "seq_id")) for seq in list(values)}
    except Exception:
        return set()


def _scheduler_snapshot(scheduler: object) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in (
        "running",
        "running_seqs",
        "active_seqs",
        "waiting",
        "waiting_queue",
        "swapped",
        "seqs",
        "requests",
    ):
        if hasattr(scheduler, name):
            value = getattr(scheduler, name)
            size = _safe_len(value)
            snapshot[name] = size if size is not None else str(type(value).__name__)
    return snapshot


def _prompt_token_count(engine: object, prompt: str) -> int:
    tokenizer = getattr(engine, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(prompt))
        except Exception:
            pass
    return max(1, len(prompt))


def _scheduler_prefill_chunk_size(scheduler: object) -> int:
    config = getattr(scheduler, "config", None)
    try:
        return int(getattr(config, "rwkv_prefill_chunk_size"))
    except (AttributeError, TypeError, ValueError):
        return -1


def _seq_prefill_step_tokens(seq: object, chunk_size: int) -> int:
    method = getattr(seq, "prefill_step_tokens", None)
    if callable(method):
        return max(0, int(method(chunk_size)))
    try:
        remaining = int(getattr(seq, "num_prompt_tokens")) - int(getattr(seq, "num_cached_tokens", 0))
    except (TypeError, ValueError):
        return 0
    remaining = max(0, remaining)
    if int(chunk_size) == -1:
        return remaining
    return min(remaining, max(0, int(chunk_size)))


def _can_emit_sequence_token(scheduler: object, seq: object, is_prefill: bool) -> bool:
    if not is_prefill:
        return True
    chunk_size = _scheduler_prefill_chunk_size(scheduler)
    try:
        cached = int(getattr(seq, "num_cached_tokens", 0))
        prompt_tokens = int(getattr(seq, "num_prompt_tokens"))
    except (TypeError, ValueError):
        return True
    return cached + _seq_prefill_step_tokens(seq, chunk_size) >= prompt_tokens


def _score_choices_from_logits(logits_row: torch.Tensor, record: _NanoActiveScoreRecord) -> dict[str, float]:
    if logits_row.ndim != 1:
        logits_row = logits_row.reshape(-1)
    return {
        text: float(logits_row[int(token_id)].detach().to("cpu").item())
        for text, token_id in zip(record.choice_token_texts, record.choice_token_ids, strict=True)
    }


def _dedupe_token_seqs(token_seqs: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    deduped: list[tuple[int, ...]] = []
    for token_seq in token_seqs:
        normalized = tuple(int(token_id) for token_id in token_seq)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def _trim_at_stop_suffix(text: str, stop_suffixes: Sequence[str] | None) -> tuple[str, bool]:
    if not stop_suffixes:
        return text, False
    stop_at: int | None = None
    for suffix in stop_suffixes:
        if not suffix:
            continue
        index = text.find(str(suffix))
        if index < 0:
            continue
        stop_at = index if stop_at is None else min(stop_at, index)
    if stop_at is None:
        return text, False
    return text[:stop_at], True


def _finish_reason(seq: object, *, trimmed: bool) -> str:
    if trimmed:
        return "stop_token"
    try:
        if int(getattr(seq, "num_raw_completion_tokens")) >= int(getattr(seq, "max_tokens")):
            return "max_length"
    except (TypeError, ValueError):
        pass
    return "stop_token"


__all__ = [
    "DEFAULT_NANO_VLLM_PATH",
    "NanoVLLMBackendConfig",
    "NanoVLLMInferenceBackend",
    "load_nano_vllm_runtime",
]
