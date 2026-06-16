from __future__ import annotations

"""Batching infer service built on top of the local RWKV backend."""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import json
from queue import Empty, Queue
import threading
import time
from typing import Sequence

import torch

from .api import (
    ChoiceScore,
    CompletionChoice,
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    current_unix_seconds,
    next_completion_id,
)
from .backend import InferenceBackend
from .sampling import GeneratedTextDelta, GeneratedToken, GenerationOutput


@dataclass(slots=True)
class _PendingRequest:
    request: CompletionRequest
    response_id: str
    created: int
    future: Future[CompletionResponse]
    enqueued_at: int
    stream_queue: Queue[GeneratedTextDelta | None] | None = None


@dataclass(slots=True, frozen=True)
class CompletionStreamHandle:
    response_id: str
    created: int
    future: Future[CompletionResponse]
    token_queue: Queue[GeneratedTextDelta | None]


@dataclass(slots=True)
class _BatchMetricsSnapshot:
    timestamp_ms: int
    batch_size: int
    prefill_tokens: int
    generated_tokens: int
    queue_wait_ms: float
    elapsed_ms: float
    output_tok_s: float
    total_tok_s: float
    gpu_memory_used_mb: int | None
    gpu_memory_total_mb: int | None
    backend_stats: dict[str, object]
    success: bool
    error: str | None = None


class InferenceService:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        max_batch_size: int = 16,
        batch_collect_ms: int = 5,
    ) -> None:
        self.backend = backend
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_collect_ms = max(0, int(batch_collect_ms))
        submit = getattr(backend, "submit", None)
        self._backend_submit = submit if callable(submit) else None
        self._metrics_lock = threading.Lock()
        self._batch_metrics: deque[_BatchMetricsSnapshot] = deque(maxlen=256)
        self._batch_totals = {
            "total_batches": 0,
            "total_requests": 0,
            "completed_requests": 0,
            "error_requests": 0,
            "total_prefill_tokens": 0,
            "total_generated_tokens": 0,
            "total_queue_wait_ms": 0.0,
            "total_elapsed_ms": 0.0,
            "failed_batches": 0,
            "last_output_tok_s": 0.0,
            "last_total_tok_s": 0.0,
            "last_gpu_memory_used_mb": None,
            "last_gpu_memory_total_mb": None,
        }
        self._queue: Queue[_PendingRequest | None] = Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, name="rwkv-infer-worker", daemon=True)
        self._worker.start()

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    def submit_completion(self, request: CompletionRequest) -> Future[CompletionResponse]:
        return self._submit_pending(request).future

    def submit_streaming_completion(self, request: CompletionRequest) -> CompletionStreamHandle:
        token_queue: Queue[GeneratedTextDelta | None] = Queue()
        pending = self._submit_pending(request, stream_queue=token_queue)
        return CompletionStreamHandle(
            response_id=pending.response_id,
            created=pending.created,
            future=pending.future,
            token_queue=token_queue,
        )

    def _submit_pending(
        self,
        request: CompletionRequest,
        *,
        stream_queue: Queue[GeneratedTextDelta | None] | None = None,
    ) -> _PendingRequest:
        future: Future[CompletionResponse] = Future()
        pending = _PendingRequest(
            request=request,
            response_id=next_completion_id(),
            created=current_unix_seconds(),
            future=future,
            stream_queue=stream_queue,
            enqueued_at=time.monotonic_ns(),
        )
        self._queue.put(pending)
        return pending

    def batch_metrics(self) -> dict[str, object]:
        with self._metrics_lock:
            recent = list(self._batch_metrics)
            totals = dict(self._batch_totals)
        backend_stats = _backend_batch_stats(self.backend)
        pending = _pending_snapshot(
            service_queue=self._queue.qsize(),
            backend_stats=backend_stats,
        )
        avg_batch_size = (
            totals["total_requests"] / totals["total_batches"] if totals["total_batches"] else 0.0
        )
        total_elapsed_s = float(totals["total_elapsed_ms"]) / 1000.0
        avg_output_tok_s = (
            float(totals["total_generated_tokens"]) / total_elapsed_s if total_elapsed_s > 0 else 0.0
        )
        avg_total_tok_s = (
            float(totals["total_prefill_tokens"] + totals["total_generated_tokens"]) / total_elapsed_s
            if total_elapsed_s > 0
            else 0.0
        )
        return {
            "model_name": self.model_name,
            "max_batch_size": self.max_batch_size,
            "batch_collect_ms": self.batch_collect_ms,
            "totals": {
                "total_batches": totals["total_batches"],
                "total_requests": totals["total_requests"],
                "completed_requests": totals["completed_requests"],
                "error_requests": totals["error_requests"],
                "total_prefill_tokens": totals["total_prefill_tokens"],
                "total_generated_tokens": totals["total_generated_tokens"],
                "total_queue_wait_ms": totals["total_queue_wait_ms"],
                "total_elapsed_ms": totals["total_elapsed_ms"],
                "failed_batches": totals["failed_batches"],
                "avg_batch_size": round(avg_batch_size, 4),
                "avg_output_tok_s": round(avg_output_tok_s, 4),
                "avg_total_tok_s": round(avg_total_tok_s, 4),
                "last_output_tok_s": totals["last_output_tok_s"],
                "last_total_tok_s": totals["last_total_tok_s"],
                "last_gpu_memory_used_mb": totals["last_gpu_memory_used_mb"],
                "last_gpu_memory_total_mb": totals["last_gpu_memory_total_mb"],
            },
            "recent_batches": [
                {
                    "timestamp_ms": snapshot.timestamp_ms,
                    "batch_size": snapshot.batch_size,
                    "prefill_tokens": snapshot.prefill_tokens,
                    "generated_tokens": snapshot.generated_tokens,
                    "queue_wait_ms": snapshot.queue_wait_ms,
                    "elapsed_ms": snapshot.elapsed_ms,
                    "output_tok_s": snapshot.output_tok_s,
                    "total_tok_s": snapshot.total_tok_s,
                    "gpu_memory_used_mb": snapshot.gpu_memory_used_mb,
                    "gpu_memory_total_mb": snapshot.gpu_memory_total_mb,
                    "backend_stats": snapshot.backend_stats,
                    "success": snapshot.success,
                    "error": snapshot.error,
                }
                for snapshot in recent
            ],
            "pending": pending,
            "backend_live": backend_stats,
        }

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=5.0)
        self._cancel_service_queue(RuntimeError("inference service shut down"))
        shutdown = getattr(self.backend, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _run_worker(self) -> None:
        pending: deque[_PendingRequest] = deque()
        while True:
            if not pending:
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    if self._stop.is_set():
                        return
                    continue
                if item is None:
                    if self._stop.is_set():
                        return
                    continue
                pending.append(item)
                self._collect_pending(pending)
            else:
                self._collect_pending(pending, block=False)

            if not pending:
                if self._stop.is_set():
                    return
                continue

            score_index = next(
                (index for index, item in enumerate(pending) if item.request.is_choice_scoring_request()),
                None,
            )
            if score_index is not None:
                item = pending[score_index]
                del pending[score_index]
                self._execute_score(item)
                continue

            if self._backend_submit is not None:
                batch = self._take_submit_generation_batch(pending)
                for item in batch:
                    self._execute_submit_generation(item)
                continue

            batch = self._take_generation_batch(pending)
            self._execute_generation_batch(batch)

    def _collect_pending(self, pending: deque[_PendingRequest], *, block: bool = True) -> None:
        timeout_s = self.batch_collect_ms / 1000.0
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if not block and remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=max(remaining, 0.0) if block else 0.0)
            except Empty:
                break
            if item is None:
                if self._stop.is_set():
                    break
                continue
            pending.append(item)
            block = False
            if len(pending) >= self.max_batch_size * 4:
                break

    def _take_generation_batch(self, pending: deque[_PendingRequest]) -> list[_PendingRequest]:
        first = pending.popleft()
        key = first.request.generation_batch_key()
        batch = [first]
        remainder: deque[_PendingRequest] = deque()
        while pending:
            item = pending.popleft()
            if len(batch) < self.max_batch_size and item.request.generation_batch_key() == key:
                batch.append(item)
            else:
                remainder.append(item)
        pending.extend(remainder)
        return batch

    def _take_submit_generation_batch(self, pending: deque[_PendingRequest]) -> list[_PendingRequest]:
        batch: list[_PendingRequest] = []
        remainder: deque[_PendingRequest] = deque()
        while pending:
            item = pending.popleft()
            if len(batch) < self.max_batch_size and not item.request.is_choice_scoring_request():
                batch.append(item)
            else:
                remainder.append(item)
        pending.extend(remainder)
        return batch

    def _execute_submit_generation(self, item: _PendingRequest) -> None:
        submit = self._backend_submit
        if submit is None:
            self._execute_generation_batch([item])
            return
        request = item.request
        sampling = request.to_sampling_config()
        stop_suffixes = _request_stop_suffixes(request)
        prefill_tokens = self._prompt_token_count(request.prompt)
        queue_wait_ms = (time.monotonic_ns() - item.enqueued_at) / 1e6
        start = time.perf_counter()

        def _on_token(_prompt_index: int, delta: GeneratedTextDelta) -> None:
            if item.stream_queue is not None:
                item.stream_queue.put(delta)

        def _finish_with_exception(exc: BaseException) -> None:
            if item.stream_queue is not None:
                item.stream_queue.put(None)
            if not item.future.done():
                item.future.set_exception(exc)

        def _on_done(done_future: Future[GenerationOutput]) -> None:
            generated_tokens = 0
            success = True
            error_text: str | None = None
            try:
                output = done_future.result()
                generated_tokens = len(getattr(output, "token_ids", []))
                if not item.future.done():
                    item.future.set_result(self._build_generation_response(item, output))
            except BaseException as exc:
                success = False
                error_text = str(exc)
                _finish_with_exception(exc)
            else:
                if item.stream_queue is not None:
                    item.stream_queue.put(None)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                backend_stats = _backend_batch_stats(self.backend)
                gpu_memory_used_mb, gpu_memory_total_mb = _gpu_memory_mb()
                self._record_batch_metrics(
                    batch_size=1,
                    prefill_tokens=prefill_tokens,
                    generated_tokens=generated_tokens,
                    queue_wait_ms=queue_wait_ms,
                    elapsed_ms=elapsed_ms,
                    backend_stats=backend_stats,
                    gpu_memory_used_mb=gpu_memory_used_mb,
                    gpu_memory_total_mb=gpu_memory_total_mb,
                    success=success,
                    error=error_text,
                )

        try:
            backend_future = submit(
                request.prompt,
                sampling=sampling,
                prompt_index=0,
                prompt_stop_suffixes=stop_suffixes,
                prompt_seed=request.seed,
                top_logprobs=max(int(request.logprobs or 0), 0),
                prefill_chunk_size=request.effective_prefill_chunk_size(),
                on_token=_on_token if item.stream_queue is not None else None,
            )
        except BaseException as exc:
            _finish_with_exception(exc)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record_batch_metrics(
                batch_size=1,
                prefill_tokens=prefill_tokens,
                generated_tokens=0,
                queue_wait_ms=queue_wait_ms,
                elapsed_ms=elapsed_ms,
                backend_stats=_backend_batch_stats(self.backend),
                gpu_memory_used_mb=None,
                gpu_memory_total_mb=None,
                success=False,
                error=str(exc),
            )
            return

        backend_future.add_done_callback(_safe_submit_done_callback(_on_done))

    def _execute_generation_batch(self, batch: Sequence[_PendingRequest]) -> None:
        if not batch:
            return
        request0 = batch[0].request
        sampling = request0.to_sampling_config()
        prompts = [item.request.prompt for item in batch]
        seeds = [item.request.seed for item in batch]
        stop_suffixes = [_request_stop_suffixes(item.request) for item in batch]
        max_top_logprobs = max(max(int(item.request.logprobs or 0), 0) for item in batch)
        prefill_tokens = [self._prompt_token_count(item.request.prompt) for item in batch]
        total_prefill_tokens = int(sum(prefill_tokens))
        queue_wait_ms = (time.monotonic_ns() - min(item.enqueued_at for item in batch)) / 1e6
        completed_outputs: dict[int, GenerationOutput] = {}
        completed_lock = threading.Lock()

        def _on_token(prompt_index: int, delta: GeneratedTextDelta) -> None:
            item = batch[prompt_index]
            if item.stream_queue is not None:
                item.stream_queue.put(delta)

        def _complete_output(prompt_index: int, output: GenerationOutput) -> bool:
            if prompt_index < 0 or prompt_index >= len(batch):
                return False
            with completed_lock:
                if prompt_index in completed_outputs:
                    return False
                completed_outputs[prompt_index] = output
            item = batch[prompt_index]
            if not item.future.done():
                item.future.set_result(self._build_generation_response(item, output))
            if item.stream_queue is not None:
                item.stream_queue.put(None)
            return True

        def _on_complete(output: GenerationOutput) -> None:
            try:
                prompt_index = int(output.prompt_index)
            except (TypeError, ValueError):
                return
            _complete_output(prompt_index, output)

        start = time.perf_counter()
        generated_tokens = 0
        success = True
        error_text: str | None = None
        try:
            outputs = self.backend.generate(
                prompts,
                sampling=sampling,
                batch_size=len(prompts),
                progress_desc="Infer",
                prompt_stop_suffixes=stop_suffixes,
                prompt_seeds=seeds,
                prefill_chunk_size=request0.effective_prefill_chunk_size(),
                on_token=_on_token if any(item.stream_queue is not None for item in batch) else None,
                on_complete=_on_complete,
                top_logprobs=max_top_logprobs,
                show_progress=False,
            )
            for fallback_index, output in enumerate(outputs):
                try:
                    prompt_index = int(output.prompt_index)
                except (TypeError, ValueError):
                    prompt_index = fallback_index
                if not (0 <= prompt_index < len(batch)):
                    prompt_index = fallback_index
                _complete_output(prompt_index, output)
        except BaseException as exc:
            success = False
            error_text = str(exc)
            for item in batch:
                if item.stream_queue is not None:
                    item.stream_queue.put(None)
                if not item.future.done():
                    item.future.set_exception(exc)
        finally:
            with completed_lock:
                generated_tokens = sum(
                    len(getattr(output, "token_ids", [])) for output in completed_outputs.values()
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            backend_stats = _backend_batch_stats(self.backend)
            gpu_memory_used_mb, gpu_memory_total_mb = _gpu_memory_mb()
            self._record_batch_metrics(
                batch_size=len(batch),
                prefill_tokens=total_prefill_tokens,
                generated_tokens=generated_tokens,
                queue_wait_ms=queue_wait_ms,
                elapsed_ms=elapsed_ms,
                backend_stats=backend_stats,
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_memory_total_mb=gpu_memory_total_mb,
                success=success,
                error=error_text,
            )

    def _execute_score(self, item: _PendingRequest) -> None:
        request = item.request
        assert request.candidate_token_texts is not None
        try:
            raw_scores, best_text = self.backend.score_choice_tokens(
                prompt=request.prompt,
                choice_token_texts=request.candidate_token_texts,
            )
            scored = _normalize_choice_scores(raw_scores)
            top_map = {entry.text: entry.logprob for entry in scored}
            pred_logprob = top_map.get(best_text)
            item.future.set_result(
                CompletionResponse(
                    id=item.response_id,
                    created=item.created,
                    model=self.model_name,
                    choices=[
                        CompletionChoice(
                            text=best_text,
                            index=0,
                            finish_reason="logprobs",
                            logprobs=CompletionLogprobs(
                                tokens=[best_text],
                                token_logprobs=[pred_logprob],
                                top_logprobs=[top_map],
                                text_offset=[len(request.prompt)],
                            ),
                        )
                    ],
                )
            )
        except BaseException as exc:
            item.future.set_exception(exc)

    def _build_generation_response(
        self,
        item: _PendingRequest,
        output,
    ) -> CompletionResponse:
        request = item.request
        logprobs = None
        requested_top = max(int(request.logprobs or 0), 0)
        if requested_top > 0 and output.tokens:
            logprobs = _build_completion_logprobs(output.tokens, requested_top)
        return CompletionResponse(
            id=item.response_id,
            created=item.created,
            model=self.model_name,
            choices=[
                CompletionChoice(
                    text=output.text,
                    index=0,
                    finish_reason=output.finish_reason,
                    logprobs=logprobs,
                )
            ],
        )

    def _prompt_token_count(self, prompt: str) -> int:
        tokenizer = getattr(self.backend, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                token_ids = encode(prompt)
                return len(token_ids)
            except Exception:
                pass
        return max(1, len(prompt))

    def _record_batch_metrics(
        self,
        *,
        batch_size: int,
        prefill_tokens: int,
        generated_tokens: int,
        queue_wait_ms: float,
        elapsed_ms: float,
        backend_stats: dict[str, object],
        gpu_memory_used_mb: int | None,
        gpu_memory_total_mb: int | None,
        success: bool,
        error: str | None,
    ) -> None:
        elapsed_s = max(float(elapsed_ms) / 1000.0, 1e-9)
        output_tok_s = max(0.0, float(generated_tokens)) / elapsed_s
        total_tok_s = max(0.0, float(prefill_tokens) + float(generated_tokens)) / elapsed_s
        snapshot = _BatchMetricsSnapshot(
            timestamp_ms=int(time.time() * 1000),
            batch_size=max(0, int(batch_size)),
            prefill_tokens=max(0, int(prefill_tokens)),
            generated_tokens=max(0, int(generated_tokens)),
            queue_wait_ms=float(queue_wait_ms),
            elapsed_ms=float(elapsed_ms),
            output_tok_s=float(output_tok_s),
            total_tok_s=float(total_tok_s),
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            backend_stats=dict(backend_stats),
            success=bool(success),
            error=error,
        )
        with self._metrics_lock:
            self._batch_metrics.append(snapshot)
            self._batch_totals["total_batches"] += 1
            self._batch_totals["total_requests"] += max(0, int(batch_size))
            if success:
                self._batch_totals["completed_requests"] += max(0, int(batch_size))
            else:
                self._batch_totals["error_requests"] += max(0, int(batch_size))
            self._batch_totals["total_prefill_tokens"] += max(0, int(prefill_tokens))
            self._batch_totals["total_generated_tokens"] += max(0, int(generated_tokens))
            self._batch_totals["total_queue_wait_ms"] += float(queue_wait_ms)
            self._batch_totals["total_elapsed_ms"] += float(elapsed_ms)
            self._batch_totals["last_output_tok_s"] = float(output_tok_s)
            self._batch_totals["last_total_tok_s"] = float(total_tok_s)
            self._batch_totals["last_gpu_memory_used_mb"] = gpu_memory_used_mb
            self._batch_totals["last_gpu_memory_total_mb"] = gpu_memory_total_mb
            if not success:
                self._batch_totals["failed_batches"] += 1
        print(
            "RWKV_INFER_BATCH_METRICS "
            + json.dumps(
                {
                    "model_name": self.model_name,
                    "timestamp_ms": snapshot.timestamp_ms,
                    "batch_size": snapshot.batch_size,
                    "prefill_tokens": snapshot.prefill_tokens,
                    "generated_tokens": snapshot.generated_tokens,
                    "queue_wait_ms": round(snapshot.queue_wait_ms, 4),
                    "elapsed_ms": round(snapshot.elapsed_ms, 4),
                    "output_tok_s": round(snapshot.output_tok_s, 4),
                    "total_tok_s": round(snapshot.total_tok_s, 4),
                    "gpu_memory_used_mb": snapshot.gpu_memory_used_mb,
                    "gpu_memory_total_mb": snapshot.gpu_memory_total_mb,
                    "backend_stats": snapshot.backend_stats,
                    "success": snapshot.success,
                    "error": snapshot.error,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    def _cancel_service_queue(self, exc: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                return
            if item is None:
                continue
            if item.stream_queue is not None:
                item.stream_queue.put(None)
            if not item.future.done():
                item.future.set_exception(exc)


def _backend_batch_stats(backend: InferenceBackend) -> dict[str, object]:
    getter = getattr(backend, "last_batch_metrics", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, dict):
                return dict(value)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return {}


def _pending_snapshot(*, service_queue: int, backend_stats: dict[str, object]) -> dict[str, object]:
    engine_inbox = _optional_int(backend_stats.get("engine_inbox")) or 0
    active_records = _optional_int(backend_stats.get("active_records")) or 0
    scheduler_waiting = _optional_int(backend_stats.get("scheduler_waiting")) or 0
    scheduler_running = _optional_int(backend_stats.get("scheduler_running")) or 0
    service_queue = max(0, int(service_queue))
    return {
        "pending_queue": service_queue + engine_inbox + active_records,
        "service_queue": service_queue,
        "engine_inbox": engine_inbox,
        "active_records": active_records,
        "scheduler_waiting": scheduler_waiting,
        "scheduler_running": scheduler_running,
    }


def _safe_submit_done_callback(callback):
    def _wrapped(future: Future[GenerationOutput]) -> None:
        try:
            callback(future)
        except Exception:
            return

    return _wrapped


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gpu_memory_mb() -> tuple[int | None, int | None]:
    try:
        if not torch.cuda.is_available():
            return None, None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_bytes = int(total_bytes) - int(free_bytes)
        return used_bytes // (1024 * 1024), int(total_bytes) // (1024 * 1024)
    except Exception:
        return None, None


def _normalize_choice_scores(scores: dict[str, float]) -> list[ChoiceScore]:
    items = list(scores.items())
    if not items:
        raise ValueError("choice scores cannot be empty")
    logits = torch.tensor([value for _key, value in items], dtype=torch.float32)
    logprobs = torch.log_softmax(logits, dim=0)
    return [
        ChoiceScore(text=text, logprob=float(logprob))
        for (text, _value), logprob in zip(items, logprobs.tolist(), strict=True)
    ]


def _request_stop_suffixes(request: CompletionRequest) -> tuple[str, ...]:
    stop = request.stop
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,) if stop else ()
    return tuple(item for item in stop if item)


def _build_completion_logprobs(
    tokens: Sequence[GeneratedToken],
    requested_top_logprobs: int,
) -> CompletionLogprobs:
    top_limit = max(int(requested_top_logprobs), 0)
    text_offset: list[int] = []
    current_offset = 0
    token_texts: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float]] = []
    for token in tokens:
        token_texts.append(token.text)
        token_logprobs.append(token.logprob)
        text_offset.append(current_offset)
        current_offset += len(token.text)
        top_logprobs.append(
            {
                candidate.text: float(candidate.logprob)
                for candidate in token.top_logprobs[:top_limit]
            }
        )
    return CompletionLogprobs(
        tokens=token_texts,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=text_offset,
    )

__all__ = ["CompletionStreamHandle", "InferenceService"]
