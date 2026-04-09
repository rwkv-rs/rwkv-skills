from __future__ import annotations

"""Batching infer service built on top of the local RWKV backend."""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
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
from .sampling import GeneratedTextDelta, GeneratedToken


@dataclass(slots=True)
class _PendingRequest:
    request: CompletionRequest
    response_id: str
    created: int
    future: Future[CompletionResponse]
    stream_queue: Queue[GeneratedTextDelta | None] | None = None


@dataclass(slots=True, frozen=True)
class CompletionStreamHandle:
    response_id: str
    created: int
    future: Future[CompletionResponse]
    token_queue: Queue[GeneratedTextDelta | None]


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
        )
        self._queue.put(pending)
        return pending

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=5.0)
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

    def _execute_generation_batch(self, batch: Sequence[_PendingRequest]) -> None:
        if not batch:
            return
        request0 = batch[0].request
        sampling = request0.to_sampling_config()
        prompts = [item.request.prompt for item in batch]
        seeds = [item.request.seed for item in batch]
        stop_suffixes = [_request_stop_suffixes(item.request) for item in batch]
        max_top_logprobs = max(max(int(item.request.logprobs or 0), 0) for item in batch)

        def _on_token(prompt_index: int, delta: GeneratedTextDelta) -> None:
            item = batch[prompt_index]
            if item.stream_queue is not None:
                item.stream_queue.put(delta)

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
                top_logprobs=max_top_logprobs,
                show_progress=False,
            )
            for item, output in zip(batch, outputs, strict=True):
                item.future.set_result(self._build_generation_response(item, output))
                if item.stream_queue is not None:
                    item.stream_queue.put(None)
        except BaseException as exc:
            for item in batch:
                if item.stream_queue is not None:
                    item.stream_queue.put(None)
                item.future.set_exception(exc)

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
