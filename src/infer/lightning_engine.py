from __future__ import annotations

"""Engine-selection utilities for the staged RWKV lightning migration."""

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Callable, Literal, Protocol, Sequence

import torch
from tqdm import tqdm

from .constraints import ConstraintRuntime, DecodeConstraint, build_token_constraint_cache
from .engine import (
    DEFAULT_PREFILL_CHUNK_SIZE,
    InferenceEngine,
    TokenizerProtocol,
    _build_generated_token,
    _decode_tokens,
    _finish_generated_text,
    _infer_device,
    _infer_vocab_size,
    _normalize_stop_suffixes,
    _prepare_state_container,
    _push_generated_token_text,
    _record_generated_text_delta,
    _token_bytes,
)
from .rapid_sampling_loader import get_rapid_sampling_module
from .sampling import GeneratedTextDelta, GeneratedToken, GenerationOutput, SamplingConfig
from .state_pool import DEFAULT_PREFIX_CACHE_BUCKETS, StateCacheManager, StatePoolConfig

LocalEngineMode = Literal["classic", "lightning"]


class LocalEngineProtocol(Protocol):
    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling: SamplingConfig,
        batch_size: int,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        progress_desc: str = "Generating",
        probe_only: bool = False,
        on_complete=None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None = None,
        prompt_stop_suffixes: Sequence[Sequence[str] | None] | None = None,
        prompt_constraints: Sequence[DecodeConstraint | None] | None = None,
        prompt_seeds: Sequence[int | None] | None = None,
        top_logprobs: int = 0,
        show_progress: bool = True,
    ) -> list[GenerationOutput]:
        ...

    def shutdown(self) -> None:
        ...


@dataclass(slots=True, frozen=True)
class LightningEngineConfig:
    state_db_path: str = "rwkv_sessions.db"
    state_l1_capacity: int = 16
    state_l2_capacity: int = 64
    prefix_cache_buckets: tuple[int, ...] = DEFAULT_PREFIX_CACHE_BUCKETS
    prefix_bucket_capacity: int = 16


@dataclass(slots=True)
class _LightningTask:
    prompt_index: int
    prompt: str
    prompt_tokens: tuple[int, ...]
    pending_tokens: deque[int]
    prompt_tokens_remaining: int
    processed_prompt_tokens: int
    generated_token_count: int
    generated_tokens: list[int]
    generated_events: list[GeneratedToken]
    emitted_text_parts: list[str]
    utf8_tokens_in_buffer: list[GeneratedToken]
    pending_stop_tokens: list[GeneratedToken]
    stop_suffixes: tuple[tuple[str, bytes], ...]
    max_stop_suffix_len: int
    pending_token: GeneratedToken | None
    ready_logits: torch.Tensor | None
    constraint_runtime: ConstraintRuntime | None
    finish_reason: str | None


class ClassicInferenceEngineAdapter:
    def __init__(self, model: object, tokenizer: TokenizerProtocol) -> None:
        self._delegate = InferenceEngine(model, tokenizer)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling: SamplingConfig,
        batch_size: int,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        progress_desc: str = "Generating",
        probe_only: bool = False,
        on_complete=None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None = None,
        prompt_stop_suffixes: Sequence[Sequence[str] | None] | None = None,
        prompt_constraints: Sequence[DecodeConstraint | None] | None = None,
        prompt_seeds: Sequence[int | None] | None = None,
        top_logprobs: int = 0,
        show_progress: bool = True,
    ) -> list[GenerationOutput]:
        return self._delegate.generate(
            prompts,
            sampling=sampling,
            batch_size=batch_size,
            prefill_chunk_size=prefill_chunk_size,
            progress_desc=progress_desc,
            probe_only=probe_only,
            on_complete=on_complete,
            on_token=on_token,
            prompt_stop_suffixes=prompt_stop_suffixes,
            prompt_constraints=prompt_constraints,
            prompt_seeds=prompt_seeds,
            top_logprobs=top_logprobs,
            show_progress=show_progress,
        )

    def shutdown(self) -> None:
        return None


class LightningInferenceEngineAdapter:
    def __init__(
        self,
        model: object,
        tokenizer: TokenizerProtocol,
        *,
        config: LightningEngineConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        effective = config or LightningEngineConfig()
        self.config = effective
        self.state_cache = StateCacheManager(
            StatePoolConfig(
                l1_capacity=effective.state_l1_capacity,
                l2_capacity=effective.state_l2_capacity,
                db_path=effective.state_db_path,
                prefix_cache_buckets=effective.prefix_cache_buckets,
                prefix_bucket_capacity=effective.prefix_bucket_capacity,
            )
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling: SamplingConfig,
        batch_size: int,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        progress_desc: str = "Generating",
        probe_only: bool = False,
        on_complete=None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None = None,
        prompt_stop_suffixes: Sequence[Sequence[str] | None] | None = None,
        prompt_constraints: Sequence[DecodeConstraint | None] | None = None,
        prompt_seeds: Sequence[int | None] | None = None,
        top_logprobs: int = 0,
        show_progress: bool = True,
    ) -> list[GenerationOutput]:
        return _lightning_continuous_batching(
            self.model,
            self.tokenizer,
            self.state_cache,
            prompts,
            sampling,
            batch_size,
            prefill_chunk_size,
            progress_desc,
            probe_only,
            on_complete,
            on_token,
            prompt_stop_suffixes,
            prompt_constraints,
            prompt_seeds,
            top_logprobs,
            show_progress,
            self.config.prefix_cache_buckets,
        )

    def shutdown(self) -> None:
        self.state_cache.close()


def build_local_engine(
    model: object,
    tokenizer: TokenizerProtocol,
    *,
    mode: LocalEngineMode = "classic",
    state_db_path: str | None = None,
) -> LocalEngineProtocol:
    if mode == "classic":
        return ClassicInferenceEngineAdapter(model, tokenizer)
    if mode == "lightning":
        config = LightningEngineConfig(
            state_db_path=state_db_path or LightningEngineConfig.state_db_path,
        )
        return LightningInferenceEngineAdapter(model, tokenizer, config=config)
    raise ValueError(f"unknown local engine mode: {mode!r}")


__all__ = [
    "ClassicInferenceEngineAdapter",
    "LightningEngineConfig",
    "LightningInferenceEngineAdapter",
    "LocalEngineMode",
    "LocalEngineProtocol",
    "build_local_engine",
]


def _lightning_continuous_batching(
    model: object,
    tokenizer: TokenizerProtocol,
    state_cache: StateCacheManager,
    prompts: Sequence[str],
    sampling: SamplingConfig,
    batch_size: int,
    prefill_chunk_size: int,
    progress_desc: str,
    probe_only: bool,
    on_complete: Callable[[GenerationOutput], None] | None,
    on_token: Callable[[int, GeneratedTextDelta], None] | None,
    prompt_stop_suffixes: Sequence[Sequence[str] | None] | None,
    prompt_constraints: Sequence[DecodeConstraint | None] | None,
    prompt_seeds: Sequence[int | None] | None,
    top_logprobs: int,
    show_progress: bool,
    prefix_cache_buckets: Sequence[int],
) -> list[GenerationOutput]:
    if not prompts:
        return []
    if prompt_seeds is not None and len(prompt_seeds) != len(prompts):
        raise ValueError("prompt_seeds 长度必须与 prompts 一致")
    if prompt_stop_suffixes is not None and len(prompt_stop_suffixes) != len(prompts):
        raise ValueError("prompt_stop_suffixes 长度必须与 prompts 一致")
    if prompt_constraints is not None and len(prompt_constraints) != len(prompts):
        raise ValueError("prompt_constraints 长度必须与 prompts 一致")

    batch_size = max(1, min(int(batch_size), len(prompts)))
    prefill_chunk_size = max(1, int(prefill_chunk_size))
    top_logprobs = max(0, int(top_logprobs))

    vocab_size = _infer_vocab_size(model)  # type: ignore[arg-type]
    device = _infer_device(model)  # type: ignore[arg-type]
    states = _prepare_state_container(model.generate_zero_state(batch_size))  # type: ignore[attr-defined]
    sampling = sampling.checked(vocab_size)
    prefix_buckets = tuple(sorted({int(bucket) for bucket in prefix_cache_buckets if int(bucket) > 0}))

    rapid_sampler = get_rapid_sampling_module()
    sampler_states = rapid_sampler.setup_rand(int(torch.initial_seed() & 0x7FFFFFFFFFFFFFFF), batch_size)
    if not isinstance(sampler_states, torch.Tensor) or sampler_states.ndim != 1:
        raise RuntimeError("rapid-sampling setup_rand 返回格式异常，期望 1D Tensor")
    if sampler_states.numel() % batch_size != 0:
        raise RuntimeError("rapid-sampling 随机状态长度与 batch_size 不匹配")
    sampler_state_row_width = sampler_states.numel() // batch_size

    stop_tokens = set(sampling.stop_tokens)
    ban_token_ids = [token_id for token_id in tuple(sampling.ban_tokens or ()) if 0 <= token_id < vocab_size]
    valid_no_penalty_ids = sorted(token_id for token_id in set(sampling.no_penalty_token_ids) if 0 <= token_id < vocab_size)
    no_penalty_ids = (
        torch.tensor(valid_no_penalty_ids, dtype=torch.int64, device=device)
        if valid_no_penalty_ids
        else None
    )
    penalties = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)

    sampler_temperature = float(sampling.temperature or 1.0)
    sampler_top_k = int(sampling.top_k) if sampling.top_k is not None else -1
    sampler_top_p = float(sampling.top_p) if sampling.top_p is not None else 1.0
    presence_penalty_value = float(sampling.presence_penalty)
    repetition_penalty_value = float(sampling.repetition_penalty)
    penalty_decay_value = float(sampling.penalty_decay)

    encoded = deque()
    for idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        if sampling.pad_zero:
            tokens = [0] + tokens
        seed = (
            None
            if prompt_seeds is None or prompt_seeds[idx] is None
            else int(prompt_seeds[idx])
        )
        stop_suffixes = None if prompt_stop_suffixes is None else prompt_stop_suffixes[idx]
        normalized_stop_suffixes, max_stop_suffix_len = _normalize_stop_suffixes(stop_suffixes)
        encoded.append((idx, prompt, tokens, seed, normalized_stop_suffixes, max_stop_suffix_len))

    constraint_cache = None
    if prompt_constraints is not None and any(constraint is not None for constraint in prompt_constraints):
        constraint_cache = build_token_constraint_cache(tokenizer, vocab_size=vocab_size)

    active_tasks: list[_LightningTask] = []

    pbar = tqdm(
        total=len(prompts),
        desc=progress_desc,
        unit=" sequence",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
        disable=not show_progress,
    )
    outputs: list[GenerationOutput] = []
    start_time = time.time()
    tokens_processed = 0
    window_start_time = start_time
    window_start_tokens = 0
    throughput_ema: float | None = None

    def _sampler_states_view(active_count: int) -> torch.Tensor:
        byte_len = active_count * sampler_state_row_width
        return sampler_states.narrow(0, 0, byte_len)

    def _set_sampler_seed(slot_idx: int, seed: int) -> None:
        normalized = int(seed) & 0x7FFFFFFFFFFFFFFF
        slot_state = rapid_sampler.setup_rand(normalized, 1)
        if not isinstance(slot_state, torch.Tensor) or slot_state.ndim != 1:
            raise RuntimeError("rapid-sampling setup_rand(seed, 1) 返回格式异常")
        if slot_state.numel() != sampler_state_row_width:
            raise RuntimeError("rapid-sampling setup_rand(seed, 1) 返回长度异常")
        start = slot_idx * sampler_state_row_width
        sampler_states.narrow(0, start, sampler_state_row_width).copy_(
            slot_state.to(device=sampler_states.device, dtype=sampler_states.dtype)
        )

    def _swap_sampler_state_rows(dst_idx: int, src_idx: int) -> None:
        if dst_idx == src_idx:
            return
        dst = dst_idx * sampler_state_row_width
        src = src_idx * sampler_state_row_width
        tmp = sampler_states[dst : dst + sampler_state_row_width].clone()
        sampler_states[dst : dst + sampler_state_row_width] = sampler_states[src : src + sampler_state_row_width]
        sampler_states[src : src + sampler_state_row_width] = tmp

    def _reset_slot(slot_idx: int) -> None:
        penalties[slot_idx, :] = 0
        states[0][:, :, slot_idx, :] = 0
        states[1][:, slot_idx, :, :, :] = 0
        states[2][slot_idx] = 0

    def _validate_sampled_tokens(sampled_tokens: torch.Tensor | Sequence[int], active_count: int) -> torch.Tensor:
        sampled = torch.as_tensor(sampled_tokens, device=device)
        if sampled.ndim == 0:
            sampled = sampled.unsqueeze(0)
        sampled = sampled.reshape(-1).to(dtype=torch.int64)
        if sampled.numel() != active_count:
            raise RuntimeError(f"rapid-sampling 返回 {sampled.numel()} 个 token，但 active_count={active_count}")
        invalid = (sampled < 0) | (sampled >= vocab_size)
        if torch.any(invalid):
            preview = sampled[: min(4, sampled.numel())].tolist()
            raise RuntimeError(f"rapid-sampling 返回非法 token id: {preview}")
        return sampled

    def _sample_rows(logits: torch.Tensor, active_count: int, sample_rows: list[int]) -> list[int]:
        if not sample_rows:
            return []
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[0] != len(sample_rows):
            raise RuntimeError(
                f"sample logits row count {logits.shape[0]} does not match sample_rows {len(sample_rows)}"
            )
        row_index = torch.tensor(sample_rows, dtype=torch.int64, device=device)
        logits_subset = logits.contiguous()
        penalties_view = penalties[:active_count]
        penalties_subset = penalties_view.index_select(0, row_index).clone()
        if no_penalty_ids is not None:
            penalties_subset[:, no_penalty_ids] = 0.0

        if len(sample_rows) == active_count:
            sampler_subset = _sampler_states_view(active_count)
        else:
            sampler_subset = torch.empty(
                (len(sample_rows) * sampler_state_row_width,),
                dtype=sampler_states.dtype,
                device=sampler_states.device,
            )
            for dst_idx, src_idx in enumerate(sample_rows):
                dst_start = dst_idx * sampler_state_row_width
                src_start = src_idx * sampler_state_row_width
                sampler_subset.narrow(0, dst_start, sampler_state_row_width).copy_(
                    sampler_states.narrow(0, src_start, sampler_state_row_width)
                )

        if sampling.penalties_enabled():
            sampled_raw = rapid_sampler.batch_sampling_repetition_temperature_topk_topp(
                logits_subset,
                penalties_subset,
                sampler_subset,
                presence_penalty_value,
                repetition_penalty_value,
                penalty_decay_value,
                sampler_temperature,
                sampler_top_k,
                sampler_top_p,
            )
        else:
            sampled_raw = rapid_sampler.batch_sampling_temperature_topk_topp(
                logits_subset,
                sampler_subset,
                sampler_temperature,
                sampler_top_k,
                sampler_top_p,
            )
        sampled_tensor = _validate_sampled_tokens(sampled_raw, len(sample_rows))

        penalties_view.index_copy_(0, row_index, penalties_subset)
        if len(sample_rows) != active_count:
            for src_idx, dst_idx in enumerate(sample_rows):
                src_start = src_idx * sampler_state_row_width
                dst_start = dst_idx * sampler_state_row_width
                sampler_states.narrow(0, dst_start, sampler_state_row_width).copy_(
                    sampler_subset.narrow(0, src_start, sampler_state_row_width)
                )
        return sampled_tensor.tolist()

    def _copy_state_into_slot(slot_idx: int, slot_state: list[torch.Tensor]) -> None:
        if slot_state[0].ndim == 4:
            states[0][:, :, slot_idx, :] = slot_state[0][:, :, 0, :].to(device=states[0].device, dtype=states[0].dtype)
        else:
            states[0][:, :, slot_idx, :] = slot_state[0].to(device=states[0].device, dtype=states[0].dtype)
        if slot_state[1].ndim == 5:
            states[1][:, slot_idx, :, :, :] = slot_state[1][:, 0, :, :, :].to(device=states[1].device, dtype=states[1].dtype)
        else:
            states[1][:, slot_idx, :, :, :] = slot_state[1].to(device=states[1].device, dtype=states[1].dtype)
        slot2 = slot_state[2]
        if slot2.ndim == 1:
            states[2][slot_idx] = slot2[0].to(device=states[2].device, dtype=states[2].dtype)
        else:
            states[2][slot_idx] = slot2.to(device=states[2].device, dtype=states[2].dtype)

    def _extract_slot_state(slot_idx: int) -> list[torch.Tensor]:
        return [
            states[0][:, :, slot_idx : slot_idx + 1, :].detach().clone(),
            states[1][:, slot_idx : slot_idx + 1, :, :, :].detach().clone(),
            states[2][slot_idx : slot_idx + 1].detach().clone(),
        ]

    def _extract_slot_logits(logits: torch.Tensor, local_idx: int) -> torch.Tensor:
        return logits[local_idx].detach().clone()

    def _build_constraint_runtime(prompt_idx: int) -> ConstraintRuntime | None:
        if prompt_constraints is None:
            return None
        constraint = prompt_constraints[prompt_idx]
        if constraint is None:
            return None
        if constraint_cache is None:
            raise RuntimeError("constraint cache was not initialized")
        return ConstraintRuntime(constraint=constraint.clone(), cache=constraint_cache)

    def _activate_slot(slot_idx: int) -> bool:
        if not encoded:
            return False
        prompt_idx, prompt, tokens, seed, stop_suffixes, max_stop_suffix_len = encoded.popleft()
        _reset_slot(slot_idx)
        ready_logits: torch.Tensor | None = None
        processed_prompt_tokens = 0
        pending = deque(tokens)
        prompt_tokens_remaining = len(tokens)

        cache_match = state_cache.match_prefix_state(tokens, device=device)
        if cache_match is not None:
            cache_state = cache_match.get("state")
            matched_tokens = int(cache_match.get("matched_tokens", 0))
            cache_logits = cache_match.get("logits")
            if isinstance(cache_state, list) and matched_tokens > 0:
                _copy_state_into_slot(slot_idx, cache_state)
                processed_prompt_tokens = matched_tokens
                pending = deque(tokens[matched_tokens:])
                prompt_tokens_remaining = len(tokens) - matched_tokens
                if prompt_tokens_remaining == 0 and isinstance(cache_logits, torch.Tensor):
                    ready_logits = cache_logits.to(device=device, dtype=torch.float32).reshape(-1)
                elif prompt_tokens_remaining == 0:
                    _reset_slot(slot_idx)
                    processed_prompt_tokens = 0
                    pending = deque(tokens)
                    prompt_tokens_remaining = len(tokens)

        task = _LightningTask(
            prompt_index=prompt_idx,
            prompt=prompt,
            prompt_tokens=tuple(tokens),
            pending_tokens=pending,
            prompt_tokens_remaining=prompt_tokens_remaining,
            processed_prompt_tokens=processed_prompt_tokens,
            generated_token_count=0,
            generated_tokens=[],
            generated_events=[],
            emitted_text_parts=[],
            utf8_tokens_in_buffer=[],
            pending_stop_tokens=[],
            stop_suffixes=stop_suffixes,
            max_stop_suffix_len=max_stop_suffix_len,
            pending_token=None,
            ready_logits=ready_logits,
            constraint_runtime=_build_constraint_runtime(prompt_idx),
            finish_reason=None,
        )
        if slot_idx < len(active_tasks):
            active_tasks[slot_idx] = task
        else:
            active_tasks.append(task)
        if seed is not None:
            _set_sampler_seed(slot_idx, seed)
        return True

    def _remove_slot(remove_idx: int) -> None:
        last_idx = len(active_tasks) - 1
        if remove_idx != last_idx:
            states[0][:, :, remove_idx, :] = states[0][:, :, last_idx, :]
            states[1][:, remove_idx, :, :, :] = states[1][:, last_idx, :, :, :]
            states[2][remove_idx] = states[2][last_idx]
            penalties[remove_idx, :] = penalties[last_idx, :]
            _swap_sampler_state_rows(remove_idx, last_idx)
            active_tasks[remove_idx] = active_tasks[last_idx]
        active_tasks.pop()

    def _next_bucket_boundary(processed: int, total: int) -> int | None:
        for bucket in prefix_buckets:
            if processed < bucket <= total:
                return bucket
        return None

    for slot_idx in range(batch_size):
        if not _activate_slot(slot_idx):
            break

    while active_tasks:
        accomplished: list[int] = []
        for idx, task in enumerate(active_tasks):
            if task.finish_reason is not None:
                if not task.pending_tokens and task.pending_token is None and task.ready_logits is None:
                    for delta in _finish_generated_text(task):
                        _record_generated_text_delta(task, delta, on_token=on_token, probe_only=probe_only)
                    output = GenerationOutput(
                        prompt_index=task.prompt_index,
                        prompt=task.prompt,
                        token_ids=list(task.generated_tokens),
                        text="".join(task.emitted_text_parts),
                        finish_reason=task.finish_reason,
                        tokens=list(task.generated_events),
                    )
                    if on_complete is not None and not probe_only:
                        on_complete(output)
                    outputs.append(output)
                    pbar.update(1)
                    if not _activate_slot(idx):
                        accomplished.append(idx)
                continue
            if task.pending_tokens or task.ready_logits is not None:
                continue
            generated_token = task.pending_token
            if generated_token is None:
                continue
            task.pending_token = None
            token_id = generated_token.token_id
            if token_id is None:
                raise RuntimeError("generated token event is missing token id")
            max_generated_tokens = 1 if probe_only else sampling.max_generate_tokens
            reached_stop = token_id in stop_tokens
            matched_stop_suffix = False
            reached_constraint = False
            if not reached_stop:
                task.generated_token_count += 1
                deltas, matched_stop_suffix = _push_generated_token_text(task, generated_token)
                for delta in deltas:
                    _record_generated_text_delta(task, delta, on_token=on_token, probe_only=probe_only)
                if task.constraint_runtime is not None:
                    if not task.constraint_runtime.commit_token_bytes(_token_bytes(generated_token)):
                        task.finish_reason = "constraint_violation"
                    elif task.constraint_runtime.is_complete():
                        reached_constraint = True
            reached_length = task.generated_token_count >= max_generated_tokens
            if not reached_stop and not matched_stop_suffix and not reached_length and not reached_constraint and task.finish_reason is None:
                task.pending_tokens.append(token_id)
            if reached_stop or matched_stop_suffix or reached_length or reached_constraint or task.finish_reason is not None:
                if not matched_stop_suffix:
                    for delta in _finish_generated_text(task):
                        _record_generated_text_delta(task, delta, on_token=on_token, probe_only=probe_only)
                output = GenerationOutput(
                    prompt_index=task.prompt_index,
                    prompt=task.prompt,
                    token_ids=list(task.generated_tokens),
                    text="".join(task.emitted_text_parts),
                    finish_reason=(
                        task.finish_reason
                        or ("constraint_stop" if reached_constraint else ("stop_token" if (reached_stop or matched_stop_suffix) else "max_length"))
                    ),
                    tokens=list(task.generated_events),
                )
                if on_complete is not None and not probe_only:
                    on_complete(output)
                outputs.append(output)
                pbar.update(1)
                if not _activate_slot(idx):
                    accomplished.append(idx)

        if accomplished:
            for remove_idx in sorted(accomplished, reverse=True):
                _remove_slot(remove_idx)
            if not active_tasks:
                break

        direct_sample_rows: list[int] = []
        direct_sample_logits: list[torch.Tensor] = []
        forward_rows: list[int] = []
        forward_tokens: list[list[int]] = []
        forward_sample_flags: list[bool] = []
        forward_cache_buckets: list[int | None] = []

        for task_idx, task in enumerate(active_tasks):
            if task.ready_logits is not None:
                direct_sample_rows.append(task_idx)
                direct_sample_logits.append(task.ready_logits.reshape(1, -1))
                task.ready_logits = None
                continue
            if task.prompt_tokens_remaining > 0:
                next_bucket = _next_bucket_boundary(
                    task.processed_prompt_tokens,
                    task.processed_prompt_tokens + task.prompt_tokens_remaining,
                )
                take_count = min(prefill_chunk_size, task.prompt_tokens_remaining)
                if next_bucket is not None:
                    take_count = min(take_count, next_bucket - task.processed_prompt_tokens)
                step_tokens = [task.pending_tokens.popleft() for _ in range(take_count)]
                task.prompt_tokens_remaining -= take_count
                task.processed_prompt_tokens += take_count
                forward_rows.append(task_idx)
                forward_tokens.append(step_tokens)
                forward_sample_flags.append(task.prompt_tokens_remaining == 0)
                forward_cache_buckets.append(
                    task.processed_prompt_tokens if task.processed_prompt_tokens in prefix_buckets else None
                )
                continue
            if task.pending_tokens:
                forward_rows.append(task_idx)
                forward_tokens.append([task.pending_tokens.popleft()])
                forward_sample_flags.append(True)
                forward_cache_buckets.append(None)
                continue
            raise RuntimeError("lightning engine reached an idle task without logits or pending tokens")

        active_count = len(active_tasks)
        sample_rows: list[int] = list(direct_sample_rows)
        sample_logits_parts: list[torch.Tensor] = list(direct_sample_logits)

        if forward_rows:
            state_subset = [
                states[0][:, :, forward_rows, :].clone(),
                states[1][:, forward_rows, :, :, :].clone(),
                states[2][forward_rows].clone(),
            ]
            logits_forward = model.forward_batch(forward_tokens, state_subset).float().contiguous()  # type: ignore[attr-defined]
            for local_idx, global_idx in enumerate(forward_rows):
                states[0][:, :, global_idx, :] = state_subset[0][:, :, local_idx, :]
                states[1][:, global_idx, :, :, :] = state_subset[1][:, local_idx, :, :, :]
                states[2][global_idx] = state_subset[2][local_idx]

            for local_idx, global_idx in enumerate(forward_rows):
                bucket_len = forward_cache_buckets[local_idx]
                if bucket_len is not None:
                    task = active_tasks[global_idx]
                    state_cache.put_prefix_state(
                        _reconstruct_prompt_prefix(task, bucket_len),
                        _extract_slot_state(global_idx),
                        _extract_slot_logits(logits_forward, local_idx),
                    )
                if forward_sample_flags[local_idx]:
                    sample_rows.append(global_idx)
                    sample_logits_parts.append(logits_forward[local_idx : local_idx + 1])

            tokens_processed += sum(len(tokens) for tokens in forward_tokens)

        if not sample_rows:
            now = time.time()
            elapsed = max(now - start_time, 1e-6)
            window_elapsed = now - window_start_time
            if window_elapsed >= 0.5:
                recent_tokens = tokens_processed - window_start_tokens
                inst_throughput = recent_tokens / max(window_elapsed, 1e-6)
                throughput_ema = inst_throughput if throughput_ema is None else 0.9 * throughput_ema + 0.1 * inst_throughput
                window_start_time = now
                window_start_tokens = tokens_processed
            inst_display = throughput_ema if throughput_ema is not None else 0.0
            pbar.set_postfix_str(f"tok/s avg {tokens_processed / elapsed:.1f} cur {inst_display:.1f}")
            pbar.update(0)
            continue

        combined_logits = torch.cat(sample_logits_parts, dim=0).to(device=device, dtype=torch.float32)
        if ban_token_ids:
            combined_logits[:, ban_token_ids] = -math.inf
        eligible_sample_rows: list[int] = []
        eligible_logits_parts: list[torch.Tensor] = []
        for local_idx, task_idx in enumerate(sample_rows):
            runtime = active_tasks[task_idx].constraint_runtime
            row_logits = combined_logits[local_idx]
            if runtime is not None:
                allowed_ids = runtime.allowed_token_ids()
                if not allowed_ids:
                    active_tasks[task_idx].finish_reason = "constraint_dead_end"
                    continue
                allowed_tensor = torch.tensor(sorted(allowed_ids), dtype=torch.int64, device=device)
                masked = torch.full_like(row_logits, -math.inf)
                masked.index_copy_(0, allowed_tensor, row_logits.index_select(0, allowed_tensor))
                row_logits = masked
            eligible_sample_rows.append(task_idx)
            eligible_logits_parts.append(row_logits.reshape(1, -1))
        if not eligible_sample_rows:
            continue
        sampled_logits = torch.cat(eligible_logits_parts, dim=0)
        sampled_list = _sample_rows(sampled_logits, active_count, eligible_sample_rows)
        for local_idx, task_idx in enumerate(eligible_sample_rows):
            token_id = int(sampled_list[local_idx])
            active_tasks[task_idx].pending_token = _build_generated_token(
                tokenizer,
                sampled_logits[local_idx],
                token_id,
                top_logprobs=top_logprobs,
                include_logprobs=top_logprobs > 0,
            )

        now = time.time()
        elapsed = max(now - start_time, 1e-6)
        window_elapsed = now - window_start_time
        if window_elapsed >= 0.5:
            recent_tokens = tokens_processed - window_start_tokens
            inst_throughput = recent_tokens / max(window_elapsed, 1e-6)
            throughput_ema = inst_throughput if throughput_ema is None else 0.9 * throughput_ema + 0.1 * inst_throughput
            window_start_time = now
            window_start_tokens = tokens_processed
        inst_display = throughput_ema if throughput_ema is not None else 0.0
        pbar.set_postfix_str(f"tok/s avg {tokens_processed / elapsed:.1f} cur {inst_display:.1f}")
        pbar.update(0)

    pbar.close()
    for output in outputs:
        if not output.text:
            output.text = _decode_tokens(tokenizer, output.token_ids)
    outputs.sort(key=lambda item: item.prompt_index)
    return outputs


def _reconstruct_prompt_prefix(task: _LightningTask, bucket_len: int) -> list[int]:
    if len(task.prompt_tokens) < bucket_len:
        raise ValueError("bucket_len exceeds prompt prefill length")
    if task.processed_prompt_tokens < bucket_len:
        raise ValueError("processed prompt length is shorter than bucket_len")
    return list(task.prompt_tokens[:bucket_len])
