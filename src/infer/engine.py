from __future__ import annotations

"""RWKV 推理引擎，封装连续批量生成、state 管理等逻辑。"""

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import torch
from tqdm import tqdm

from .rapid_sampling_loader import get_rapid_sampling_module
from .sampling import GenerationOutput, SamplingConfig


class TokenizerProtocol:
    def encode(self, text: str) -> list[int]:  # pragma: no cover - protocol
        ...

    def decode(self, token_ids: Sequence[int]) -> str:  # pragma: no cover - protocol
        ...


class RWKVModelProtocol:
    def generate_zero_state(self, batch_size: int):  # pragma: no cover - protocol
        ...

    def forward(self, tokens: Sequence[int], state, full_output: bool = False):  # pragma: no cover - protocol
        ...

    def forward_batch(self, tokens: Sequence[Sequence[int] | None], state, full_output: bool = False):  # pragma: no cover - protocol
        ...


class InferenceEngine:
    def __init__(self, model: RWKVModelProtocol, tokenizer: TokenizerProtocol) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling: SamplingConfig,
        batch_size: int,
        progress_desc: str = "Generating",
        probe_only: bool = False,
        on_complete: Callable[[GenerationOutput], None] | None = None,
    ) -> list[GenerationOutput]:
        return _continuous_batching(
            self.model,
            self.tokenizer,
            prompts,
            sampling,
            batch_size,
            progress_desc,
            probe_only,
            on_complete,
        )


@dataclass(slots=True)
class _ActiveTask:
    prompt_index: int
    prompt: str
    pending_tokens: deque[int]
    generated_tokens: list[int]
    new_token: int | None
    finish_reason: str | None


def _continuous_batching(
    model: RWKVModelProtocol,
    tokenizer: TokenizerProtocol,
    prompts: Sequence[str],
    sampling: SamplingConfig,
    batch_size: int,
    progress_desc: str,
    probe_only: bool = False,
    on_complete: Callable[[GenerationOutput], None] | None = None,
) -> list[GenerationOutput]:
    if not prompts:
        return []
    batch_size = max(1, min(batch_size, len(prompts)))

    vocab_size = _infer_vocab_size(model)
    device = _infer_device(model)
    states = _prepare_state_container(model.generate_zero_state(batch_size))

    rapid_sampler = get_rapid_sampling_module()
    sampler_states: torch.Tensor | None = None
    sampler_state_row_width = 0
    random_seed = int(torch.initial_seed() & 0x7FFFFFFFFFFFFFFF)
    sampler_states = rapid_sampler.setup_rand(random_seed, batch_size)
    if not isinstance(sampler_states, torch.Tensor) or sampler_states.ndim != 1:
        raise RuntimeError("rapid-sampling setup_rand 返回格式异常，期望 1D Tensor")
    if sampler_states.numel() % batch_size != 0:
        raise RuntimeError("rapid-sampling 随机状态长度与 batch_size 不匹配")
    sampler_state_row_width = sampler_states.numel() // batch_size

    def _sampler_states_view(active_count: int) -> torch.Tensor:
        if sampler_states is None:
            raise RuntimeError("rapid-sampling 状态未初始化")
        if active_count <= 0:
            raise RuntimeError("active_count 必须大于 0")
        byte_len = active_count * sampler_state_row_width
        return sampler_states.narrow(0, 0, byte_len)

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

    encoded = deque()
    for idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        if sampling.pad_zero:
            tokens = [0] + tokens
        encoded.append((idx, prompt, tokens))

    stop_tokens = set(sampling.stop_tokens)
    ban_tokens = tuple(sampling.ban_tokens or ())
    ban_token_ids = [token_id for token_id in ban_tokens if 0 <= token_id < vocab_size]
    no_penalty = set(sampling.no_penalty_token_ids)

    sampler_temperature = float(sampling.temperature or 1.0)
    sampler_top_k = int(sampling.top_k) if sampling.top_k is not None else -1
    sampler_top_p = float(sampling.top_p) if sampling.top_p is not None else 1.0

    alpha_presence_value = float(sampling.alpha_presence)
    alpha_frequency_value = float(sampling.alpha_frequency)
    alpha_decay_value = float(sampling.alpha_decay)

    penalties = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    no_penalty_ids: torch.Tensor | None = None
    valid_no_penalty_ids = sorted(token_id for token_id in no_penalty if 0 <= token_id < vocab_size)
    if valid_no_penalty_ids:
        no_penalty_ids = torch.tensor(valid_no_penalty_ids, dtype=torch.int64, device=device)

    active_tasks: list[_ActiveTask] = []
    for _ in range(batch_size):
        prompt_idx, prompt, tokens = encoded.popleft()
        pending = deque(tokens)
        active_tasks.append(_ActiveTask(prompt_idx, prompt, pending, [], None, None))

    pbar = tqdm(
        total=len(prompts),
        desc=progress_desc,
        unit=" sequence",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
    )

    outputs: list[GenerationOutput] = []
    start_time = time.time()
    tokens_processed = 0
    window_start_time = start_time
    window_start_tokens = 0
    throughput_ema: float | None = None

    def _reset_slot(slot_idx: int) -> None:
        penalties[slot_idx, :] = 0
        states[0][:, :, slot_idx, :] = 0
        states[1][:, slot_idx, :, :, :] = 0
        states[2][slot_idx] = 0

    def _swap_sampler_state_rows(dst_idx: int, src_idx: int) -> None:
        if sampler_states is None:
            return
        dst = dst_idx * sampler_state_row_width
        src = src_idx * sampler_state_row_width
        if dst == src:
            return
        tmp = sampler_states[dst : dst + sampler_state_row_width].clone()
        sampler_states[dst : dst + sampler_state_row_width] = sampler_states[src : src + sampler_state_row_width]
        sampler_states[src : src + sampler_state_row_width] = tmp

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

    while active_tasks:
        accomplished: list[int] = []

        for idx, task in enumerate(active_tasks):
            if task.pending_tokens:
                continue
            new_token = task.new_token
            if new_token is None:
                continue
            reached_stop = new_token in stop_tokens
            reached_length = len(task.generated_tokens) >= (sampling.max_generate_tokens if not probe_only else 1)
            if not reached_stop:
                task.pending_tokens.append(new_token)
                task.generated_tokens.append(new_token)
            if reached_stop or reached_length:
                output = GenerationOutput(
                    prompt_index=task.prompt_index,
                    prompt=task.prompt,
                    token_ids=list(task.generated_tokens),
                    text="",
                    finish_reason="stop_token" if reached_stop else "max_length",
                )
                if on_complete is not None and not probe_only:
                    output.text = _decode_tokens(tokenizer, output.token_ids)
                    on_complete(output)
                outputs.append(output)
                pbar.update(1)
                if encoded:
                    prompt_idx, prompt, tokens = encoded.popleft()
                    pending = deque(tokens)
                    active_tasks[idx] = _ActiveTask(prompt_idx, prompt, pending, [], None, None)
                    _reset_slot(idx)
                else:
                    accomplished.append(idx)

        if accomplished:
            for remove_idx in sorted(accomplished, reverse=True):
                _remove_slot(remove_idx)

        next_tokens: list[list[int]] = []
        active_count = len(active_tasks)
        for task in active_tasks:
            token = task.pending_tokens.popleft()
            next_tokens.append([token])
        tokens_processed += active_count

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

        if not active_tasks:
            break

        state_view = [
            states[0][:, :, :active_count, :],
            states[1][:, :active_count, :, :, :],
            states[2][:active_count],
        ]
        logits = model.forward_batch(next_tokens, state_view).float().contiguous()

        if ban_token_ids:
            logits[:, ban_token_ids] = -math.inf

        sampler_states_view = _sampler_states_view(active_count)
        sampled_raw = rapid_sampler.batch_sampling_repetition_temperature_topk_topp(
            logits,
            penalties[:active_count],
            sampler_states_view,
            alpha_presence_value,
            alpha_frequency_value,
            alpha_decay_value,
            sampler_temperature,
            sampler_top_k,
            sampler_top_p,
        )
        if no_penalty_ids is not None:
            penalties[:active_count, no_penalty_ids] = 0.0
        sampled_tensor = _validate_sampled_tokens(sampled_raw, active_count)

        sampled_list = sampled_tensor.tolist()
        for idx, task in enumerate(active_tasks):
            task.new_token = int(sampled_list[idx])

    pbar.close()

    for output in outputs:
        if output.text:
            continue
        output.text = _decode_tokens(tokenizer, output.token_ids)

    outputs.sort(key=lambda item: item.prompt_index)
    return outputs


def _infer_vocab_size(model: RWKVModelProtocol) -> int:
    args = getattr(model, "args", None)
    if args is not None and hasattr(args, "vocab_size"):
        return int(args.vocab_size)
    candidate = getattr(model, "vocab_size", None)
    if candidate:
        return int(candidate)
    return 65536


def _infer_device(model: RWKVModelProtocol) -> torch.device:
    tensor = None
    if hasattr(model, "z") and isinstance(model.z, dict):
        tensor = model.z.get("head.weight")
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    if hasattr(model, "parameters"):
        try:
            first = next(model.parameters())  # type: ignore[arg-type]
            if isinstance(first, torch.Tensor):
                return first.device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_state_container(state):
    if isinstance(state, list):
        return state
    if isinstance(state, tuple):
        return list(state)
    raise TypeError("generate_zero_state 必须返回 list 或 tuple")


def _decode_tokens(tokenizer: TokenizerProtocol, token_ids: Sequence[int]) -> str:
    tokens = list(token_ids)
    text = ""
    while tokens:
        try:
            text = tokenizer.decode(tokens)
            break
        except Exception:
            tokens = tokens[:-1]
    return text


__all__ = ["InferenceEngine", "GenerationOutput"]
