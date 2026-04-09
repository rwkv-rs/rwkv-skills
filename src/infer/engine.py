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
from .sampling import GeneratedTextDelta, GeneratedToken, GeneratedTokenCandidate, GenerationOutput, SamplingConfig

DEFAULT_PREFILL_CHUNK_SIZE = 16


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
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        progress_desc: str = "Generating",
        probe_only: bool = False,
        on_complete: Callable[[GenerationOutput], None] | None = None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None = None,
        prompt_stop_suffixes: Sequence[Sequence[str] | None] | None = None,
        prompt_seeds: Sequence[int | None] | None = None,
        top_logprobs: int = 0,
        show_progress: bool = True,
    ) -> list[GenerationOutput]:
        return _continuous_batching(
            self.model,
            self.tokenizer,
            prompts,
            sampling,
            batch_size,
            prefill_chunk_size,
            progress_desc,
            probe_only,
            on_complete,
            on_token,
            prompt_stop_suffixes,
            prompt_seeds,
            top_logprobs,
            show_progress,
        )


@dataclass(slots=True)
class _ActiveTask:
    prompt_index: int
    prompt: str
    pending_tokens: deque[int]
    prompt_tokens_remaining: int
    generated_token_count: int
    generated_tokens: list[int]
    generated_events: list[GeneratedToken]
    emitted_text_parts: list[str]
    utf8_tokens_in_buffer: list[GeneratedToken]
    pending_stop_tokens: list[GeneratedToken]
    stop_suffixes: tuple[tuple[str, bytes], ...]
    max_stop_suffix_len: int
    pending_token: GeneratedToken | None
    finish_reason: str | None


def _continuous_batching(
    model: RWKVModelProtocol,
    tokenizer: TokenizerProtocol,
    prompts: Sequence[str],
    sampling: SamplingConfig,
    batch_size: int,
    prefill_chunk_size: int,
    progress_desc: str,
    probe_only: bool = False,
    on_complete: Callable[[GenerationOutput], None] | None = None,
    on_token: Callable[[int, GeneratedTextDelta], None] | None = None,
    prompt_stop_suffixes: Sequence[Sequence[str] | None] | None = None,
    prompt_seeds: Sequence[int | None] | None = None,
    top_logprobs: int = 0,
    show_progress: bool = True,
) -> list[GenerationOutput]:
    if not prompts:
        return []
    if prompt_seeds is not None and len(prompt_seeds) != len(prompts):
        raise ValueError("prompt_seeds 长度必须与 prompts 一致")
    if prompt_stop_suffixes is not None and len(prompt_stop_suffixes) != len(prompts):
        raise ValueError("prompt_stop_suffixes 长度必须与 prompts 一致")
    batch_size = max(1, min(batch_size, len(prompts)))
    prefill_chunk_size = max(1, int(prefill_chunk_size))
    top_logprobs = max(0, int(top_logprobs))

    vocab_size = _infer_vocab_size(model)
    device = _infer_device(model)
    states = _prepare_state_container(model.generate_zero_state(batch_size))
    sampling = sampling.checked(vocab_size)

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

    def _set_sampler_seed(slot_idx: int, seed: int) -> None:
        if sampler_states is None:
            raise RuntimeError("rapid-sampling 状态未初始化")
        normalized = int(seed) & 0x7FFFFFFFFFFFFFFF
        slot_state = rapid_sampler.setup_rand(normalized, 1)
        if not isinstance(slot_state, torch.Tensor) or slot_state.ndim != 1:
            raise RuntimeError("rapid-sampling setup_rand(seed, 1) 返回格式异常")
        if slot_state.numel() != sampler_state_row_width:
            raise RuntimeError("rapid-sampling setup_rand(seed, 1) 返回长度异常")
        start = slot_idx * sampler_state_row_width
        target = sampler_states.narrow(0, start, sampler_state_row_width)
        target.copy_(slot_state.to(device=target.device, dtype=target.dtype))

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
        seed = (
            None
            if prompt_seeds is None or prompt_seeds[idx] is None
            else int(prompt_seeds[idx])
        )
        stop_suffixes = None if prompt_stop_suffixes is None else prompt_stop_suffixes[idx]
        normalized_stop_suffixes, max_stop_suffix_len = _normalize_stop_suffixes(stop_suffixes)
        encoded.append((idx, prompt, tokens, seed, normalized_stop_suffixes, max_stop_suffix_len))

    stop_tokens = set(sampling.stop_tokens)
    ban_tokens = tuple(sampling.ban_tokens or ())
    ban_token_ids = [token_id for token_id in ban_tokens if 0 <= token_id < vocab_size]
    no_penalty = set(sampling.no_penalty_token_ids)

    sampler_temperature = float(sampling.temperature or 1.0)
    sampler_top_k = int(sampling.top_k) if sampling.top_k is not None else -1
    sampler_top_p = float(sampling.top_p) if sampling.top_p is not None else 1.0

    presence_penalty_value = float(sampling.presence_penalty)
    repetition_penalty_value = float(sampling.repetition_penalty)
    penalty_decay_value = float(sampling.penalty_decay)

    penalties = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    no_penalty_ids: torch.Tensor | None = None
    valid_no_penalty_ids = sorted(token_id for token_id in no_penalty if 0 <= token_id < vocab_size)
    if valid_no_penalty_ids:
        no_penalty_ids = torch.tensor(valid_no_penalty_ids, dtype=torch.int64, device=device)

    active_tasks: list[_ActiveTask] = []
    for slot_idx in range(batch_size):
        prompt_idx, prompt, tokens, seed, stop_suffixes, max_stop_suffix_len = encoded.popleft()
        pending = deque(tokens)
        active_tasks.append(
            _ActiveTask(
                prompt_idx,
                prompt,
                pending,
                len(tokens),
                0,
                [],
                [],
                [],
                [],
                [],
                stop_suffixes,
                max_stop_suffix_len,
                None,
                None,
            )
        )
        if seed is not None:
            _set_sampler_seed(slot_idx, seed)

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

    def _sample_subset(logits: torch.Tensor, active_count: int, sample_rows: list[int]) -> list[int]:
        if not sample_rows:
            return []

        if sampler_states is None:
            raise RuntimeError("rapid-sampling 状态未初始化")

        row_index = torch.tensor(sample_rows, dtype=torch.int64, device=device)
        logits_subset = logits.index_select(0, row_index).contiguous()
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

    while active_tasks:
        accomplished: list[int] = []

        for idx, task in enumerate(active_tasks):
            if task.pending_tokens:
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
            if not reached_stop:
                task.generated_token_count += 1
                deltas, matched_stop_suffix = _push_generated_token_text(task, generated_token)
                for delta in deltas:
                    _record_generated_text_delta(task, delta, on_token=on_token, probe_only=probe_only)
            reached_length = task.generated_token_count >= max_generated_tokens
            if not reached_stop and not matched_stop_suffix and not reached_length:
                task.pending_tokens.append(token_id)
            if reached_stop or matched_stop_suffix or reached_length:
                if not matched_stop_suffix:
                    for delta in _finish_generated_text(task):
                        _record_generated_text_delta(task, delta, on_token=on_token, probe_only=probe_only)
                output = GenerationOutput(
                    prompt_index=task.prompt_index,
                    prompt=task.prompt,
                    token_ids=list(task.generated_tokens),
                    text="".join(task.emitted_text_parts),
                    finish_reason="stop_token" if (reached_stop or matched_stop_suffix) else "max_length",
                    tokens=list(task.generated_events),
                )
                if on_complete is not None and not probe_only:
                    on_complete(output)
                outputs.append(output)
                pbar.update(1)
                if encoded:
                    prompt_idx, prompt, tokens, seed, stop_suffixes, max_stop_suffix_len = encoded.popleft()
                    pending = deque(tokens)
                    active_tasks[idx] = _ActiveTask(
                        prompt_idx,
                        prompt,
                        pending,
                        len(tokens),
                        0,
                        [],
                        [],
                        [],
                        [],
                        [],
                        stop_suffixes,
                        max_stop_suffix_len,
                        None,
                        None,
                    )
                    _reset_slot(idx)
                    if seed is not None:
                        _set_sampler_seed(idx, seed)
                else:
                    accomplished.append(idx)

        if accomplished:
            for remove_idx in sorted(accomplished, reverse=True):
                _remove_slot(remove_idx)

        next_tokens: list[list[int]] = []
        rows_to_sample: list[int] = []
        active_count = len(active_tasks)
        for task_idx, task in enumerate(active_tasks):
            if task.prompt_tokens_remaining > 0:
                take_count = min(prefill_chunk_size, task.prompt_tokens_remaining)
                step_tokens = [task.pending_tokens.popleft() for _ in range(take_count)]
                task.prompt_tokens_remaining -= take_count
                if task.prompt_tokens_remaining == 0:
                    rows_to_sample.append(task_idx)
            else:
                step_tokens = [task.pending_tokens.popleft()]
                rows_to_sample.append(task_idx)
            next_tokens.append(step_tokens)
        tokens_processed += sum(len(tokens) for tokens in next_tokens)

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

        sampled_list = _sample_subset(logits, active_count, rows_to_sample)
        for local_idx, task_idx in enumerate(rows_to_sample):
            token_id = int(sampled_list[local_idx])
            active_tasks[task_idx].pending_token = _build_generated_token(
                tokenizer,
                logits[task_idx],
                token_id,
                top_logprobs=top_logprobs,
                include_logprobs=top_logprobs > 0,
            )

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


def _decode_token_bytes(tokenizer: TokenizerProtocol, token_ids: Sequence[int]) -> bytes:
    decode_bytes = getattr(tokenizer, "decodeBytes", None)
    if callable(decode_bytes):
        data = decode_bytes(token_ids)
        if isinstance(data, bytes):
            return data
    decode_bytes = getattr(tokenizer, "decode_bytes", None)
    if callable(decode_bytes):
        data = decode_bytes(token_ids)
        if isinstance(data, bytes):
            return data
    try:
        return tokenizer.decode(token_ids).encode("utf-8")
    except Exception:
        return b""


def _normalize_stop_suffixes(stop_suffixes: Sequence[str] | None) -> tuple[tuple[tuple[str, bytes], ...], int]:
    normalized = tuple(
        (suffix, suffix.encode("utf-8"))
        for suffix in (str(item) for item in (stop_suffixes or ()))
        if suffix
    )
    max_len = max((len(bytes_) for _suffix, bytes_ in normalized), default=0)
    return normalized, max_len


def _record_generated_text_delta(
    task: _ActiveTask,
    delta: GeneratedTextDelta,
    *,
    on_token: Callable[[int, GeneratedTextDelta], None] | None,
    probe_only: bool,
) -> None:
    if not delta.text and not delta.tokens:
        return
    task.emitted_text_parts.append(delta.text)
    task.generated_tokens.extend(
        int(token.token_id)
        for token in delta.tokens
        if token.token_id is not None
    )
    task.generated_events.extend(delta.tokens)
    if on_token is not None and not probe_only:
        on_token(task.prompt_index, delta)


def _push_generated_token_text(
    task: _ActiveTask,
    token: GeneratedToken,
) -> tuple[list[GeneratedTextDelta], bool]:
    task.utf8_tokens_in_buffer.append(token)
    emit_count = _longest_valid_utf8_token_prefix(task.utf8_tokens_in_buffer)
    if emit_count <= 0:
        return [], False
    stable_tokens = task.utf8_tokens_in_buffer[:emit_count]
    del task.utf8_tokens_in_buffer[:emit_count]
    return _push_stop_tokens(task, stable_tokens)


def _push_stop_tokens(
    task: _ActiveTask,
    tokens: Sequence[GeneratedToken],
) -> tuple[list[GeneratedTextDelta], bool]:
    if not tokens:
        return [], False
    task.pending_stop_tokens.extend(tokens)
    pending_bytes = _collect_token_bytes(task.pending_stop_tokens)
    matched = _find_stop_suffix(pending_bytes, task.stop_suffixes)
    if matched is not None:
        delta = _take_stop_output(task, matched[0], allow_partial_text=True)
        task.pending_stop_tokens.clear()
        task.utf8_tokens_in_buffer.clear()
        return ([delta] if delta.text or delta.tokens else []), True
    emit_limit = len(pending_bytes)
    if task.max_stop_suffix_len > 0:
        emit_limit = max(0, emit_limit - (task.max_stop_suffix_len - 1))
    delta = _take_stop_output(task, emit_limit, allow_partial_text=False)
    return ([delta] if delta.text or delta.tokens else []), False


def _finish_generated_text(task: _ActiveTask) -> list[GeneratedTextDelta]:
    task.utf8_tokens_in_buffer.clear()
    if not task.pending_stop_tokens:
        return []
    emit_limit = len(_collect_token_bytes(task.pending_stop_tokens))
    delta = _take_stop_output(task, emit_limit, allow_partial_text=False)
    return [delta] if delta.text or delta.tokens else []


def _take_stop_output(
    task: _ActiveTask,
    emit_limit: int,
    *,
    allow_partial_text: bool,
) -> GeneratedTextDelta:
    if emit_limit <= 0 or not task.pending_stop_tokens:
        return GeneratedTextDelta(text="", tokens=[])
    emit_count = 0
    emitted_bytes = 0
    for token in task.pending_stop_tokens:
        next_bytes = emitted_bytes + len(token.bytes)
        if next_bytes > emit_limit:
            break
        emitted_bytes = next_bytes
        emit_count += 1
    tokens = task.pending_stop_tokens[:emit_count]
    del task.pending_stop_tokens[:emit_count]
    text_bytes = bytearray(_collect_token_bytes(tokens))
    if allow_partial_text and emitted_bytes < emit_limit and task.pending_stop_tokens:
        partial_len = emit_limit - emitted_bytes
        partial_bytes = task.pending_stop_tokens[0].bytes[:partial_len]
        valid_prefix_len = _longest_valid_utf8_prefix_len(partial_bytes)
        if valid_prefix_len > 0:
            text_bytes.extend(partial_bytes[:valid_prefix_len])
    return GeneratedTextDelta(
        text=bytes(text_bytes).decode("utf-8", errors="replace"),
        tokens=list(tokens),
    )


def _token_bytes(token: GeneratedToken) -> bytes:
    if token.bytes:
        return bytes(token.bytes)
    if token.text:
        return token.text.encode("utf-8")
    return b""


def _collect_token_bytes(tokens: Sequence[GeneratedToken]) -> bytes:
    return b"".join(_token_bytes(token) for token in tokens)


def _longest_valid_utf8_token_prefix(tokens: Sequence[GeneratedToken]) -> int:
    if not tokens:
        return 0
    buffer = bytearray()
    last_valid = 0
    for index, token in enumerate(tokens):
        buffer.extend(_token_bytes(token))
        try:
            bytes(buffer).decode("utf-8")
            last_valid = index + 1
        except UnicodeDecodeError as exc:
            if exc.reason == "unexpected end of data" and exc.end == len(buffer):
                continue
            last_valid = index + 1
            break
    return last_valid


def _longest_valid_utf8_prefix_len(data: bytes) -> int:
    try:
        data.decode("utf-8")
        return len(data)
    except UnicodeDecodeError as exc:
        return int(exc.start)


def _find_stop_suffix(
    data: bytes,
    stop_suffixes: Sequence[tuple[str, bytes]],
) -> tuple[int, int] | None:
    matched: tuple[int, int, int] | None = None
    for suffix_index, (_suffix, suffix_bytes) in enumerate(stop_suffixes):
        if not suffix_bytes or len(suffix_bytes) > len(data):
            continue
        start = data.find(suffix_bytes)
        if start < 0:
            continue
        if matched is None or start < matched[0] or (start == matched[0] and len(suffix_bytes) > matched[2]):
            matched = (start, suffix_index, len(suffix_bytes))
    if matched is None:
        return None
    return matched[0], matched[1]


def _build_generated_token(
    tokenizer: TokenizerProtocol,
    logits_row: torch.Tensor,
    token_id: int,
    *,
    top_logprobs: int,
    include_logprobs: bool,
) -> GeneratedToken:
    token_bytes = _decode_token_bytes(tokenizer, [int(token_id)])
    text = token_bytes.decode("utf-8", errors="replace")
    if not include_logprobs:
        return GeneratedToken(token_id=int(token_id), text=text, bytes=token_bytes)

    row = logits_row.to(dtype=torch.float32)
    logprobs = torch.log_softmax(row, dim=0)
    limit = max(1, min(int(top_logprobs), int(logprobs.shape[0])))
    top_values, top_indices = torch.topk(logprobs, k=limit)
    return GeneratedToken(
        token_id=int(token_id),
        text=text,
        bytes=token_bytes,
        logprob=float(logprobs[int(token_id)].item()),
        top_logprobs=[
            _build_generated_token_candidate(tokenizer, int(candidate_id), float(candidate_logprob))
            for candidate_id, candidate_logprob in zip(top_indices.tolist(), top_values.tolist(), strict=True)
        ],
    )


def _build_generated_token_candidate(
    tokenizer: TokenizerProtocol,
    token_id: int,
    logprob: float,
) -> GeneratedTokenCandidate:
    token_bytes = _decode_token_bytes(tokenizer, [token_id])
    return GeneratedTokenCandidate(
        token_id=token_id,
        text=token_bytes.decode("utf-8", errors="replace"),
        bytes=token_bytes,
        logprob=logprob,
    )


__all__ = ["InferenceEngine", "GenerationOutput"]
