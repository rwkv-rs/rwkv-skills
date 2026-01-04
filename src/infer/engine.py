from __future__ import annotations

"""RWKV 推理引擎，封装连续批量生成、state 管理等逻辑。"""

from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Sequence

if "FLASHINFER_WORKSPACE_BASE" not in os.environ:
    # flashinfer JIT cache defaults to ~/.cache; some sandboxed environments may have an unwritable HOME.
    try:
        home = Path.home()
        if not (home.exists() and os.access(home, os.W_OK)):
            os.environ["FLASHINFER_WORKSPACE_BASE"] = "/tmp"
    except Exception:  # noqa: BLE001
        os.environ["FLASHINFER_WORKSPACE_BASE"] = "/tmp"

import flashinfer
import torch
from tqdm import tqdm

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
    ) -> list[GenerationOutput]:
        return _continuous_batching(self.model, self.tokenizer, prompts, sampling, batch_size, progress_desc, probe_only)


@dataclass(slots=True)
class _ActiveTask:
    prompt_index: int
    prompt: str
    pending_tokens: deque[int]
    generated_tokens: list[int]
    new_token: int | None
    finish_reason: str | None


def _torch_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Lightweight torch fallback for top-k/top-p sampling on CUDA."""

    use_top_k = top_k is not None and 0 < top_k < logits.size(-1)
    if use_top_k:
        logits, topk_idx = torch.topk(logits, top_k, dim=-1)
    else:
        topk_idx = None

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        # keep at least one token even if top_p is extremely small
        mask = cum > top_p
        mask[..., 0] = False
        filtered = sorted_probs.masked_fill(mask, 0.0)
        filtered_sum = filtered.sum(dim=-1, keepdim=True)
        needs_fallback = ~torch.isfinite(filtered_sum) | (filtered_sum <= 0)
        if torch.any(needs_fallback):
            # fallback to the top candidate so we always have a valid distribution
            fallback = torch.zeros_like(filtered)
            fallback[..., 0] = 1.0
            filtered = torch.where(needs_fallback, fallback, filtered)
            filtered_sum = filtered.sum(dim=-1, keepdim=True)
        filtered = filtered / filtered_sum.clamp_min(torch.finfo(filtered.dtype).eps)
        local_choice = torch.multinomial(filtered, 1).squeeze(-1)
        local_idx = torch.gather(sorted_idx, -1, local_choice.unsqueeze(-1)).squeeze(-1)
    else:
        prob_sum = probs.sum(dim=-1, keepdim=True)
        needs_fallback = ~torch.isfinite(prob_sum) | (prob_sum <= 0)
        if torch.any(needs_fallback):
            # prefer the highest logit in the current view when the distribution is degenerate
            fallback = torch.zeros_like(probs)
            fallback.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
            probs = torch.where(needs_fallback, fallback, probs)
            prob_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / prob_sum.clamp_min(torch.finfo(probs.dtype).eps)
        local_idx = torch.multinomial(probs, 1).squeeze(-1)

    if topk_idx is not None:
        local_idx = torch.gather(topk_idx, -1, local_idx.unsqueeze(-1)).squeeze(-1)
    return local_idx


def _continuous_batching(
    model: RWKVModelProtocol,
    tokenizer: TokenizerProtocol,
    prompts: Sequence[str],
    sampling: SamplingConfig,
    batch_size: int,
    progress_desc: str,
    probe_only: bool = False,
) -> list[GenerationOutput]:
    if not prompts:
        return []
    batch_size = max(1, min(batch_size, len(prompts)))

    sample_mode = (sampling.sample_mode or "normal").strip().lower()
    if sample_mode not in {"normal", "simple"}:
        print(f"⚠️ unknown sample_mode={sampling.sample_mode!r}; falling back to 'normal'.")
        sample_mode = "normal"
    is_simple = sample_mode == "simple"
    noise = float(sampling.noise or 0.0)
    if noise < 0:
        noise = 0.0

    def _torch_sample(logits_tensor: torch.Tensor, *, force_cpu: bool = False) -> torch.Tensor:
        if force_cpu and logits_tensor.is_cuda:
            logits_tensor = logits_tensor.cpu()
        try:
            return _torch_top_k_top_p(logits_tensor, sampling.top_k, sampling.top_p)
        except RuntimeError as exc:
            if logits_tensor.is_cuda:
                print(f"⚠️ torch sampling on CUDA failed: {exc}; retrying on CPU.")
                return _torch_top_k_top_p(logits_tensor.cpu(), sampling.top_k, sampling.top_p)
            raise

    encoded = deque()
    for idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        if sampling.pad_zero:
            tokens = [0] + tokens
        encoded.append((idx, prompt, tokens))

    vocab_size = _infer_vocab_size(model)
    device = _infer_device(model)
    states = _prepare_state_container(model.generate_zero_state(batch_size))

    stop_tokens = set(sampling.stop_tokens)
    ban_tokens = tuple(sampling.ban_tokens or ())
    no_penalty = set(sampling.no_penalty_token_ids)

    # Frequency/presence penalty tracking can be very expensive for large (batch, vocab).
    # Avoid allocating these tensors when they are not needed (simple mode / zero penalties).
    alpha_presence_value = 0.0 if is_simple else float(sampling.alpha_presence)
    alpha_frequency_value = 0.0 if is_simple else float(sampling.alpha_frequency)
    alpha_decay_value = 1.0 if is_simple else float(sampling.alpha_decay)
    use_penalties = (not is_simple) and (alpha_presence_value != 0.0 or alpha_frequency_value != 0.0)

    occurrence: torch.Tensor | None = None
    alpha_presence_vector: torch.Tensor | None = None
    alpha_presence: torch.Tensor | None = None
    if use_penalties:
        # Keep float32 so the post-penalty logits stay float32 (flashinfer expects float32 inputs).
        occurrence = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
        alpha_presence_vector = torch.zeros_like(occurrence)
        alpha_presence = torch.tensor(alpha_presence_value, dtype=torch.float32, device=device)

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
    flashinfer_ok = True
    start_time = time.time()
    tokens_generated = 0
    window_start_time = start_time
    window_start_tokens = 0
    throughput_ema: float | None = None

    def _reset_slot(slot_idx: int) -> None:
        if occurrence is not None:
            occurrence[slot_idx, :] = 0
        if alpha_presence_vector is not None:
            alpha_presence_vector[slot_idx, :] = 0
        states[0][:, :, slot_idx, :] = 0
        states[1][:, slot_idx, :, :, :] = 0

    def _remove_slot(remove_idx: int) -> None:
        last_idx = len(active_tasks) - 1
        if remove_idx != last_idx:
            states[0][:, :, remove_idx, :] = states[0][:, :, last_idx, :]
            states[1][:, remove_idx, :, :, :] = states[1][:, last_idx, :, :, :]
            if occurrence is not None:
                occurrence[remove_idx, :] = occurrence[last_idx, :]
            if alpha_presence_vector is not None:
                alpha_presence_vector[remove_idx, :] = alpha_presence_vector[last_idx, :]
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
                tokens_generated += 1
            if reached_stop or reached_length:
                outputs.append(
                    GenerationOutput(
                        prompt_index=task.prompt_index,
                        prompt=task.prompt,
                        token_ids=list(task.generated_tokens),
                        text="",
                        finish_reason="stop_token" if reached_stop else "max_length",
                    )
                )
                pbar.update(1)
                if encoded:
                    prompt_idx, prompt, tokens = encoded.popleft()
                    pending = deque(tokens)
                    active_tasks[idx] = _ActiveTask(prompt_idx, prompt, pending, [], None, None)
                    _reset_slot(idx)
                else:
                    accomplished.append(idx)
            else:
                if not is_simple and new_token not in no_penalty:
                    if occurrence is not None:
                        occurrence[idx, new_token] += 1.0
                    if alpha_presence_vector is not None and alpha_presence is not None:
                        alpha_presence_vector[idx, new_token] = alpha_presence

        if accomplished:
            for remove_idx in sorted(accomplished, reverse=True):
                _remove_slot(remove_idx)

        now = time.time()
        elapsed = max(now - start_time, 1e-6)
        window_elapsed = now - window_start_time
        if window_elapsed >= 0.5:
            recent_tokens = tokens_generated - window_start_tokens
            inst_throughput = recent_tokens / max(window_elapsed, 1e-6)
            throughput_ema = inst_throughput if throughput_ema is None else 0.9 * throughput_ema + 0.1 * inst_throughput
            window_start_time = now
            window_start_tokens = tokens_generated
        inst_display = throughput_ema if throughput_ema is not None else 0.0
        pbar.set_postfix_str(f"tok/s avg {tokens_generated / elapsed:.1f} cur {inst_display:.1f}")
        pbar.update(0)

        if not active_tasks:
            break

        next_tokens: list[list[int]] = []
        active_count = len(active_tasks)
        for task in active_tasks:
            token = task.pending_tokens.popleft()
            next_tokens.append([token])

        state_view = [
            states[0][:, :, :active_count, :],
            states[1][:, :active_count, :, :, :],
        ]
        out = model.forward_batch(next_tokens, state_view)

        if use_penalties and occurrence is not None:
            if alpha_decay_value != 1.0:
                occurrence[:active_count].mul_(alpha_decay_value)
            if alpha_frequency_value != 0.0:
                out = out - occurrence[:active_count] * alpha_frequency_value
            if alpha_presence_value != 0.0 and alpha_presence_vector is not None:
                out = out - alpha_presence_vector[:active_count]

        if ban_tokens:
            out[:, list(ban_tokens)] = -math.inf

        if sampling.temperature != 1.0:
            temp = sampling.temperature or 1.0
            out /= temp

        if is_simple:
            logits = out
            if noise:
                logits = logits + torch.empty_like(logits).uniform_(0.0, noise)
            new_tokens = torch.argmax(logits, dim=-1)
        elif flashinfer_ok:
            logits = out.float().contiguous()
            try:
                new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                    logits, sampling.top_k, sampling.top_p
                )
            except RuntimeError as exc:
                print(f"⚠️ flashinfer sampling failed: {exc}; falling back to torch sampling.")
                flashinfer_ok = False
                error_text = str(exc).lower()
                force_cpu = logits.is_cuda and ("illegal memory access" in error_text or "cuda error" in error_text)
                new_tokens = _torch_sample(logits, force_cpu=force_cpu)
        else:
            logits = out.float().contiguous()
            new_tokens = _torch_sample(logits)
        new_tokens = new_tokens.tolist()
        for idx, task in enumerate(active_tasks):
            task.new_token = new_tokens[idx]

    pbar.close()

    for output in outputs:
        tokens = list(output.token_ids)
        text = ""
        while tokens:
            try:
                text = tokenizer.decode(tokens)
                break
            except:
                tokens = tokens[:-1]
        output.text = text

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


__all__ = ["InferenceEngine", "GenerationOutput"]
