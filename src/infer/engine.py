from __future__ import annotations

"""RWKV 推理引擎，封装连续批量生成、state 管理等逻辑。"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

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
    ) -> list[GenerationOutput]:
        return _continuous_batching(self.model, self.tokenizer, prompts, sampling, batch_size, progress_desc)


@dataclass(slots=True)
class _ActiveTask:
    prompt_index: int
    prompt: str
    pending_tokens: list[int]
    state_pos: int
    generated_tokens: list[int]
    new_token: int | None
    finish_reason: str | None


def _torch_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Lightweight torch fallback for top-k/top-p sampling on CUDA."""

    use_top_k = top_k is not None and top_k > 0 and top_k < logits.size(-1)
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
) -> list[GenerationOutput]:
    if not prompts:
        return []
    batch_size = max(1, min(batch_size, len(prompts)))

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

    occurrence = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    alpha_presence_vector = torch.zeros_like(occurrence)
    alpha_presence = torch.tensor(sampling.alpha_presence, dtype=torch.float32, device=device)
    stop_tokens = set(sampling.stop_tokens)
    ban_tokens = tuple(sampling.ban_tokens or ())
    no_penalty = set(sampling.no_penalty_token_ids)

    active_tasks: list[_ActiveTask] = []
    for slot in range(batch_size):
        prompt_idx, prompt, tokens = encoded.popleft()
        active_tasks.append(_ActiveTask(prompt_idx, prompt, tokens, slot, [], None, None))

    pbar = tqdm(
        total=len(prompts),
        desc=progress_desc,
        unit=" sequence",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    outputs: list[GenerationOutput] = []
    flashinfer_ok = True

    while active_tasks:
        accomplished: list[int] = []
        slots_to_remove: set[int] = set()

        for idx, task in enumerate(active_tasks):
            if task.pending_tokens:
                continue
            new_token = task.new_token
            if new_token is None:
                continue
            reached_stop = new_token in stop_tokens
            reached_length = len(task.generated_tokens) >= sampling.max_generate_tokens
            if not reached_stop:
                task.pending_tokens.append(new_token)
                task.generated_tokens.append(new_token)
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
                    active_tasks[idx] = _ActiveTask(prompt_idx, prompt, tokens, task.state_pos, [], None, None)
                    occurrence[task.state_pos, :] = 0
                    alpha_presence_vector[task.state_pos, :] = 0
                    states[0][:, :, task.state_pos, :] = 0
                    states[1][:, task.state_pos, :, :] = 0
                else:
                    accomplished.append(idx)
                    slots_to_remove.add(task.state_pos)
            else:
                if new_token not in no_penalty:
                    occurrence[task.state_pos, new_token] += 1.0
                    alpha_presence_vector[task.state_pos, new_token] = alpha_presence

        if accomplished:
            for slot in sorted(slots_to_remove, reverse=True):
                states[0] = torch.cat([states[0][:, :, :slot, :], states[0][:, :, slot + 1 :, :]], dim=2)
                states[1] = torch.cat([states[1][:, :slot, :, :, :], states[1][:, slot + 1 :, :, :, :]], dim=1)
                occurrence = torch.cat([occurrence[:slot, :], occurrence[slot + 1 :, :]], dim=0)
                alpha_presence_vector = torch.cat(
                    [alpha_presence_vector[:slot, :], alpha_presence_vector[slot + 1 :, :]], dim=0
                )
            for idx in sorted(accomplished, reverse=True):
                del active_tasks[idx]
            state_positions = sorted({task.state_pos for task in active_tasks})
            pos_map = {old: new for new, old in enumerate(state_positions)}
            for task in active_tasks:
                task.state_pos = pos_map[task.state_pos]

        if not active_tasks:
            break

        max_state_idx = max(task.state_pos for task in active_tasks)
        next_tokens: list[list[int] | None] = [None] * (max_state_idx + 1)
        for task in active_tasks:
            token = task.pending_tokens.pop(0)
            next_tokens[task.state_pos] = [token]

        out = model.forward_batch(next_tokens, states)
        occurrence *= sampling.alpha_decay
        out = out - alpha_presence_vector - occurrence * sampling.alpha_frequency

        if ban_tokens:
            out[:, list(ban_tokens)] = -math.inf

        if sampling.temperature != 1.0:
            temp = sampling.temperature or 1.0
            out /= temp

        logits = out.float().contiguous()
        if flashinfer_ok:
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
            new_tokens = _torch_sample(logits)
        new_tokens = new_tokens.tolist()
        for task in active_tasks:
            task.new_token = new_tokens[task.state_pos]

    pbar.close()

    for output in outputs:
        tokens = list(output.token_ids)
        text = ""
        while tokens:
            try:
                text = tokenizer.decode(tokens)
                break
            except Exception:
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
