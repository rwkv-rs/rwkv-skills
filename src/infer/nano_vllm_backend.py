from __future__ import annotations

"""Inference backend backed by the external nano-vLLM RWKV engine."""

import importlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Literal, Sequence

from tqdm import tqdm

from .constraints import DecodeConstraint
from .engine import DEFAULT_PREFILL_CHUNK_SIZE
from .sampling import GeneratedTextDelta, GenerationOutput, SamplingConfig


DEFAULT_NANO_VLLM_PATH = Path("/tmp/nano-vllm-rwkv-315cf53")


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
class NanoVLLMInferenceBackend:
    config: NanoVLLMBackendConfig
    runtime: _NanoVLLMRuntime
    engine: object
    scheduler: object
    model_runner: object
    _lock: threading.Lock

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
        return cls(
            config=config,
            runtime=runtime,
            engine=engine,
            scheduler=scheduler,
            model_runner=model_runner,
            _lock=threading.Lock(),
        )

    @property
    def model_name(self) -> str:
        raw_name = self.config.model_name
        if raw_name:
            return str(raw_name)
        return Path(str(self.config.model_path)).stem

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
        outputs: list[GenerationOutput | None] = [None] * len(prompts)
        max_batch = max(1, min(int(batch_size), len(prompts)))
        progress = tqdm(total=len(prompts), desc=progress_desc, unit=" prompt", disable=not show_progress)
        try:
            with self._lock:
                for start in range(0, len(prompts), max_batch):
                    stop = min(start + max_batch, len(prompts))
                    self._generate_chunk(
                        start=start,
                        prompts=prompts[start:stop],
                        sampling=effective_sampling,
                        prompt_stop_suffixes=(
                            None if prompt_stop_suffixes is None else prompt_stop_suffixes[start:stop]
                        ),
                        outputs=outputs,
                        on_complete=on_complete if not probe_only else None,
                        on_token=on_token,
                        progress=progress,
                    )
        finally:
            progress.close()
        return [output for output in outputs if output is not None]

    def _generate_chunk(
        self,
        *,
        start: int,
        prompts: Sequence[str],
        sampling: SamplingConfig,
        prompt_stop_suffixes: Sequence[Sequence[str] | None] | None,
        outputs: list[GenerationOutput | None],
        on_complete: Callable[[GenerationOutput], None] | None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None,
        progress: tqdm,
    ) -> None:
        seq_records: dict[int, tuple[int, str, object, Sequence[str] | None]] = {}
        sampling_params = self._to_sampling_params(sampling)
        for local_index, prompt in enumerate(prompts):
            seq = self.engine.add_request(prompt, sampling_params)
            stop_suffixes = None if prompt_stop_suffixes is None else prompt_stop_suffixes[local_index]
            stop_token_seqs = self._build_stop_token_seqs(sampling=sampling, stop_suffixes=stop_suffixes)
            if stop_token_seqs:
                setattr(seq, "stop_token_seqs", stop_token_seqs)
            seq_records[int(getattr(seq, "seq_id"))] = (start + local_index, prompt, seq, stop_suffixes)

        while not self.engine.is_finished():
            raw_outputs, _num_tokens = self.engine.step()
            for seq_id, token_ids in raw_outputs:
                prompt_index, prompt, seq, stop_suffixes = seq_records[int(seq_id)]
                text = self._decode_token_ids(token_ids)
                text, trimmed = _trim_at_stop_suffix(text, stop_suffixes)
                output = GenerationOutput(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    token_ids=[int(token_id) for token_id in token_ids],
                    text=text,
                    finish_reason=_finish_reason(seq, trimmed=trimmed),
                )
                outputs[prompt_index] = output
                if on_token is not None and output.text:
                    on_token(prompt_index, GeneratedTextDelta(text=output.text, tokens=list(output.tokens)))
                if on_complete is not None:
                    on_complete(output)
                progress.update(1)

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
        return str(decode(list(token_ids)))

    def score_choice_tokens(
        self,
        *,
        prompt: str,
        choice_token_texts: Sequence[str],
    ) -> tuple[dict[str, float], str]:
        _ = prompt
        _ = choice_token_texts
        raise NotImplementedError("nano-vLLM backend does not support candidate choice scoring")

    def shutdown(self) -> None:
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
