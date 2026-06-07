from __future__ import annotations

"""Shared local/remote inference backends for evaluation pipelines."""

import argparse
import concurrent.futures
import http.client
import json
import socket
import time
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch
from tqdm import tqdm

from .constraints import DecodeConstraint
from .engine import DEFAULT_PREFILL_CHUNK_SIZE, TokenizerProtocol
from .lightning_engine import LocalEngineProtocol, build_local_engine
from .sampling import GeneratedTextDelta, GenerationOutput, SamplingConfig

if TYPE_CHECKING:
    from .model import ModelLoadConfig


class RemoteHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"remote infer request failed: HTTP {status_code}: {detail}")
        self.status_code = int(status_code)
        self.detail = str(detail)


_REMOTE_TRANSIENT_ERRORS = (
    urllib_error.URLError,
    http.client.HTTPException,
    TimeoutError,
    ConnectionError,
    socket.timeout,
)

RemoteInferenceProtocol = Literal["openai", "vllm", "nano-vllm-contents"]
RemoteInferenceSeedPolicy = Literal["preserve", "omit-for-contents"]
REMOTE_INFERENCE_PROTOCOL_CHOICES: tuple[RemoteInferenceProtocol, ...] = ("openai", "vllm")
REMOTE_INFERENCE_LEGACY_PROTOCOL_CHOICES: tuple[RemoteInferenceProtocol, ...] = ("nano-vllm-contents",)
REMOTE_INFERENCE_ALL_PROTOCOL_CHOICES: tuple[RemoteInferenceProtocol, ...] = (
    *REMOTE_INFERENCE_PROTOCOL_CHOICES,
    *REMOTE_INFERENCE_LEGACY_PROTOCOL_CHOICES,
)
REMOTE_INFERENCE_SEED_POLICY_CHOICES: tuple[RemoteInferenceSeedPolicy, ...] = (
    "preserve",
    "omit-for-contents",
)


def normalize_api_base(base_url: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("infer base URL cannot be empty")
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def normalize_local_device(device: str) -> str:
    raw = str(device or "").strip() or "cuda"
    parsed = torch.device(raw)
    if parsed.type == "cuda" and parsed.index is None:
        return "cuda:0"
    return str(parsed)


def _safe_tqdm_update(progress: tqdm, amount: int = 1) -> None:
    try:
        progress.update(amount)
    except (AttributeError, ValueError):
        return


def _safe_tqdm_close(progress: tqdm) -> None:
    try:
        progress.close()
    except (AttributeError, ValueError):
        return


def add_inference_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", help="Path to RWKV weights (.pth)")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--engine-mode",
        choices=("classic", "lightning"),
        default="classic",
        help="Local inference engine implementation to use",
    )
    parser.add_argument(
        "--state-db-path",
        help="Path to the local sqlite state cache database used by the lightning engine",
    )
    parser.add_argument("--infer-base-url", help="OpenAI-compatible infer service base URL")
    parser.add_argument("--infer-model", help="Model name exposed by the remote infer service")
    parser.add_argument("--infer-api-key", default="", help="API key for the remote infer service")
    parser.add_argument("--infer-timeout-s", type=float, default=600.0, help="Timeout for remote infer requests")
    parser.add_argument(
        "--infer-max-workers",
        type=int,
        default=32,
        help="Max concurrent HTTP workers used by the eval-side remote client",
    )
    parser.add_argument(
        "--infer-protocol",
        choices=REMOTE_INFERENCE_PROTOCOL_CHOICES,
        default="openai",
        help="Remote request protocol: generic OpenAI compatibility, or vLLM-tuned OpenAI chat requests",
    )
    parser.add_argument(
        "--infer-seed-policy",
        choices=REMOTE_INFERENCE_SEED_POLICY_CHOICES,
        default="preserve",
        help=(
            "Remote seed handling. preserve keeps per-prompt seeds, falling back to OpenAI requests when needed; "
            "omit-for-contents drops seeds only for nano-vLLM contents batching."
        ),
    )


def validate_inference_backend_args(args: argparse.Namespace) -> None:
    model_path = str(getattr(args, "model_path", "") or "").strip()
    infer_base_url = str(getattr(args, "infer_base_url", "") or "").strip()
    infer_model = str(getattr(args, "infer_model", "") or "").strip()
    has_local = bool(model_path)
    has_remote = bool(infer_base_url or infer_model)
    if has_local and has_remote:
        raise ValueError("请二选一：使用本地 --model-path，或远端 --infer-base-url/--infer-model。")
    if not has_local and not has_remote:
        raise ValueError("必须提供 --model-path，或同时提供 --infer-base-url 和 --infer-model。")
    if has_remote and not infer_base_url:
        raise ValueError("远端推理模式缺少 --infer-base-url。")
    if has_remote and not infer_model:
        raise ValueError("远端推理模式缺少 --infer-model。")


def resolve_backend_model_name(args: argparse.Namespace) -> str:
    validate_inference_backend_args(args)
    infer_model = str(getattr(args, "infer_model", "") or "").strip()
    if infer_model:
        return infer_model
    model_path = str(getattr(args, "model_path", "") or "").strip()
    return Path(model_path).stem


class InferenceBackend(Protocol):
    model_name: str

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
        ...

    def score_choice_tokens(
        self,
        *,
        prompt: str,
        choice_token_texts: Sequence[str],
    ) -> tuple[dict[str, float], str]:
        ...


@dataclass(slots=True)
class LocalInferenceBackend:
    model_name: str
    model: object
    tokenizer: TokenizerProtocol
    engine: LocalEngineProtocol
    engine_mode: str = "classic"

    @classmethod
    def from_model_config(
        cls,
        config: ModelLoadConfig,
        *,
        engine_mode: str = "classic",
        state_db_path: str | None = None,
    ) -> LocalInferenceBackend:
        from .model import load_rwkv_model

        normalized_device = normalize_local_device(str(getattr(config, "device", "cuda") or "cuda"))
        if getattr(config, "device", None) != normalized_device:
            if is_dataclass(config):
                config = replace(config, device=normalized_device)
            else:
                setattr(config, "device", normalized_device)
        model, tokenizer = load_rwkv_model(config)
        model_name = Path(config.weights_path).stem
        return cls(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            engine=build_local_engine(model, tokenizer, mode=engine_mode, state_db_path=state_db_path),
            engine_mode=engine_mode,
        )

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
        effective_constraints = _resolve_effective_constraints(
            constraints=constraints,
            constraint_mode=constraint_mode,
        )
        return self.engine.generate(
            prompts,
            sampling=sampling,
            batch_size=batch_size,
            prefill_chunk_size=prefill_chunk_size,
            progress_desc=progress_desc,
            probe_only=probe_only,
            on_complete=on_complete,
            on_token=on_token,
            prompt_stop_suffixes=prompt_stop_suffixes,
            prompt_constraints=effective_constraints,
            prompt_seeds=prompt_seeds,
            top_logprobs=top_logprobs,
            show_progress=show_progress,
        )

    def score_choice_tokens(
        self,
        *,
        prompt: str,
        choice_token_texts: Sequence[str],
    ) -> tuple[dict[str, float], str]:
        if not choice_token_texts:
            raise ValueError("choice_token_texts cannot be empty")
        tokens = [0] + list(self.tokenizer.encode(prompt.strip()))
        state = _blank_state(self.model)
        with torch.no_grad():
            logits = self.model.forward(tokens, state, full_output=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.to(torch.float32)
        choice_token_ids = [_single_token_id(self.tokenizer, token_text) for token_text in choice_token_texts]
        slice_values = logits[choice_token_ids]
        scores = {
            token_text: float(value)
            for token_text, value in zip(choice_token_texts, slice_values.cpu(), strict=True)
        }
        pred_idx = int(torch.argmax(slice_values).item())
        return scores, choice_token_texts[pred_idx]

    def shutdown(self) -> None:
        self.engine.shutdown()


@dataclass(slots=True, frozen=True)
class RemoteInferenceConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout_s: float = 600.0
    max_workers: int = 32
    max_retries: int = 3
    retry_initial_delay_s: float = 1.0
    retry_max_delay_s: float = 10.0
    prefer_chat_completions: bool = True
    protocol: RemoteInferenceProtocol = "openai"
    seed_policy: RemoteInferenceSeedPolicy = "preserve"
    request_latency_callback: Callable[[str, float, bool], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def completions_url(self) -> str:
        return f"{normalize_api_base(self.base_url)}/completions"

    def chat_completions_url(self) -> str:
        return f"{normalize_api_base(self.base_url)}/chat/completions"


@dataclass(slots=True)
class RemoteInferenceBackend:
    config: RemoteInferenceConfig
    _legacy_choice_scoring_supported: bool | None = field(default=None, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.config.model

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
        effective_constraints = _resolve_effective_constraints(
            constraints=constraints,
            constraint_mode=constraint_mode,
        )
        if effective_constraints is not None and any(constraint is not None for constraint in effective_constraints):
            raise NotImplementedError("remote infer backend does not support prompt constraints")
        if not prompts:
            return []
        if prompt_seeds is not None and len(prompt_seeds) != len(prompts):
            raise ValueError("prompt_seeds length must match prompts length")
        if prompt_stop_suffixes is not None and len(prompt_stop_suffixes) != len(prompts):
            raise ValueError("prompt_stop_suffixes length must match prompts length")
        effective_sampling = sampling.clamp(1) if probe_only else sampling
        force_openai_single_requests = False
        use_contents_protocol = self.config.protocol == "nano-vllm-contents"
        if prompt_seeds is not None:
            has_prompt_seeds = any(seed is not None for seed in prompt_seeds)
            if use_contents_protocol and has_prompt_seeds and self.config.seed_policy == "omit-for-contents":
                prompt_seeds = None
            elif has_prompt_seeds:
                force_openai_single_requests = True
            else:
                prompt_seeds = None
        if use_contents_protocol and not force_openai_single_requests:
            return self._generate_contents_batches(
                prompts,
                sampling=effective_sampling,
                batch_size=batch_size,
                progress_desc=progress_desc,
                probe_only=probe_only,
                on_complete=on_complete,
                on_token=on_token,
                prompt_stop_suffixes=prompt_stop_suffixes,
                show_progress=show_progress,
            )
        outputs: list[GenerationOutput | None] = [None] * len(prompts)
        max_workers = max(1, min(int(batch_size), int(self.config.max_workers), len(prompts)))
        progress = tqdm(total=len(prompts), desc=progress_desc, unit=" request", disable=not show_progress)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        self._generate_one,
                        prompt_index,
                        prompt,
                        effective_sampling,
                        None if prompt_seeds is None else int(prompt_seeds[prompt_index]),
                        None if prompt_stop_suffixes is None else prompt_stop_suffixes[prompt_index],
                        prefill_chunk_size,
                    ): prompt_index
                    for prompt_index, prompt in enumerate(prompts)
                }
                for future in concurrent.futures.as_completed(future_map):
                    output = future.result()
                    outputs[output.prompt_index] = output
                    if on_token is not None and output.text:
                        on_token(output.prompt_index, GeneratedTextDelta(text=output.text, tokens=list(output.tokens)))
                    if on_complete is not None and not probe_only:
                        on_complete(output)
                    _safe_tqdm_update(progress, 1)
        finally:
            _safe_tqdm_close(progress)
        return [output for output in outputs if output is not None]

    def _generate_contents_batches(
        self,
        prompts: Sequence[str],
        *,
        sampling: SamplingConfig,
        batch_size: int,
        progress_desc: str,
        probe_only: bool,
        on_complete: Callable[[GenerationOutput], None] | None,
        on_token: Callable[[int, GeneratedTextDelta], None] | None,
        prompt_stop_suffixes: Sequence[Sequence[str] | None] | None,
        show_progress: bool,
    ) -> list[GenerationOutput]:
        outputs: list[GenerationOutput | None] = [None] * len(prompts)
        max_batch = max(1, min(int(batch_size), len(prompts)))
        progress = tqdm(total=len(prompts), desc=progress_desc, unit=" request", disable=not show_progress)
        try:
            for indices, stop_suffixes in _iter_contents_batches(
                prompt_count=len(prompts),
                max_batch=max_batch,
                prompt_stop_suffixes=prompt_stop_suffixes,
            ):
                batch_outputs = self._generate_contents_batch(
                    indices=indices,
                    prompts=[prompts[index] for index in indices],
                    sampling=sampling,
                    stop_suffixes=stop_suffixes,
                )
                for output in batch_outputs:
                    outputs[output.prompt_index] = output
                    if on_token is not None and output.text:
                        on_token(output.prompt_index, GeneratedTextDelta(text=output.text, tokens=list(output.tokens)))
                    if on_complete is not None and not probe_only:
                        on_complete(output)
                _safe_tqdm_update(progress, len(indices))
        finally:
            _safe_tqdm_close(progress)
        return [output for output in outputs if output is not None]

    def _generate_contents_batch(
        self,
        *,
        indices: Sequence[int],
        prompts: Sequence[str],
        sampling: SamplingConfig,
        stop_suffixes: tuple[str, ...],
    ) -> list[GenerationOutput]:
        payload: dict[str, object] = {
            "model": self.model_name,
            "contents": list(prompts),
            "max_tokens": int(sampling.max_generate_tokens),
            "temperature": float(sampling.temperature),
            "top_k": int(sampling.top_k),
            "top_p": float(sampling.top_p),
            "alpha_presence": float(sampling.alpha_presence),
            "alpha_frequency": float(sampling.alpha_frequency),
            "alpha_decay": float(sampling.alpha_decay),
            "pad_zero": bool(sampling.pad_zero),
            "stream": False,
            "stop_tokens": list(stop_suffixes),
        }
        response = self._post_json(self.config.chat_completions_url(), payload)
        choices = response.get("choices")
        if not isinstance(choices, list):
            raise RuntimeError("remote infer response missing choices")
        by_index: dict[int, dict[str, object]] = {}
        for choice in choices:
            if not isinstance(choice, dict):
                raise RuntimeError("remote infer response choice format is invalid")
            raw_index = choice.get("index", len(by_index))
            try:
                choice_index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("remote infer response choice index is invalid") from exc
            by_index[choice_index] = choice
        if len(by_index) != len(indices):
            raise RuntimeError("remote infer response choice count does not match contents batch")

        outputs: list[GenerationOutput] = []
        for local_index, prompt_index in enumerate(indices):
            choice = by_index.get(local_index)
            if choice is None:
                raise RuntimeError("remote infer response missing contents batch choice")
            text = _extract_chat_choice_text(choice)
            outputs.append(
                GenerationOutput(
                    prompt_index=int(prompt_index),
                    prompt=prompts[local_index],
                    token_ids=[],
                    text=text,
                    finish_reason=_normalize_remote_finish_reason(choice.get("finish_reason")),
                )
            )
        return outputs

    def score_choice_tokens(
        self,
        *,
        prompt: str,
        choice_token_texts: Sequence[str],
    ) -> tuple[dict[str, float], str]:
        if not choice_token_texts:
            raise ValueError("choice_token_texts cannot be empty")
        if self.config.protocol == "vllm":
            self._legacy_choice_scoring_supported = False
            raise NotImplementedError("vllm remote protocol does not support candidate choice scoring")
        if self._legacy_choice_scoring_supported is False:
            raise NotImplementedError("remote infer service does not support candidate choice scoring")
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 1,
            "candidate_token_texts": list(choice_token_texts),
        }
        try:
            response = self._post_json(self.config.completions_url(), payload)
        except RemoteHTTPError as exc:
            if exc.status_code in {400, 404, 405, 422, 501}:
                self._legacy_choice_scoring_supported = False
                raise NotImplementedError(
                    "remote infer service does not support candidate choice scoring"
                ) from exc
            raise
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("remote infer response missing choices")
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise RuntimeError("remote infer response choice format is invalid")
        logprobs = choice0.get("logprobs")
        if not isinstance(logprobs, dict):
            self._legacy_choice_scoring_supported = False
            raise NotImplementedError("remote infer response missing choice-scoring logprobs")
        top_logprobs = logprobs.get("top_logprobs")
        if not isinstance(top_logprobs, list) or not top_logprobs or not isinstance(top_logprobs[0], dict):
            self._legacy_choice_scoring_supported = False
            raise NotImplementedError("remote infer response missing choice-scoring top_logprobs")
        self._legacy_choice_scoring_supported = True
        scores = {str(key): float(value) for key, value in top_logprobs[0].items()}
        best_text = max(choice_token_texts, key=lambda item: scores.get(item, float("-inf")))
        return scores, best_text

    def _generate_one(
        self,
        prompt_index: int,
        prompt: str,
        sampling: SamplingConfig,
        seed: int | None,
        stop_suffixes: Sequence[str] | None,
        prefill_chunk_size: int,
    ) -> GenerationOutput:
        include_private_fields = self.config.protocol == "nano-vllm-contents"
        payload = _completion_payload_from_sampling(
            model=self.model_name,
            prompt=prompt,
            sampling=sampling,
            seed=seed,
            stop_suffixes=stop_suffixes,
            prefill_chunk_size=prefill_chunk_size,
            include_private_fields=include_private_fields,
        )
        chat_payload = _chat_payload_from_completion_payload(
            payload,
            prompt,
            include_private_fields=include_private_fields,
        )
        if self.config.protocol == "vllm":
            response = self._post_json(self.config.chat_completions_url(), chat_payload)
            is_chat_response = True
        elif self.config.prefer_chat_completions:
            try:
                response = self._post_json(self.config.chat_completions_url(), chat_payload)
                is_chat_response = True
            except RemoteHTTPError as exc:
                if exc.status_code not in {404, 405}:
                    raise
                response = self._post_json(self.config.completions_url(), payload)
                is_chat_response = False
        else:
            try:
                response = self._post_json(self.config.completions_url(), payload)
                is_chat_response = False
            except RemoteHTTPError as exc:
                if exc.status_code not in {404, 405}:
                    raise
                response = self._post_json(self.config.chat_completions_url(), chat_payload)
                is_chat_response = True
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("remote infer response missing choices")
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise RuntimeError("remote infer response choice format is invalid")
        text = _extract_chat_choice_text(choice0) if is_chat_response else _extract_completion_choice_text(choice0)
        return GenerationOutput(
            prompt_index=prompt_index,
            prompt=prompt,
            token_ids=[],
            text=text,
            finish_reason=_normalize_remote_finish_reason(choice0.get("finish_reason")),
        )

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        ok = False
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key or 'rwkv-skills'}",
            },
        )
        try:
            raw = self._urlopen_with_retries(req)
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise RuntimeError("remote infer response must be a JSON object")
            ok = True
            return data
        finally:
            callback = self.config.request_latency_callback
            if callback is not None:
                callback(str(url), max(0.0, time.perf_counter() - started), ok)

    def _urlopen_with_retries(self, req: urllib_request.Request) -> str:
        attempts = max(1, int(self.config.max_retries) + 1)
        timeout_s = max(float(self.config.timeout_s), 1.0)
        delay_s = max(float(self.config.retry_initial_delay_s), 0.0)
        max_delay_s = max(float(self.config.retry_max_delay_s), delay_s)
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                with urllib_request.urlopen(req, timeout=timeout_s) as resp:
                    return resp.read().decode("utf-8")
            except urllib_error.HTTPError as exc:  # pragma: no cover - exercised through integration
                detail = exc.read().decode("utf-8", errors="replace")
                raise RemoteHTTPError(exc.code, detail) from exc
            except _REMOTE_TRANSIENT_ERRORS as exc:  # pragma: no cover - exercised through integration
                last_exc = exc
                if attempt >= attempts:
                    break
                if delay_s > 0:
                    time.sleep(delay_s)
                    delay_s = min(delay_s * 2, max_delay_s)
        if isinstance(last_exc, urllib_error.URLError):
            raise RuntimeError(f"remote infer request failed after {attempts} attempts: {last_exc.reason}") from last_exc
        raise RuntimeError(f"remote infer request failed after {attempts} attempts: {last_exc}") from last_exc


def _completion_payload_from_sampling(
    *,
    model: str,
    prompt: str,
    sampling: SamplingConfig,
    seed: int | None,
    stop_suffixes: Sequence[str] | None,
    prefill_chunk_size: int,
    include_private_fields: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(sampling.max_generate_tokens),
        "temperature": float(sampling.temperature),
        "top_p": float(sampling.top_p),
        "presence_penalty": float(sampling.alpha_presence),
        "frequency_penalty": float(sampling.alpha_frequency),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if stop_suffixes:
        payload["stop"] = list(stop_suffixes)
    if include_private_fields:
        payload.update(
            {
                "top_k": int(sampling.top_k),
                "repetition_penalty": float(sampling.alpha_frequency),
                "penalty_decay": float(sampling.alpha_decay),
                "stop_tokens": [int(token_id) for token_id in sampling.stop_tokens],
                "ban_tokens": [int(token_id) for token_id in sampling.ban_tokens or ()],
                "pad_zero": bool(sampling.pad_zero),
                "no_penalty_token_ids": [int(token_id) for token_id in sampling.no_penalty_token_ids],
                "prefill_chunk_size": int(prefill_chunk_size),
            }
        )
    return payload


def _chat_payload_from_completion_payload(
    payload: dict[str, object],
    prompt: str,
    *,
    include_private_fields: bool = True,
) -> dict[str, object]:
    chat_payload: dict[str, object] = {
        "model": payload["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": payload["max_tokens"],
        "temperature": float(payload.get("temperature", 0.0) or 0.0),
        "stream": False,
    }
    if "top_p" in payload:
        chat_payload["top_p"] = payload["top_p"]
    for key in ("presence_penalty", "frequency_penalty"):
        if key in payload:
            chat_payload[key] = payload[key]
    if "seed" in payload:
        chat_payload["seed"] = payload["seed"]
    if "stop" in payload:
        chat_payload["stop"] = payload["stop"]
    if include_private_fields:
        if "top_k" in payload:
            chat_payload["top_k"] = payload["top_k"]
        for key in (
            "repetition_penalty",
            "penalty_decay",
            "ban_tokens",
            "pad_zero",
            "no_penalty_token_ids",
            "prefill_chunk_size",
        ):
            if key in payload:
                chat_payload[key] = payload[key]
        if "stop_tokens" in payload:
            # The current RWKV chat serving schema accepts token ids as strings,
            # while the local/legacy completion path keeps them as ints.
            chat_payload["stop_tokens"] = [str(token_id) for token_id in payload["stop_tokens"]]  # type: ignore[index]
    return chat_payload


def _extract_completion_choice_text(choice: dict[str, object]) -> str:
    text = choice.get("text")
    if isinstance(text, str):
        return text
    raise RuntimeError("remote infer response missing completion text")


def _extract_chat_choice_text(choice: dict[str, object]) -> str:
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("remote infer response missing chat message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "".join(parts)
    raise RuntimeError("remote infer chat message content format is invalid")


def _normalize_remote_finish_reason(finish_reason: object) -> str:
    value = str(finish_reason or "stop")
    mapping: dict[str, Literal["stop_token", "max_length"]] = {
        "stop": "stop_token",
        "length": "max_length",
    }
    return mapping.get(value, value)


def build_inference_backend_from_args(args: argparse.Namespace) -> InferenceBackend:
    validate_inference_backend_args(args)
    infer_base_url = str(getattr(args, "infer_base_url", "") or "").strip()
    if infer_base_url:
        return RemoteInferenceBackend(
            RemoteInferenceConfig(
                base_url=infer_base_url,
                model=str(getattr(args, "infer_model", "") or "").strip(),
                api_key=str(getattr(args, "infer_api_key", "") or ""),
                timeout_s=float(getattr(args, "infer_timeout_s", 600.0) or 600.0),
                max_workers=max(1, int(getattr(args, "infer_max_workers", 32) or 32)),
                protocol=_normalize_remote_protocol(getattr(args, "infer_protocol", "openai")),
                seed_policy=_normalize_remote_seed_policy(getattr(args, "infer_seed_policy", "preserve")),
            )
        )
    model_path = str(getattr(args, "model_path", "") or "").strip()
    from .model import ModelLoadConfig

    return LocalInferenceBackend.from_model_config(
        ModelLoadConfig(
            weights_path=model_path,
            device=str(getattr(args, "device", "cuda") or "cuda"),
        ),
        engine_mode=str(getattr(args, "engine_mode", "classic") or "classic"),
        state_db_path=(
            None
            if getattr(args, "state_db_path", None) in (None, "")
            else str(getattr(args, "state_db_path"))
        ),
    )


def _blank_state(model: object):
    try:
        return model.generate_zero_state(0)
    except TypeError:
        return model.generate_zero_state()


def _single_token_id(tokenizer: TokenizerProtocol, text: str) -> int:
    token_ids = list(tokenizer.encode(text))
    if len(token_ids) != 1:
        raise ValueError(f"candidate token text {text!r} must map to a single token, got {token_ids}")
    return int(token_ids[0])


def _normalize_constraint_mode(mode: str | None) -> Literal["off", "soft", "strict"]:
    normalized = str(mode or "off").strip().lower()
    if normalized not in {"off", "soft", "strict"}:
        raise ValueError("constraint_mode must be one of: off, soft, strict")
    return normalized  # type: ignore[return-value]


def _normalize_remote_protocol(protocol: object) -> RemoteInferenceProtocol:
    value = str(protocol or "openai").strip().lower().replace("_", "-")
    if value in {"vllm-openai", "vllm-chat", "vllm-compatible"}:
        value = "vllm"
    if value in {"nano-vllm", "nanovllm", "contents", "lightning"}:
        value = "nano-vllm-contents"
    if value not in REMOTE_INFERENCE_ALL_PROTOCOL_CHOICES:
        choices = ", ".join(REMOTE_INFERENCE_PROTOCOL_CHOICES)
        raise ValueError(f"infer_protocol must be one of: {choices}")
    return value  # type: ignore[return-value]


def _normalize_remote_seed_policy(seed_policy: object) -> RemoteInferenceSeedPolicy:
    value = str(seed_policy or "preserve").strip().lower().replace("_", "-")
    aliases = {
        "keep": "preserve",
        "keep-seeds": "preserve",
        "preserve-seeds": "preserve",
        "omit": "omit-for-contents",
        "drop": "omit-for-contents",
        "drop-for-contents": "omit-for-contents",
        "omit-seeds": "omit-for-contents",
    }
    value = aliases.get(value, value)
    if value not in REMOTE_INFERENCE_SEED_POLICY_CHOICES:
        choices = ", ".join(REMOTE_INFERENCE_SEED_POLICY_CHOICES)
        raise ValueError(f"infer_seed_policy must be one of: {choices}")
    return value  # type: ignore[return-value]


def _contents_stop_key(
    prompt_stop_suffixes: Sequence[Sequence[str] | None] | None,
    index: int,
) -> tuple[str, ...]:
    if prompt_stop_suffixes is None:
        return ()
    raw = prompt_stop_suffixes[index]
    if raw is None:
        return ()
    return tuple(str(item) for item in raw if str(item))


def _iter_contents_batches(
    *,
    prompt_count: int,
    max_batch: int,
    prompt_stop_suffixes: Sequence[Sequence[str] | None] | None,
) -> list[tuple[tuple[int, ...], tuple[str, ...]]]:
    batches: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    start = 0
    while start < prompt_count:
        stop_key = _contents_stop_key(prompt_stop_suffixes, start)
        indices = [start]
        next_index = start + 1
        while next_index < prompt_count and len(indices) < max_batch:
            if _contents_stop_key(prompt_stop_suffixes, next_index) != stop_key:
                break
            indices.append(next_index)
            next_index += 1
        batches.append((tuple(indices), stop_key))
        start = next_index
    return batches


def _resolve_effective_constraints(
    *,
    constraints: Sequence[DecodeConstraint | None] | None,
    constraint_mode: str | None,
) -> Sequence[DecodeConstraint | None] | None:
    mode = _normalize_constraint_mode(constraint_mode)
    if mode == "off":
        return None
    return constraints


__all__ = [
    "InferenceBackend",
    "LocalInferenceBackend",
    "REMOTE_INFERENCE_PROTOCOL_CHOICES",
    "REMOTE_INFERENCE_LEGACY_PROTOCOL_CHOICES",
    "REMOTE_INFERENCE_ALL_PROTOCOL_CHOICES",
    "REMOTE_INFERENCE_SEED_POLICY_CHOICES",
    "RemoteInferenceProtocol",
    "RemoteInferenceSeedPolicy",
    "RemoteInferenceBackend",
    "RemoteInferenceConfig",
    "add_inference_backend_arguments",
    "build_inference_backend_from_args",
    "normalize_api_base",
    "normalize_local_device",
    "resolve_backend_model_name",
    "validate_inference_backend_args",
]
