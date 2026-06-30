from __future__ import annotations

"""Shared remote inference backend for evaluation pipelines."""

import argparse
import concurrent.futures
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, Sequence

import httpx
from tqdm import tqdm

from .constraints import DecodeConstraint
from .sampling import GeneratedTextDelta, GenerationOutput, SamplingConfig


class RemoteHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"remote infer request failed: HTTP {status_code}: {detail}")
        self.status_code = int(status_code)
        self.detail = str(detail)


_REMOTE_TRANSIENT_ERRORS = (httpx.RequestError,)
_RETRYABLE_REMOTE_HTTP_STATUS_CODES = frozenset({408, 429, 502, 503, 504})
DEFAULT_PREFILL_CHUNK_SIZE = 16
DEFAULT_REMOTE_MAX_WORKERS = 128

RemoteInferenceProtocol = Literal["openai", "vllm", "completions"]
RemoteInferenceSeedPolicy = Literal["preserve", "omit"]
REMOTE_INFERENCE_PROTOCOL_CHOICES: tuple[RemoteInferenceProtocol, ...] = (
    "openai",
    "vllm",
    "completions",
)
REMOTE_INFERENCE_SEED_POLICY_CHOICES: tuple[RemoteInferenceSeedPolicy, ...] = (
    "preserve",
    "omit",
)


def normalize_api_root(base_url: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("infer base URL cannot be empty")
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    if base.endswith("/v1") or base.endswith("/v2"):
        return base.rsplit("/", 1)[0]
    return base


def normalize_api_base_for_version(base_url: str, version: Literal["v1", "v2"]) -> str:
    return f"{normalize_api_root(base_url)}/{version}"


def normalize_api_base(base_url: str) -> str:
    return normalize_api_base_for_version(base_url, "v1")


def normalize_local_device(device: str) -> str:
    import torch

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


def _is_retryable_remote_http_error(exc: RemoteHTTPError) -> bool:
    return int(exc.status_code) in _RETRYABLE_REMOTE_HTTP_STATUS_CODES


def add_inference_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", help="Deprecated; local model loading has been removed from this branch")
    parser.add_argument("--device", default="cuda", help="Deprecated compatibility flag")
    parser.add_argument(
        "--engine-mode",
        choices=("vllm-rwkv",),
        default="vllm-rwkv",
        help="Deprecated compatibility flag; vllm-rwkv is the only supported engine",
    )
    parser.add_argument(
        "--state-db-path",
        help="Deprecated compatibility flag; local Lightning state caches have been removed",
    )
    parser.add_argument("--infer-base-url", help="OpenAI-compatible infer service base URL")
    parser.add_argument("--infer-model", help="Model name exposed by the remote infer service")
    parser.add_argument("--infer-api-key", default="", help="API key for the remote infer service")
    parser.add_argument("--infer-timeout-s", type=float, default=600.0, help="Timeout for remote infer requests")
    parser.add_argument(
        "--infer-max-workers",
        type=int,
        default=DEFAULT_REMOTE_MAX_WORKERS,
        help="Max concurrent HTTP workers used by the eval-side remote client",
    )
    parser.add_argument(
        "--infer-protocol",
        choices=REMOTE_INFERENCE_PROTOCOL_CHOICES,
        default="openai",
        help="Remote request protocol: generic OpenAI compatibility, vLLM chat requests, or raw completions",
    )
    parser.add_argument(
        "--infer-seed-policy",
        choices=REMOTE_INFERENCE_SEED_POLICY_CHOICES,
        default="preserve",
        help=(
            "Remote seed handling. preserve keeps per-prompt seeds when the protocol supports them; "
            "omit drops per-prompt seeds for higher-throughput vLLM/completions requests."
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
    if has_local and not has_remote:
        raise ValueError("本分支已移除本地推理；请使用 --infer-base-url 和 --infer-model 连接 vllm-rwkv 服务。")
    if not has_remote:
        raise ValueError("必须同时提供 --infer-base-url 和 --infer-model。")
    if has_remote and not infer_base_url:
        raise ValueError("远端推理模式缺少 --infer-base-url。")
    if has_remote and not infer_model:
        raise ValueError("远端推理模式缺少 --infer-model。")


def resolve_backend_model_name(args: argparse.Namespace) -> str:
    validate_inference_backend_args(args)
    infer_model = str(getattr(args, "infer_model", "") or "").strip()
    return infer_model


def require_completion_style_remote_protocol(
    args: argparse.Namespace,
    *,
    benchmark_name: str,
) -> bool:
    """Force legacy prefilled prompts onto raw completions when using remote infer."""

    validate_inference_backend_args(args)
    infer_base_url = str(getattr(args, "infer_base_url", "") or "").strip()
    if not infer_base_url:
        return False
    protocol = _normalize_remote_protocol(getattr(args, "infer_protocol", "openai"))
    if protocol in {"openai", "vllm"}:
        setattr(args, "infer_protocol", "completions")
        return True
    if protocol == "completions":
        return True
    raise ValueError(
        f"{benchmark_name} requires completion-style remote generation; "
        f"unsupported infer protocol: {protocol}"
    )


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


@dataclass(slots=True, frozen=True)
class RemoteInferenceConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout_s: float = 600.0
    max_workers: int = DEFAULT_REMOTE_MAX_WORKERS
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
        return f"{normalize_api_base_for_version(self.base_url, 'v1')}/completions"

    def chat_completions_url(self) -> str:
        return f"{normalize_api_base_for_version(self.base_url, 'v1')}/chat/completions"


def resolve_generation_prompt_batch_size(
    backend: object,
    batch_size: int,
    *,
    max_inflight_batches: int = 4,
) -> int:
    del backend, max_inflight_batches
    return max(1, int(batch_size))


@dataclass(slots=True)
class RemoteInferenceBackend:
    config: RemoteInferenceConfig
    _http_client: httpx.Client | None = field(default=None, init=False, repr=False)
    _http_client_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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
            raise NotImplementedError("remote infer backend does not support prompt constraints; use a local backend")
        if not prompts:
            return []
        if prompt_seeds is not None and len(prompt_seeds) != len(prompts):
            raise ValueError("prompt_seeds length must match prompts length")
        if prompt_stop_suffixes is not None and len(prompt_stop_suffixes) != len(prompts):
            raise ValueError("prompt_stop_suffixes length must match prompts length")
        effective_sampling = sampling.clamp(1) if probe_only else sampling
        omit_prompt_seeds = (
            self.config.protocol in {"vllm", "completions"}
            or self.config.seed_policy == "omit"
        )
        if prompt_seeds is not None:
            has_prompt_seeds = any(seed is not None for seed in prompt_seeds)
            if has_prompt_seeds and omit_prompt_seeds:
                prompt_seeds = None
            elif not has_prompt_seeds:
                prompt_seeds = None
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

    def shutdown(self) -> None:
        with self._http_client_lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            client.close()

    def _generate_one(
        self,
        prompt_index: int,
        prompt: str,
        sampling: SamplingConfig,
        seed: int | None,
        stop_suffixes: Sequence[str] | None,
        prefill_chunk_size: int,
    ) -> GenerationOutput:
        include_private_fields = self.config.protocol == "completions"
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
        elif self.config.protocol == "completions":
            response = self._post_json(self.config.completions_url(), payload)
            is_chat_response = False
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
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key or 'rwkv-skills'}",
        }
        try:
            raw = self._post_bytes_with_retries(str(url), body, headers)
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise RuntimeError("remote infer response must be a JSON object")
            ok = True
            return data
        finally:
            callback = self.config.request_latency_callback
            if callback is not None:
                callback(str(url), max(0.0, time.perf_counter() - started), ok)

    def _http_client_for_requests(self) -> httpx.Client:
        client = self._http_client
        if client is not None:
            return client
        with self._http_client_lock:
            client = self._http_client
            if client is None:
                timeout_s = max(float(self.config.timeout_s), 1.0)
                max_connections = max(1, int(self.config.max_workers))
                client = httpx.Client(
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=max_connections,
                        keepalive_expiry=60.0,
                    ),
                    timeout=httpx.Timeout(timeout_s),
                )
                self._http_client = client
            return client

    def _post_bytes_with_retries(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> str:
        attempts = max(1, int(self.config.max_retries) + 1)
        transient_attempts = max(attempts, 12)
        timeout_s = max(float(self.config.timeout_s), 1.0)
        delay_s = max(float(self.config.retry_initial_delay_s), 0.0)
        max_delay_s = max(float(self.config.retry_max_delay_s), delay_s)
        last_exc: BaseException | None = None
        for attempt in range(1, transient_attempts + 1):
            try:
                response = self._http_client_for_requests().post(
                    url,
                    content=body,
                    headers=headers,
                    timeout=timeout_s,
                )
                try:
                    if int(response.status_code) >= 400:
                        raise RemoteHTTPError(int(response.status_code), response.text)
                    content = response.content
                finally:
                    close_response = getattr(response, "close", None)
                    if callable(close_response):
                        close_response()
                return content.decode("utf-8")
            except RemoteHTTPError as exc:
                last_exc = exc
                if not _is_retryable_remote_http_error(exc) or attempt >= attempts:
                    raise
                if delay_s > 0:
                    time.sleep(delay_s)
                    delay_s = min(delay_s * 2, max_delay_s)
            except _REMOTE_TRANSIENT_ERRORS as exc:  # pragma: no cover - exercised through integration
                last_exc = exc
                if attempt >= transient_attempts:
                    break
                if delay_s > 0:
                    time.sleep(delay_s)
                    delay_s = min(delay_s * 2, max_delay_s)
        raise RuntimeError(f"remote infer request failed after {transient_attempts} attempts: {last_exc}") from last_exc


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
    return RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url=infer_base_url,
            model=str(getattr(args, "infer_model", "") or "").strip(),
            api_key=str(getattr(args, "infer_api_key", "") or ""),
            timeout_s=float(getattr(args, "infer_timeout_s", 600.0) or 600.0),
            max_workers=max(
                1,
                int(getattr(args, "infer_max_workers", DEFAULT_REMOTE_MAX_WORKERS) or DEFAULT_REMOTE_MAX_WORKERS),
            ),
            protocol=_normalize_remote_protocol(getattr(args, "infer_protocol", "openai")),
            seed_policy=_normalize_remote_seed_policy(getattr(args, "infer_seed_policy", "preserve")),
        )
    )


def _normalize_constraint_mode(mode: str | None) -> Literal["off", "soft", "strict"]:
    normalized = str(mode or "off").strip().lower()
    if normalized not in {"off", "soft", "strict"}:
        raise ValueError("constraint_mode must be one of: off, soft, strict")
    return normalized  # type: ignore[return-value]


def _normalize_remote_protocol(protocol: object) -> RemoteInferenceProtocol:
    value = str(protocol or "openai").strip().lower().replace("_", "-")
    if value in {"vllm-openai", "vllm-chat", "vllm-compatible"}:
        value = "vllm"
    if value in {"completion", "raw-completion", "raw-completions", "text-completion", "text-completions"}:
        value = "completions"
    if value in {"rwkv-lightning", "lightning", "lightning-v2", "nano-vllm", "nanovllm", "contents"}:
        raise ValueError("旧 Lightning/nano-vLLM 协议已移除；请使用 vllm、openai 或 completions。")
    if value not in REMOTE_INFERENCE_PROTOCOL_CHOICES:
        choices = ", ".join(REMOTE_INFERENCE_PROTOCOL_CHOICES)
        raise ValueError(f"infer_protocol must be one of: {choices}")
    return value  # type: ignore[return-value]


def _normalize_remote_seed_policy(seed_policy: object) -> RemoteInferenceSeedPolicy:
    value = str(seed_policy or "preserve").strip().lower().replace("_", "-")
    aliases = {
        "keep": "preserve",
        "keep-seeds": "preserve",
        "preserve-seeds": "preserve",
        "drop": "omit",
        "omit-for-contents": "omit",
        "drop-for-contents": "omit",
        "omit-seeds": "omit",
    }
    value = aliases.get(value, value)
    if value not in REMOTE_INFERENCE_SEED_POLICY_CHOICES:
        choices = ", ".join(REMOTE_INFERENCE_SEED_POLICY_CHOICES)
        raise ValueError(f"infer_seed_policy must be one of: {choices}")
    return value  # type: ignore[return-value]


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
    "DEFAULT_REMOTE_MAX_WORKERS",
    "InferenceBackend",
    "REMOTE_INFERENCE_PROTOCOL_CHOICES",
    "REMOTE_INFERENCE_SEED_POLICY_CHOICES",
    "RemoteInferenceProtocol",
    "RemoteInferenceSeedPolicy",
    "RemoteInferenceBackend",
    "RemoteInferenceConfig",
    "add_inference_backend_arguments",
    "build_inference_backend_from_args",
    "normalize_api_base",
    "normalize_local_device",
    "require_completion_style_remote_protocol",
    "resolve_backend_model_name",
    "resolve_generation_prompt_batch_size",
    "validate_inference_backend_args",
]
