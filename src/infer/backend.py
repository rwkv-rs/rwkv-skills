from __future__ import annotations

"""Shared local/remote inference backends for evaluation pipelines."""

import argparse
import concurrent.futures
import json
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
        prompt_seeds: Sequence[int] | None = None,
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
        prompt_seeds: Sequence[int] | None = None,
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
        prompt_seeds: Sequence[int] | None = None,
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
        outputs: list[GenerationOutput | None] = [None] * len(prompts)
        max_workers = max(1, min(int(batch_size), int(self.config.max_workers), len(prompts)))
        progress = tqdm(total=len(prompts), desc=progress_desc, unit=" request", disable=not show_progress)
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
                progress.update(1)
        progress.close()
        return [output for output in outputs if output is not None]

    def score_choice_tokens(
        self,
        *,
        prompt: str,
        choice_token_texts: Sequence[str],
    ) -> tuple[dict[str, float], str]:
        if not choice_token_texts:
            raise ValueError("choice_token_texts cannot be empty")
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
        _ = prefill_chunk_size
        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
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
        response = self._post_json(self.config.chat_completions_url(), payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("remote infer response missing choices")
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise RuntimeError("remote infer response choice format is invalid")
        text = _extract_chat_choice_text(choice0)
        return GenerationOutput(
            prompt_index=prompt_index,
            prompt=prompt,
            token_ids=[],
            text=text,
            finish_reason=_normalize_remote_finish_reason(choice0.get("finish_reason")),
        )

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
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
            with urllib_request.urlopen(req, timeout=max(float(self.config.timeout_s), 1.0)) as resp:
                raw = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:  # pragma: no cover - exercised through integration
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteHTTPError(exc.code, detail) from exc
        except urllib_error.URLError as exc:  # pragma: no cover - exercised through integration
            raise RuntimeError(f"remote infer request failed: {exc.reason}") from exc
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError("remote infer response must be a JSON object")
        return data


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
    "RemoteInferenceBackend",
    "RemoteInferenceConfig",
    "add_inference_backend_arguments",
    "build_inference_backend_from_args",
    "normalize_api_base",
    "normalize_local_device",
    "resolve_backend_model_name",
    "validate_inference_backend_args",
]
