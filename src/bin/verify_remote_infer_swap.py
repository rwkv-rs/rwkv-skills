from __future__ import annotations

"""Smoke-check remote OpenAI-compatible inference protocols."""

import argparse
from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path
from typing import Sequence

from src.infer.backend import (
    REMOTE_INFERENCE_PROTOCOL_CHOICES,
    RemoteInferenceBackend,
    RemoteInferenceConfig,
    RemoteInferenceProtocol,
)
from src.infer.sampling import GenerationOutput, SamplingConfig


DEFAULT_VERIFY_PROMPT = "User: Reply with exactly one short sentence about remote inference.\n\nAssistant:"
DEFAULT_PROTOCOLS: tuple[RemoteInferenceProtocol, ...] = ("vllm",)


@dataclass(slots=True, frozen=True)
class ProtocolVerification:
    protocol: str
    status: str
    elapsed_s: float
    request_count: int
    output_count: int
    nonempty_output_count: int
    output_chars: int
    first_output_preview: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.status == "ok"
            and self.output_count == self.request_count
            and self.nonempty_output_count == self.request_count
        )


@dataclass(slots=True, frozen=True)
class RemoteInferSwapVerification:
    base_url: str
    model: str
    prompt: str
    max_tokens: int
    batch_size: int
    protocols: tuple[ProtocolVerification, ...]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.protocols)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "base_url": self.base_url,
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "protocols": [asdict(item) | {"ok": item.ok} for item in self.protocols],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify remote inference protocol swap readiness")
    parser.add_argument("--infer-base-url", "--base-url", required=True, help="Remote inference base URL")
    parser.add_argument("--infer-model", "--model", required=True, help="Remote inference model name")
    parser.add_argument("--infer-api-key", "--api-key", default="", help="Remote inference bearer token")
    parser.add_argument("--infer-timeout-s", "--timeout-s", type=float, default=600.0, help="Request timeout")
    parser.add_argument(
        "--protocols",
        default=",".join(DEFAULT_PROTOCOLS),
        help="Comma-separated protocols to verify: openai,vllm",
    )
    parser.add_argument("--prompt", default=DEFAULT_VERIFY_PROMPT, help="Prompt used for both protocols")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max generated tokens per request")
    parser.add_argument("--batch-size", type=int, default=2, help="Prompt count and remote batch size per protocol")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Sampling top-p")
    parser.add_argument("--top-k", type=int, default=50, help="Sampling top-k")
    parser.add_argument("--output-path", help="Optional JSON summary path")
    return parser.parse_args(argv)


def verify_remote_infer_swap(
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    timeout_s: float = 600.0,
    protocols: Sequence[RemoteInferenceProtocol] = DEFAULT_PROTOCOLS,
    prompt: str = DEFAULT_VERIFY_PROMPT,
    max_tokens: int = 16,
    batch_size: int = 2,
    temperature: float = 0.0,
    top_p: float = 0.8,
    top_k: int = 50,
) -> RemoteInferSwapVerification:
    normalized_protocols = tuple(_normalize_protocols(protocols))
    sampling = SamplingConfig(
        max_generate_tokens=max(1, int(max_tokens)),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=max(1, int(top_k)),
    )
    verifications = tuple(
        _verify_protocol(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            protocol=protocol,
            prompt=prompt,
            sampling=sampling,
            batch_size=batch_size,
        )
        for protocol in normalized_protocols
    )
    return RemoteInferSwapVerification(
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=max(1, int(max_tokens)),
        batch_size=max(1, int(batch_size)),
        protocols=verifications,
    )


def write_verification_result(path: Path, result: RemoteInferSwapVerification) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _verify_protocol(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float,
    protocol: RemoteInferenceProtocol,
    prompt: str,
    sampling: SamplingConfig,
    batch_size: int,
) -> ProtocolVerification:
    request_count = max(1, int(batch_size))
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            max_workers=request_count,
            protocol=protocol,
            seed_policy="preserve",
        )
    )
    prompts = [prompt] * request_count
    started = time.perf_counter()
    try:
        outputs = backend.generate(prompts, sampling=sampling, batch_size=request_count, show_progress=False)
    except BaseException as exc:
        return ProtocolVerification(
            protocol=protocol,
            status="failed",
            elapsed_s=max(0.0, time.perf_counter() - started),
            request_count=request_count,
            output_count=0,
            nonempty_output_count=0,
            output_chars=0,
            error=str(exc),
        )
    elapsed_s = max(0.0, time.perf_counter() - started)
    return _summarize_outputs(
        protocol=protocol,
        request_count=request_count,
        elapsed_s=elapsed_s,
        outputs=outputs,
    )


def _summarize_outputs(
    *,
    protocol: str,
    request_count: int,
    elapsed_s: float,
    outputs: Sequence[GenerationOutput],
) -> ProtocolVerification:
    texts = [str(output.text or "") for output in outputs]
    output_chars = sum(len(text) for text in texts)
    first_preview = texts[0].replace("\n", "\\n")[:160] if texts else ""
    return ProtocolVerification(
        protocol=protocol,
        status="ok",
        elapsed_s=elapsed_s,
        request_count=request_count,
        output_count=len(outputs),
        nonempty_output_count=sum(1 for text in texts if text.strip()),
        output_chars=output_chars,
        first_output_preview=first_preview,
    )


def _normalize_protocols(protocols: Sequence[str]) -> tuple[RemoteInferenceProtocol, ...]:
    normalized: list[RemoteInferenceProtocol] = []
    for raw in protocols:
        for item in str(raw).split(","):
            protocol = item.strip()
            if not protocol:
                continue
            if protocol not in REMOTE_INFERENCE_PROTOCOL_CHOICES:
                choices = ", ".join(REMOTE_INFERENCE_PROTOCOL_CHOICES)
                raise ValueError(f"protocol must be one of: {choices}")
            normalized.append(protocol)  # type: ignore[arg-type]
    if not normalized:
        raise ValueError("at least one protocol is required")
    return tuple(normalized)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = verify_remote_infer_swap(
        base_url=str(args.infer_base_url),
        model=str(args.infer_model),
        api_key=str(args.infer_api_key or ""),
        timeout_s=float(args.infer_timeout_s),
        protocols=_normalize_protocols([str(args.protocols)]),
        prompt=str(args.prompt),
        max_tokens=int(args.max_tokens),
        batch_size=int(args.batch_size),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
    )
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if args.output_path:
        write_verification_result(Path(args.output_path).expanduser(), result)
    return 0 if result.ok else 1


__all__ = [
    "ProtocolVerification",
    "RemoteInferSwapVerification",
    "main",
    "parse_args",
    "verify_remote_infer_swap",
    "write_verification_result",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
