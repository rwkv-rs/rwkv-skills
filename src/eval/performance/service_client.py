from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Any, Literal

from openai import OpenAI

from src.infer.backend import normalize_api_base


ChatProtocol = Literal["openai-chat"]


@dataclass(slots=True)
class ServiceRequestResult:
    request_index: int
    text: str
    ttft_s: float
    e2el_s: float
    finish_reason: str | None
    error: str | None = None


class OpenAIChatServiceClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = normalize_api_base(base_url)
        self.model = str(model)
        self.api_key = str(api_key or "rwkv-skills")
        self.timeout_s = max(float(timeout_s), 1.0)
        self._local = threading.local()

    def _client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_s,
            )
            self._local.client = client
        return client

    def list_models(self) -> list[str]:
        client = self._client()
        response = client.models.list()
        names: list[str] = []
        for item in response.data:
            model_id = getattr(item, "id", None)
            if isinstance(model_id, str) and model_id:
                names.append(model_id)
        return names

    def benchmark_one(
        self,
        *,
        request_index: int,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> ServiceRequestResult:
        client = self._client()
        started = time.perf_counter()
        first_token_at: float | None = None
        finish_reason: str | None = None
        fragments: list[str] = []

        stream = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=seed,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice0 = choices[0]
            delta = getattr(choice0, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str) and content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                fragments.append(content)
            chunk_finish = getattr(choice0, "finish_reason", None)
            if chunk_finish:
                finish_reason = str(chunk_finish)
        finished = time.perf_counter()
        if first_token_at is None:
            first_token_at = finished

        return ServiceRequestResult(
            request_index=request_index,
            text="".join(fragments),
            ttft_s=first_token_at - started,
            e2el_s=finished - started,
            finish_reason=finish_reason,
        )

    def benchmark_many(
        self,
        *,
        prompts: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        base_seed: int | None,
        max_workers: int,
    ) -> list[ServiceRequestResult]:
        worker_count = max(1, min(int(max_workers), len(prompts)))
        results: list[ServiceRequestResult | None] = [None] * len(prompts)

        def _run(index: int, prompt: str) -> ServiceRequestResult:
            seed = None if base_seed is None else int(base_seed) + int(index)
            try:
                return self.benchmark_one(
                    request_index=index,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            except BaseException as exc:
                return ServiceRequestResult(
                    request_index=index,
                    text="",
                    ttft_s=0.0,
                    e2el_s=0.0,
                    finish_reason=None,
                    error=str(exc),
                )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run, index, prompt) for index, prompt in enumerate(prompts)]
            for future in futures:
                result = future.result()
                results[result.request_index] = result
        return [result for result in results if result is not None]


__all__ = [
    "ChatProtocol",
    "OpenAIChatServiceClient",
    "ServiceRequestResult",
]
