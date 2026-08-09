#!/usr/bin/env python3
"""Verify that ordinary vLLM protocol generation preserves a raw prompt."""

from __future__ import annotations

import argparse
import json

from src.infer.backend import RemoteInferenceBackend, RemoteInferenceConfig
from src.infer.sampling import SamplingConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="rwkv-skills")
    args = parser.parse_args()

    prompt = (
        "User: Output exactly ALPHA and nothing else.\n\n"
        "Assistant: <think></think>\n"
    )
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            protocol="vllm",
            max_workers=1,
        )
    )
    try:
        output = backend.generate(
            [prompt],
            sampling=SamplingConfig(
                max_generate_tokens=16,
                temperature=1e-5,
                top_p=1.0,
                top_k=0,
            ),
            batch_size=1,
            show_progress=False,
        )[0]
    finally:
        backend.shutdown()

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "prompt": prompt,
        "returned_prompt_matches": output.prompt == prompt,
        "text": output.text,
        "finish_reason": output.finish_reason,
        "leading_orphan_close": output.text.lstrip().startswith("></think>"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if output.prompt != prompt or result["leading_orphan_close"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
