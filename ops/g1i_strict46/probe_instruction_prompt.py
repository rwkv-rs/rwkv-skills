"""Probe empty-think prompt variants against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


VARIANTS = {
    "compact": "Assistant: <think></think>\n",
    "line_break": "Assistant: <think>\n</think>\n",
    "blank_line": "Assistant: <think>\n\n</think>\n",
    "compact_blank_line": "Assistant: <think></think>\n\n",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="rwkv-skills")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--endpoint-mode", choices=("completions", "chat"), default="completions")
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + (
        "/chat/completions" if args.endpoint_mode == "chat" else "/completions"
    )
    rows = []
    for name, suffix in VARIANTS.items():
        for repeat in range(max(1, args.repeats)):
            prompt = (
                "User: Reply with exactly the word ALPHA and no other text.\n\n"
                + suffix
            )
            payload = {
                "model": args.model,
                "max_tokens": 64,
                "temperature": 0.3,
                "top_p": 0.3,
                "top_k": 50,
                "stream": False,
            }
            if args.endpoint_mode == "chat":
                payload["messages"] = [{"role": "user", "content": prompt}]
            else:
                payload["prompt"] = prompt
            request = Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {args.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(request, timeout=300) as response:  # noqa: S310 - explicit trusted endpoint
                body = json.load(response)
            choice = body["choices"][0]
            if args.endpoint_mode == "chat":
                text = str((choice.get("message") or {}).get("content") or "")
            else:
                text = str(choice.get("text") or "")
            rows.append(
                {
                    "variant": name,
                    "repeat": repeat,
                    "prompt_suffix": suffix,
                    "text": text,
                    "finish_reason": choice.get("finish_reason"),
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
