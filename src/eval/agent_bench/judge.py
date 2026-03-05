from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class NLAssertionResult:
    assertion: str
    met: bool
    justification: str


@dataclass(slots=True)
class NLAssertionJudge:
    model_name: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    num_retries: int = 2

    def evaluate(self, assertions: list[str], messages: list[Any]) -> list[NLAssertionResult]:
        if not assertions:
            return []

        completion = _import_litellm_completion()
        conversation = _stringify_messages(messages)
        prompt = (
            "You are a strict trajectory judge.\n"
            "Given expected outcomes and a conversation between assistant and user, "
            "determine whether each outcome is satisfied.\n"
            "Return JSON only in this shape:\n"
            '{"results":[{"assertion":"...","met":true,"reason":"..."}]}\n\n'
            f"Conversation:\n{conversation}\n\n"
            f"Expected outcomes:\n{json.dumps(assertions, ensure_ascii=False)}"
        )
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            api_base=self.base_url,
            temperature=self.temperature,
            num_retries=self.num_retries,
        )

        text = response.choices[0].message.content or ""
        parsed = _extract_json_object(text)
        if not isinstance(parsed, dict):
            return [
                NLAssertionResult(assertion=item, met=False, justification="judge_output_parse_error")
                for item in assertions
            ]

        rows = parsed.get("results")
        if not isinstance(rows, list):
            return [
                NLAssertionResult(assertion=item, met=False, justification="judge_results_missing")
                for item in assertions
            ]

        mapped: dict[str, NLAssertionResult] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            assertion = row.get("assertion")
            if not isinstance(assertion, str):
                continue
            mapped[assertion] = NLAssertionResult(
                assertion=assertion,
                met=bool(row.get("met", False)),
                justification=str(row.get("reason") or ""),
            )

        results: list[NLAssertionResult] = []
        for item in assertions:
            hit = mapped.get(item)
            if hit is not None:
                results.append(hit)
            else:
                results.append(
                    NLAssertionResult(assertion=item, met=False, justification="judge_result_missing")
                )
        return results


def _stringify_messages(messages: Iterable[Any]) -> str:
    rows: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "unknown"))
            content = message.get("content")
            if content is None and message.get("tool_calls"):
                content = json.dumps(message.get("tool_calls"), ensure_ascii=False)
            rows.append(f"{role}: {str(content or '').strip()}")
            continue

        role = getattr(message, "role", "unknown")
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        if content is None and tool_calls:
            content = json.dumps(
                [
                    {
                        "name": getattr(call, "name", ""),
                        "arguments": getattr(call, "arguments", {}),
                    }
                    for call in tool_calls
                ],
                ensure_ascii=False,
            )
        rows.append(f"{role}: {str(content or '').strip()}")
    return "\n".join(rows)


def _extract_json_object(text: str) -> Any | None:
    candidates = [text.strip()]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(match.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _import_litellm_completion():
    try:
        from litellm import completion
    except Exception as exc:  # pragma: no cover - dependency is optional during unit tests
        raise RuntimeError("litellm is required for NL assertion judging.") from exc
    return completion


__all__ = [
    "NLAssertionResult",
    "NLAssertionJudge",
]
