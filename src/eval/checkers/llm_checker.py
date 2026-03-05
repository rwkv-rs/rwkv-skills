from __future__ import annotations

import json
import os
import time
import threading
import heapq
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import jsonschema
from json import JSONDecodeError
from openai import OpenAI, OpenAIError

from src.eval.env_config import load_env_file
from src.eval.results.layout import check_details_path
from src.eval.results.io import iter_jsonl
from src.eval.results.schema import strict_nonneg_int
from src.eval.scheduler.config import REPO_ROOT


CHECKER_FIELD_ANSWER_CORRECT = "answer_correct"
CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR = "instruction_following_error"
CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR = "world_knowledge_error"
CHECKER_FIELD_MATH_ERROR = "math_error"
CHECKER_FIELD_REASONING_LOGIC_ERROR = "reasoning_logic_error"
CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER = "thought_contains_correct_answer"
CHECKER_FIELD_REASON = "reason"
CHECKER_FIELD_COT = "checker_cot"

CHECKER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        CHECKER_FIELD_ANSWER_CORRECT: {
            "type": "boolean",
            "description": "模型答案实际正确（可能是参考答案不全导致误判）。",
        },
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR: {
            "type": "boolean",
            "description": "指令遵循错误：模型未能正确理解题目意图/格式要求/约束。",
        },
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR: {
            "type": "boolean",
            "description": "世界知识错误：推理过程中引入明显不合理/错误的事实或常识。",
        },
        CHECKER_FIELD_MATH_ERROR: {
            "type": "boolean",
            "description": "数学运算错误：推理中出现不成立的算术/代数/数值计算。",
        },
        CHECKER_FIELD_REASONING_LOGIC_ERROR: {
            "type": "boolean",
            "description": "推理逻辑错误：推理链不严谨/跳步/自相矛盾。",
        },
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER: {
            "type": "boolean",
            "description": "思考过程是否包含正确答案（不一定等同于参考答案）。",
        },
        CHECKER_FIELD_REASON: {
            "type": "string",
            "description": "简要概述判定原因（面向人类阅读，短句即可）。",
        },
        CHECKER_FIELD_COT: {
            "type": "string",
            "description": "分析过程/推理过程（可分点描述，允许较长）。",
        },
    },
    "required": [
        CHECKER_FIELD_ANSWER_CORRECT,
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR,
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR,
        CHECKER_FIELD_MATH_ERROR,
        CHECKER_FIELD_REASONING_LOGIC_ERROR,
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER,
        CHECKER_FIELD_REASON,
        CHECKER_FIELD_COT,
    ],
}


_PROMPT_TEMPLATE = """你是一个专业的大语言模型评估任务的专家，你需要帮助我分析某大语言模型对以下题目的作答情况。

【题目与模型作答上下文】
{context}

【评估器提取到的答案】
{answer}

【参考答案】
{ref_answer}

请你分析是否有以下情况出现：
1) 答案正确（参考答案不全面导致被判为错题）
2) 指令遵循错误（模型未能正确理解题目意图）
3) 世界知识错误（推理过程中引入了明显不合理的世界知识）
4) 数学运算错误（推理过程中有明显不成立的数学运算）
5) 推理逻辑错误（推理过程不严谨）
6) 思考过程是否包含正确答案（不一定是参考答案）

要求：
- 你必须仅返回一个 JSON 对象，且必须符合固定字段名（不要输出多余字段或额外文本）。
- 字段名固定为：{field_names}
"""


def _truncate_text(value: str, *, max_chars: int) -> str:
    if not value:
        return ""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    # Keep both head and tail (tail often contains the final answer).
    head = max(1, int(max_chars * 0.7))
    tail = max(1, max_chars - head - 64)
    return f"{value[:head]}\n\n...[truncated {len(value) - head - tail} chars]...\n\n{value[-tail:]}"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "n", "off"}


def _load_existing_keys(path: Path) -> set[tuple[str, int, int]]:
    """Return {(dataset_split, sample_index, repeat_index)} already written."""
    if not path.exists():
        return set()
    keys: set[tuple[str, int, int]] = set()
    for row in iter_jsonl(path, loads=orjson.loads):
        split = str(row.get("dataset_split", ""))
        sample_index = strict_nonneg_int(row.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(row.get("repeat_index"), "repeat_index")
        keys.add((split, sample_index, repeat_index))
    return keys


def _validate_checker_payload(payload: dict[str, Any]) -> None:
    jsonschema.validate(instance=payload, schema=CHECKER_JSON_SCHEMA)


def _coerce_checker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort coercion for common provider quirks (e.g. 'true'/'false' strings)."""
    normalized: dict[str, Any] = dict(payload)

    def coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    for key in (
        CHECKER_FIELD_ANSWER_CORRECT,
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR,
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR,
        CHECKER_FIELD_MATH_ERROR,
        CHECKER_FIELD_REASONING_LOGIC_ERROR,
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER,
    ):
        normalized[key] = coerce_bool(normalized.get(key))
    for key in (CHECKER_FIELD_REASON, CHECKER_FIELD_COT):
        value = normalized.get(key)
        normalized[key] = "" if value is None else str(value)
    return normalized


@dataclass(slots=True)
class LLMCheckerConfig:
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.0
    max_workers: int = 16
    max_prompt_chars: int = 20000
    # Set to -1 to retry forever (handled by the scheduler loop with backoff).
    max_retries: int = -1

    @classmethod
    def from_env(cls) -> LLMCheckerConfig | None:
        api_key = (
            os.environ.get("CHECKER_API_KEY")
            or os.environ.get("LLM_CHECKER_API_KEY")
            or os.environ.get("JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("API_KEY")
        )
        model = (
            os.environ.get("CHECKER_MODEL")
            or os.environ.get("LLM_CHECKER_MODEL")
            or os.environ.get("JUDGE_MODEL")
        )
        base_url = (
            os.environ.get("CHECKER_BASE_URL")
            or os.environ.get("LLM_CHECKER_BASE_URL")
            or os.environ.get("JUDGE_BASE_URL")
            or os.environ.get("LLM_JUDGE_BASE_URL")
            or os.environ.get("API_BASE")
        )
        max_workers_raw = os.environ.get("CHECKER_MAX_WORKERS") or os.environ.get("LLM_CHECKER_MAX_WORKERS")
        max_prompt_chars_raw = os.environ.get("CHECKER_MAX_PROMPT_CHARS") or os.environ.get(
            "LLM_CHECKER_MAX_PROMPT_CHARS"
        )
        max_retries_raw = os.environ.get("CHECKER_MAX_RETRIES") or os.environ.get("LLM_CHECKER_MAX_RETRIES")
        if not api_key or not model:
            return None
        kwargs: dict[str, Any] = {}
        if max_workers_raw:
            try:
                kwargs["max_workers"] = max(1, int(max_workers_raw))
            except ValueError:
                pass
        if max_prompt_chars_raw:
            try:
                kwargs["max_prompt_chars"] = max(1000, int(max_prompt_chars_raw))
            except ValueError:
                pass
        if max_retries_raw:
            try:
                kwargs["max_retries"] = int(max_retries_raw)
            except ValueError:
                pass
        return cls(api_key=api_key, model=model, base_url=base_url, **kwargs)


def _build_prompt(context: str, answer: str, ref_answer: str) -> str:
    fields = [
        CHECKER_FIELD_COT,
        CHECKER_FIELD_ANSWER_CORRECT,
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR,
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR,
        CHECKER_FIELD_MATH_ERROR,
        CHECKER_FIELD_REASONING_LOGIC_ERROR,
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER,
        CHECKER_FIELD_REASON,
    ]
    return _PROMPT_TEMPLATE.format(
        context=context,
        answer=answer,
        ref_answer=ref_answer,
        field_names=", ".join(fields),
    )


class LLMCheckerFailure(RuntimeError):
    """Raised when the checker cannot obtain/validate a response after retries."""


class _LLMCheckerOutputError(ValueError):
    """Raised when the provider output cannot be parsed/validated as required JSON."""


def _call_llm_checker(client: OpenAI, *, config: LLMCheckerConfig, prompt: str) -> dict[str, Any]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "rwkv_skills_llm_checker",
            "schema": CHECKER_JSON_SCHEMA,
            "strict": True,
        },
    }

    last_exc: Exception | None = None
    max_retries = int(config.max_retries)
    if max_retries < 0:
        # "Retry forever" is implemented by the caller (scheduler loop), so we
        # always keep a single _call_llm_checker invocation bounded.
        max_retries = 0
    attempt = 0
    use_schema = True

    while True:
        attempt += 1
        if use_schema:
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    stream=False,
                    temperature=float(config.temperature),
                    response_format=response_format,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = (response.choices[0].message.content or "").strip()
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise _LLMCheckerOutputError("LLM checker output is not a JSON object")
                data = _coerce_checker_payload(data)
                _validate_checker_payload(data)
                return data
            except (
                OpenAIError,
                JSONDecodeError,
                jsonschema.exceptions.ValidationError,
                _LLMCheckerOutputError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                last_exc = exc

        # Fall back to plain JSON (for providers that don't support json_schema response_format).
        try:
            response = client.chat.completions.create(
                model=config.model,
                stream=False,
                temperature=float(config.temperature),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                        + "\n\n仅输出 JSON（不要输出任何额外文本），并确保字段类型正确：布尔值必须是 true/false。",
                    }
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            data = json.loads(content)
            if not isinstance(data, dict):
                raise _LLMCheckerOutputError("LLM checker output is not a JSON object")
            data = _coerce_checker_payload(data)
            _validate_checker_payload(data)
            return data
        except (
            OpenAIError,
            JSONDecodeError,
            jsonschema.exceptions.ValidationError,
            _LLMCheckerOutputError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc2:
            last_exc = exc2
            use_schema = False

        if max_retries >= 0 and attempt >= max_retries + 1:
            break

        # Exponential backoff to avoid hammering the provider.
        delay = min(30.0, 0.5 * (2 ** min(6, attempt - 1)))
        time.sleep(delay)

    raise LLMCheckerFailure(f"LLM checker failed after retries: {last_exc}") from last_exc


_thread_local = threading.local()


def _get_thread_client(cfg: LLMCheckerConfig) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        setattr(_thread_local, "client", client)
    return client


def run_llm_checker(
    eval_results_path: str | Path,
    *,
    model_name: str,
    config: LLMCheckerConfig | None = None,
) -> Path | None:
    """Run wrong-answer checker over a single eval JSONL file.

    Returns the written check JSONL path, or None if skipped.
    """

    load_env_file((REPO_ROOT / ".env").resolve())
    if _env_flag("RWKV_SKILLS_DISABLE_CHECKER"):
        print("ℹ️  LLM checker disabled (RWKV_SKILLS_DISABLE_CHECKER=1)")
        return None
    cfg = config or LLMCheckerConfig.from_env()
    if cfg is None:
        print("⚠️  LLM checker skipped: missing API_KEY/JUDGE_MODEL (see .env)")
        return None

    eval_path = Path(eval_results_path).expanduser().resolve()
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    failed_rows: list[dict[str, Any]] = []
    benchmark_name: str | None = None

    for row in iter_jsonl(eval_path, loads=orjson.loads):
        if benchmark_name is None:
            benchmark_name = str(row.get("benchmark_name", "") or "")
        if bool(row.get("is_passed", False)):
            continue
        failed_rows.append(row)

    if not benchmark_name:
        print(f"⚠️  LLM checker skipped: empty eval file {eval_path}")
        return None

    if not failed_rows:
        print(f"✅ LLM checker: no failed samples for {benchmark_name} ({eval_path.name})")
        return None

    out_path = check_details_path(benchmark_name, model_name=model_name)
    seen = _load_existing_keys(out_path)

    to_check: list[dict[str, Any]] = []
    for row in failed_rows:
        split = str(row.get("dataset_split", "") or "")
        sample_index = strict_nonneg_int(row.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(row.get("repeat_index"), "repeat_index")
        key = (split, sample_index, repeat_index)
        if key in seen:
            continue
        to_check.append(row)

    if not to_check:
        print(f"✅ LLM checker: all failed samples already checked -> {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, int(cfg.max_workers))
    max_workers = min(max_workers, len(to_check))

    def _check_one(row: dict[str, Any]) -> dict[str, Any]:
        context = _truncate_text(str(row.get("context", "") or ""), max_chars=int(cfg.max_prompt_chars))
        answer = _truncate_text(
            str(row.get("answer", "") or ""),
            max_chars=max(1, int(cfg.max_prompt_chars // 4)),
        )
        ref_answer = _truncate_text(
            str(row.get("ref_answer", "") or ""),
            max_chars=max(1, int(cfg.max_prompt_chars // 4)),
        )
        prompt = _build_prompt(context, answer, ref_answer)
        client = _get_thread_client(cfg)
        checked = _call_llm_checker(client, config=cfg, prompt=prompt)
        return {
            "benchmark_name": str(row.get("benchmark_name", "") or ""),
            "dataset_split": str(row.get("dataset_split", "") or ""),
            "sample_index": strict_nonneg_int(row.get("sample_index"), "sample_index"),
            "repeat_index": strict_nonneg_int(row.get("repeat_index"), "repeat_index"),
            CHECKER_FIELD_COT: str(checked.get(CHECKER_FIELD_COT, "") or ""),
            CHECKER_FIELD_ANSWER_CORRECT: bool(checked.get(CHECKER_FIELD_ANSWER_CORRECT, False)),
            CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR: bool(
                checked.get(CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR, False)
            ),
            CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR: bool(checked.get(CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR, False)),
            CHECKER_FIELD_MATH_ERROR: bool(checked.get(CHECKER_FIELD_MATH_ERROR, False)),
            CHECKER_FIELD_REASONING_LOGIC_ERROR: bool(checked.get(CHECKER_FIELD_REASONING_LOGIC_ERROR, False)),
            CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER: bool(
                checked.get(CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER, False)
            ),
            CHECKER_FIELD_REASON: str(checked.get(CHECKER_FIELD_REASON, "") or ""),
        }

    retry_forever = int(cfg.max_retries) < 0

    def _row_key(row: dict[str, Any]) -> tuple[str, int, int]:
        return (
            str(row.get("dataset_split", "") or ""),
            strict_nonneg_int(row.get("sample_index"), "sample_index"),
            strict_nonneg_int(row.get("repeat_index"), "repeat_index"),
        )

    def _describe_row(row: dict[str, Any]) -> str:
        return (
            f"benchmark={row.get('benchmark_name','')} "
            f"split={row.get('dataset_split','')} "
            f"sample_index={row.get('sample_index',0)} "
            f"repeat_index={row.get('repeat_index',0)}"
        )

    def _backoff_seconds(attempt: int) -> float:
        return min(30.0, 0.5 * (2 ** min(6, max(0, int(attempt) - 1))))

    wrote = 0
    failed = 0
    stop_scheduling = False
    todo = deque(to_check)
    retry_heap: list[tuple[float, tuple[str, int, int], dict[str, Any]]] = []
    attempts_by_key: dict[tuple[str, int, int], int] = {}
    pending: dict[object, dict[str, Any]] = {}

    with out_path.open("ab") as out_f, ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or todo or retry_heap:
            now = time.monotonic()
            while retry_heap and retry_heap[0][0] <= now:
                _, _, row = heapq.heappop(retry_heap)
                todo.append(row)

            while not stop_scheduling and len(pending) < max_workers and todo:
                row = todo.popleft()
                future = executor.submit(_check_one, row)
                pending[future] = row

            if not pending:
                if retry_heap:
                    sleep_for = max(0.0, retry_heap[0][0] - time.monotonic())
                    time.sleep(min(1.0, sleep_for))
                    continue
                break

            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = pending.pop(future)
                if future.cancelled():
                    continue

                try:
                    output_row = future.result()
                except LLMCheckerFailure as exc:
                    failed += 1

                    if retry_forever:
                        key = _row_key(row)
                        attempt = attempts_by_key.get(key, 0) + 1
                        attempts_by_key[key] = attempt
                        delay = _backoff_seconds(attempt)
                        heapq.heappush(retry_heap, (time.monotonic() + delay, key, row))
                        print(
                            "⚠️  LLM checker failed; will retry: "
                            f"{_describe_row(row)} attempt={attempt} backoff={delay:.1f}s -> {exc}"
                        )
                        continue

                    if not stop_scheduling:
                        stop_scheduling = True
                        print(f"⚠️  LLM checker failed; stop scheduling new requests: {_describe_row(row)} -> {exc}")
                        for pending_future in pending:
                            pending_future.cancel()
                    continue

                out_f.write(orjson.dumps(output_row, option=orjson.OPT_APPEND_NEWLINE))
                wrote += 1

    if wrote:
        suffix = f" (+{wrote} rows)"
        if failed:
            suffix += f", failed={failed}"
        if retry_forever:
            suffix += ", retry_forever=true"
        print(f"🧩 LLM checker saved: {out_path}{suffix}")
        return out_path
    if failed:
        print(f"⚠️  LLM checker: no new rows written (failed={failed}) -> {out_path}")
        return out_path if out_path.exists() else None
    print(f"✅ LLM checker: no new rows written -> {out_path}")
    return out_path if out_path.exists() else None


__all__ = [
    "LLMCheckerConfig",
    "run_llm_checker",
    "CHECKER_JSON_SCHEMA",
    "CHECKER_FIELD_COT",
    "CHECKER_FIELD_REASON",
    "CHECKER_FIELD_ANSWER_CORRECT",
    "CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR",
    "CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR",
    "CHECKER_FIELD_MATH_ERROR",
    "CHECKER_FIELD_REASONING_LOGIC_ERROR",
    "CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER",
]
