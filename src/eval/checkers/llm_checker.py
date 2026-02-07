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
from typing import Any, Iterable

import jsonschema
from json import JSONDecodeError
from openai import OpenAI, OpenAIError

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


def _load_env_file(path: Path) -> None:
    """Lightweight .env loader (key=value, optional quotes, ignores comments)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


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


def _extract_context_text(row: dict[str, Any]) -> str:
    """从 eval row 中提取 context 文本。"""
    context = row.get("context")
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        # 从 stages 中提取 prompt 和 completion
        stages = context.get("stages", [])
        parts = []
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            prompt = stage.get("prompt")
            completion = stage.get("completion")
            if prompt:
                parts.append(f"[Prompt]\n{prompt}")
            if completion:
                parts.append(f"[Completion]\n{completion}")
        return "\n\n".join(parts)
    return str(context)


def run_llm_checker_db(
    *,
    task_id: str,
    config: LLMCheckerConfig | None = None,
    checker_type: str = "llm_checker",
) -> int:
    """从 DB 读取失败的 eval 记录，运行 LLM checker，结果写回 DB。

    Args:
        task_id: 任务 ID
        config: LLM checker 配置，如果为 None 则从环境变量读取
        checker_type: checker 类型标识，用于 fail_reason 中的键名

    Returns:
        成功处理的记录数
    """
    from src.db.orm import init_orm
    from src.db.eval_db_service import EvalDbService
    from src.db.async_writer import CheckerWriteWorker
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from datetime import datetime

    _load_env_file((REPO_ROOT / ".env").resolve())
    if _env_flag("RWKV_SKILLS_DISABLE_CHECKER"):
        print("ℹ️  LLM checker disabled (RWKV_SKILLS_DISABLE_CHECKER=1)")
        return 0

    cfg = config or LLMCheckerConfig.from_env()
    if cfg is None:
        print("⚠️  LLM checker skipped: missing API_KEY/JUDGE_MODEL (see .env)")
        return 0

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()

    # 先统计需要处理的记录数（流式计数，不加载全部到内存）
    total_count = 0
    for _ in service.iter_failed_evals_for_checker(
        task_id=task_id,
        checker_type=checker_type,
    ):
        total_count += 1

    if total_count == 0:
        print(f"✅ LLM checker: no failed samples need checking for task {task_id}")
        return 0

    print(f"🔍 LLM checker: {total_count} samples to check for task {task_id}")

    max_workers = max(1, min(int(cfg.max_workers), total_count))

    def _check_one(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        eval_id = row["eval_id"]
        context = _truncate_text(_extract_context_text(row), max_chars=int(cfg.max_prompt_chars))
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

        # 构建 checker 结果
        result = {
            "checked_at": datetime.utcnow().isoformat() + "Z",
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
        return eval_id, result

    def _describe_row(row: dict[str, Any]) -> str:
        return (
            f"eval_id={row.get('eval_id')} "
            f"sample_index={row.get('sample_index')} "
            f"repeat_index={row.get('repeat_index')}"
        )

    def _backoff_seconds(attempt: int) -> float:
        return min(30.0, 0.5 * (2 ** min(6, max(0, int(attempt) - 1))))

    retry_forever = int(cfg.max_retries) < 0

    # 使用异步写入器
    writer = CheckerWriteWorker(
        service=service,
        checker_type=checker_type,
        batch_size=50,
    )

    wrote = 0
    failed = 0
    stop_scheduling = False
    # 使用流式迭代器，避免一次性加载全部到内存
    row_iter = iter(service.iter_failed_evals_for_checker(
        task_id=task_id,
        checker_type=checker_type,
    ))
    iter_exhausted = False
    todo: deque[dict[str, Any]] = deque()
    retry_heap: list[tuple[float, int, dict[str, Any]]] = []  # (time, eval_id, row)
    attempts_by_id: dict[int, int] = {}
    pending: dict[object, dict[str, Any]] = {}

    def _fill_todo() -> None:
        """从迭代器填充 todo 队列，最多填充到 max_workers * 2 条。"""
        nonlocal iter_exhausted
        while not iter_exhausted and len(todo) < max_workers * 2:
            try:
                row = next(row_iter)
                todo.append(row)
            except StopIteration:
                iter_exhausted = True
                break

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            _fill_todo()  # 初始填充
            while pending or todo or retry_heap or not iter_exhausted:
                now = time.monotonic()
                while retry_heap and retry_heap[0][0] <= now:
                    _, _, row = heapq.heappop(retry_heap)
                    todo.append(row)

                # 按需从迭代器补充 todo
                if not stop_scheduling:
                    _fill_todo()

                while not stop_scheduling and len(pending) < max_workers and todo:
                    row = todo.popleft()
                    future = executor.submit(_check_one, row)
                    pending[future] = row

                if not pending:
                    if retry_heap:
                        sleep_for = max(0.0, retry_heap[0][0] - time.monotonic())
                        time.sleep(min(1.0, sleep_for))
                        continue
                    if not iter_exhausted:
                        _fill_todo()
                        continue
                    break

                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    if future.cancelled():
                        continue

                    try:
                        eval_id, result = future.result()
                        writer.enqueue(eval_id, result)
                        wrote += 1
                    except LLMCheckerFailure as exc:
                        failed += 1
                        eval_id = row.get("eval_id", 0)

                        if retry_forever:
                            attempt = attempts_by_id.get(eval_id, 0) + 1
                            attempts_by_id[eval_id] = attempt
                            delay = _backoff_seconds(attempt)
                            heapq.heappush(retry_heap, (time.monotonic() + delay, eval_id, row))
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
    finally:
        updated = writer.close()

    suffix = f" (checked={wrote}, updated={updated})"
    if failed:
        suffix += f", failed={failed}"
    if retry_forever:
        suffix += ", retry_forever=true"
    print(f"🧩 LLM checker done for task {task_id}{suffix}")
    return updated


__all__ = [
    "LLMCheckerConfig",
    "run_llm_checker_db",
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
