from __future__ import annotations

import json

from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.evaluators.function_call import FunctionCallPipeline
from src.eval.metrics.function_call import evaluate_function_call


def test_function_call_prompt_renders_json_call_context() -> None:
    record = FunctionCallTaskRecord(
        task_id="calendar-1",
        tools=[
            {
                "name": "find_free_slots",
                "description": "Find free calendar slots",
                "arguments": {
                    "date": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                    "time_window": {"type": "string"},
                },
            },
            {
                "name": "create_calendar_event",
                "description": "Create a calendar event",
                "arguments": {
                    "title": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
            },
        ],
        messages=[
            {
                "role": "user",
                "content": "Schedule a 30-minute sync with Bob on 2026-05-08 afternoon.",
            },
            {
                "role": "assistant",
                "tool_call": {
                    "name": "find_free_slots",
                    "arguments": {
                        "date": "2026-05-08",
                        "duration_minutes": 30,
                        "time_window": "afternoon",
                    },
                },
            },
            {
                "role": "function",
                "content": {
                    "free_slots": [
                        {
                            "start": "2026-05-08T15:00:00+09:00",
                            "end": "2026-05-08T15:30:00+09:00",
                        }
                    ],
                    "bob_email": "bob@example.com",
                },
            },
        ],
        expected_call={
            "name": "create_calendar_event",
            "arguments": {
                "title": "Sync with Bob",
                "start_time": "2026-05-08T15:00:00+09:00",
                "end_time": "2026-05-08T15:30:00+09:00",
                "attendees": ["bob@example.com"],
            },
        },
    )
    pipeline = object.__new__(FunctionCallPipeline)

    prompt = pipeline._make_prompt(record, config=None)

    assert "System: Tools:\n[" in prompt
    assert "Return only a JSON function call." in prompt
    assert "User: Schedule a 30-minute sync with Bob on 2026-05-08 afternoon." in prompt
    assert 'Assistant: {"name":"find_free_slots","arguments":{"date":"2026-05-08"' in prompt
    assert 'User: Function output:\n{"free_slots":[{"start":"2026-05-08T15:00:00+09:00"' in prompt
    assert prompt.endswith("Assistant:")


def test_function_call_loader_requires_expected_json_call(tmp_path) -> None:
    path = tmp_path / "function_call.jsonl"
    row = {
        "task_id": "translate-1",
        "instruction": 'Translate "Will it rain tomorrow?" into Japanese.',
        "tools": [
            {
                "name": "translate_text",
                "arguments": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string"},
                },
            }
        ],
        "expected_call": {
            "name": "translate_text",
            "arguments": {
                "text": "Will it rain tomorrow?",
                "target_language": "Japanese",
            },
        },
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    record = JsonlFunctionCallTaskLoader(path).load()[0]

    assert record.messages == [
        {"role": "user", "content": 'Translate "Will it rain tomorrow?" into Japanese.'}
    ]
    assert record.env == {"type": "json_function_call"}
    assert record.scorer == {"type": "json_function_call_exact"}
    assert record.expected_call["name"] == "translate_text"


def test_function_call_metric_scores_strict_json_calls(tmp_path) -> None:
    path = tmp_path / "function_call.jsonl"
    row = {
        "task_id": "weather-1",
        "instruction": "What is the weather in Tokyo in Celsius?",
        "tools": [{"name": "get_weather", "arguments": {"location": {"type": "string"}}}],
        "expected_call": {
            "name": "get_weather",
            "arguments": {"location": "Tokyo", "unit": "celsius"},
        },
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = evaluate_function_call(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "final_answer": json.dumps(
                    {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo", "unit": "celsius"},
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "sample_index": 0,
                "repeat_index": 1,
                "final_answer": '```json\n{"name":"get_weather","arguments":{"location":"Tokyo","unit":"celsius"}}\n```',
            },
        ],
        dataset_path=str(path),
    )

    assert metrics.success_rate == 0.5
    assert metrics.payloads is not None
    assert metrics.payloads[0]["is_passed"] is True
    assert metrics.payloads[1]["fail_reason"] == "invalid_json"
