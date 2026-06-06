from __future__ import annotations

import json

from src.db.eval_service import create_eval_service, init_eval_store, use_json_eval_store
from src.eval.evaluating import RunMode, prepare_task_execution


def test_eval_store_defaults_to_db(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_EVAL_STORE", raising=False)
    monkeypatch.delenv("RWKV_EVAL_PERSISTENCE", raising=False)
    monkeypatch.delenv("RWKV_EVAL_BACKEND", raising=False)

    assert not use_json_eval_store()


def test_json_eval_store_persists_structured_completion_and_resume(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RWKV_EVAL_STORE", "json")
    monkeypatch.setenv("RWKV_EVAL_JSON_ROOT", str(tmp_path))

    assert use_json_eval_store()
    init_eval_store()
    service = create_eval_service()
    sampling_config = {"stage1": {"temperature": 0.1}}

    state = prepare_task_execution(
        service=service,
        dataset="tau3_bench_airline_base",
        model="rwkv7-g1g-7.2b-20260523-ctx8192",
        is_param_search=False,
        job_name="function_tau3_bench",
        sampling_config=sampling_config,
        run_mode=RunMode.AUTO,
    )

    service.insert_completion_payload(
        task_id=state.task_id,
        payload={
            "benchmark_name": "tau3_bench_airline",
            "dataset_split": "base",
            "sample_index": 0,
            "repeat_index": 0,
            "pass_index": 0,
            "sampling_config": sampling_config,
            "prompt1": "User: hello\nAssistant:",
            "completion1": '{"name":"calculate","arguments":{"expression":"2+2"}}',
            "stop_reason1": "stop_token",
            "agent_trace": [{"role": "user", "content": "hello"}],
            "agent_result": {"is_passed": False, "num_turns": 1, "error": "demo"},
            "perf": {"total_attempt_s": 1.25, "agent_generation_s": 0.75},
        },
    )

    completion_path = tmp_path / "tasks" / state.task_id / "completions.jsonl"
    rows = [json.loads(line) for line in completion_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["context"]["agent_trace"] == [{"role": "user", "content": "hello"}]
    assert rows[0]["context"]["stages"][0]["prompt"] == "User: hello\nAssistant:"
    assert rows[0]["context"]["perf"] == {"agent_generation_s": 0.75, "total_attempt_s": 1.25}

    resumed = prepare_task_execution(
        service=create_eval_service(),
        dataset="tau3_bench_airline_base",
        model="rwkv7-g1g-7.2b-20260523-ctx8192",
        is_param_search=False,
        job_name="function_tau3_bench",
        sampling_config=sampling_config,
        run_mode=RunMode.AUTO,
    )
    assert resumed.task_id == state.task_id
    assert resumed.run_mode is RunMode.RESUME
    assert resumed.skip_keys == {(0, 0, 0)}


def test_json_eval_store_records_eval_and_score(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RWKV_EVAL_STORE", "json")
    monkeypatch.setenv("RWKV_EVAL_JSON_ROOT", str(tmp_path))

    service = create_eval_service()
    state = prepare_task_execution(
        service=service,
        dataset="ifeval_test",
        model="rwkv",
        is_param_search=False,
        job_name="instruction_following",
        run_mode=RunMode.RERUN,
    )
    service.insert_completion_payload(
        task_id=state.task_id,
        payload={
            "benchmark_name": "ifeval",
            "dataset_split": "test",
            "sample_index": 2,
            "repeat_index": 0,
            "pass_index": 0,
            "prompt1": "prompt",
            "completion1": "answer",
        },
    )
    inserted = service.ingest_eval_payloads(
        task_id=state.task_id,
        payloads=[
            {
                "sample_index": 2,
                "repeat_index": 0,
                "pass_index": 0,
                "context": "promptanswer",
                "answer": "answer",
                "ref_answer": "",
                "is_passed": True,
                "fail_reason": "",
            }
        ],
    )
    service.record_score_payload(task_id=state.task_id, payload={"metrics": {"accuracy": 1.0}})

    assert inserted == 1
    assert service.get_score_payload(task_id=state.task_id)["metrics"] == {"accuracy": 1.0}
    assert (tmp_path / "tasks" / state.task_id / "eval.jsonl").exists()
    assert (tmp_path / "tasks" / state.task_id / "score.json").exists()
