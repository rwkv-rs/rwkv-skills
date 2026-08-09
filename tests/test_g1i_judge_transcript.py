from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.g1i_strict46.judge_transcript import (
    JudgeTranscriptIntegrityError,
    JudgeTranscriptRecorder,
    JudgeTranscriptReplayer,
    JudgeTranscriptUsageError,
    load_judge_transcript,
    sanitize_endpoint_identity,
)
from src.eval.metrics.free_response import (
    LLMJudgeConfig,
    LLMJudgeStats,
    llm_judge_protocol,
)


ENDPOINT_WITH_SECRETS = (
    "https://alice:password@judge.example:8443/v1/chat?api_key=super-secret#token"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _resign(payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "transcript_sha256"}
    payload["transcript_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()


def _write_canonical(path: Path, payload: object) -> None:
    path.write_bytes(_canonical(payload) + b"\n")


class FakeJudge:
    def __init__(self, *, max_workers: int = 4) -> None:
        self.config = LLMJudgeConfig(
            api_key="sk-must-never-be-persisted",
            model="qwen3-judge",
            base_url=ENDPOINT_WITH_SECRETS,
            temperature=0.0,
            max_workers=max_workers,
        )
        self.client = SimpleNamespace(base_url=ENDPOINT_WITH_SECRETS)
        self.last_run_stats: LLMJudgeStats | None = None
        self.calls: list[list[tuple[str, str, str]]] = []

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]:
        self.calls.append(list(items))
        results = [prediction != "wrong" for _question, _reference, prediction in items]
        self.last_run_stats = LLMJudgeStats(
            total=len(items),
            parsed_count=len(items),
            protocol=llm_judge_protocol(self.config),
        )
        return results


def _record_fixture(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeJudge, object]:
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    delegate = FakeJudge()
    recorder = JudgeTranscriptRecorder(path)
    first = recorder.wrap(delegate, scope="task:101")
    assert first.judge(
        [
            ("question\r\nline", "reference", "right"),
            ("question\nline", "reference", "right"),
            ("other", "reference", "wrong"),
        ]
    ) == [True, True, False]
    second = recorder.wrap(delegate, scope="task:202")
    assert second.judge([("question\nline", "reference", "right")]) == [True]
    return delegate, recorder.persist()


def test_record_deduplicates_normalized_inputs_and_redacts_endpoint_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "judge.json"
    delegate, artifact = _record_fixture(path, monkeypatch)

    assert sum(len(call) for call in delegate.calls) == 2
    called_questions = {
        question for call in delegate.calls for question, _reference, _prediction in call
    }
    assert called_questions == {"question\nline", "other"}
    assert artifact.payload["statistics"] == {
        "protocol_count": 1,
        "unique_input_count": 2,
        "actual_judge_call_count": 2,
        "coordinate_count": 4,
        "true_coordinate_count": 3,
        "false_coordinate_count": 1,
        "scope_count": 2,
    }
    serialized = path.read_text(encoding="utf-8")
    assert "sk-must-never-be-persisted" not in serialized
    assert "password" not in serialized
    assert "super-secret" not in serialized
    assert "alice" not in serialized
    assert "https://judge.example:8443/v1/chat" in serialized
    assert load_judge_transcript(path, expected_sha256=artifact.sha256).sha256 == artifact.sha256


def test_replay_is_network_free_order_independent_and_requires_full_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "judge.json"
    _delegate, artifact = _record_fixture(path, monkeypatch)
    replayer = JudgeTranscriptReplayer(artifact)
    first = replayer.wrap(
        FakeJudge().config,
        scope="task:101",
        endpoint_url=ENDPOINT_WITH_SECRETS,
    )
    second = replayer.wrap(
        FakeJudge().config,
        scope="task:202",
        endpoint_url=ENDPOINT_WITH_SECRETS,
    )

    assert first.judge(
        [
            ("other", "reference", "wrong"),
            ("question\nline", "reference", "right"),
            ("question\r\nline", "reference", "right"),
        ]
    ) == [False, True, True]
    assert second.judge([("question\nline", "reference", "right")]) == [True]
    replayer.assert_consumed()


def test_replay_fails_closed_for_missing_duplicate_and_drifted_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "judge.json"
    _delegate, artifact = _record_fixture(path, monkeypatch)

    missing = JudgeTranscriptReplayer(artifact)
    judge = missing.wrap(
        FakeJudge().config,
        scope="task:101",
        endpoint_url=ENDPOINT_WITH_SECRETS,
    )
    missing.wrap(
        FakeJudge().config,
        scope="task:202",
        endpoint_url=ENDPOINT_WITH_SECRETS,
    )
    judge.judge([("other", "reference", "wrong")])
    with pytest.raises(JudgeTranscriptUsageError, match="missing=3"):
        missing.assert_consumed()

    duplicate = JudgeTranscriptReplayer(artifact)
    judge = duplicate.wrap(
        FakeJudge().config,
        scope="task:202",
        endpoint_url=ENDPOINT_WITH_SECRETS,
    )
    judge.judge([("question\nline", "reference", "right")])
    with pytest.raises(JudgeTranscriptUsageError, match="duplicate/excess"):
        judge.judge([("question\nline", "reference", "right")])

    drift = JudgeTranscriptReplayer(artifact).wrap(
        FakeJudge().config,
        scope="task:101",
        endpoint_url=ENDPOINT_WITH_SECRETS,
    )
    with pytest.raises(JudgeTranscriptUsageError, match="unrecorded or drifted"):
        drift.judge([("question changed", "reference", "right")])


def test_replay_rejects_protocol_and_endpoint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "judge.json"
    _delegate, artifact = _record_fixture(path, monkeypatch)

    with pytest.raises(JudgeTranscriptUsageError, match="protocol or endpoint"):
        JudgeTranscriptReplayer(artifact).wrap(
            FakeJudge(max_workers=8).config,
            scope="task:101",
            endpoint_url=ENDPOINT_WITH_SECRETS,
        ).judge([("other", "reference", "wrong")])
    with pytest.raises(JudgeTranscriptUsageError, match="protocol or endpoint"):
        JudgeTranscriptReplayer(artifact).wrap(
            FakeJudge().config,
            scope="task:101",
            endpoint_url="https://different.example/v1",
        ).judge([("other", "reference", "wrong")])


@pytest.mark.parametrize("mutation", ["result", "input", "response_hash"])
def test_load_detects_entry_tampering(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "judge.json"
    _delegate, _artifact = _record_fixture(path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["entries"][0]
    if mutation == "result":
        entry["result"] = not entry["result"]
    elif mutation == "input":
        entry["input"]["question"] += " tampered"
    else:
        entry["response_sha256"] = "0" * 64
    _resign(payload)
    _write_canonical(path, payload)

    with pytest.raises(JudgeTranscriptIntegrityError):
        load_judge_transcript(path)


def test_load_rejects_duplicate_and_noncanonical_entry_order_even_if_resigned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_path = tmp_path / "duplicate.json"
    _delegate, _artifact = _record_fixture(duplicate_path, monkeypatch)
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    duplicate["entries"].append(dict(duplicate["entries"][0]))
    _resign(duplicate)
    _write_canonical(duplicate_path, duplicate)
    with pytest.raises(JudgeTranscriptIntegrityError, match="duplicate or not canonical"):
        load_judge_transcript(duplicate_path)

    order_path = tmp_path / "order.json"
    _delegate, _artifact = _record_fixture(order_path, monkeypatch)
    reordered = json.loads(order_path.read_text(encoding="utf-8"))
    reordered["entries"].reverse()
    _resign(reordered)
    _write_canonical(order_path, reordered)
    with pytest.raises(JudgeTranscriptIntegrityError, match="duplicate or not canonical"):
        load_judge_transcript(order_path)


def test_recording_is_thread_safe_and_calls_each_unique_input_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    path = tmp_path / "concurrent.json"
    delegate = FakeJudge()
    recorder = JudgeTranscriptRecorder(path)
    judge = recorder.wrap(delegate, scope="task:concurrent")
    item = ("same", "reference", "right")

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _index: judge.judge([item])[0], range(64)))
    assert results == [True] * 64
    with ThreadPoolExecutor(max_workers=4) as executor:
        artifacts = list(executor.map(lambda _index: recorder.persist(), range(4)))

    assert sum(len(call) for call in delegate.calls) == 1
    assert {artifact.sha256 for artifact in artifacts} == {artifacts[0].sha256}
    assert artifacts[0].payload["statistics"]["coordinate_count"] == 64


def test_existing_identical_transcript_replays_in_record_mode_without_api_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "judge.json"
    _first_delegate, first_artifact = _record_fixture(path, monkeypatch)

    second_delegate = FakeJudge()
    recorder = JudgeTranscriptRecorder(path)
    task_101 = recorder.wrap(second_delegate, scope="task:101")
    task_202 = recorder.wrap(second_delegate, scope="task:202")
    assert task_101.judge(
        [
            ("other", "reference", "wrong"),
            ("question\nline", "reference", "right"),
            ("question\nline", "reference", "right"),
        ]
    ) == [False, True, True]
    assert task_202.judge([("question\nline", "reference", "right")]) == [True]
    second_artifact = recorder.persist()

    assert second_delegate.calls == []
    assert second_artifact.sha256 == first_artifact.sha256
    assert path.read_bytes() == _canonical(first_artifact.payload) + b"\n"


def test_record_requires_seed_42_and_endpoint_sanitizer_strips_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(JudgeTranscriptUsageError, match="PYTHONHASHSEED=42"):
        JudgeTranscriptRecorder(tmp_path / "judge.json")

    assert sanitize_endpoint_identity(ENDPOINT_WITH_SECRETS) == (
        "https://judge.example:8443/v1/chat"
    )
    assert sanitize_endpoint_identity("https://judge.example/v1/") == (
        "https://judge.example/v1"
    )


def test_zero_call_scope_is_persisted_and_must_be_declared_during_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    path = tmp_path / "zero-call.json"
    delegate = FakeJudge()
    recorder = JudgeTranscriptRecorder(path)
    recorder.wrap(delegate, scope="task:all-exact")
    artifact = recorder.persist()

    assert delegate.calls == []
    assert artifact.payload["entries"] == []
    assert artifact.payload["statistics"]["coordinate_count"] == 0
    replayer = JudgeTranscriptReplayer(artifact)
    replayer.wrap(
        delegate.config,
        scope="task:all-exact",
        endpoint_url=ENDPOINT_WITH_SECRETS,
    )
    replayer.assert_consumed()
