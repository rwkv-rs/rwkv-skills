from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.oneoff import audit_free_response_extractor_global as audit
from scripts.oneoff import merge_free_response_global_audit as merge
from scripts.oneoff import merge_free_response_global_audit_parts as legacy_merge
from scripts.oneoff import run_free_response_global_audit_parts as launcher


class _SequenceScorer:
    _verify_with_configured_timeout = True

    def __init__(self, outcomes: list[tuple[bool, str, str]]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    @staticmethod
    def _strategy_a_prompt(_payload: dict) -> str:
        return "question"

    def score_free_response_strategy(self, *_args, **_kwargs):
        passed, answer, fail_reason = self._outcomes[self.calls]
        self.calls += 1
        return SimpleNamespace(
            math_passed=passed,
            display_answer=answer,
            fail_reason=fail_reason,
        )


def _score(module: _SequenceScorer):
    stats: Counter[str] = Counter()
    result = audit._score_math(
        module,
        "strategy_a",
        {"sample_index": 0, "repeat_index": 0},
        "question",
        "7",
        stats,
    )
    return result, stats


def test_timeout_detection_uses_exact_fail_reason() -> None:
    module = _SequenceScorer([(False, "", "non_timeout_parse_error")])

    result, stats = _score(module)

    assert module.calls == 1
    assert result[4] is False
    assert result[5]["status"] == "not_retried"
    assert stats == Counter()


def test_one_resolved_retry_and_one_timeout_is_indeterminate() -> None:
    module = _SequenceScorer(
        [
            (False, "", "math_verify_timeout"),
            (True, "7", ""),
            (False, "", "prediction_parse_timeout"),
        ]
    )

    result, stats = _score(module)

    assert module.calls == 3
    assert result[4] is True
    assert result[5]["status"] == "partially_resolved_timeout"
    assert stats["unresolved"] == 1
    assert stats["indeterminate"] == 1


def test_conflicting_resolved_retries_are_indeterminate() -> None:
    module = _SequenceScorer(
        [
            (False, "", "reference_parse_timeout"),
            (True, "7", ""),
            (False, "8", "value_mismatch"),
        ]
    )

    result, stats = _score(module)

    assert result[4] is True
    assert result[5]["status"] == "conflicting_resolved_outcomes"
    assert stats["conflicting"] == 1
    assert stats["indeterminate"] == 1


def test_matching_resolved_retries_are_accepted() -> None:
    module = _SequenceScorer(
        [
            (False, "", "math_verify_timeout"),
            (True, "7", ""),
            (True, "7", ""),
        ]
    )

    result, stats = _score(module)

    assert result[:3] == (True, "7", "")
    assert result[4] is False
    assert result[5]["status"] == "resolved_consistently"
    assert stats["resolved"] == 1


def test_score_surface_preserves_distinct_math_and_final_passed() -> None:
    class _Scorer:
        _verify_with_configured_timeout = True

        @staticmethod
        def score_free_response_strategy(*_args, **_kwargs):
            return SimpleNamespace(
                math_passed=False,
                final_passed=True,
                display_answer="7",
                fail_reason="",
                judge_eligible=True,
            )

    result = audit._score_math(
        _Scorer(),
        "strategy_a",
        {"sample_index": 0, "repeat_index": 0},
        "q",
        "7",
        Counter(),
    )

    assert result[0] is False
    assert result[7] is True
    assert audit._score_signature(result)["math_passed"] is False
    assert audit._score_signature(result)["final_passed"] is True


def test_dataset_snapshot_is_content_addressed_and_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"question":"q","answer":"a"}\n')

    frozen, digest = audit._freeze_dataset_file(source, tmp_path / "snapshots")

    assert frozen.name == f"{digest}.jsonl"
    assert frozen.read_bytes() == source.read_bytes()
    assert frozen.stat().st_mode & 0o222 == 0
    source.write_bytes(b'{"question":"changed","answer":"a"}\n')
    second, second_digest = audit._freeze_dataset_file(
        source, tmp_path / "snapshots"
    )
    assert second_digest != digest
    assert second != frozen
    assert frozen.read_bytes() == b'{"question":"q","answer":"a"}\n'


def test_metadata_snapshot_digest_survives_integer_key_json_roundtrip() -> None:
    document = {
        "groups": {2: "strategy_a", 10: "strategy_b"},
        "task_counts": {2: 20, 10: 100},
        "nested": {10: {2: "value"}},
    }
    roundtripped = json.loads(json.dumps(document))

    audit_digest = audit._snapshot_digest(document)

    assert audit_digest == audit._snapshot_digest(roundtripped)
    assert audit_digest == launcher._snapshot_digest(document)
    assert audit_digest == launcher._snapshot_digest(roundtripped)


def test_dataset_snapshot_rejects_preexisting_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    payload = b'{"question":"q","answer":"a"}\n'
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    root = tmp_path / "snapshots"
    directory = root / digest[:2]
    directory.mkdir(parents=True)
    external = tmp_path / "external.jsonl"
    external.write_bytes(payload)
    external.chmod(0o444)
    (directory / f"{digest}.jsonl").symlink_to(external)

    with pytest.raises(RuntimeError, match="symlink dataset snapshot"):
        audit._freeze_dataset_file(source, root)


def test_judge_route_uses_final_passed_even_when_answer_is_unchanged() -> None:
    baseline = audit._judge_request(
        final_passed=False,
        judge_eligible=True,
        question="q",
        reference="7",
        answer="7",
    )
    candidate = audit._judge_request(
        final_passed=True,
        judge_eligible=True,
        question="q",
        reference="7",
        answer="7",
    )

    assert baseline == ("q", "7", "7")
    assert candidate is None
    assert baseline != candidate


def test_strategy_a_replay_expands_to_entire_task_family() -> None:
    families = {3: [3, 4, 5], 4: [3, 4, 5], 5: [3, 4, 5]}

    assert audit._replay_tasks_for_change(
        group="strategy_a", task_id=3, task_families=families
    ) == [3, 4, 5]
    assert audit._replay_tasks_for_change(
        group="strategy_b", task_id=4, task_families=families
    ) == [4]


def test_task_group_inventory_builds_complete_family_and_rejects_conflict() -> None:
    task_rows = [
        {
            "task_id": task_id,
            "config_path": "x",
            "evaluator": f"free_response:{group}",
            "sampling_config": {},
            "model_id": 1,
            "benchmark_id": 2,
            "task_created_at": "now",
            "benchmark_name": "b",
            "benchmark_split": "test",
            "model_name": "m",
            "arch_version": "g1i",
            "num_params": "1.5B",
        }
        for task_id, group in (
            (10, "strategy_a"),
            (11, "strategy_b"),
            (12, "strategy_c"),
        )
    ]

    class _Rows:
        def __init__(self, values):
            self.values = values

        def fetchall(self):
            return self.values

    class _Connection:
        def __init__(self, score_rows):
            self.score_rows = score_rows

        def execute(self, query):
            query = str(query)
            if "from scores" in query:
                assert "order by score_id, task_id" in query.lower()
                return _Rows(self.score_rows)
            assert "order by t.task_id" in query.lower()
            return _Rows(task_rows)

    mapping = {
        "strategy_a": 10,
        "strategy_b": 11,
        "strategy_c": 12,
    }
    groups, primary_c, _metadata, families = audit._task_groups(
        _Connection([{"score_id": 1, "task_id": 12, "metrics": {
            "strategy_task_ids": mapping
        }}])
    )
    assert groups == {10: "strategy_a", 11: "strategy_b", 12: "strategy_c"}
    assert primary_c == {12}
    assert families == {10: [10, 11, 12], 11: [10, 11, 12], 12: [10, 11, 12]}

    conflicting = [
        {"score_id": 1, "task_id": 12, "metrics": {
            "strategy_task_ids": mapping
        }},
        {"score_id": 2, "task_id": 12, "metrics": {
            "strategy_task_ids": {"strategy_b": 10}
        }},
    ]
    with pytest.raises(RuntimeError, match="conflicting strategy family"):
        audit._task_groups(_Connection(conflicting))


def test_frozen_code_artifact_is_content_addressed_and_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    frozen, digest = launcher._freeze_file(source, tmp_path / "frozen")

    assert frozen.stem == digest
    assert frozen.stat().st_mode & 0o222 == 0
    assert audit._verify_frozen_file(frozen, digest, label="test") == digest
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second, second_digest = launcher._freeze_file(source, tmp_path / "frozen")
    assert second_digest != digest
    assert second != frozen


def test_project_local_evaluator_cannot_drift_from_frozen_src_contract(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    evaluator = project / "src" / "evaluator.py"
    evaluator.write_text("VALUE = 1\n")
    _, _, records = launcher._freeze_project_contract(
        project,
        tmp_path / "contract",
    )
    evaluator.write_text("VALUE = 2\n")
    _, changed_digest = launcher._freeze_file(
        evaluator,
        tmp_path / "artifacts",
    )

    with pytest.raises(RuntimeError, match="changed while"):
        launcher._verify_artifact_matches_project_contract(
            evaluator,
            changed_digest,
            project_root=project,
            records=records,
            label="candidate module",
        )


def test_frozen_src_import_detects_post_snapshot_file_replacement(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    (project / "src" / "__init__.py").write_text("", encoding="utf-8")
    source = project / "src" / "audit_fixture.py"
    source.write_text("VALUE = 'frozen'\n", encoding="utf-8")
    frozen_root, _, records = launcher._freeze_project_contract(
        project, tmp_path / "contract"
    )
    manifest = {
        "local_src_contract": {
            "files": [
                value for value in records if str(value["path"]).startswith("src/")
            ]
        }
    }
    importer = audit._FrozenSrcImporter(frozen_root, manifest)
    spec = importer.find_spec("src.audit_fixture")
    assert spec is not None

    frozen_source = frozen_root / "src" / "audit_fixture.py"
    frozen_source.chmod(0o644)
    frozen_source.write_text("VALUE = 'replaced'\n", encoding="utf-8")
    module = importlib.util.module_from_spec(spec)
    importer.exec_module(module)

    assert module.VALUE == "frozen"
    with pytest.raises(RuntimeError, match="became writable|changed on disk"):
        audit._verify_loaded_src_modules(
            importer,
            {"src.audit_fixture": module},
        )


def test_frozen_project_contract_includes_non_python_src_resources(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / "src" / "package").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    (project / "src" / "__init__.py").write_text("", encoding="utf-8")
    (project / "src" / "package" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    resource = project / "src" / "package" / "answers.json"
    resource.write_text('{"answer": 7}\n', encoding="utf-8")

    frozen_root, _, records = launcher._freeze_project_contract(
        project,
        tmp_path / "contract",
    )

    by_path = {str(record["path"]): record for record in records}
    assert "src/package/answers.json" in by_path
    frozen_resource = frozen_root / "src" / "package" / "answers.json"
    assert frozen_resource.read_bytes() == resource.read_bytes()
    assert frozen_resource.stat().st_mode & 0o222 == 0
    manifest = {
        "local_src_contract": {
            "files": [
                value
                for value in records
                if str(value["path"]).startswith("src/")
            ]
        }
    }
    # Resource records are authenticated but cannot become import targets.
    importer = audit._FrozenSrcImporter(frozen_root, manifest)
    assert importer.find_spec("src.package.answers") is None


def test_dependency_manifest_bootstraps_only_from_frozen_project_tree(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    code_root = tmp_path / "code"
    frozen_project, project_digest, _ = launcher._freeze_project_contract(
        project_root,
        code_root,
    )
    artifacts = code_root / "artifacts"
    frozen_audit, audit_digest = launcher._freeze_file(
        Path(audit.__file__),
        artifacts,
    )
    frozen_evaluator, evaluator_digest = launcher._freeze_file(
        project_root / "src" / "eval" / "metrics" / "free_response.py",
        artifacts,
    )
    artifacts.chmod(0o555)
    manifest_path = tmp_path / "dependencies.json"
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(frozen_audit),
            "--baseline-module",
            str(frozen_evaluator),
            "--candidate-module",
            str(frozen_evaluator),
            "--output",
            str(tmp_path / "unused.json"),
            "--expected-audit-script-sha256",
            audit_digest,
            "--expected-baseline-module-sha256",
            evaluator_digest,
            "--expected-candidate-module-sha256",
            evaluator_digest,
            "--emit-dependency-manifest",
            str(manifest_path),
        ],
        check=True,
        cwd=frozen_project,
        env=launcher._child_environment(frozen_project, {}),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["project_contract"]["sha256"] == project_digest
    assert manifest["frozen_project_root"] == str(frozen_project)
    for name in (
        "math-verify",
        "sympy",
        "latex2sympy2-extended",
        "antlr4-python3-runtime",
        "psycopg",
    ):
        assert manifest["packages"][name]["version"]
        assert manifest["packages"][name]["file_count"] > 0
    for loaded_distribution in (
        "openai",
        "httpx",
        "tqdm",
        "mpmath",
        "psycopg-binary",
    ):
        assert loaded_distribution in manifest["packages"]
    assert manifest_path.stat().st_mode & 0o222 == 0

    probe = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            """
import importlib.util
import json
from pathlib import Path
import sys

audit_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("dependency_probe", audit_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
module._load_psycopg()
from math_verify import parse, verify
parsed = parse("1")
verify(parsed, parsed)
module._verify_dependency_environment(manifest)
""",
            str(frozen_audit),
            str(manifest_path),
        ],
        check=True,
        cwd=frozen_project,
        env=launcher._child_environment(frozen_project, {}),
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0


def test_dependency_verifier_rejects_a_late_unmanifested_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"packages": {"known-package": {}}}
    monkeypatch.setattr(
        audit,
        "_dependency_environment",
        lambda *, package_names: expected,
    )
    monkeypatch.setattr(
        audit,
        "_loaded_distribution_names",
        lambda: {"known-package", "late-package"},
    )

    with pytest.raises(RuntimeError, match="late-package"):
        audit._verify_dependency_environment(expected)


def test_dependency_discovery_settles_after_runtime_and_metadata_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: set[str] = set()
    observed_package_sets: list[set[str]] = []

    monkeypatch.setattr(
        audit,
        "_load_psycopg",
        lambda: loaded.add("selected-driver-backend"),
    )

    def load_scorer(_module, *, label: str) -> None:
        loaded.add(f"{label}-runtime")

    monkeypatch.setattr(
        audit,
        "_load_scorer_runtime_dependencies",
        load_scorer,
    )
    monkeypatch.setattr(
        audit,
        "_loaded_distribution_names",
        lambda: set(loaded),
    )

    def dependency_environment(*, package_names: set[str]):
        observed_package_sets.append(set(package_names))
        if len(observed_package_sets) == 1:
            loaded.add("metadata-runtime")
        return {"packages": {name: {} for name in sorted(package_names)}}

    monkeypatch.setattr(
        audit,
        "_dependency_environment",
        dependency_environment,
    )
    monkeypatch.setattr(
        audit,
        "_verify_dependency_environment",
        lambda manifest: manifest,
    )

    manifest = audit._settled_dependency_environment(
        baseline=object(),
        candidate=object(),
    )

    expected_runtime = {
        "selected-driver-backend",
        "baseline scorer-runtime",
        "candidate scorer-runtime",
        "metadata-runtime",
    }
    assert expected_runtime.issubset(manifest["packages"])
    assert len(observed_package_sets) == 2
    assert "metadata-runtime" not in observed_package_sets[0]
    assert "metadata-runtime" in observed_package_sets[1]


def _production_parts() -> list[dict]:
    task_inventory = {
        "3": {"group": "strategy_a", "rows": 1},
        "4": {"group": "strategy_b", "rows": 1},
        "5": {"group": "strategy_c", "rows": 1},
    }
    inventory_digest = hashlib.sha256(
        merge._canonical_json_bytes(task_inventory)
    ).hexdigest()
    task_families = {"3": [3], "4": [4], "5": [5]}
    task_families_digest = hashlib.sha256(
        merge._canonical_json_bytes(task_families)
    ).hexdigest()
    audit_mode = {
        "full_scan_a": True,
        "full_real_scorer": True,
        "max_structural_rows": None,
        "database_snapshot_imported": True,
        "metadata_snapshot_digest_verified": True,
        "question_source": "current_production_snapshot",
        "order_consistency_probe_rows": 8,
        "strategy_a_selection": "all_rows",
        "independent_order_probe": True,
        "stable_row_order": "task_id,completions_id,eval_id",
        "frozen_code_verified": True,
        "frozen_src_contract_verified": True,
        "dependency_manifest_verified": True,
        "atomic_part_artifact": True,
    }
    metadata_snapshot = {
        "schema_version": "free-response-global-audit-metadata.v2",
        "digest": "a" * 64,
        "database_identity": {"exported_snapshot_id": "00000003-00000001-1"},
        "task_inventory_digest": inventory_digest,
        "task_families_digest": task_families_digest,
        "dataset_digests": {
            "dataset": {
                "file_sha256": "b" * 64,
                "records_sha256": "c" * 64,
                "record_count": 1,
            }
        },
    }
    def tree_contract(records: list[dict]) -> dict:
        normalized = sorted(records, key=lambda value: str(value["path"]))
        return {
            "file_count": len(normalized),
            "total_bytes": sum(int(value["bytes"]) for value in normalized),
            "sha256": hashlib.sha256(
                merge._canonical_json_bytes(normalized)
            ).hexdigest(),
            "files": normalized,
        }

    src_records = [
        {"path": "src/__init__.py", "bytes": 0, "sha256": "2" * 64}
    ]
    metadata_records = [
        {"path": "pyproject.toml", "bytes": 1, "sha256": "3" * 64}
    ]
    local_src_contract = tree_contract(src_records)
    metadata_contract = tree_contract(metadata_records)
    project_contract = tree_contract([*src_records, *metadata_records])
    packages = {
        name: {
            **tree_contract(
                [
                    {
                        "path": f"/site-packages/{name}/module.py",
                        "bytes": 1,
                        "sha256": "4" * 64,
                    }
                ]
            ),
            "version": "1",
        }
        for name in (
            "math-verify",
            "sympy",
            "latex2sympy2-extended",
            "antlr4-python3-runtime",
            "psycopg",
        )
    }
    dependency_environment = {
        "schema_version": "free-response-global-audit-dependencies.v1",
        "python": "test",
        "python_executable": "/python",
        "python_cache_tag": "cpython-test",
        "platform": "test",
        "machine": "test",
        "packages": packages,
        "frozen_project_root": f"/frozen/project-{project_contract['sha256']}",
        "local_src_contract": local_src_contract,
        "project_metadata_contract": metadata_contract,
        "project_contract": project_contract,
        "scoring_environment": {},
    }
    dependency_environment["sha256"] = hashlib.sha256(
        merge._canonical_json_bytes(dependency_environment)
    ).hexdigest()
    dependency_file_sha = hashlib.sha256(
        merge._canonical_json_bytes(dependency_environment)
    ).hexdigest()
    common = {
        "schema_version": "free-response-global-audit-part.v3",
        "database": "db",
        "database_rows": 3,
        "tasks": 3,
        "strategy_totals": {
            "strategy_a": 1,
            "strategy_b": 1,
            "strategy_c": 1,
        },
        "primary_c_tasks": 0,
        "baseline_module_sha256": "d" * 64,
        "candidate_module_sha256": "e" * 64,
        "audit_script_sha256": "1" * 64,
        "dependency_manifest_file_sha256": dependency_file_sha,
        "dependency_environment": dependency_environment,
        "audit_mode": audit_mode,
        "metadata_snapshot": metadata_snapshot,
        "task_inventory": task_inventory,
        "task_inventory_digest": inventory_digest,
        "task_families": task_families,
        "math_fast_integer_match_env": None,
        "math_fast_integer_match_enabled": {
            "baseline": False,
            "candidate": False,
        },
        "sql_answer_cue_regex": "x",
        "structural_rows_scanned": 1,
        "changed_verification_windows": 0,
        "proof_equivalent_rows": 0,
        "judgement_rows": 0,
        "stored_noncomparable_rows": 0,
        "stored_reference_drift_rows": 0,
        "real_scorer_rows": 1,
        "full_candidate_scores": 1,
        "full_baseline_scores": 1,
        "scoring_errors": 0,
        "indeterminate_rows": 0,
        "deterministic_surface_changed_rows": 0,
        "judge_input_affected_rows": 0,
        "replay_affected_rows": 0,
        "replay_affected_by_task": {},
        "replay_affected_task_ids": [],
        "replay_affected_reasons": {},
        "blocking_timeout_count": 0,
        "module_order_consistency": {
            "probe_processes": "two_independent_processes_per_row",
            "orders": ["candidate_then_baseline", "baseline_then_candidate"],
            "probed_rows": 1,
            "conflict_count": 0,
            "conflicts": [],
            "timeout_events": {},
        },
        "proof_equivalent_reasons": {},
        "proof_equivalent_rows_by_strategy": {},
        "real_scorer_reasons": {"forced_full_real_scorer": 1},
        "timeout_retries": {},
        "timeout_retries_by_implementation": {},
        "row_transitions": {"0->0": 1},
        "stored_final_transitions": {},
        "row_transitions_by_strategy": {},
        "stored_final_transitions_by_strategy": {},
        "canonical_fingerprints": {},
        "primary_c_canonical_fingerprints": {},
        "a_sql_prefilter_superset_proof": {
            "exhaustive": True,
            "complement_rows_scanned": 0,
            "violations": [],
        },
        "cell_deltas": [],
        "changes": [],
        "audit_scope": {"proved": "deterministic"},
        "historical_generation_provenance": {"status": "unbound"},
    }
    parts: list[dict] = []
    for task_id, group in ((3, "strategy_a"), (4, "strategy_b"), (5, "strategy_c")):
        part = {
                **common,
                "requested_groups": [group],
                "partition": {
                    "count": 1,
                    "index": 0,
                    "selected_task_ids": [task_id],
                    "expected_task_counts": {str(task_id): 1},
                    "scanned_task_counts": {str(task_id): 1},
                },
                "_artifact": f"{group}.json",
                "_expected_group": group,
                "_expected_partition_index": 0,
            }
        part["_artifact_sha256"] = hashlib.sha256(
            f"{group}:0".encode("utf-8")
        ).hexdigest()
        part["_artifact_bytes"] = 1
        parts.append(part)
    return parts


def test_merge_preflight_accepts_exact_production_inventory() -> None:
    result = merge._validate_production_parts(
        _production_parts(), groups=list(audit.STRATEGIES), partitions=1
    )

    assert result["database_snapshot_id"] == "00000003-00000001-1"
    assert result["task_count"] == 3


def test_merge_preflight_rejects_legacy_mode() -> None:
    parts = _production_parts()
    parts[0]["audit_mode"] = {
        **parts[0]["audit_mode"],
        "full_real_scorer": False,
    }

    with pytest.raises(RuntimeError, match="legacy/non-production"):
        merge._validate_production_parts(
            parts, groups=list(audit.STRATEGIES), partitions=1
        )


def test_merge_preflight_rejects_per_task_scan_count_mismatch() -> None:
    parts = _production_parts()
    parts[1]["partition"] = {
        **parts[1]["partition"],
        "scanned_task_counts": {"4": 0},
    }

    with pytest.raises(RuntimeError, match="per-task scanned row count"):
        merge._validate_production_parts(
            parts, groups=list(audit.STRATEGIES), partitions=1
        )


def test_merge_preflight_rejects_dependency_manifest_drift() -> None:
    parts = _production_parts()
    for part in parts:
        part["dependency_environment"] = {
            **part["dependency_environment"],
            "python": "mutated-after-manifest",
        }

    with pytest.raises(RuntimeError, match="dependency-environment digest mismatch"):
        merge._validate_production_parts(
            parts,
            groups=list(audit.STRATEGIES),
            partitions=1,
        )


def test_merge_preflight_rejects_forged_order_evidence_and_partial_groups() -> None:
    parts = _production_parts()
    parts[0]["module_order_consistency"] = {
        **parts[0]["module_order_consistency"],
        "conflict_count": 1,
        "conflicts": [],
    }
    with pytest.raises(RuntimeError, match="module-order evidence"):
        merge._validate_production_parts(
            parts,
            groups=list(audit.STRATEGIES),
            partitions=1,
        )

    with pytest.raises(RuntimeError, match="requires exactly"):
        merge._validate_production_parts(
            _production_parts()[:1],
            groups=["strategy_a"],
            partitions=1,
        )


def test_merge_preflight_requires_exact_strategy_a_family_expansion() -> None:
    parts = _production_parts()
    families = {"3": [3, 4, 5], "4": [3, 4, 5], "5": [3, 4, 5]}
    digest = hashlib.sha256(merge._canonical_json_bytes(families)).hexdigest()
    for part in parts:
        part["task_families"] = families
        part["metadata_snapshot"] = {
            **part["metadata_snapshot"],
            "task_families_digest": digest,
        }
    target = parts[0]
    target["replay_affected_rows"] = 1
    target["replay_affected_by_task"] = {"3": 1, "4": 1, "5": 1}
    target["replay_affected_task_ids"] = [3, 4, 5]
    target["changes"] = [
        {
            "task_id": 3,
            "group": "strategy_a",
            "judge_input_affected": False,
            "replay_affected": True,
        }
    ]

    merge._validate_production_parts(
        parts,
        groups=list(audit.STRATEGIES),
        partitions=1,
    )
    target["replay_affected_by_task"] = {"3": 1}
    target["replay_affected_task_ids"] = [3]
    with pytest.raises(RuntimeError, match="task expansion mismatch"):
        merge._validate_production_parts(
            parts,
            groups=list(audit.STRATEGIES),
            partitions=1,
        )


def test_merge_preflight_rejects_asymmetric_or_noncanonical_task_families() -> None:
    for families, message in (
        ({"3": [3, 4], "4": [4], "5": [5]}, "asymmetric"),
        ({"3": [3, 3], "4": [4], "5": [5]}, "non-canonical"),
    ):
        parts = _production_parts()
        digest = hashlib.sha256(merge._canonical_json_bytes(families)).hexdigest()
        for part in parts:
            part["task_families"] = families
            part["metadata_snapshot"] = {
                **part["metadata_snapshot"],
                "task_families_digest": digest,
            }
        with pytest.raises(RuntimeError, match=message):
            merge._validate_production_parts(
                parts,
                groups=list(audit.STRATEGIES),
                partitions=1,
            )


def _write_parts(prefix: Path, parts: list[dict]) -> None:
    for part in parts:
        group = str(part["_expected_group"])
        index = int(part["_expected_partition_index"])
        path = Path(f"{prefix}_{group}_p{index}.json")
        serialized = {
            key: value for key, value in part.items() if not key.startswith("_")
        }
        path.write_text(json.dumps(serialized), encoding="utf-8")
        path.chmod(0o444)


@pytest.mark.parametrize(
    "failure",
    (
        "scoring_error",
        "timeout",
        "indeterminate",
        "one_to_zero",
        "judge_input_affected",
        "replay_affected",
    ),
)
def test_strict_merge_gate_rejects_every_blocking_outcome(
    monkeypatch,
    tmp_path: Path,
    failure: str,
) -> None:
    parts = _production_parts()
    target = parts[0]
    if failure == "scoring_error":
        target["scoring_errors"] = 1
    elif failure == "timeout":
        target["timeout_retries"] = {"initial_timeout": 1}
    elif failure == "indeterminate":
        target["indeterminate_rows"] = 1
    elif failure == "one_to_zero":
        target["row_transitions"] = {"1->0": 1}
        target["replay_affected_rows"] = 1
        target["replay_affected_by_task"] = {"3": 1}
        target["replay_affected_task_ids"] = [3]
        target["changes"] = [
            {
                "task_id": 3,
                "group": "strategy_a",
                "transition": "1->0",
                "explanation": "explicit_retraction",
                "judge_input_affected": False,
                "replay_affected": True,
            }
        ]
    elif failure == "judge_input_affected":
        target["judge_input_affected_rows"] = 1
        target["replay_affected_rows"] = 1
        target["replay_affected_by_task"] = {"3": 1}
        target["replay_affected_task_ids"] = [3]
        target["changes"] = [
            {
                "task_id": 3,
                "group": "strategy_a",
                "judge_input_affected": True,
                "replay_affected": True,
            }
        ]
    elif failure == "replay_affected":
        target["replay_affected_rows"] = 1
        target["replay_affected_by_task"] = {"3": 1}
        target["replay_affected_task_ids"] = [3]
        target["changes"] = [
            {
                "task_id": 3,
                "group": "strategy_a",
                "judge_input_affected": False,
                "replay_affected": True,
            }
        ]
    prefix = tmp_path / "parts"
    output_json = tmp_path / "merged.json"
    output_md = tmp_path / "merged.md"
    _write_parts(prefix, parts)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--input-prefix",
            str(prefix),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--partitions",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        merge.main()

    assert exc_info.value.code == 1
    failed_json = merge._artifact_path(output_json, "failed")
    assert not output_json.exists()
    assert not merge._acceptance_manifest_path(output_json).exists()
    assert json.loads(failed_json.read_text(encoding="utf-8"))["gate"][
        "passed"
    ] is False


def test_strict_merge_rejects_missing_expected_part(tmp_path: Path) -> None:
    prefix = tmp_path / "parts"
    _write_parts(prefix, _production_parts()[:-1])

    with pytest.raises(FileNotFoundError, match="missing audit artifacts"):
        merge._load_parts(prefix, list(audit.STRATEGIES), 1)


def test_merge_rejects_writable_or_symlinked_part_artifacts(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "parts"
    _write_parts(prefix, _production_parts())
    writable = Path(f"{prefix}_strategy_a_p0.json")
    writable.chmod(0o644)
    with pytest.raises(RuntimeError, match="audit part is writable"):
        merge._load_parts(prefix, list(audit.STRATEGIES), 1)

    writable.chmod(0o444)
    target = Path(f"{prefix}_strategy_b_p0.real.json")
    original = Path(f"{prefix}_strategy_b_p0.json")
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        merge._load_parts(prefix, list(audit.STRATEGIES), 1)


def test_legacy_merge_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="legacy free-response audit merge"):
        legacy_merge.main()


def test_order_probe_runs_each_order_in_an_independent_process(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frozen_root = tmp_path / "project-test"
    frozen_root.mkdir()
    frozen_root.chmod(0o555)
    monkeypatch.chdir(frozen_root)
    monkeypatch.setenv("RWKV_AUDIT_FROZEN_PROJECT_ROOT", str(frozen_root))
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append({"command": command, **kwargs})
        mode = command[command.index("--order-probe-mode") + 1]
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "order": mode,
                    "results": {"baseline": {}, "candidate": {}},
                    "timeout_stats": {},
                }
            )
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    args = SimpleNamespace(
        baseline_module=tmp_path / "baseline.py",
        candidate_module=tmp_path / "candidate.py",
        expected_audit_script_sha256="a" * 64,
        expected_baseline_module_sha256="b" * 64,
        expected_candidate_module_sha256="c" * 64,
        dependency_manifest=tmp_path / "dependency.json",
        expected_dependency_manifest_sha256="d" * 64,
    )
    request = {
        "group": "strategy_a",
        "payload": {},
        "question": "q",
        "reference": "a",
    }

    audit._run_isolated_order_probe(
        args,
        mode="candidate_then_baseline",
        request=request,
    )
    audit._run_isolated_order_probe(
        args,
        mode="baseline_then_candidate",
        request=request,
    )

    assert len(calls) == 2
    assert calls[0]["command"] is not calls[1]["command"]
    assert calls[0]["cwd"] == frozen_root.resolve()
    assert calls[1]["cwd"] == frozen_root.resolve()
    assert calls[0]["env"]["PYTHONPATH"] == str(frozen_root.resolve())


def test_successful_merge_uses_acceptance_manifest_as_atomic_commit_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "parts"
    output_json = tmp_path / "merged.json"
    output_md = tmp_path / "merged.md"
    _write_parts(prefix, _production_parts())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--input-prefix",
            str(prefix),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--partitions",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        merge.main()

    assert exc_info.value.code == 0
    acceptance_path = merge._acceptance_manifest_path(output_json)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    merged = json.loads(output_json.read_text(encoding="utf-8"))
    assert acceptance["accepted"] is True
    assert acceptance["merge_script_sha256"] == merged["merge_script_sha256"]
    assert acceptance["production_provenance"]["input_artifacts_sha256"] == (
        merged["input_artifacts_sha256"]
    )
    for record in merged["input_artifacts"]:
        assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == (
            record["sha256"]
        )
    assert acceptance["json"]["sha256"] == hashlib.sha256(
        output_json.read_bytes()
    ).hexdigest()
    assert acceptance["markdown"]["sha256"] == hashlib.sha256(
        output_md.read_bytes()
    ).hexdigest()
    for path in (output_json, output_md, acceptance_path):
        assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(SystemExit) as second:
        merge.main()
    assert second.value.code != 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_terminate_all_kills_and_reaps_entire_process_group() -> None:
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        launcher._terminate_all([process])
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_terminate_all_signals_group_after_leader_already_exited(
    monkeypatch,
) -> None:
    calls: list[tuple[int, object]] = []
    process = SimpleNamespace(
        pid=4321,
        poll=lambda: 1,
        wait=lambda **_kwargs: 1,
    )

    state = {"killed": False}

    def fake_killpg(pid: int, sig: object) -> None:
        calls.append((pid, sig))
        if sig == launcher.signal.SIGKILL:
            state["killed"] = True
        elif sig == 0 and state["killed"]:
            raise ProcessLookupError

    monkeypatch.setattr(
        launcher.os,
        "killpg",
        fake_killpg,
    )
    ticks = iter((0.0, 11.0, 11.0))
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(ticks))

    launcher._terminate_all([process])

    assert (4321, launcher.signal.SIGTERM) in calls
    assert (4321, launcher.signal.SIGKILL) in calls


def test_bounded_worker_queue_never_exceeds_resident_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processes: list[SimpleNamespace] = []
    max_resident = 0
    status_counts: list[dict[str, int]] = []

    def fake_start(spec, *, cwd, env):
        nonlocal max_resident
        assert cwd == tmp_path
        assert env == {"FROZEN": "1"}
        process = SimpleNamespace(
            pid=1000 + len(processes),
            returncode=None,
        )
        process.poll = lambda process=process: process.returncode
        processes.append(process)
        max_resident = max(
            max_resident,
            sum(item.returncode is None for item in processes),
        )
        return launcher._WorkerHandle(spec=spec, process=process)

    def finish_resident_wave(_seconds: float) -> None:
        for process in processes:
            if process.returncode is None:
                process.returncode = 0

    monkeypatch.setattr(launcher, "_start_worker", fake_start)
    monkeypatch.setattr(launcher.time, "sleep", finish_resident_wave)
    specs = [
        launcher._WorkerSpec(
            group="strategy_a",
            index=index,
            command=["audit", str(index)],
            stdout_path=tmp_path / f"p{index}.out",
            stderr_path=tmp_path / f"p{index}.err",
        )
        for index in range(5)
    ]

    handles = launcher._run_bounded_workers(
        specs,
        max_workers=2,
        cwd=tmp_path,
        env={"FROZEN": "1"},
        publish_status=lambda _records, counts: status_counts.append(
            dict(counts)
        ),
        poll_interval_seconds=0,
    )

    assert len(handles) == len(specs)
    assert max_resident == 2
    assert all(counts["running"] <= 2 for counts in status_counts)
    assert status_counts[-1] == {
        "pending": 0,
        "running": 0,
        "completed": 5,
        "failed": 0,
    }


def test_bounded_worker_queue_fails_closed_and_does_not_launch_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processes: list[SimpleNamespace] = []
    terminated: list[object] = []
    statuses: list[dict[str, int]] = []

    def fake_start(spec, *, cwd, env):
        del cwd, env
        process = SimpleNamespace(
            pid=2000 + len(processes),
            returncode=17 if not processes else None,
        )
        process.poll = lambda process=process: process.returncode
        processes.append(process)
        return launcher._WorkerHandle(spec=spec, process=process)

    monkeypatch.setattr(launcher, "_start_worker", fake_start)
    monkeypatch.setattr(
        launcher,
        "_terminate_all",
        lambda values: terminated.extend(values),
    )
    specs = [
        launcher._WorkerSpec(
            group="strategy_b",
            index=index,
            command=["audit", str(index)],
            stdout_path=tmp_path / f"p{index}.out",
            stderr_path=tmp_path / f"p{index}.err",
        )
        for index in range(3)
    ]

    with pytest.raises(RuntimeError, match="strategy_b/p0:17"):
        launcher._run_bounded_workers(
            specs,
            max_workers=2,
            cwd=tmp_path,
            env={},
            publish_status=lambda _records, counts: statuses.append(
                dict(counts)
            ),
            poll_interval_seconds=0,
        )

    assert len(processes) == 2
    assert terminated == processes
    assert statuses[-1] == {
        "pending": 1,
        "running": 1,
        "completed": 0,
        "failed": 1,
    }


class _FakeExporter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    def execute(self, query: str):
        self.events.append(str(query))
        if "pg_export_snapshot" in str(query):
            return SimpleNamespace(
                fetchone=lambda: {"snapshot_id": "00000003-00000001-1"}
            )
        return SimpleNamespace(fetchone=lambda: {})

    def rollback(self) -> None:
        self.events.append("rollback")

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


class _CompletedProcess:
    pid = 123

    @staticmethod
    def poll() -> int:
        return 0


def test_inventory_failure_is_bounded_redacted_and_not_inherited(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    diagnostic = tmp_path / "inventory.stderr.txt"
    secret = "do-not-leak-this-password"
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        kwargs["stderr"].write(
            (
                "inventory failed\n"
                f"PG_PASSWORD={secret}\n"
                f"password={secret}\n"
            ).encode("utf-8")
        )
        kwargs["stderr"].flush()
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 23") as error:
        launcher._run_inventory_command(
            ["python", "inventory.py"],
            cwd=tmp_path,
            env={"PG_PASSWORD": secret},
            configured_env={"PG_PASSWORD": secret},
            diagnostic_path=diagnostic,
        )

    assert observed["check"] is False
    assert observed["stdout"] is subprocess.DEVNULL
    assert observed["stderr"] is not None
    assert secret not in str(error.value)
    assert "inventory failed" in str(error.value)
    assert "<redacted>" in str(error.value)
    stored = diagnostic.read_text(encoding="utf-8")
    assert secret not in stored
    assert "inventory failed" in stored
    assert "<redacted>" in stored
    assert diagnostic.stat().st_mode & 0o222 == 0


def test_inventory_diagnostic_tail_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    diagnostic = tmp_path / "inventory.stderr.txt"

    def fake_run(_command, **kwargs):
        kwargs["stderr"].write(
            b"x" * (launcher.INVENTORY_DIAGNOSTIC_TAIL_BYTES * 3)
        )
        kwargs["stderr"].flush()
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 9"):
        launcher._run_inventory_command(
            ["python", "inventory.py"],
            cwd=tmp_path,
            env={},
            configured_env={},
            diagnostic_path=diagnostic,
        )

    assert diagnostic.stat().st_size <= (
        launcher.INVENTORY_DIAGNOSTIC_TAIL_BYTES
    )


def test_launcher_holds_exported_snapshot_and_passes_it_to_every_child(
    monkeypatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "scripts" / "oneoff").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='audit-test'\n")
    (project / "src" / "__init__.py").write_text("VALUE = 1\n")
    audit_source = project / "scripts" / "oneoff" / "audit.py"
    audit_source.write_text("# frozen audit\n")
    baseline_source = tmp_path / "baseline.py"
    baseline_source.write_text("VALUE = 'baseline'\n")
    candidate_source = tmp_path / "candidate.py"
    candidate_source.write_text("VALUE = 'candidate'\n")
    env = tmp_path / ".env"
    env.write_text("PG_HOST=h\nPG_PORT=1\nPG_USER=u\nPG_PASSWORD=p\n")
    prefix = tmp_path / "audit"
    status = tmp_path / "status.json"
    events: list[str] = []
    exporter = _FakeExporter(events)
    run_commands: list[list[str]] = []
    worker_commands: list[list[str]] = []

    monkeypatch.setattr(
        launcher,
        "psycopg",
        SimpleNamespace(connect=lambda *_a, **_k: exporter),
    )

    def fake_run(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
        stdout=None,
        stderr=None,
    ):
        assert Path(cwd).name.startswith("project-")
        assert env["PYTHONPATH"] == str(Path(cwd).resolve())
        assert env["RWKV_AUDIT_FROZEN_PROJECT_ROOT"] == str(
            Path(cwd).resolve()
        )
        run_commands.append(command)
        if "--emit-dependency-manifest" in command:
            assert check is True
            assert stdout is None
            assert stderr is None
            dependency_path = Path(
                command[command.index("--emit-dependency-manifest") + 1]
            )
            dependency_path.write_text(
                json.dumps({"schema_version": "test", "sha256": "test"}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)
        assert check is False
        assert stdout is subprocess.DEVNULL
        assert stderr is not None
        stderr.write(b"inventory diagnostic\n")
        stderr.flush()
        assert not exporter.closed
        metadata_path = Path(command[command.index("--metadata-cache") + 1])
        document = {
            "database_identity": {
                "exported_snapshot_id": "00000003-00000001-1"
            },
            "dataset_sources": {
                "d": {"file_sha256": "a", "records_sha256": "b"}
            },
        }
        document["snapshot_digest"] = launcher._snapshot_digest(document)
        metadata_path.write_text(json.dumps(document), encoding="utf-8")
        metadata_path.chmod(0o444)
        return SimpleNamespace(returncode=0)

    def fake_popen(command: list[str], **_kwargs):
        assert not exporter.closed
        assert Path(_kwargs["cwd"]).name.startswith("project-")
        assert _kwargs["start_new_session"] is True
        worker_commands.append(command)
        return _CompletedProcess()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        launcher.sys,
        "argv",
        [
            "launcher",
            "--audit-script",
            str(audit_source),
            "--baseline-module",
            str(baseline_source),
            "--candidate-module",
            str(candidate_source),
            "--output-prefix",
            str(prefix),
            "--status",
            str(status),
            "--env",
            str(env),
            "--partitions",
            "1",
            "--max-workers",
            "2",
        ],
    )

    launcher.main()

    assert len(run_commands) == 2
    assert "--emit-dependency-manifest" in run_commands[0]
    inventory_command = run_commands[1]
    assert "--refresh-metadata-snapshot" in inventory_command
    assert inventory_command[
        inventory_command.index("--database-snapshot-id") + 1
    ] == (
        "00000003-00000001-1"
    )
    assert "--dataset-source-root" in inventory_command
    assert "--expected-dependency-manifest-sha256" in inventory_command
    inventory_diagnostic = Path(f"{prefix}_inventory.stderr.txt")
    assert inventory_diagnostic.read_text(encoding="utf-8") == (
        "inventory diagnostic\n"
    )
    assert inventory_diagnostic.stat().st_mode & 0o222 == 0
    assert len(worker_commands) == 3
    for command in worker_commands:
        assert command[command.index("--database-snapshot-id") + 1] == (
            "00000003-00000001-1"
        )
        assert "--metadata-snapshot-digest" in command
        assert "--full-scan-a" in command
        assert "--full-real-scorer" in command
    assert events.index("rollback") > max(
        index for index, value in enumerate(events) if "pg_export_snapshot" in value
    )
    assert exporter.closed
    assert json.loads(status.read_text(encoding="utf-8"))[
        "database_snapshot_id"
    ] == "00000003-00000001-1"
    final_status = json.loads(status.read_text(encoding="utf-8"))
    assert final_status["audit_mode"]["max_workers"] == 2
    assert final_status["part_counts"] == {
        "pending": 0,
        "running": 0,
        "completed": 3,
        "failed": 0,
    }
