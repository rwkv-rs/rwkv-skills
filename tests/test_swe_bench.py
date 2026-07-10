from __future__ import annotations

import json
from pathlib import Path

from src.eval.tasks.coding.swe_bench import (
    build_swebench_prompt,
    build_swebench_prompt_with_trace,
    extract_swebench_patch,
    load_swebench_harness_result,
    write_swebench_predictions,
)
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.long_doc_evidence import LongDocEvidenceConfig


def test_extract_swebench_patch_prefers_diff_fence() -> None:
    completion = """
<think>analysis</think>
```diff
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-old
+new
```
"""

    patch = extract_swebench_patch(completion)

    assert patch.startswith("diff --git a/pkg/mod.py b/pkg/mod.py")
    assert "<think>" not in patch


def test_build_swebench_prompt_excludes_gold_patch() -> None:
    record = CodeGenerationRecord(
        task_id="sympy__sympy-20590",
        prompt="Fix sympify.",
        metadata={
            "repo": "sympy/sympy",
            "base_commit": "abc123",
            "instance_id": "sympy__sympy-20590",
            "patch": "diff --git a/gold b/gold",
            "retrieved_context": "sympy/core/sympify.py: class SympifyError",
        },
    )

    prompt = build_swebench_prompt(record)

    assert "Fix sympify." in prompt
    assert "sympy/core/sympify.py" in prompt
    assert "diff --git a/gold b/gold" not in prompt


def test_build_swebench_naive_prompt_uses_raw_issue_only() -> None:
    record = CodeGenerationRecord(
        task_id="sympy__sympy-20590",
        prompt="Fix sympify.",
        metadata={
            "repo": "sympy/sympy",
            "base_commit": "abc123",
            "instance_id": "sympy__sympy-20590",
            "retrieved_context": "sympy/core/sympify.py: class SympifyError",
        },
    )

    prompt, trace = build_swebench_prompt_with_trace(record, prompt_profile="naive")

    assert prompt == "User: Fix sympify.\n\nAssistant: <think>\n</think>\n```diff\n"
    assert trace["prompt_profile"] == "naive"
    assert "sympy/core/sympify.py" not in prompt
    assert "Return only a unified git diff patch" not in prompt


def test_build_swebench_prompt_clamps_full_prompt_and_preserves_prefill() -> None:
    record = CodeGenerationRecord(
        task_id="sympy__sympy-20590",
        prompt="Fix sympify.",
        metadata={
            "instance_id": "sympy__sympy-20590",
            "retrieved_context": "\n".join(f"ctx line {index:04d} abcdefghijklmnop" for index in range(500)),
        },
    )

    prompt, trace = build_swebench_prompt_with_trace(record, max_prompt_chars=1200)

    assert len(prompt) <= 1200
    assert prompt.endswith("Assistant: <think>\n</think>\n```diff\n")
    assert "[...truncated...]" in prompt
    assert trace["prompt_chars"] == len(prompt)
    assert trace["max_prompt_chars"] == 1200
    assert trace["prompt_trimmed_chars"] > 0


def test_build_swebench_prompt_can_compact_retrieved_context() -> None:
    context = "\n".join(
        [f"noise row {index:03d}" for index in range(35)]
        + ["astropy/modeling/separable.py has nested CompoundModel separability logic"]
        + [f"archive row {index:03d}" for index in range(35)]
    )
    record = CodeGenerationRecord(
        task_id="astropy__astropy-12907",
        prompt="Fix separability_matrix for nested CompoundModel.",
        metadata={
            "instance_id": "astropy__astropy-12907",
            "retrieved_context": context,
            "patch": "diff --git a/gold b/gold",
        },
    )

    prompt, trace = build_swebench_prompt_with_trace(
        record,
        long_doc_config=LongDocEvidenceConfig(
            mode="lexical",
            max_chunk_chars=160,
            overlap_lines=1,
            min_long_text_chars=200,
            max_evidence_chunks=1,
            max_evidence_chars=260,
        ),
    )

    assert trace["mode"] == "lexical"
    assert trace["compacted"] is True
    assert trace["chunk_count"] > 1
    assert trace["selected_chunk_ids"]
    assert "mode=lexical" in prompt
    assert "separable.py has nested CompoundModel" in prompt
    assert "diff --git a/gold b/gold" not in prompt


def test_write_swebench_predictions_uses_official_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "sympy__sympy-20590",
                "instance_id": "sympy__sympy-20590",
                "prompt": "Fix sympify.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completions = [
        {
            "sample_index": 0,
            "repeat_index": 0,
            "pass_index": 0,
            "completion1": "```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n```",
        }
    ]
    output_path = tmp_path / "predictions.jsonl"

    write_swebench_predictions(completions, dataset_path=dataset_path, model_name="demo", output_path=output_path)

    [row] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert row["instance_id"] == "sympy__sympy-20590"
    assert row["model_name_or_path"] == "demo"
    assert row["model_patch"].startswith("diff --git")


def test_load_swebench_harness_result_normalizes_instance_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "evaluation_results" / "demo-run"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "instance_results.jsonl").write_text(
        json.dumps({"instance_id": "a__b-1", "resolved": True}) + "\n"
        + json.dumps({"instance_id": "a__b-2", "resolved": False}) + "\n",
        encoding="utf-8",
    )

    result = load_swebench_harness_result(run_id="demo-run", root=tmp_path / "evaluation_results")

    assert result.metrics["swebench_instances_submitted"] == 2
    assert result.metrics["swebench_instances_resolved"] == 1
    assert result.metrics["swebench_resolution_rate"] == 0.5
