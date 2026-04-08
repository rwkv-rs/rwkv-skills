from __future__ import annotations

from pathlib import Path

from src.eval.datasets.data_prepper.data_manager import prepare_dataset
from src.eval.datasets.runtime import read_jsonl_items


def test_prepare_dataset_materializes_gpqa_spec_with_requested_split(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.multiple_choice.gpqa._load_gpqa_rows",
        lambda split: [
            {
                "Question": f"{split} question",
                "Incorrect Answer 1": "wrong-1",
                "Incorrect Answer 2": "wrong-2",
                "Incorrect Answer 3": "wrong-3",
                "Correct Answer": "right",
                "Subdomain": "biology",
                "Writer's Difficulty Estimate": "hard",
                "Explanation": "because",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("gpqa", output_root, "diamond")

    assert paths == [output_root / "gpqa" / "diamond.jsonl"]
    rows = read_jsonl_items(paths[0])
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "diamond question"
    assert row["answer"] in {"A", "B", "C", "D"}
    assert sorted([row["A"], row["B"], row["C"], row["D"]]) == ["right", "wrong-1", "wrong-2", "wrong-3"]
    assert row["source"] == "gpqa"


def test_prepare_dataset_materializes_include_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.multiple_choice.include._include_config_names",
        lambda: ["english"],
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.multiple_choice.include._load_include_rows",
        lambda _config, _split: [
            {
                "id": "inc-1",
                "question": "Which option is correct?",
                "options": ["alpha", "beta", "gamma", "delta"],
                "answer": 2,
                "language": "en",
                "subject": "culture",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("include", output_root, "test")

    assert paths == [output_root / "include" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "question": "Which option is correct?",
            "answer": "C",
            "subject": "culture",
            "subset": "english",
            "source": "include-base-44",
            "A": "alpha",
            "B": "beta",
            "C": "gamma",
            "D": "delta",
            "id": "inc-1",
            "language": "en",
        }
    ]


def test_prepare_dataset_materializes_answer_judge_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.free_answer.answer_judge.iter_hf_dataset",
        lambda *_args, **_kwargs: [
            {
                "question": "What is 2+2?",
                "gt_answer": "4",
                "gen_answer": "four",
                "dataset_name": "judges-verdict",
                "item_name": "math-1",
                "annotations": [{"score": 1}, {"score": 0.5}, {"score": 1}],
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("answer_judge", output_root, "test")

    assert paths == [output_root / "answer-judge" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "problem": "What is 2+2?",
            "expected_answer": "4",
            "predicted_answer": "four",
            "expected_judgement": "Judgement: Yes",
            "comment": "judges-verdict mean_score=0.833",
            "source": "judges-verdict",
            "source_id": "math-1",
            "annotations": [{"score": 1}, {"score": 0.5}, {"score": 1}],
        }
    ]


def test_prepare_dataset_materializes_hle_all_split_with_category_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.free_answer.hle.iter_hf_dataset",
        lambda *_args, **_kwargs: [
            {
                "id": "math-1",
                "question": "math q",
                "answer": "4",
                "answer_type": "number",
                "rationale": "math rationale",
                "raw_subject": "algebra",
                "category": "Math",
                "author_name": "author-a",
                "canary": "c1",
                "image": None,
            },
            {
                "id": "cs-1",
                "question": "cs q",
                "answer": "stack",
                "answer_type": "text",
                "rationale": "cs rationale",
                "raw_subject": "algorithms",
                "category": "Computer Science/AI",
                "author_name": "author-b",
                "canary": "c2",
                "image": None,
            },
            {
                "id": "img-1",
                "question": "ignored",
                "answer": "ignored",
                "answer_type": "image",
                "rationale": "ignored",
                "raw_subject": "ignored",
                "category": "Math",
                "author_name": "author-c",
                "canary": "c3",
                "image": {"bytes": "not-text"},
            },
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("hle", output_root, "all")

    expected = [
        output_root / "hle" / "all.jsonl",
        output_root / "hle" / "text.jsonl",
        output_root / "hle" / "other.jsonl",
        output_root / "hle" / "human.jsonl",
        output_root / "hle" / "math.jsonl",
        output_root / "hle" / "phy.jsonl",
        output_root / "hle" / "cs.jsonl",
        output_root / "hle" / "bio.jsonl",
        output_root / "hle" / "chem.jsonl",
        output_root / "hle" / "eng.jsonl",
    ]
    assert paths == expected
    assert len(read_jsonl_items(output_root / "hle" / "all.jsonl")) == 2
    assert len(read_jsonl_items(output_root / "hle" / "text.jsonl")) == 2
    assert read_jsonl_items(output_root / "hle" / "math.jsonl") == [
        {
            "id": "math-1",
            "problem": "math q",
            "expected_answer": "4",
            "answer_type": "number",
            "reference_solution": "math rationale",
            "raw_subject": "algebra",
            "category": "Math",
            "author_name": "author-a",
            "canary": "c1",
        }
    ]
    assert read_jsonl_items(output_root / "hle" / "cs.jsonl") == [
        {
            "id": "cs-1",
            "problem": "cs q",
            "expected_answer": "stack",
            "answer_type": "text",
            "reference_solution": "cs rationale",
            "raw_subject": "algorithms",
            "category": "Computer Science/AI",
            "author_name": "author-b",
            "canary": "c2",
        }
    ]

    second = prepare_dataset("hle", output_root, "all")
    assert second == expected


def test_prepare_dataset_materializes_polymath_all_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.free_answer.polymath._polymath_config_names",
        lambda: ["en"],
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.free_answer.polymath._load_polymath_rows",
        lambda config, split: [
            {
                "id": f"{config}-{split}-0",
                "question": f"{config}:{split}: question",
                "answer": "42",
                "topic": "number_theory",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("polymath", output_root, "all")

    assert paths == [output_root / "polymath" / "all.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "id": "en-top-0",
            "problem": "en:top: question",
            "expected_answer": "42",
            "language": "en",
            "difficulty": "top",
            "source": "polymath",
            "topic": "number_theory",
        },
        {
            "id": "en-high-0",
            "problem": "en:high: question",
            "expected_answer": "42",
            "language": "en",
            "difficulty": "high",
            "source": "polymath",
            "topic": "number_theory",
        },
        {
            "id": "en-medium-0",
            "problem": "en:medium: question",
            "expected_answer": "42",
            "language": "en",
            "difficulty": "medium",
            "source": "polymath",
            "topic": "number_theory",
        },
        {
            "id": "en-low-0",
            "problem": "en:low: question",
            "expected_answer": "42",
            "language": "en",
            "difficulty": "low",
            "source": "polymath",
            "topic": "number_theory",
        },
    ]


def test_prepare_dataset_materializes_ceval_validation_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.multiple_choice.ceval._load_ceval_config_names",
        lambda: ["computer_network"],
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.multiple_choice.ceval._load_ceval_rows",
        lambda config, split: [
            {
                "question": f"{config}:{split}",
                "A": "opt-a",
                "B": "opt-b",
                "C": "opt-c",
                "D": "opt-d",
                "answer": "c",
                "id": "q1",
                "explanation": "because",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("ceval", output_root, "validation")

    assert paths == [output_root / "ceval" / "validation.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "question": "computer_network:val",
            "A": "opt-a",
            "B": "opt-b",
            "C": "opt-c",
            "D": "opt-d",
            "answer": "C",
            "subject": "computer_network",
            "dataset": "ceval",
            "id": "q1",
            "explanation": "because",
        }
    ]


def test_prepare_dataset_materializes_mmlu_pro_spec_with_canonical_registry_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.multiple_choice.mmlu_pro._load_mmlu_pro_rows",
        lambda split: [
            {
                "question": f"{split} question",
                "options": ["opt-a", "opt-b", "opt-c", "opt-d"],
                "answer": "b",
                "category": "computer science",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("mmlu_pro", output_root, "test")

    assert paths == [output_root / "mmlu-pro" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "question": "test question",
            "answer": "B",
            "subject": "computer_science",
            "subset": "computer_science",
            "examples_type": "mmlu_pro_few_shot_computer_science",
            "A": "opt-a",
            "B": "opt-b",
            "C": "opt-c",
            "D": "opt-d",
        }
    ]


def test_prepare_dataset_materializes_human_eval_cn_spec(tmp_path: Path, monkeypatch) -> None:
    def _fake_download(_cache_dir, source_root_name, files, _tasks) -> None:
        source_root = output_root / "cache" / "human_eval_cn" / source_root_name
        source_root.mkdir(parents=True, exist_ok=True)
        (source_root / files[0].relative_path).write_text(
            '{"task_id":"HumanEval/0","prompt":"def add(a, b):\\n","canonical_solution":"return a + b","test":"assert add(1, 2) == 3","declaration":"def add(a, b):"}\n',
            encoding="utf-8",
        )

    monkeypatch.setattr("src.eval.datasets.runtime.spec.download_url_files", _fake_download)

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("human_eval_cn", output_root, "test")

    assert paths == [output_root / "human_eval_cn" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "canonical_solution": "return a + b",
            "entry_point": "add",
            "test": "assert add(1, 2) == 3",
            "example_test": None,
            "text": None,
            "declaration": "def add(a, b):",
        }
    ]


def test_prepare_dataset_materializes_human_eval_fix_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.code_generation.human_eval_fix._load_human_eval_fix_rows",
        lambda split: [
            {
                "task_id": f"fix/{split}",
                "prompt": "Implement foo",
                "buggy_solution": "def foo(x):\n    return x - 1",
                "entry_point": "foo",
                "canonical_solution": "def foo(x):\n    return x + 1",
                "test": "assert foo(1) == 2",
                "example_test": "assert foo(0) == 1",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("human_eval_fix", output_root, "test")

    assert paths == [output_root / "human_eval_fix" / "test.jsonl"]
    rows = read_jsonl_items(paths[0])
    assert rows[0]["task_id"] == "fix/test"
    assert "# Buggy implementation:" in rows[0]["prompt"]
    assert "# Fix the function `foo` so it passes all tests." in rows[0]["prompt"]


def test_prepare_dataset_materializes_human_eval_plus_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.code_generation.human_eval_plus._load_human_eval_plus_problems",
        lambda cache_root: {
            "HumanEvalPlus/0": {
                "prompt": "def solve():\n",
                "canonical_solution": "return 1",
                "entry_point": "solve",
                "test": "assert solve() == 1",
                "contract": "",
                "plus_input": [[1]],
                "atol": 0.0,
            }
        },
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("human_eval_plus", output_root, "test")

    assert paths == [output_root / "human_eval_plus" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "HumanEvalPlus/0",
            "prompt": "def solve():\n",
            "canonical_solution": "return 1",
            "entry_point": "solve",
            "test": "assert solve() == 1",
            "contract": "",
            "plus_input": [[1]],
            "atol": 0.0,
        }
    ]


def test_prepare_dataset_materializes_mbpp_and_mbpp_plus_specs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.code_generation.mbpp._configure_evalplus_cache",
        lambda _cache_root: None,
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.code_generation.mbpp._load_mbpp_problems",
        lambda plus: {
            "1": {
                "task_id": "1",
                "prompt": "line1    line2",
                "question": "line1    line2",
                "canonical_solution": "pass",
                "entry_point": "solve",
                "base_input": [[1]],
                "plus_input": [[2]],
            }
        }
        if plus
        else {
            "1": {
                "task_id": "1",
                "prompt": "line1    line2",
                "question": "line1    line2",
                "canonical_solution": "pass",
                "entry_point": "solve",
                "base_input": [[1]],
                "plus_input": [[2]],
            }
        },
    )

    output_root = tmp_path / "prepared"
    mbpp_paths = prepare_dataset("mbpp", output_root, "test")
    mbpp_plus_paths = prepare_dataset("mbpp_plus", output_root, "test")

    assert mbpp_paths == [output_root / "mbpp" / "test.jsonl"]
    assert mbpp_plus_paths == [output_root / "mbpp_plus" / "test.jsonl"]
    mbpp_row = read_jsonl_items(mbpp_paths[0])[0]
    mbpp_plus_row = read_jsonl_items(mbpp_plus_paths[0])[0]
    assert mbpp_row["prompt"] == "line1\tline2"
    assert "base_input" not in mbpp_row
    assert "plus_input" not in mbpp_row
    assert mbpp_plus_row["prompt"] == "line1\tline2"
    assert mbpp_plus_row["plus_input"] == [[2]]


def test_prepare_dataset_materializes_livecodebench_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.code_generation.livecodebench._load_livecodebench_rows",
        lambda split, version_tag: [
            {"question_id": "2", "question_title": "two", "question_content": "", "starter_code": "pass"},
            {"question_id": "1", "question_title": "one", "question_content": "prompt one", "starter_code": "pass"},
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("livecodebench", output_root, "test")

    assert paths == [output_root / "livecodebench" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "question_id": "2",
            "question_title": "two",
            "question_content": "",
            "starter_code": "pass",
            "task_id": "2",
            "prompt": "two",
            "release_version": "release_v6",
            "source_dataset": "livecodebench/code_generation_lite",
        },
        {
            "question_id": "1",
            "question_title": "one",
            "question_content": "prompt one",
            "starter_code": "pass",
            "task_id": "1",
            "prompt": "prompt one",
            "release_version": "release_v6",
            "source_dataset": "livecodebench/code_generation_lite",
        },
    ]
