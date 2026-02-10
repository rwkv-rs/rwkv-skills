from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.eval.scheduler.dataset_utils import canonical_slug
from src.space.data import load_scores


class TestSpaceDataFromSql(unittest.TestCase):
    @patch("src.space.data.EvalDbService")
    @patch("src.space.data.init_orm")
    def test_load_scores_normalizes_legacy_code_metrics(self, mock_init_orm, mock_service_cls) -> None:
        mock_service = MagicMock()
        mock_service.list_latest_scores_for_space.return_value = [
            {
                "task_id": 11,
                "dataset": "human_eval_test",
                "model": "demo-humaneval",
                "cot": False,
                "metrics": {"pass1": "0.77"},
                "created_at": "2025-02-07T10:00:00",
                "samples": 1,
                "problems": 1,
                "task": "code_humaneval",
                "task_details": None,
                "log_path": "/tmp/human_eval.jsonl",
                "is_param_search": False,
            },
            {
                "task_id": 12,
                "dataset": "mbpp_test",
                "model": "demo-mbpp",
                "cot": False,
                "metrics": {"score": {"pass1": 0.5}},
                "created_at": "2025-02-07T10:00:01",
                "samples": 1,
                "problems": 1,
                "task": "code_mbpp",
                "task_details": None,
                "log_path": "/tmp/mbpp.jsonl",
                "is_param_search": False,
            },
            {
                "task_id": 13,
                "dataset": "HumanEval-Dev",
                "model": "demo-humaneval-dev",
                "cot": False,
                "metrics": {"pass1": "0.11"},
                "created_at": "2025-02-07T10:00:02",
                "samples": 1,
                "problems": 1,
                "task": "code_humaneval",
                "task_details": None,
                "log_path": "/tmp/human_eval_dev.jsonl",
                "is_param_search": False,
            },
        ]
        mock_service_cls.return_value = mock_service

        entries = load_scores()
        by_dataset = {entry.dataset: entry for entry in entries}

        self.assertIn("human_eval_test", by_dataset)
        self.assertEqual(by_dataset["human_eval_test"].metrics["pass@1"], 0.77)

        self.assertIn("mbpp_test", by_dataset)
        self.assertEqual(by_dataset["mbpp_test"].metrics["pass@1"], 0.5)

        dev_slug = canonical_slug("HumanEval-Dev")
        self.assertIn(dev_slug, by_dataset)
        self.assertAlmostEqual(by_dataset[dev_slug].metrics["pass@1"], 0.11)

        mock_init_orm.assert_called_once()
        mock_service.list_latest_scores_for_space.assert_called_once_with(include_param_search=False)

    @patch("src.space.data.EvalDbService")
    @patch("src.space.data.init_orm")
    def test_load_scores_reports_init_failure(self, mock_init_orm, mock_service_cls) -> None:
        mock_init_orm.side_effect = RuntimeError("db down")

        errors: list[str] = []
        entries = load_scores(errors=errors)

        self.assertEqual(entries, [])
        self.assertTrue(any("初始化数据库失败" in msg for msg in errors))
        mock_service_cls.assert_not_called()

    @patch("src.space.data.EvalDbService")
    @patch("src.space.data.init_orm")
    def test_load_scores_reports_invalid_integer_fields(self, mock_init_orm, mock_service_cls) -> None:
        mock_service = MagicMock()
        mock_service.list_latest_scores_for_space.return_value = [
            {
                "task_id": 14,
                "dataset": "mmlu_test",
                "model": "demo-mmlu",
                "cot": False,
                "metrics": {"accuracy": 0.61},
                "created_at": "2025-02-07T11:00:00",
                "samples": "abc",
                "problems": "xyz",
                "task": "multiple_choice",
                "task_details": None,
                "log_path": "",
                "is_param_search": False,
            }
        ]
        mock_service_cls.return_value = mock_service

        errors: list[str] = []
        entries = load_scores(errors=errors)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].samples, 0)
        self.assertIsNone(entries[0].problems)
        self.assertTrue(any("samples" in msg for msg in errors))
        self.assertTrue(any("problems" in msg for msg in errors))


if __name__ == "__main__":
    unittest.main()
