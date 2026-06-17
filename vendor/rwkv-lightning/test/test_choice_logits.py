import sys
import types
import unittest
from pathlib import Path

import torch

VENDOR_ROOT = Path(__file__).resolve().parents[1]
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

sampler_module = types.ModuleType("infer.rwkv_batch.sampler")
sampler_module.sample = lambda *args, **kwargs: None
sys.modules["infer.rwkv_batch.sampler"] = sampler_module

from infer.inference import InferenceEngine


class DummyTokenizer:
    def encode(self, text):
        choice_tokens = {
            " down": [1],
            " same": [2],
            " up": [3],
            "multi": [4, 5],
            " badid": [99],
        }
        return choice_tokens.get(text, [9, 8, 7])


class DummyModel:
    def generate_zero_state(self, batch_size):
        return {"batch_size": batch_size}

    def forward(self, tokens, state):
        logits = torch.zeros(8, dtype=torch.float32)
        logits[1] = -2.0
        logits[2] = -1.0
        logits[3] = 1.0
        return logits

    def forward_batch_same_length(self, encoded_prompts, state, full_output=False):
        rows = []
        for index, _prompt in enumerate(encoded_prompts):
            logits = torch.zeros(8, dtype=torch.float32)
            logits[1] = -2.0 + index
            logits[2] = -1.0
            logits[3] = 1.0 - index
            rows.append(logits)
        return torch.stack(rows)


def build_engine():
    return InferenceEngine(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        args=types.SimpleNamespace(vocab_size=8),
        rocm_flag=False,
    )


class ChoiceLogitsTestCase(unittest.TestCase):
    def test_score_choice_logits_scores_single_token_choices(self):
        engine = build_engine()
        try:
            result = engine.score_choice_logits(
                "prompt",
                {"SHORT": " down", "FLAT": " same", "LONG": " up"},
            )
        finally:
            engine.shutdown()

        self.assertEqual(
            result["choice_token_ids"], {"SHORT": 1, "FLAT": 2, "LONG": 3}
        )
        self.assertEqual(
            result["choice_logits"],
            {"SHORT": -2.0, "FLAT": -1.0, "LONG": 1.0},
        )
        self.assertEqual(result["best_choice"], "LONG")
        self.assertGreater(
            result["choice_probabilities"]["LONG"],
            result["choice_probabilities"]["FLAT"],
        )
        self.assertAlmostEqual(sum(result["choice_probabilities"].values()), 1.0)

    def test_score_choice_logits_rejects_multi_token_choice(self):
        engine = build_engine()
        try:
            with self.assertRaisesRegex(ValueError, "exactly one token"):
                engine.score_choice_logits("prompt", {"BAD": "multi"})
        finally:
            engine.shutdown()

    def test_score_choice_logits_rejects_out_of_vocab_choice(self):
        engine = build_engine()
        try:
            with self.assertRaisesRegex(ValueError, "outside logits vocab size"):
                engine.score_choice_logits("prompt", {"BAD": " badid"})
        finally:
            engine.shutdown()

    def test_score_choice_logits_batch_scores_each_prompt(self):
        engine = build_engine()
        try:
            results = engine.score_choice_logits_batch(
                ["prompt-a", "prompt-b"],
                {"SHORT": " down", "FLAT": " same", "LONG": " up"},
            )
        finally:
            engine.shutdown()

        self.assertEqual([row["index"] for row in results], [0, 1])
        self.assertEqual(results[0]["choice_logits"]["LONG"], 1.0)
        self.assertEqual(results[0]["best_choice"], "LONG")
        self.assertEqual(results[1]["choice_logits"]["LONG"], 0.0)
        self.assertAlmostEqual(sum(results[0]["choice_probabilities"].values()), 1.0)
        self.assertAlmostEqual(sum(results[1]["choice_probabilities"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
