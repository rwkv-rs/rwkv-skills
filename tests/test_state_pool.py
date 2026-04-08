from __future__ import annotations

import torch

from src.infer.state_pool import StateCacheManager, StatePoolConfig


def test_prefix_cache_longest_match_and_disk_reload(tmp_path) -> None:
    manager = StateCacheManager(StatePoolConfig(db_path=str(tmp_path / "prefix-cache.db")))
    try:
        state_64 = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor(64, dtype=torch.int32),
        ]
        logits_64 = torch.tensor([0.1, 0.2, 0.3])
        tokens_64 = list(range(64))

        state_128 = [
            torch.tensor([10.0]),
            torch.tensor([20.0]),
            torch.tensor(128, dtype=torch.int32),
        ]
        logits_128 = torch.tensor([0.5, 0.6, 0.7])
        tokens_128 = list(range(128))

        assert manager.put_prefix_state(tokens_64, state_64, logits_64) is True
        assert manager.put_prefix_state(tokens_128, state_128, logits_128) is True

        prompt_tokens = list(range(160))
        match = manager.match_prefix_state(prompt_tokens, device="cpu")
        assert match is not None
        assert match["matched_tokens"] == 128
        assert match["cache_source"] == "l2_ram"
        assert torch.equal(match["state"][2], torch.tensor(128, dtype=torch.int32))

        manager.clear_prefix_memory()

        disk_match = manager.match_prefix_state(prompt_tokens, device="cpu")
        assert disk_match is not None
        assert disk_match["matched_tokens"] == 128
        assert disk_match["cache_source"] == "disk"
        assert torch.equal(disk_match["logits"], logits_128)
    finally:
        manager.close()


def test_session_cache_round_trip_from_disk(tmp_path) -> None:
    manager = StateCacheManager(
        StatePoolConfig(
            db_path=str(tmp_path / "sessions.db"),
            l1_capacity=1,
            l2_capacity=1,
        )
    )
    try:
        state_a = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor(1, dtype=torch.int32)]
        state_b = [torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor(2, dtype=torch.int32)]
        state_c = [torch.tensor([5.0]), torch.tensor([6.0]), torch.tensor(3, dtype=torch.int32)]

        manager.put_state("a", state_a)
        manager.put_state("b", state_b)
        manager.put_state("c", state_c)

        restored = manager.get_state("a", device="cpu")
        assert restored is not None
        assert torch.equal(restored[0], state_a[0])
        assert torch.equal(restored[2], state_a[2])
    finally:
        manager.close()
