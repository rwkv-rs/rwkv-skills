from __future__ import annotations

from ops.g1i_strict46 import probe_choice_sampling as probe


def test_stored_rows_samples_distinct_questions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Result:
        def fetchall(self):
            return [
                {
                    "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
                    "sample_index": 0,
                }
            ]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params
            return Result()

    monkeypatch.setattr(probe.psycopg, "connect", lambda *_args, **_kwargs: Connection())

    model, rows = probe._stored_rows(28619, 8)

    assert model == "rwkv7-g1i-2.9b-20260805-ctx16384"
    assert len(rows) == 1
    assert "DISTINCT ON (c.sample_index)" in str(captured["query"])
    assert captured["params"] == (28619, 8)

