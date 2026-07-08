from __future__ import annotations

"""Evaluate one WideSearch response file through the official WideSearch code.

The agent-loop verifier writes one JSONL file with rows shaped as
``{"question_id": ..., "response": ..., "trial": ...}`` and provides output
locations through ``WIDESEARCH_RESPONSE_PATH`` / ``WIDESEARCH_RESULT_DIR``.
The upstream WideSearch batch script expects its own response-root naming
scheme, so this adapter bridges the two contracts while still using the
official ``evaluate_single_query`` implementation and HF gold data loader.
"""

import json
import os
import sys
import tempfile
import types
from dataclasses import asdict
from pathlib import Path


def _official_root() -> Path:
    root = os.environ.get("RWKV_WIDESEARCH_OFFICIAL_ROOT")
    return Path(root).expanduser().resolve() if root else Path.cwd().resolve()


def _patch_eval_config() -> None:
    try:
        from src.utils.config import model_config
    except Exception:
        return

    judge_model = os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL")
    judge_base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    judge_api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not (judge_model and judge_api_key):
        return
    cfg = dict(model_config.get("default_eval_config") or {})
    cfg["model_name"] = judge_model
    cfg["api_key"] = judge_api_key
    if judge_base_url:
        cfg["base_url"] = judge_base_url
    model_config["default_eval_config"] = cfg


def _install_optional_dependency_shims() -> None:
    try:
        import pandarallel  # noqa: F401
    except ModuleNotFoundError:
        import pandas as pd

        class _PandarallelShim:
            @staticmethod
            def initialize(*_args: object, **_kwargs: object) -> None:
                if not hasattr(pd.DataFrame, "parallel_apply"):
                    pd.DataFrame.parallel_apply = pd.DataFrame.apply  # type: ignore[attr-defined]
                if not hasattr(pd.Series, "parallel_apply"):
                    pd.Series.parallel_apply = pd.Series.apply  # type: ignore[attr-defined]

        module = types.ModuleType("pandarallel")
        module.pandarallel = _PandarallelShim
        sys.modules["pandarallel"] = module

    try:
        import dateparser  # noqa: F401
    except ModuleNotFoundError:
        from dateutil import parser as dateutil_parser

        module = types.ModuleType("dateparser")

        def _parse(value: object, *_args: object, **_kwargs: object) -> object | None:
            try:
                return dateutil_parser.parse(str(value), fuzzy=True)
            except Exception:
                return None

        module.parse = _parse  # type: ignore[attr-defined]
        sys.modules["dateparser"] = module

    try:
        import volcenginesdkarkruntime  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("volcenginesdkarkruntime")

        class _MissingArk:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise ModuleNotFoundError(
                    "volcenginesdkarkruntime is required only for Ark model configs"
                )

        module.Ark = _MissingArk  # type: ignore[attr-defined]
        sys.modules["volcenginesdkarkruntime"] = module


def main() -> int:
    root = _official_root()
    sys.path.insert(0, str(root))
    _install_optional_dependency_shims()

    from src.evaluation.data_loader import WideSearchDataLoader, WideSearchDataLoaderHF, WideSearchResponse
    from src.evaluation.evaluation import evaluate_single_query

    _patch_eval_config()

    response_path = Path(os.environ["WIDESEARCH_RESPONSE_PATH"]).resolve()
    result_dir = Path(os.environ["WIDESEARCH_RESULT_DIR"]).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    data_path = os.environ.get("WIDESEARCH_DATA_PATH")
    answer_root = os.environ.get("WIDESEARCH_ANSWER_ROOT")
    if data_path and answer_root:
        data_path = str(_normalized_local_data_path(Path(data_path)))
        loader = WideSearchDataLoader(data_path, answer_root)
    else:
        loader = WideSearchDataLoaderHF()
    rows = [
        json.loads(line)
        for line in response_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no WideSearch responses found in {response_path}")

    for row in rows:
        instance_id = str(row.get("question_id") or row.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError(f"WideSearch response row missing question_id: {row}")
        query = loader.load_query_by_instance_id(instance_id)
        response = WideSearchResponse(
            instance_id=instance_id,
            response=str(row.get("response") or ""),
            messages=None,
            trial_idx=int(row.get("trial") or 0),
        )
        result_csv = result_dir / f"{instance_id}_eval_result.csv"
        result = evaluate_single_query(
            query,
            response,
            str(result_csv),
            os.environ.get("WIDESEARCH_EVAL_MODEL_CONFIG", "default_eval_config"),
        )
        (result_dir / f"{instance_id}.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


def _normalized_local_data_path(path: Path) -> Path:
    rows: list[dict[str, object]] = []
    changed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        evaluation = row.get("evaluation")
        if isinstance(evaluation, str):
            row["evaluation"] = json.loads(evaluation)
            changed = True
        rows.append(row)
    if not changed:
        return path
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix="widesearch-data-",
        delete=False,
    )
    with temp:
        for row in rows:
            temp.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            temp.write("\n")
    return Path(temp.name)


if __name__ == "__main__":
    raise SystemExit(main())
