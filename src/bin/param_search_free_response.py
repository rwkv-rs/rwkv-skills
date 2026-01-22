from __future__ import annotations

"""Run a full CoT sampling grid search and persist per-trial artifacts under results/param_search/."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.evaluators.free_response import FreeResponsePipeline
from src.eval.metrics.free_response import compute_avg_at_k, compute_pass_at_k, evaluate_free_response
from src.eval.param_search.cot_grid import grid_size_by_mode, iter_cot_sampling_grid, NORMAL_COT_GRID, SIMPLE_COT_GRID
from src.eval.results.layout import (
    PARAM_SEARCH_COMPLETIONS_ROOT,
    PARAM_SEARCH_EVAL_RESULTS_ROOT,
    PARAM_SEARCH_SCORES_ROOT,
    make_scores_payload,
    param_search_completion_trial_path,
    param_search_eval_trial_path,
    param_search_scores_trial_path,
    write_scores_json_to_path,
)
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import canonical_slug, infer_dataset_slug_from_path, safe_slug
from src.infer.model import ModelLoadConfig
from src.infer.sampling import SamplingConfig


DEFAULT_PASS_K = (1,)
DEFAULT_AVG_K: tuple[int, ...] = ()


def _round_float(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _sampling_config_to_dict(config: SamplingConfig) -> dict[str, object]:
    data = asdict(config)
    normalized: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, tuple):
            normalized[key] = [
                _round_float(item) if isinstance(item, float) else item for item in value
            ]
        elif isinstance(value, float):
            normalized[key] = _round_float(value)
        else:
            normalized[key] = value
    return normalized


def _filter_metrics_by_k(metric_map: dict[str, float] | None, ks: tuple[int, ...], prefix: str) -> dict[str, float]:
    if not metric_map or not ks:
        return {}
    allowed = {int(k) for k in ks if int(k) > 0}
    filtered: dict[str, float] = {}
    for key, value in metric_map.items():
        if not key.startswith(prefix):
            continue
        suffix_text = key[len(prefix) :]
        if not suffix_text.isdigit():
            continue
        suffix = int(suffix_text)
        if suffix in allowed:
            filtered[key] = value
    return filtered


def _max_k(values: Sequence[int] | None) -> int:
    return max(values) if values else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV param-search (free-response, exact-match)")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--cot-max-tokens", type=int, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, help="Clamp final answer generation length")
    parser.add_argument("--output", help="Ignored (scheduler compatibility)")
    parser.add_argument("--para-grid-normal", default=None, type=str, 
                        help="""Grid search parameter space as a dictionary: \n
                         e.g. \'{\"temperature\":[0.3,0.4],\"top_k\":[50],\"top_p\":[0.3,0.4],\"alpha_presence\":[0.0,0.1],\"alpha_frequency\":[0.1],\"alpha_decay\":[0.99]}\'
                         """)
    parser.add_argument("--para-grid-simple", default=None, type=str, 
                        help="""Grid search parameter space as a dictionary: \n
                         e.g. \'{\"temperature\":[0.3,0.4],\"noise\":[1.0, 2.0, 3.0]}\'
                         """)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe and exit.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to generate for and compute (default: 1)",
    )
    parser.add_argument(
        "--avg-k",
        type=int,
        action="append",
        help="avg@k values to compute from generated samples (default: none)",
    )
    parser.add_argument(
        "--scan-mode",
        choices=("both", "normal", "simple"),
        default="both",
        help="Which sampling grid(s) to scan (default: both)",
    )
    return parser.parse_args(argv)

def _check_para_grid(args):
    if args.scan_mode in ['normal', 'both']:
        if args.para_grid_normal is None: 
            raise ValueError("When scan_mode is 'normal' or 'both', --para-grid-normal must be provided.")
        else: 
            args.para_grid_normal = json.loads(args.para_grid_normal)
    if args.scan_mode in ['simple', 'both']:
        if args.para_grid_simple is None: raise ValueError("When scan_mode is 'simple' or 'both', --para-grid-simple must be provided.")
        else: args.para_grid_simple = json.loads(args.para_grid_simple)

    # Set default as a placeholder if not provided
    if args.para_grid_normal is None: args.para_grid_normal = NORMAL_COT_GRID
    if args.para_grid_simple is None: args.para_grid_simple = SIMPLE_COT_GRID
    return args

def _cleanup_previous_trials(model_name: str, dataset_slug: str) -> None:
    model_dir = safe_slug(model_name)
    dataset_dir = canonical_slug(dataset_slug)
    for root in (PARAM_SEARCH_COMPLETIONS_ROOT, PARAM_SEARCH_EVAL_RESULTS_ROOT, PARAM_SEARCH_SCORES_ROOT):
        target = (root / model_dir / dataset_dir).resolve()
        shutil.rmtree(target, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args = _check_para_grid(args)
    try:
        dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    slug = infer_dataset_slug_from_path(str(dataset_path))
    model_name = Path(args.model_path).stem

    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = FreeResponsePipeline(config)

    pass_k = tuple(args.pass_k) if args.pass_k else DEFAULT_PASS_K
    avg_k = tuple(args.avg_k) if args.avg_k else DEFAULT_AVG_K
    samples_per_task = max(_max_k(pass_k), _max_k(avg_k), 1)

    cot_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    final_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="final",
        fallback_templates="free_response_final_default",
    )
    if cot_sampling is None or final_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    final_sampling = final_sampling.clamp(args.final_max_tokens)

    if args.probe_only:
        batch_size = max(1, args.batch_size)
        _ = pipeline.run(
            dataset_path=str(dataset_path),
            output_path=str(param_search_completion_trial_path(slug, model_name=model_name, trial_index=0)),
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=batch_size,
            pad_to_batch=True,
            pass_k=(1,),
            samples_per_task=1,
            probe_only=True,
            write_output=False,
        )
        print(f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}.")
        return 0

    _cleanup_previous_trials(model_name, slug)
    sizes = grid_size_by_mode(args.para_grid_normal, args.para_grid_simple)
    if args.scan_mode == "both":
        total = sizes["normal"] + sizes["simple"]
        print(f"🔍 Param-search grid: normal={sizes['normal']} + simple={sizes['simple']} (total={total})")
    else:
        print(f"🔍 Param-search grid: {args.scan_mode}={sizes[args.scan_mode]}")
    print(f"    Dataset: {slug} | Model: {model_name}")

    best_key: str | None = None
    best_score: float | None = None
    best_trial: int | None = None

    for trial_idx, trial_cot, params in iter_cot_sampling_grid(cot_sampling, 
                                                               NORMAL_COT_GRID=args.para_grid_normal, 
                                                               SIMPLE_COT_GRID=args.para_grid_simple, 
                                                               scan_mode=args.scan_mode):
        completion_path = param_search_completion_trial_path(slug, model_name=model_name, trial_index=trial_idx)
        eval_path = param_search_eval_trial_path(slug, model_name=model_name, trial_index=trial_idx)
        score_path = param_search_scores_trial_path(slug, model_name=model_name, trial_index=trial_idx)
        completion_path.unlink(missing_ok=True)
        eval_path.unlink(missing_ok=True)
        score_path.unlink(missing_ok=True)

        print(f"🔍 trial {trial_idx} ({params['sample_mode']}): {completion_path}")
        result = pipeline.run(
            dataset_path=str(dataset_path),
            output_path=str(completion_path),
            cot_sampling=trial_cot,
            final_sampling=final_sampling,
            batch_size=max(1, args.batch_size),
            sample_limit=args.max_samples,
            pass_k=pass_k,
            samples_per_task=samples_per_task,
        )
        evaluation = evaluate_free_response(
            completion_path,
            dataset_path=str(dataset_path),
            eval_output_path=eval_path,
            judge=None,
        )

        pass_metrics_all = compute_pass_at_k(evaluation.rows, pass_k)
        avg_metrics_all = compute_avg_at_k(evaluation.rows, avg_k)
        metrics_payload: dict[str, object] = {
            "exact_accuracy": float(evaluation.exact_accuracy),
            "judge_accuracy": evaluation.judge_accuracy,
        }
        pass_payload = _filter_metrics_by_k(pass_metrics_all, pass_k, "pass@") or (pass_metrics_all or {})
        if pass_payload:
            metrics_payload.update(pass_payload)
        avg_payload = _filter_metrics_by_k(avg_metrics_all, avg_k, "avg@") or (avg_metrics_all or {})
        if avg_payload:
            metrics_payload.update(avg_payload)

        task_details: dict[str, object] = {
            "eval_details_path": str(eval_path),
            "param_search_trial": {
                "trial": int(trial_idx),
                "params": params,
                "cot_sampling": _sampling_config_to_dict(trial_cot),
                "final_sampling": _sampling_config_to_dict(final_sampling),
            },
        }
        if pass_metrics_all and pass_payload != pass_metrics_all:
            task_details["pass_curve"] = pass_metrics_all
        if avg_metrics_all and avg_payload != avg_metrics_all:
            task_details["avg_curve"] = avg_metrics_all

        payload = make_scores_payload(
            slug,
            is_cot=True,
            model_name=model_name,
            metrics=metrics_payload,
            samples=evaluation.samples,
            problems=result.problem_count,
            log_path=completion_path,
            task="free_response",
            task_details=task_details,
        )
        write_scores_json_to_path(score_path, payload)

        objective = float(metrics_payload.get("exact_accuracy", 0.0))
        param_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        if best_score is None or objective > best_score:
            best_score = objective
            best_key = param_key
            best_trial = int(trial_idx)

    if best_trial is not None:
        print(f"✅ param-search done: best trial={best_trial} exact_accuracy={best_score:.4f} params={best_key}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
