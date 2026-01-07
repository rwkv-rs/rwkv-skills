from __future__ import annotations

"""Job catalogue & dataset bookkeeping for the scheduler."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

from src.eval.datasets.data_prepper.data_manager import (
    available_code_generation_datasets,
    available_free_answer_datasets,
    available_instruction_following_datasets,
    available_multiple_choice_datasets,
    prepare_dataset,
)

from .dataset_utils import (
    DATASET_SLUG_ALIASES,
    canonical_slug,
    make_dataset_slug,
    safe_slug,
)


MULTICHOICE_DEFAULT_SPLITS: dict[str, str] = {
    "gpqa": "main",
    "ceval": "test",
}
MATH_DEFAULT_SPLITS: dict[str, str] = {
    "hle": "all",
    "simpleqa": "verified",
}
SPECIAL_DEFAULT_SPLITS: dict[str, str] = {
    "flores200": "devtest",
    "ifeval": "test",
    "arena-hard": "test",
    "ifbench": "test",
    "wmt24pp": "test",
}
CODE_DEFAULT_SPLITS: dict[str, str] = {
    "mbpp": "test",
    "human_eval": "test",
    "mbpp_plus": "test",
    "human_eval_plus": "test",
    "human_eval_fix": "test",
    "human_eval_cn": "test",
}


@dataclass(frozen=True)
class DatasetPrepSpec:
    dataset: str
    split: str


@dataclass(frozen=True)
class JobSpec:
    name: str
    module: str
    dataset_slugs: tuple[str, ...]
    is_cot: bool
    domain: str
    extra_args: tuple[str, ...] = ()
    batch_flag: str | None = None
    probe_flag: str | None = None
    probe_max_generate_flag: str | None = None
    probe_dataset_required: bool = False
    probe_extra_args: tuple[str, ...] = ()
    # 用于探测 batch size 时估算实际样本量（例如 pass@k>1 时每题会展开多条样本）
    probe_samples_per_task: int = 1

    @property
    def id_prefix(self) -> str:
        return f"{self.name}__"


def _build_dataset_catalogues() -> tuple[
    dict[str, DatasetPrepSpec],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    specs: dict[str, DatasetPrepSpec] = {}

    def register(name: str, split: str) -> str:
        slug = make_dataset_slug(name, split)
        if slug not in specs:
            specs[slug] = DatasetPrepSpec(name, split)
        return slug

    multi_choice_slugs: list[str] = []
    for dataset in available_multiple_choice_datasets():
        split = MULTICHOICE_DEFAULT_SPLITS.get(dataset, "test")
        multi_choice_slugs.append(register(dataset, split))

    multi_choice_slugs.append(canonical_slug("ceval_exam_test"))

    math_slugs: list[str] = []
    for dataset in available_free_answer_datasets():
        split = MATH_DEFAULT_SPLITS.get(dataset, "test")
        math_slugs.append(register(dataset, split))

    special_slugs: list[str] = []
    for dataset in available_instruction_following_datasets():
        split = SPECIAL_DEFAULT_SPLITS.get(dataset, "test")
        special_slugs.append(register(dataset, split))

    code_slugs: list[str] = []
    for dataset in available_code_generation_datasets():
        split = CODE_DEFAULT_SPLITS.get(dataset, "test")
        code_slugs.append(register(dataset, split))

    for alias, target in DATASET_SLUG_ALIASES.items():
        canonical = canonical_slug(target)
        if canonical in specs:
            alias_slug = canonical_slug(alias)
            specs.setdefault(alias_slug, specs[canonical])

    multi_choice_tuple = tuple(sorted({canonical_slug(s) for s in multi_choice_slugs}))
    math_tuple = tuple(sorted({canonical_slug(s) for s in math_slugs}))
    special_tuple = tuple(sorted({canonical_slug(s) for s in special_slugs}))
    code_tuple = tuple(sorted({canonical_slug(s) for s in code_slugs}))
    return specs, multi_choice_tuple, math_tuple, special_tuple, code_tuple


(
    DATASET_PREP_SPECS,
    MULTICHOICE_DATASET_SLUGS,
    MATH_DATASET_SLUGS,
    SPECIAL_DATASET_SLUGS,
    CODE_DATASET_SLUGS,
) = _build_dataset_catalogues()

LLM_JUDGE_DATASET_SLUGS: Final[tuple[str, ...]] = tuple(
    canonical_slug(slug)
    for slug in ("gsm8k_test", "math_500_test", "answer_judge_test", "gaokao2023en_test")
)
# judge-only 数据集不再调度到 free_response，避免 math_500 反复被 free_response 拉起
MATH_DATASET_SLUGS_FOR_FREE_RESPONSE: Final[tuple[str, ...]] = tuple(
    slug for slug in MATH_DATASET_SLUGS if slug not in LLM_JUDGE_DATASET_SLUGS
)

ifeval_related = [slug for slug in SPECIAL_DATASET_SLUGS if slug.startswith("ifeval")]
if not ifeval_related:
    ifeval_related = [canonical_slug("ifeval_test")]
INSTRUCTION_FOLLOWING_DATASET_SLUGS: Final[tuple[str, ...]] = tuple(sorted(set(ifeval_related)))

HUMAN_EVAL_CODE_SLUGS: Final[tuple[str, ...]] = tuple(
    sorted(slug for slug in CODE_DATASET_SLUGS if "human_eval" in slug)
)
MBPP_CODE_SLUGS: Final[tuple[str, ...]] = tuple(
    sorted(slug for slug in CODE_DATASET_SLUGS if slug.startswith("mbpp"))
)


JOB_CATALOGUE: dict[str, JobSpec] = {
    "multi_choice_plain": JobSpec(
        name="multi_choice_plain",
        module="src.bin.eval_multi_choice",
        dataset_slugs=MULTICHOICE_DATASET_SLUGS,
        is_cot=False,
        domain="multi_choice",
    ),
    "multi_choice_cot": JobSpec(
        name="multi_choice_cot",
        module="src.bin.eval_multi_choice_cot",
        dataset_slugs=MULTICHOICE_DATASET_SLUGS,
        is_cot=True,
        domain="multi_choice",
        extra_args=("--no-param-search",),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
    ),
    "free_response": JobSpec(
        name="free_response",
        module="src.bin.eval_free_response",
        dataset_slugs=MATH_DATASET_SLUGS_FOR_FREE_RESPONSE,
        is_cot=True,
        domain="free_response",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
        # 高难数学默认 pass@k 较大，按 256 倍展开估算样本量，避免 batch 探测过小
        probe_samples_per_task=256,
    ),
    "free_response_judge": JobSpec(
        name="free_response_judge",
        module="src.bin.eval_free_response_judge",
        dataset_slugs=LLM_JUDGE_DATASET_SLUGS,
        is_cot=True,
        domain="free_response",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
    ),
    "param_search_free_response": JobSpec(
        name="param_search_free_response",
        module="src.bin.param_search_free_response",
        dataset_slugs=(canonical_slug("math_500_test"),),
        is_cot=True,
        domain="param_search",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
    ),
    "param_search_free_response_judge": JobSpec(
        name="param_search_free_response_judge",
        module="src.bin.param_search_free_response_judge",
        dataset_slugs=(canonical_slug("gsm8k_test"),),
        is_cot=True,
        domain="param_search",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
    ),
    "param_search_select": JobSpec(
        name="param_search_select",
        module="src.bin.param_search_select",
        dataset_slugs=(canonical_slug("gsm8k_test"),),
        is_cot=True,
        domain="param_search",
    ),
    "code_human_eval": JobSpec(
        name="code_human_eval",
        module="src.bin.eval_code_human_eval",
        dataset_slugs=HUMAN_EVAL_CODE_SLUGS or (canonical_slug("human_eval_test"),),
        is_cot=False,
        domain="code",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_max_generate_flag="--max-tokens",
        probe_samples_per_task=1,  # coding 默认只做 pass@1
    ),
    "code_mbpp": JobSpec(
        name="code_mbpp",
        module="src.bin.eval_code_mbpp",
        dataset_slugs=MBPP_CODE_SLUGS or (canonical_slug("mbpp_test"),),
        is_cot=False,
        domain="code",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_max_generate_flag="--max-tokens",
        probe_samples_per_task=1,  # coding 默认只做 pass@1
    ),
    "instruction_following": JobSpec(
        name="instruction_following",
        module="src.bin.eval_instruction_following",
        dataset_slugs=INSTRUCTION_FOLLOWING_DATASET_SLUGS,
        is_cot=False,
        domain="instruction_following",
        extra_args=("--no-param-search",),
    ),
}

JOB_ORDER: tuple[str, ...] = tuple(JOB_CATALOGUE.keys())


def detect_job_from_dataset(dataset_slug: str, is_cot: bool) -> str | None:
    slug = canonical_slug(dataset_slug)
    # 优先判定 LLM judge 数据集，避免 math_500 被 free_response 捕获导致调度状态错误。
    if is_cot and slug in LLM_JUDGE_DATASET_SLUGS:
        return "free_response_judge"
    for job_name, spec in JOB_CATALOGUE.items():
        if spec.is_cot == is_cot and slug in spec.dataset_slugs:
            return job_name
    return None


def locate_dataset(slug: str, *, search: Sequence[Path], output_root: Path) -> Path:
    from .datasets import find_dataset_file, refresh_dataset_index

    canonical = canonical_slug(slug)
    found = find_dataset_file(canonical, search)
    if found:
        return found

    spec = DATASET_PREP_SPECS.get(canonical)
    if spec is None:
        locations = "\n".join(f"  - {root}" for root in search)
        raise FileNotFoundError(
            f"未找到数据集 {slug!r}。请将 JSONL 文件放置于以下目录之一：\n{locations}"
        )

    prepared_paths = prepare_dataset(spec.dataset, output_root, spec.split)
    refresh_dataset_index(search)
    for path in prepared_paths:
        detected = canonical_slug(path.stem)
        if detected == canonical:
            return path
    refreshed = find_dataset_file(canonical, search)
    if refreshed:
        return refreshed
    raise FileNotFoundError(f"数据集 {slug!r} 未生成 JSONL，prepare_dataset 返回 {prepared_paths}")


__all__ = [
    "JobSpec",
    "JOB_CATALOGUE",
    "JOB_ORDER",
    "DATASET_PREP_SPECS",
    "detect_job_from_dataset",
    "locate_dataset",
    "safe_slug",
]
