from __future__ import annotations

"""Question-count heuristics used to prioritise pending jobs."""

from typing import Mapping, TYPE_CHECKING

from .dataset_utils import canonical_slug

if TYPE_CHECKING:  # pragma: no cover
    from .state import CompletedRecord

# Derived from historical evaluation logs under `results_old/`.
# These represent the full number of questions/samples per dataset.
HISTORICAL_QUESTION_COUNTS: dict[str, int] = {
    "aime24_test": 30,
    "aime25_test": 30,
    "amc23_test": 40,
    "answer_judge_test": 200,
    "agieval_mcq_test": 5940,
    "arc_challenge_test": 1172,
    "arc_easy_test": 2376,
    "bbh_mcq_test": 4070,
    "beyond_aime_test": 100,
    "brumo25_test": 30,
    "ceval_test": 12342,
    "cmmlu_test": 11582,
    "commonsense_qa_validation": 1221,
    "comp_math_24_25_test": 256,
    "gaokao2023en_test": 385,
    "gpqa_diamond": 198,
    "gpqa_extended": 546,
    "gpqa_main": 448,
    "gsm8k_test": 1319,
    "hellaswag_validation": 10042,
    "hmmt_feb25_test": 30,
    "human_eval_test": 164,
    "human_eval_plus_test": 164,
    "human_eval_fix_test": 164,
    "human_eval_cn_test": 164,
    "ifbench_test": 300,
    "ifeval_test": 541,
    "kmmlu_test": 35030,
    "livecodebench_test": 1055,
    "math_500_test": 500,
    "math_odyssey_test": 387,
    "mbpp_test": 508,
    "mbpp_plus_test": 508,
    "medmcqa_validation": 4183,
    "medqa_test": 1273,
    "minerva_math_test": 272,
    "mmlu_pro_test": 12032,
    "mmlu_redux_test": 5431,
    "mmlu_sr_question_and_answer_test": 14042,
    "mmlu_test": 14042,
    "openbookqa_test": 500,
    "olympiadbench_test": 675,
    "simpleqa_verified": 1000,
    "svamp_test": 1000,
    "truthfulqa_mc1_validation": 817,
    "winogrande_validation": 1267,
}


def _normalize_count(raw: object) -> int | None:
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def derive_question_counts(completed_records: Mapping[str, "CompletedRecord"] | None = None) -> dict[str, int]:
    """Combine historical counts with the latest `samples` info from completed runs."""

    counts = dict(HISTORICAL_QUESTION_COUNTS)
    if not completed_records:
        return counts
    for record in completed_records.values():
        value = _normalize_count(getattr(record, "samples", None))
        if value is None:
            continue
        slug = canonical_slug(record.key.dataset_slug)
        cap = HISTORICAL_QUESTION_COUNTS.get(slug)
        if cap is not None and value > cap:
            value = cap
        previous = counts.get(slug)
        if previous is None or value > previous:
            counts[slug] = value
    return counts


def question_count_for_slug(dataset_slug: str, counts: Mapping[str, int] | None = None) -> int | None:
    slug = canonical_slug(dataset_slug)
    if counts and slug in counts:
        return counts[slug]
    return HISTORICAL_QUESTION_COUNTS.get(slug)


__all__ = [
    "HISTORICAL_QUESTION_COUNTS",
    "derive_question_counts",
    "question_count_for_slug",
]
