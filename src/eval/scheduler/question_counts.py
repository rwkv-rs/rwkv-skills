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
    "algebra222_test": 222,
    "amc23_test": 40,
    "answer_judge_test": 12,
    "asdiv_test": 2305,
    "beyond_aime_test": 100,
    "brumo25_test": 30,
    "ceval_test": 12342,
    "college_math_test": 2818,
    "comp_math_24_25_test": 256,
    "gaokao2023en_test": 385,
    "gpqa_main": 448,
    "gsm8k_test": 1319,
    "gsm_plus_test": 9204,
    "hendrycks_math_test": 5000,
    "hle_all": 2158,
    "hmmt_feb25_test": 30,
    "human_eval_test": 164,
    "human_eval_plus_test": 164,
    "human_eval_fix_test": 164,
    "human_eval_cn_test": 164,
    "ifeval_test": 541,
    "livecodebench_test": 1055,
    "livecodebench_v1_test": 400,
    "livecodebench_v2_test": 511,
    "livecodebench_v3_test": 612,
    "livecodebench_v4_test": 713,
    "livecodebench_v5_test": 880,
    "livecodebench_v6_test": 1055,
    "math_500_test": 500,
    "math_odyssey_test": 387,
    "mawps_test": 2065,
    "mbpp_test": 508,
    "minerva_math_test": 272,
    "mmlu_pro_test": 12032,
    "mmlu_redux_test": 5431,
    "mmlu_test": 14042,
    "mmmlu_test": 14042,
    "cmmlu_test": 11582,
    "mbpp_plus_test": 508,
    "olympiadbench_test": 675,
    "omni_math_test": 4428,
    "qp_slice_4pur8six": 2560,
    "qp_slice_4u84bsd3": 1319,
    "qp_slice_6p17rl_y": 8192,
    "qp_slice_783ylykp": 1319,
    "qp_slice_7axnbbiz": 541,
    "qp_slice__04py908": 2560,
    "qp_slice_bi4lwzol": 1319,
    "qp_slice_bjie8ahg": 541,
    "qp_slice_c2310r7i": 2560,
    "qp_slice_cu6vtzm6": 1319,
    "qp_slice_dc4e9lep": 12288,
    "qp_slice_derx3294": 12288,
    "qp_slice_etijtecw": 2560,
    "qp_slice_ffz2n8q0": 12288,
    "qp_slice_g4xeyoxo": 541,
    "qp_slice_gmsq0w_k": 2560,
    "qp_slice_jie2etsx": 12288,
    "qp_slice_jtbfap89": 500,
    "qp_slice_ncy4vpzh": 541,
    "qp_slice_oz01zy9v": 541,
    "qp_slice_q22jufq6": 12288,
    "qp_slice_qw3xqxpk": 1319,
    "qp_slice_rptl3zya": 12288,
    "qp_slice_s0q8qsuq": 12288,
    "qp_slice_t6iw1hvv": 541,
    "qp_slice_u3oysx4c": 1319,
    "qp_slice_u7trhmaq": 12288,
    "qp_slice_vmmnq3s5": 500,
    "qp_slice_xe00nxoz": 8192,
    "qp_slice_xhyx4zpo": 1280,
    "simpleqa_verified": 1000,
    "supergpqa_test": 26529,
    "svamp_test": 1000,
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
