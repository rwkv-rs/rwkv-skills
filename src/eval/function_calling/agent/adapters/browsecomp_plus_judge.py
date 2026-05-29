from __future__ import annotations

"""BrowseComp-Plus official-style OpenAI judge adapter."""

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from openai import OpenAI

from src.eval.benchmark_config import BenchmarkModelConfig
from src.eval.function_calling.agent.adapters.browsecomp_plus import (
    DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT,
    browsecomp_plus_run_from_agent_details,
)
from src.eval.results.schema import build_context_from_completions, make_eval_payload
from src.eval.scheduler.config import REPO_ROOT


DEFAULT_BROWSECOMP_PLUS_JUDGE_MODEL = "gpt-5.4-mini"

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

correct_answer: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on correct_answer, focusing only on if there are meaningful differences between correct_answer and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than correct_answer, focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the correct_answer given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()

BROWSECOMP_PLUS_JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "correct_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "correct": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
    },
    "required": [
        "extracted_final_answer",
        "correct_answer",
        "reasoning",
        "correct",
        "confidence",
    ],
}


@dataclass(frozen=True, slots=True)
class BrowseCompPlusJudgeConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str = DEFAULT_BROWSECOMP_PLUS_JUDGE_MODEL
    max_output_tokens: int = 1024
    reasoning_effort: str | None = None
    timeout_s: float | None = 120.0
    api_mode: str = "auto"

    @classmethod
    def from_env(cls, overrides: Mapping[str, Any] | None = None) -> "BrowseCompPlusJudgeConfig":
        _load_env_file(REPO_ROOT / ".env")
        override = dict(overrides or {})
        api_key = _coerce_optional_str(override.get("api_key"))
        return cls(
            api_key=(
                api_key
                or _first_env(
                    "BROWSECOMP_PLUS_JUDGE_API_KEY",
                    "JUDGE_OPENAI_API_KEY",
                    "OPENAI_API_KEY",
                    "JUDGE_API_KEY",
                    "API_KEY",
                    "REELXAI_API_KEY",
                )
            ),
            base_url=(
                _coerce_optional_str(override.get("base_url"))
                or _first_env(
                    "BROWSECOMP_PLUS_JUDGE_BASE_URL",
                    "JUDGE_OPENAI_BASE_URL",
                    "OPENAI_BASE_URL",
                    "OPENAI_API_BASE",
                    "JUDGE_BASE_URL",
                    "API_BASE",
                    "REELXAI_BASE_URL",
                )
            ),
            model=(
                _coerce_optional_str(override.get("model"))
                or _first_env(
                    "BROWSECOMP_PLUS_JUDGE_MODEL",
                    "JUDGE_OPENAI_MODEL",
                    "OPENAI_MODEL",
                    "JUDGE_MODEL",
                )
                or DEFAULT_BROWSECOMP_PLUS_JUDGE_MODEL
            ),
            max_output_tokens=(
                _positive_int_value(override.get("max_output_tokens"))
                or _positive_int_env(
                    ("BROWSECOMP_PLUS_JUDGE_MAX_OUTPUT_TOKENS", "JUDGE_MAX_TOKENS"),
                    1024,
                )
            ),
            reasoning_effort=(
                _coerce_optional_str(override.get("reasoning_effort"))
                or _first_env(
                    "BROWSECOMP_PLUS_JUDGE_REASONING_EFFORT",
                    "JUDGE_OPENAI_REASONING_EFFORT",
                )
            ),
            timeout_s=(
                _positive_float_value(override.get("timeout_s"))
                or _positive_float_env(
                    ("BROWSECOMP_PLUS_JUDGE_TIMEOUT_S", "JUDGE_TIMEOUT_S"),
                    120.0,
                )
            ),
            api_mode=(
                _coerce_optional_str(override.get("api_mode"))
                or _first_env(
                    "BROWSECOMP_PLUS_JUDGE_API_MODE",
                    "JUDGE_OPENAI_API_MODE",
                )
                or "auto"
            ),
        )

    @classmethod
    def from_benchmark_config(
        cls,
        config: BenchmarkModelConfig | None,
    ) -> "BrowseCompPlusJudgeConfig":
        overrides = config.browsecomp_plus_judge if config is not None else None
        return cls.from_env(overrides=overrides)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "BrowseComp-Plus judge requires an OpenAI-compatible API key. "
                "Set BROWSECOMP_PLUS_JUDGE_API_KEY, JUDGE_OPENAI_API_KEY, OPENAI_API_KEY, "
                "or REELXAI_API_KEY in the environment or .env."
            )
        return self.api_key


@dataclass(slots=True)
class BrowseCompPlusJudgeMetrics:
    accuracy: float
    retrieval_recall: float | None
    calibration_error: float | None
    results: list[dict[str, Any]]
    summary: dict[str, Any]
    payloads: list[dict[str, Any]] = field(default_factory=list)


def browsecomp_plus_ground_truth_path() -> Path:
    override = _first_env("RWKV_BROWSECOMP_PLUS_GROUND_TRUTH", "BROWSECOMP_PLUS_GROUND_TRUTH")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_dir():
            path = path / "browsecomp_plus_decrypted.jsonl"
        return path
    for candidate in (
        DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT / "data" / "browsecomp_plus_decrypted.jsonl",
        REPO_ROOT / "data" / "browsecomp_plus" / "test.jsonl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT / "data" / "browsecomp_plus_decrypted.jsonl").resolve()


def browsecomp_plus_qrel_evidence_path() -> Path:
    override = _first_env("RWKV_BROWSECOMP_PLUS_QREL_EVIDENCE", "BROWSECOMP_PLUS_QREL_EVIDENCE")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_dir():
            path = path / "qrel_evidence.txt"
        return path
    for candidate in (
        DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT / "topics-qrels" / "qrel_evidence.txt",
        REPO_ROOT / "references" / "BrowseComp-Plus" / "topics-qrels" / "qrel_evidence.txt",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT / "topics-qrels" / "qrel_evidence.txt").resolve()


def default_browsecomp_plus_eval_dir(task_id: str | None = None) -> Path:
    override = _first_env("RWKV_BROWSECOMP_PLUS_EVAL_DIR", "BROWSECOMP_PLUS_EVAL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    suffix = f"task_{task_id}" if task_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "out" / "browsecomp_plus_evals" / suffix).resolve()


def load_ground_truth(jsonl_path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(jsonl_path).expanduser().resolve()
    ground_truth: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
            query_id = str(row.get("query_id") or row.get("id") or metadata.get("query_id") or index)
            question = str(row.get("query") or row.get("question") or row.get("instruction") or metadata.get("query") or "")
            answer = row.get("answer", metadata.get("answer"))
            if answer is None:
                continue
            ground_truth[query_id] = {"question": question, "answer": str(answer)}
    return ground_truth


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(question=question, response=response, correct_answer=correct_answer)


def evaluate_browsecomp_plus_completions(
    completions: Iterable[dict[str, Any]],
    *,
    config: BrowseCompPlusJudgeConfig | None = None,
    ground_truth_path: str | Path | None = None,
    qrel_evidence_path: str | Path | None = None,
    eval_dir: str | Path | None = None,
    client: Any | None = None,
    force: bool = False,
) -> BrowseCompPlusJudgeMetrics:
    cfg = config or BrowseCompPlusJudgeConfig.from_env()
    gt_path = Path(ground_truth_path).expanduser().resolve() if ground_truth_path else browsecomp_plus_ground_truth_path()
    if not gt_path.exists():
        raise FileNotFoundError(f"BrowseComp-Plus ground truth not found: {gt_path}")
    ground_truth = load_ground_truth(gt_path)
    qrel_path = Path(qrel_evidence_path).expanduser().resolve() if qrel_evidence_path else browsecomp_plus_qrel_evidence_path()
    qrel_evidence = load_qrel_data(qrel_path)
    output_dir = Path(eval_dir).expanduser().resolve() if eval_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    judge_client = client or OpenAI(
        api_key=cfg.require_api_key(),
        base_url=_normalize_openai_base_url(cfg.base_url),
        timeout=cfg.timeout_s,
    )

    all_results: list[dict[str, Any]] = []
    eval_payloads: list[dict[str, Any]] = []
    for payload in completions:
        run = browsecomp_plus_run_from_completion(payload)
        result = evaluate_browsecomp_plus_run(
            run,
            payload=payload,
            ground_truth=ground_truth,
            qrel_evidence=qrel_evidence,
            config=cfg,
            client=judge_client,
            output_dir=output_dir,
            force=force,
        )
        all_results.append(result)
        eval_payloads.append(_make_browsecomp_eval_payload(payload, result))

    summary = summarize_browsecomp_plus_results(all_results)
    summary["ground_truth"] = str(gt_path)
    summary["qrel_evidence"] = str(qrel_path) if qrel_path.exists() else str(qrel_path)
    summary["judge_model"] = cfg.model
    summary["judge_base_url_configured"] = bool(cfg.base_url)
    if output_dir is not None:
        _write_summary_files(output_dir, all_results, summary)
    accuracy = float(summary.get("Accuracy (%)") or 0.0) / 100.0
    recall_percent = summary.get("Recall (%)")
    calibration_percent = summary.get("Calibration Error (%)")
    return BrowseCompPlusJudgeMetrics(
        accuracy=accuracy,
        retrieval_recall=(float(recall_percent) / 100.0 if isinstance(recall_percent, (int, float)) else None),
        calibration_error=(
            float(calibration_percent) / 100.0 if isinstance(calibration_percent, (int, float)) else None
        ),
        results=all_results,
        summary=summary,
        payloads=eval_payloads,
    )


def evaluate_browsecomp_plus_run(
    run: Mapping[str, Any],
    *,
    payload: Mapping[str, Any] | None,
    ground_truth: Mapping[str, Mapping[str, str]],
    qrel_evidence: Mapping[str, Sequence[str]],
    config: BrowseCompPlusJudgeConfig,
    client: Any,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    query_id = str(run.get("query_id") or "")
    if not query_id:
        query_id = str(payload.get("sample_index") if payload else "")
    eval_path = output_dir / f"{query_id}_eval.json" if output_dir is not None and query_id else None
    if eval_path is not None and eval_path.exists() and not force:
        with eval_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if isinstance(cached, dict):
            return cached

    gt = ground_truth.get(query_id)
    if gt is None:
        raise KeyError(f"BrowseComp-Plus ground truth missing query_id={query_id!r}")
    correct_answer = str(gt["answer"])
    question = str(gt.get("question") or "")
    response = _extract_response_from_run(run)
    is_completed = run.get("status") == "completed"
    retrieved_docids = {str(item) for item in run.get("retrieved_docids", []) if item is not None}
    positives = [str(item) for item in qrel_evidence.get(query_id, [])]
    retrieval_recall = (
        len(retrieved_docids.intersection(set(positives))) / float(len(positives))
        if positives
        else 0.0
    )

    if not response or not is_completed:
        result = {
            "query_id": query_id,
            "question": question,
            "response": response,
            "correct_answer": correct_answer,
            "is_completed": bool(is_completed),
            "judge_prompt": None,
            "judge_response": None,
            "judge_result": {
                "parse_error": True,
                "error": "Response incomplete or cannot be parsed",
                "correct": False,
            },
            "tool_call_counts": _dict_value(run.get("tool_call_counts")),
            "citations": None,
            "retrieval": {
                "retrieved_docids": sorted(retrieved_docids),
                "recall": retrieval_recall,
                "has_qrel": bool(positives),
            },
            "model_info": _model_info(config),
        }
        _write_eval_file(eval_path, result)
        return result

    judge_prompt = create_judge_prompt(question, response, correct_answer)
    judge_result, judge_text = call_openai_structured_judge(
        client=client,
        prompt=judge_prompt,
        config=config,
    )
    cited_docids = extract_citations_from_response(response)
    citation_metrics = compute_citation_metrics(cited_docids, positives)
    result = {
        "query_id": query_id,
        "question": question,
        "response": response,
        "correct_answer": correct_answer,
        "is_completed": bool(is_completed),
        "judge_prompt": judge_prompt,
        "judge_response": judge_text,
        "judge_result": judge_result,
        "tool_call_counts": _dict_value(run.get("tool_call_counts")),
        "citations": {
            "cited_docids": cited_docids,
            "metrics": citation_metrics,
        },
        "retrieval": {
            "retrieved_docids": sorted(retrieved_docids),
            "recall": retrieval_recall,
            "has_qrel": bool(positives),
        },
        "model_info": _model_info(config),
    }
    _write_eval_file(eval_path, result)
    return result


def call_openai_structured_judge(
    *,
    client: Any,
    prompt: str,
    config: BrowseCompPlusJudgeConfig,
) -> tuple[dict[str, Any], str]:
    if _resolve_api_mode(config) == "chat":
        return _call_openai_chat_structured_judge(client=client, prompt=prompt, config=config)
    return _call_openai_responses_structured_judge(client=client, prompt=prompt, config=config)


def _call_openai_responses_structured_judge(
    *,
    client: Any,
    prompt: str,
    config: BrowseCompPlusJudgeConfig,
) -> tuple[dict[str, Any], str]:
    body: dict[str, Any] = {
        "model": config.model,
        "input": prompt,
        "max_output_tokens": config.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "browsecomp_plus_judge",
                "strict": True,
                "schema": BROWSECOMP_PLUS_JUDGE_JSON_SCHEMA,
            }
        },
    }
    if config.reasoning_effort:
        body["reasoning"] = {"effort": config.reasoning_effort}
    response = client.responses.create(**body)
    text = _response_output_text(response)
    parsed = parse_structured_judge_json(text)
    return parsed, text


def _call_openai_chat_structured_judge(
    *,
    client: Any,
    prompt: str,
    config: BrowseCompPlusJudgeConfig,
) -> tuple[dict[str, Any], str]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": config.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "browsecomp_plus_judge",
                "strict": True,
                "schema": BROWSECOMP_PLUS_JUDGE_JSON_SCHEMA,
            },
        },
    }
    if config.reasoning_effort:
        body["reasoning_effort"] = config.reasoning_effort
    completion = client.chat.completions.create(**body)
    text = _chat_completion_output_text(completion)
    parsed = parse_structured_judge_json(text)
    return parsed, text


def parse_structured_judge_json(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"BrowseComp-Plus judge did not return JSON: {text!r}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"BrowseComp-Plus judge JSON must be an object: {raw!r}")
    missing = [field for field in BROWSECOMP_PLUS_JUDGE_JSON_SCHEMA["required"] if field not in raw]
    if missing:
        raise ValueError(f"BrowseComp-Plus judge JSON missing required fields: {missing}")
    correct = _coerce_correct(raw.get("correct"))
    confidence = _coerce_confidence(raw.get("confidence"))
    return {
        "extracted_final_answer": str(raw.get("extracted_final_answer") or ""),
        "correct_answer": str(raw.get("correct_answer") or ""),
        "reasoning": str(raw.get("reasoning") or ""),
        "correct": correct,
        "confidence": confidence,
        "parse_error": False,
    }


def browsecomp_plus_run_from_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    run = payload.get("browsecomp_plus_run")
    if isinstance(run, Mapping):
        return dict(run)
    agent_details = payload.get("agent_details")
    if isinstance(agent_details, Mapping):
        run = browsecomp_plus_run_from_agent_details(agent_details)
        if run is not None:
            return run
    context = payload.get("context")
    if isinstance(context, Mapping):
        run = context.get("browsecomp_plus_run")
        if isinstance(run, Mapping):
            return dict(run)
        details = context.get("agent_details")
        if isinstance(details, Mapping):
            run = browsecomp_plus_run_from_agent_details(details)
            if run is not None:
                return run
    raise ValueError("completion payload does not contain browsecomp_plus_run")


def load_qrel_data(qrel_path: str | Path) -> dict[str, list[str]]:
    path = Path(qrel_path).expanduser().resolve()
    qrel_data: defaultdict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Expected 4 qrel columns in line: {line}")
            qrel_data[str(parts[0])].append(str(parts[2]))
    return dict(qrel_data)


def extract_citations_from_response(response_text: str) -> list[str]:
    if not response_text:
        return []
    single_matches = re.findall(r"\[(\d+)\]", response_text)
    multi_matches = re.findall(r"\[([^\[\]]*?)\]", response_text)
    single_fullwidth_matches = re.findall(r"【(\d+)】", response_text)
    multi_fullwidth_matches = re.findall(r"【([^【】]*?)】", response_text)
    all_docids: set[str] = set(single_matches)
    all_docids.update(single_fullwidth_matches)
    for match in multi_matches:
        if match in single_matches:
            continue
        all_docids.update(re.findall(r"\d+", match))
    for match in multi_fullwidth_matches:
        if match in single_fullwidth_matches:
            continue
        all_docids.update(re.findall(r"\d+", match))
    return sorted(all_docids)


def compute_citation_metrics(cited_docids: Sequence[str], relevant_docids: Sequence[str]) -> dict[str, float]:
    metrics = {
        "num_citations": float(len(cited_docids)),
        "num_relevant": float(len(relevant_docids)),
        "precision": 0.0,
        "recall": 0.0,
    }
    if not cited_docids:
        return metrics
    cited_set = set(cited_docids)
    relevant_set = set(relevant_docids)
    relevant_cited = cited_set & relevant_set
    metrics["precision"] = len(relevant_cited) / len(cited_docids)
    if relevant_docids:
        metrics["recall"] = len(relevant_cited) / len(relevant_docids)
    return metrics


def summarize_browsecomp_plus_results(all_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(all_results)
    all_tool_counts: defaultdict[str, float] = defaultdict(float)
    for result in all_results:
        for tool_name, count in _dict_value(result.get("tool_call_counts")).items():
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                all_tool_counts[str(tool_name)] += float(count)
    if total:
        for tool_name in list(all_tool_counts):
            all_tool_counts[tool_name] = all_tool_counts[tool_name] / total

    confidences: list[float] = []
    correctness: list[bool] = []
    for result in all_results:
        judge = result.get("judge_result")
        if not isinstance(judge, Mapping):
            continue
        if judge.get("parse_error") or judge.get("correct") is None:
            continue
        conf = judge.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            confidences.append(float(conf))
            correctness.append(bool(judge.get("correct")))

    calibration_error = calculate_calibration_error(confidences, correctness) if len(confidences) >= 100 else 0.0
    retrieval_recalls: list[float] = []
    for result in all_results:
        retrieval = result.get("retrieval")
        if not isinstance(retrieval, Mapping) or not retrieval.get("has_qrel"):
            continue
        recall = retrieval.get("recall")
        if isinstance(recall, (int, float)) and not isinstance(recall, bool):
            retrieval_recalls.append(float(recall))
    retrieval_recall_avg = float(np.mean(retrieval_recalls)) if retrieval_recalls else None
    correct_count = sum(1 for result in all_results if result.get("judge_result", {}).get("correct", False))
    accuracy_percent = round((correct_count / total) * 100.0, 2) if total else 0.0
    recall_percent = round(retrieval_recall_avg * 100.0, 2) if retrieval_recall_avg is not None else None

    per_query_metrics = []
    for result in all_results:
        recall_val = result.get("retrieval", {}).get("recall", None)
        per_query_metrics.append(
            {
                "query_id": result.get("query_id"),
                "correct": bool(result.get("judge_result", {}).get("correct", False)),
                "recall": round(float(recall_val) * 100.0, 2) if isinstance(recall_val, (int, float)) else None,
            }
        )

    citation_summary = _citation_summary(all_results)
    return {
        "LLM": "change me when submitting",
        "Accuracy (%)": accuracy_percent,
        "Recall (%)": recall_percent,
        "avg_tool_stats": dict(all_tool_counts),
        "Calibration Error (%)": round(calibration_error, 2),
        "Retriever": "rwkv-skills local BrowseComp-Plus adapter",
        "Link": "change me when submitting",
        "Evaluation Date": datetime.now().date().isoformat(),
        "per_query_metrics": per_query_metrics,
        "citation_summary": citation_summary,
    }


def calculate_calibration_error(confidences: Sequence[float], correctness: Sequence[bool], beta: int = 100) -> float:
    if len(confidences) != len(correctness) or not confidences:
        return 0.0
    confidence = np.array(confidences) / 100.0
    correct = np.array(correctness, dtype=float)
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if not bins:
        return 0.0
    bins[-1] = [bins[-1][0], len(confidence)]
    cerr = 0.0
    total_examples = len(confidence)
    for start, end in bins[:-1]:
        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        num_examples = len(bin_confidence)
        if num_examples > 0:
            diff = abs(float(np.nanmean(bin_confidence)) - float(np.nanmean(bin_correct)))
            cerr += num_examples / total_examples * np.square(diff)
    return float(np.sqrt(cerr) * 100.0)


def _make_browsecomp_eval_payload(
    completions_payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    judge_result = result.get("judge_result") if isinstance(result.get("judge_result"), Mapping) else {}
    passed = bool(judge_result.get("correct", False))
    fail_reason = "" if passed else str(judge_result.get("error") or "judge_incorrect")
    payload = make_eval_payload(
        dict(completions_payload),
        is_passed=passed,
        fail_reason=fail_reason,
        answer=str(result.get("response") or ""),
        ref_answer=str(result.get("correct_answer") or ""),
    )
    if payload.get("context") == "":
        payload["context"] = build_context_from_completions(dict(completions_payload))
    return payload


def _citation_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results_with_citations = [
        result
        for result in results
        if isinstance(result.get("citations"), Mapping) and result.get("citations", {}).get("cited_docids")
    ]
    total = len(results)
    cited_total = len(results_with_citations)
    avg_citations = (
        sum(len(result["citations"]["cited_docids"]) for result in results_with_citations) / cited_total
        if cited_total
        else 0.0
    )
    precision_avg = (
        sum(result["citations"].get("metrics", {}).get("precision", 0.0) for result in results_with_citations)
        / cited_total
        if cited_total
        else 0.0
    )
    recall_avg = (
        sum(result["citations"].get("metrics", {}).get("recall", 0.0) for result in results_with_citations)
        / cited_total
        if cited_total
        else 0.0
    )
    return {
        "responses_with_citations": cited_total,
        "total_responses": total,
        "citation_coverage_percent": round((cited_total / total) * 100.0, 2) if total else 0.0,
        "avg_citations_per_response": round(avg_citations, 2),
        "precision_percent": round(precision_avg * 100.0, 2),
        "recall_percent": round(recall_avg * 100.0, 2),
    }


def _write_summary_files(output_dir: Path, results: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with (output_dir / "detailed_judge_results.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "query_id",
            "predicted_answer",
            "correct_answer",
            "judge_correct",
            "confidence",
            "is_completed",
            "parse_error",
            "num_citations",
            "precision_positives",
            "recall_positives",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            judge = result.get("judge_result") if isinstance(result.get("judge_result"), Mapping) else {}
            citations = result.get("citations") if isinstance(result.get("citations"), Mapping) else {}
            citation_metrics = citations.get("metrics") if isinstance(citations.get("metrics"), Mapping) else {}
            writer.writerow(
                {
                    "query_id": result.get("query_id", ""),
                    "predicted_answer": judge.get("extracted_final_answer") or "",
                    "correct_answer": result.get("correct_answer", ""),
                    "judge_correct": judge.get("correct", ""),
                    "confidence": judge.get("confidence", ""),
                    "is_completed": result.get("is_completed", ""),
                    "parse_error": judge.get("parse_error", False),
                    "num_citations": len(citations.get("cited_docids", [])),
                    "precision_positives": citation_metrics.get("precision", 0.0),
                    "recall_positives": citation_metrics.get("recall", 0.0),
                }
            )


def _write_eval_file(eval_path: Path | None, result: Mapping[str, Any]) -> None:
    if eval_path is None:
        return
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)


def _response_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    if isinstance(response, Mapping):
        text = response.get("output_text")
        if isinstance(text, str):
            return text
        return _response_output_text_from_mapping(response)
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, Mapping):
            return _response_output_text_from_mapping(dumped)
    return ""


def _response_output_text_from_mapping(response: Mapping[str, Any]) -> str:
    parts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, Mapping):
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _chat_completion_output_text(completion: Any) -> str:
    if isinstance(completion, Mapping):
        choices = completion.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, Mapping):
                message = choice.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    return content if isinstance(content, str) else ""
        return ""
    choices = getattr(completion, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""
    if hasattr(completion, "model_dump"):
        dumped = completion.model_dump()
        if isinstance(dumped, Mapping):
            return _chat_completion_output_text(dumped)
    return ""


def _extract_response_from_run(run: Mapping[str, Any]) -> str:
    result = run.get("result")
    if isinstance(result, list) and result:
        last = result[-1]
        if isinstance(last, Mapping) and last.get("type") == "output_text":
            return str(last.get("output") or "")
    return ""


def _model_info(config: BrowseCompPlusJudgeConfig) -> dict[str, Any]:
    return {
        "judge_model": config.model,
        "max_output_tokens": config.max_output_tokens,
        "reasoning_effort": config.reasoning_effort,
        "base_url_configured": bool(config.base_url),
        "api_mode": _resolve_api_mode(config),
    }


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("BrowseComp-Plus judge confidence must be numeric")
    if not isinstance(value, (int, float)):
        raise ValueError(f"BrowseComp-Plus judge confidence must be numeric: {value!r}")
    return max(0.0, min(100.0, float(value)))


def _coerce_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "yes":
            return True
        if normalized == "no":
            return False
    raise ValueError(f"BrowseComp-Plus judge field 'correct' must be yes/no: {value!r}")


def _resolve_api_mode(config: BrowseCompPlusJudgeConfig) -> str:
    mode = str(config.api_mode or "auto").strip().lower()
    if mode in {"responses", "chat"}:
        return mode
    base_url = str(config.base_url or "").rstrip("/").lower()
    if base_url.endswith("/chat/completions"):
        return "chat"
    return "responses"


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    normalized = str(base_url).strip().rstrip("/")
    lowered = normalized.lower()
    for suffix in ("/chat/completions", "/responses"):
        if lowered.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip("'").strip('"')


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def _positive_int_env(names: Sequence[str], default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            try:
                parsed = int(raw)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    return default


def _positive_float_env(names: Sequence[str], default: float | None) -> float | None:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            try:
                parsed = float(raw)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    return default


def _coerce_optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _positive_int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _positive_float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


__all__ = [
    "BROWSECOMP_PLUS_JUDGE_JSON_SCHEMA",
    "BrowseCompPlusJudgeConfig",
    "BrowseCompPlusJudgeMetrics",
    "browsecomp_plus_ground_truth_path",
    "browsecomp_plus_qrel_evidence_path",
    "call_openai_structured_judge",
    "create_judge_prompt",
    "default_browsecomp_plus_eval_dir",
    "evaluate_browsecomp_plus_completions",
    "evaluate_browsecomp_plus_run",
    "extract_citations_from_response",
    "load_ground_truth",
    "load_qrel_data",
    "parse_structured_judge_json",
    "summarize_browsecomp_plus_results",
]
