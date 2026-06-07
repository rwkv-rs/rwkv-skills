from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS, normalize_rwkv_text
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES
from src.eval.function_calling.simple_tool_call import (
    SimpleToolCallRecord,
    build_simple_tool_call_prompt,
    decode_simple_tool_call_response,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

OFFICIAL_BFCL_EXEC_SOURCE = "ShishirPatil/gorilla@28a0f42"
BFCL_OFFICIAL_AST_DEPENDENCIES: tuple[str, ...] = (
    "tree_sitter",
    "tree_sitter_java",
    "tree_sitter_javascript",
)


@dataclass(frozen=True, slots=True)
class BfclExecRecord:
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_executable_calls: tuple[str, ...]
    execution_result_type: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BfclExecCallResult:
    call: str
    success: bool
    result: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BfclExecEvaluation:
    reward: float
    is_passed: bool
    fail_reason: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BfclOfficialAstStatus:
    available: bool
    official_root: str
    missing_dependencies: tuple[str, ...]
    import_error: str = ""


def load_bfcl_exec_rows_from_sources(
    question_path: str | Path,
    possible_answer_path: str | Path,
    *,
    category: str,
) -> list[dict[str, Any]]:
    questions = _read_json_or_jsonl_items(Path(question_path))
    answer_lookup = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_json_or_jsonl_items(Path(possible_answer_path))
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{category}_{index}")
        answer = answer_lookup.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL executable possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        if not instruction:
            raise ValueError(f"BFCL executable row {task_id!r} is missing question content")
        expected_calls = _ground_truth_to_exec_calls(answer.get("ground_truth") if answer else None)
        execution_types = [str(item).strip() for item in _coerce_list(answer.get("execution_result_type") if answer else None)]
        row_execution_types = [item or "exact_match" for item in execution_types]
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "tools": [_normalize_tool_schema(tool) for tool in _coerce_list(item.get("function"))],
                "expected_tool_calls": _expected_tool_calls_from_exec(expected_calls),
                "expected_executable_calls": [call for call in expected_calls if call],
                "execution_result_type": row_execution_types,
                "metadata": {
                    "source_format": "official_bfcl_v4_exec",
                    "category": category,
                    "source_path": str(Path(question_path)),
                    "possible_answer_path": str(Path(possible_answer_path)),
                    "execution_result_type": row_execution_types,
                },
            }
        )
    return rows


def load_bfcl_exec_manifest_records(path: str | Path) -> list[BfclExecRecord]:
    records: list[BfclExecRecord] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            records.append(normalize_bfcl_exec_manifest_row(payload, index=index, source_path=target))
    return records


def normalize_bfcl_exec_manifest_row(
    payload: Mapping[str, Any],
    *,
    index: int,
    source_path: str | Path | None = None,
) -> BfclExecRecord:
    task_id = str(payload.get("task_id") or payload.get("id") or f"bfcl_exec_{index:04d}")
    instruction = str(payload.get("instruction") or payload.get("question") or "").strip()
    if not instruction:
        raise ValueError(f"BFCL executable row {task_id!r} is missing instruction")
    metadata = dict(payload.get("metadata") or {})
    if source_path is not None:
        metadata.setdefault("manifest_path", str(Path(source_path)))
    execution_types = tuple(str(item).strip() or "exact_match" for item in _coerce_list(payload.get("execution_result_type")))
    expected_calls = tuple(str(item).strip() for item in _coerce_list(payload.get("expected_executable_calls")) if str(item).strip())
    if not expected_calls:
        expected_calls = tuple(str(item).strip() for item in _coerce_list(payload.get("ground_truth")) if str(item).strip())
    return BfclExecRecord(
        task_id=task_id,
        instruction=instruction,
        tools=tuple(dict(item) for item in _coerce_list(payload.get("tools")) if isinstance(item, Mapping)),
        expected_executable_calls=expected_calls,
        execution_result_type=execution_types or tuple("exact_match" for _ in expected_calls),
        metadata=metadata,
    )


def build_bfcl_exec_prompt(record: BfclExecRecord, *, history_max_chars: int) -> str:
    prompt = build_simple_tool_call_prompt(
        SimpleToolCallRecord(
            task_id=record.task_id,
            instruction=record.instruction,
            tools=record.tools,
            expected_tool_calls=(),
            metadata=record.metadata,
        ),
        history_max_chars=history_max_chars,
    )
    if _force_bfcl_exec_array_prefix(record):
        prompt += "[\n"
    return prompt


def render_bfcl_exec_call(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or call.get("tool_name") or "").strip()
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    rendered = ", ".join(f"{key}={_python_literal(value)}" for key, value in arguments.items())
    return f"{name}({rendered})" if rendered else f"{name}()"


def evaluate_bfcl_exec_calls(
    record: BfclExecRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
    sandbox: "BfclExecSandbox | None" = None,
) -> BfclExecEvaluation:
    sandbox = sandbox or BfclExecSandbox()
    expected_calls = list(record.expected_executable_calls)
    model_calls = [render_bfcl_exec_call(call) for call in decoded_calls]
    match_types = list(record.execution_result_type) or ["exact_match"] * len(expected_calls)
    while len(match_types) < len(expected_calls):
        match_types.append("exact_match")

    details: dict[str, Any] = {
        "expected_executable_calls": expected_calls,
        "decoded_executable_calls": model_calls,
        "execution_result_type": match_types,
        "tool_count_ok": len(model_calls) == len(expected_calls),
        "call_matches": [],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return BfclExecEvaluation(0.0, False, parse_error, details)

    expected_results = [sandbox.execute(call) for call in expected_calls]
    model_results = [sandbox.execute(call) for call in model_calls]
    details["expected_execution_results"] = [_result_payload(item) for item in expected_results]
    details["decoded_execution_results"] = [_result_payload(item) for item in model_results]

    failure_bits: list[str] = []
    passed_count = 0
    if _is_parallel_exec_record(record):
        matched_model_indices: set[int] = set()
        for expected_index, expected in enumerate(expected_results):
            match_type = match_types[expected_index] if expected_index < len(match_types) else "exact_match"
            candidate_reasons: list[str] = []
            for model_index, actual in enumerate(model_results):
                if model_index in matched_model_indices:
                    continue
                ok, reason = _execution_result_matches(actual, expected, match_type)
                if ok:
                    matched_model_indices.add(model_index)
                    details["call_matches"].append(
                        {
                            "expected_index": expected_index,
                            "decoded_index": model_index,
                            "ok": True,
                            "reason": reason,
                            "match_type": match_type,
                        }
                    )
                    passed_count += 1
                    break
                candidate_reasons.append(f"decoded_{model_index}:{reason}")
            else:
                reason = candidate_reasons[0].split(":", 1)[1] if candidate_reasons else "missing_call"
                details["call_matches"].append(
                    {
                        "expected_index": expected_index,
                        "decoded_index": None,
                        "ok": False,
                        "reason": reason,
                        "match_type": match_type,
                        "candidate_reasons": candidate_reasons,
                    }
                )
                failure_bits.append(f"call_{expected_index}:{reason}")
        for model_index in range(len(model_results)):
            if model_index in matched_model_indices:
                continue
            details["call_matches"].append(
                {
                    "expected_index": None,
                    "decoded_index": model_index,
                    "ok": False,
                    "reason": "unexpected_extra_call",
                }
            )
            failure_bits.append(f"call_{model_index}:unexpected_extra_call")
        denominator = max(1, len(expected_results))
        reward = passed_count / denominator
        is_passed = len(model_results) == len(expected_results) and passed_count == len(expected_results)
        return BfclExecEvaluation(float(reward), bool(is_passed), "; ".join(failure_bits), details)

    max_len = max(len(expected_results), len(model_results))
    for index in range(max_len):
        if index >= len(expected_results):
            details["call_matches"].append({"index": index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(model_results):
            details["call_matches"].append({"index": index, "ok": False, "reason": "missing_call"})
            failure_bits.append(f"call_{index}:missing_call")
            continue
        expected = expected_results[index]
        actual = model_results[index]
        match_type = match_types[index] if index < len(match_types) else "exact_match"
        ok, reason = _execution_result_matches(actual, expected, match_type)
        details["call_matches"].append({"index": index, "ok": ok, "reason": reason, "match_type": match_type})
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")

    denominator = max(1, len(expected_results))
    reward = passed_count / denominator
    is_passed = len(model_results) == len(expected_results) and passed_count == len(expected_results)
    return BfclExecEvaluation(float(reward), bool(is_passed), "; ".join(failure_bits), details)


def _is_parallel_exec_record(record: BfclExecRecord) -> bool:
    category = str(record.metadata.get("category") or record.metadata.get("dataset") or "").strip().lower()
    return "parallel" in category


def _force_bfcl_exec_array_prefix(record: BfclExecRecord) -> bool:
    return _is_parallel_exec_record(record) and len(record.expected_executable_calls) > 1


def _complete_bfcl_exec_forced_prefix(record: BfclExecRecord, completion: str) -> str:
    if not _force_bfcl_exec_array_prefix(record):
        return completion
    stripped = completion.lstrip()
    if stripped.startswith("["):
        return completion
    if stripped.startswith("{") and not completion.rstrip().endswith("]"):
        return "[\n" + completion.rstrip() + "\n]"
    return "[\n" + completion


class BfclExecSandbox:
    def __init__(self) -> None:
        self._functions: dict[str, Callable[..., Any]] = {
            "add_binary_numbers": lambda a, b: bin(int(str(a), 2) + int(str(b), 2))[2:],
            "adjust_for_inflation": lambda amount, inflation_rate, years: amount * ((1 + inflation_rate) ** years),
            "apply_discount": lambda price, discount: price * (1 - discount),
            "book_room": lambda **kwargs: {"booking_confirmed": True, **kwargs},
            "calc_binomial_probability": self._calc_binomial_probability,
            "calculate_basal_metabolic_rate": self._calculate_basal_metabolic_rate,
            "calculate_cosine_similarity": self._calculate_cosine_similarity,
            "calculate_daily_energy_expenditure": lambda bmr, activity_factor: bmr * activity_factor,
            "calculate_density": lambda mass, volume: mass / volume,
            "calculate_displacement": lambda initial_velocity, acceleration, time: initial_velocity * time + 0.5 * acceleration * time**2,
            "calculate_electrostatic_potential_energy": lambda charge, voltage: charge * voltage,
            "calculate_final_velocity": lambda initial_velocity, acceleration, time: initial_velocity + acceleration * time,
            "calculate_future_value": lambda present_value, interest_rate, periods: present_value * ((1 + interest_rate) ** periods),
            "calculate_intercept": lambda x1, y1, slope: y1 - slope * x1,
            "calculate_interest_rate": lambda principal, amount, time: (amount / principal) ** (1 / time) - 1,
            "calculate_investment_value": self._calculate_investment_value,
            "calculate_mean": lambda numbers: sum(numbers) / len(numbers),
            "calculate_nutritional_needs": self._calculate_nutritional_needs,
            "calculate_permutations": lambda n, k: math.factorial(int(n)) // math.factorial(int(n) - int(k)),
            "calculate_slope": lambda x1, y1, x2, y2: (y2 - y1) / (x2 - x1),
            "calculate_standard_deviation": self._calculate_standard_deviation,
            "calculate_total": lambda price, quantity: price * quantity,
            "calculate_total_price": lambda price, quantity: price * quantity,
            "calculate_triangle_area": lambda base, height: 0.5 * base * height,
            "compound_interest": lambda principal, rate, time, n=1: principal * ((1 + rate / n) ** (n * time)),
            "confirm_booking": lambda **kwargs: {"confirmed": True, **kwargs},
            "convert_binary_to_decimal": lambda binary: int(str(binary), 2),
            "convert_coordinates": lambda latitude, longitude: {"latitude": latitude, "longitude": longitude},
            "convert_decimal_to_hex": lambda decimal: hex(int(decimal)),
            "convert_temperature": self._convert_temperature,
            "estimate_derivative": self._estimate_derivative,
            "generate_random_number": lambda min_value=0, max_value=100: int((int(min_value) + int(max_value)) / 2),
            "geometry_area_circle": lambda radius: math.pi * radius * radius,
            "get_distance": lambda pointA, pointB: math.dist(pointA, pointB),
            "get_fibonacci_number": self._get_fibonacci_number,
            "get_fibonacci_sequence": self._get_fibonacci_sequence,
            "get_prime_factors": self._get_prime_factors,
            "inflation_adjustment": lambda amount, inflation_rate, years: amount * ((1 + inflation_rate) ** years),
            "linear_regression": self._linear_regression,
            "mat_mul": self._mat_mul,
            "math_factorial": lambda n: math.factorial(int(n)),
            "math_gcd": lambda a, b: math.gcd(int(a), int(b)),
            "math_lcm": lambda a, b: abs(int(a) * int(b)) // math.gcd(int(a), int(b)),
            "maxPoints": self._max_points,
            "mortgage_calculator": self._mortgage_calculator,
            "order_food": self._order_food,
            "polygon_area": self._polygon_area,
            "predict_value": lambda slope, intercept, x: slope * x + intercept,
            "quadratic_roots": self._quadratic_roots,
            "sort_array": lambda array, reverse=False: sorted(array, reverse=bool(reverse)),
            "validate_polygon": lambda vertices: len(vertices) >= 3,
        }

    def execute(self, call: str) -> BfclExecCallResult:
        try:
            name, args, kwargs = _parse_call(call)
            func = self._functions.get(name)
            result = func(*args, **kwargs) if func is not None else self._execute_fixture(name, args, kwargs)
            return BfclExecCallResult(call=call, success=True, result=_json_safe(result))
        except Exception as exc:  # noqa: BLE001
            return BfclExecCallResult(call=call, success=False, error=str(exc))

    def _execute_fixture(self, name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        if name in {"get_stock_price_by_stock_name", "get_company_name_by_stock_name", "get_stock_history"}:
            stock = str(kwargs.get("stock_name") or (args[0] if args else "")).upper()
            if name == "get_company_name_by_stock_name":
                return {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc."}.get(stock, stock)
            if name == "get_stock_history":
                return {"symbol": stock, "history": [{"close": self._stock_price(stock)}]}
            return self._stock_price(stock)
        if name == "get_weather_data":
            coordinates = kwargs.get("coordinates") or (args[0] if args else [0, 0])
            if isinstance(coordinates, Mapping):
                latitude = coordinates.get("latitude", coordinates.get("lat", 0))
                longitude = coordinates.get("longitude", coordinates.get("lon", coordinates.get("long", 0)))
            else:
                latitude = coordinates[0] if len(coordinates) > 0 else 0
                longitude = coordinates[1] if len(coordinates) > 1 else 0
            return {"temperature": round(float(latitude) * 0.1 + float(longitude) * 0.01, 3), "unit": "celsius"}
        if name in {"get_coordinate_by_ip_address", "get_zipcode_by_ip_address"}:
            ip_address = str(kwargs.get("ip_address") or (args[0] if args else ""))
            return "private range" if ip_address.startswith("192.168.") else {"ip_address": ip_address}
        if name in {"get_coordinates_from_city", "retrieve_city_based_on_zipcode", "get_time_zone_by_coord"}:
            return self._location_fixture(name, args, kwargs)
        if name in {"get_covid_death_by_country", "get_active_covid_case_by_country"}:
            country = str(kwargs.get("country") or (args[0] if args else "")).lower()
            base = sum(ord(char) for char in country)
            return base * (1000 if "death" in name else 500)
        if name in {"get_rating_by_amazon_ASIN", "get_price_by_amazon_ASIN", "get_product_name_by_amazon_ASIN"}:
            asin = str(kwargs.get("ASIN") or kwargs.get("asin") or (args[0] if args else ""))
            if "rating" in name:
                return str(round(3.5 + (sum(ord(c) for c in asin) % 15) / 10, 1))
            if "price" in name:
                return f"${50 + (sum(ord(c) for c in asin) % 500)}.00"
            return f"Product {asin}"
        if name in {"get_movie_director", "get_director_by_movie_name", "get_movie_rating", "get_movie_genre"}:
            movie = str(kwargs.get("movie_name") or (args[0] if args else "")).lower()
            movies = {
                "avatar": {"director": "James Cameron", "rating": "PG-13", "genre": "Science Fiction"},
                "pulp fiction": {"director": "Quentin Tarantino", "rating": "R", "genre": "Crime"},
            }
            info = movies.get(movie, {"director": "Unknown", "rating": "Unknown", "genre": "Unknown"})
            if "rating" in name:
                return info["rating"]
            if "genre" in name:
                return info["genre"]
            return info["director"]
        if name == "retrieve_holiday_by_year":
            country = kwargs.get("country") or (args[0] if args else "")
            year = int(kwargs.get("year") or (args[1] if len(args) > 1 else 2023))
            return [
                {
                    "countryCode": str(country),
                    "date": f"{year:04d}-01-01",
                    "localName": "New Year",
                    "name": "New Year's Day",
                }
            ]
        if name == "find_term_on_urban_dictionary":
            term = str(kwargs.get("term") or (args[0] if args else ""))
            return {"term": term, "definition": f"Definition for {term}"}
        if name == "convert_currency":
            amount = float(kwargs.get("amount") or (args[0] if args else 0.0))
            from_currency = str(kwargs.get("from_currency") or (args[1] if len(args) > 1 else "")).upper()
            to_currency = str(kwargs.get("to_currency") or (args[2] if len(args) > 2 else "")).upper()
            rates = {
                ("USD", "EUR"): 0.92,
                ("EUR", "USD"): 1.08,
                ("USD", "GBP"): 0.79,
                ("GBP", "USD"): 1.27,
            }
            return amount * rates.get((from_currency, to_currency), 1.0)
        raise ValueError(f"unsupported official BFCL executable function: {name}")

    @staticmethod
    def _stock_price(stock: str) -> float:
        return {"AAPL": 169.02, "MSFT": 421.9, "GOOG": 175.4, "META": 477.2, "NFLX": 610.1, "BABA": 75.0}.get(stock, 100.0)

    @staticmethod
    def _location_fixture(name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        if name == "retrieve_city_based_on_zipcode":
            return {"90210": "BEVERLY HILLS", "10001": "NEW YORK", "08540": "PRINCETON"}.get(str(kwargs.get("zipcode") or ""), "UNKNOWN")
        if name == "get_coordinates_from_city":
            city = str(kwargs.get("city_name") or (args[0] if args else ""))
            return {"city": city, "latitude": "0.0", "longitude": "0.0"}
        return "UTC"

    @staticmethod
    def _calc_binomial_probability(n: int, k: int, p: float) -> float:
        return math.comb(int(n), int(k)) * (float(p) ** int(k)) * ((1 - float(p)) ** (int(n) - int(k)))

    @staticmethod
    def _calculate_cosine_similarity(vectorA: Sequence[float], vectorB: Sequence[float]) -> float:
        dot = sum(float(a) * float(b) for a, b in zip(vectorA, vectorB))
        norm_a = math.sqrt(sum(float(a) ** 2 for a in vectorA))
        norm_b = math.sqrt(sum(float(b) ** 2 for b in vectorB))
        return dot / (norm_a * norm_b)

    @staticmethod
    def _calculate_standard_deviation(numbers: Sequence[float]) -> float:
        mean = sum(numbers) / len(numbers)
        return math.sqrt(sum((float(item) - mean) ** 2 for item in numbers) / len(numbers))

    @staticmethod
    def _calculate_basal_metabolic_rate(weight: float, height: float, age: int, gender: str) -> float:
        offset = 5 if str(gender).lower() == "male" else -161
        return 10 * weight + 6.25 * height - 5 * age + offset

    def _calculate_nutritional_needs(self, weight: float, height: float, age: int, gender: str, activity_level: float, goal: str) -> dict[str, float]:
        bmr = self._calculate_basal_metabolic_rate(weight, height, age, gender)
        calories = bmr * float(activity_level)
        if str(goal).lower().startswith("lose"):
            calories -= 500
        elif str(goal).lower().startswith("gain"):
            calories += 500
        return {"calories": calories}

    @staticmethod
    def _calculate_investment_value(
        initial_investment: float,
        annual_contribution: float,
        years: int,
        annual_return: float,
        inflation_rate: Sequence[float] | float = (),
        adjust_for_inflation: bool = True,
    ) -> float:
        value = float(initial_investment)
        rates: Sequence[float]
        if isinstance(inflation_rate, (int, float)):
            rates = [float(inflation_rate)]
        elif inflation_rate is None:
            rates = []
        else:
            rates = inflation_rate
        for index in range(int(years)):
            inflation = float(rates[index]) if adjust_for_inflation and index < len(rates) else 0.0
            value = (value + float(annual_contribution)) * (1 + float(annual_return) - inflation)
        return value

    @staticmethod
    def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        source = str(from_unit).lower()
        target = str(to_unit).lower()
        celsius = (value - 32) * 5 / 9 if source.startswith("f") else value
        return celsius * 9 / 5 + 32 if target.startswith("f") else celsius

    @staticmethod
    def _estimate_derivative(function: str, x: float) -> float:
        source = str(function).strip()
        if not source.startswith("lambda"):
            source = f"lambda x: {source}"
        fn = eval(source, {"__builtins__": {}, "math": math}, {})  # noqa: S307 - BFCL executable sandbox input is restricted to math expressions.
        h = 1e-5
        return (fn(x + h) - fn(x - h)) / (2 * h)

    @staticmethod
    def _get_fibonacci_number(n: int) -> int:
        seq = BfclExecSandbox._get_fibonacci_sequence(int(n) + 1)
        return seq[-1] if seq else 0

    @staticmethod
    def _get_fibonacci_sequence(n: int) -> list[int]:
        values = [0, 1]
        while len(values) < int(n):
            values.append(values[-1] + values[-2])
        return values[: int(n)]

    @staticmethod
    def _get_prime_factors(
        n: int | None = None,
        *,
        number: int | None = None,
        formatted: bool | str | None = None,
    ) -> list[int] | str:
        factors: list[int] = []
        value = int(number if number is not None else n)
        divisor = 2
        while divisor * divisor <= value:
            while value % divisor == 0:
                factors.append(divisor)
                value //= divisor
            divisor += 1
        if value > 1:
            factors.append(value)
        if _truthy_exec_flag(formatted):
            return " * ".join(str(item) for item in factors)
        return factors

    @staticmethod
    def _order_food(
        item: Sequence[str] | None = None,
        quantity: Sequence[float] | None = None,
        price: Sequence[float] | None = None,
        *,
        items: Sequence[str] | None = None,
        **_kwargs: Any,
    ) -> float:
        selected_items = item if item is not None else items
        quantities = quantity or [1 for _ in (selected_items or [])]
        prices = price or [0 for _ in (selected_items or [])]
        return sum(float(q) * float(p) for q, p in zip(quantities, prices))

    @staticmethod
    def _linear_regression(x: Sequence[float], y: Sequence[float], point: float) -> float:
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        denom = sum((item - mean_x) ** 2 for item in x)
        slope = 0.0 if denom == 0 else sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denom
        return slope * point + (mean_y - slope * mean_x)

    @staticmethod
    def _mat_mul(matA: Sequence[Sequence[float]], matB: Sequence[Sequence[float]]) -> list[list[float]]:
        columns = list(zip(*matB))
        return [[sum(a * b for a, b in zip(row, col)) for col in columns] for row in matA]

    @staticmethod
    def _max_points(points: Sequence[Sequence[float]]) -> int:
        if len(points) <= 2:
            return len(points)
        best = 2
        for i, a in enumerate(points):
            slopes: dict[tuple[float, float], int] = {}
            for b in points[i + 1 :]:
                dx = float(b[0]) - float(a[0])
                dy = float(b[1]) - float(a[1])
                key = (0.0, 1.0) if dx == 0 else (1.0, round(dy / dx, 12))
                slopes[key] = slopes.get(key, 1) + 1
                best = max(best, slopes[key])
        return best

    @staticmethod
    def _mortgage_calculator(loan_amount: float, interest_rate: float, loan_period: int) -> float:
        monthly_rate = float(interest_rate) / 12
        payments = int(loan_period) * 12
        if monthly_rate == 0:
            return loan_amount / payments
        return loan_amount * monthly_rate * ((1 + monthly_rate) ** payments) / (((1 + monthly_rate) ** payments) - 1)

    @staticmethod
    def _polygon_area(vertices: Sequence[Sequence[float]]) -> float:
        area = 0.0
        for index, point in enumerate(vertices):
            next_point = vertices[(index + 1) % len(vertices)]
            area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
        return abs(area) / 2

    @staticmethod
    def _quadratic_roots(a: float, b: float, c: float) -> list[float | str]:
        disc = b * b - 4 * a * c
        if disc < 0:
            return ["complex"]
        root = math.sqrt(disc)
        return [(-b + root) / (2 * a), (-b - root) / (2 * a)]


def _bfcl_exec_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("fail_reason") or agent_result.get("error") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=json.dumps(agent_info.get("decoded_executable_calls") or [], ensure_ascii=False, sort_keys=True),
        ref_answer=json.dumps(agent_info.get("expected_executable_calls") or [], ensure_ascii=False, sort_keys=True),
    )


def _run_bfcl_exec(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_bfcl_exec_manifest_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        run.dataset_slug,
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]
    if not records:
        raise ValueError("BFCL executable manifest is empty")

    plan = _resolve_function_calling_plan(
        run.dataset_slug,
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    tool_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    if tool_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    tool_sampling = clamp_function_calling_sampling(tool_sampling, max(1, int(args.decision_max_tokens or 768)))
    sampling_payload = normalize_sampling_config_by_stage([(1, tool_sampling)])
    history_max_chars = max(0, int(args.history_max_chars or DEFAULT_HISTORY_MAX_CHARS))
    batch_size = max(1, int(args.batch_size or 16))
    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        prompts = [build_bfcl_exec_prompt(record, history_max_chars=history_max_chars) for _index, record in repeated]
        run.engine.generate(
            prompts,
            sampling=tool_sampling,
            batch_size=len(prompts),
            progress_desc="BFCL-Exec-Probe",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
            prompt_seeds=[sample_repeat_seed(index, 0, stage=1) for index, _record in repeated],
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    job_name = _resolve_job_name("function_bfcl_exec", run_context=run_context)
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 32),
        run_context=run_context,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    flush_partial = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_bfcl_exec_completion_to_eval_payload,
        runner_name="bfcl_exec",
    )
    sandbox = BfclExecSandbox()

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=flush_partial,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                for key, record in pending:
                    prompt = build_bfcl_exec_prompt(record, history_max_chars=history_max_chars)
                    output = run.engine.generate(
                        [prompt],
                        sampling=tool_sampling,
                        batch_size=1,
                        progress_desc=f"BFCL-Exec sample {key.sample_index}",
                        prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                        prompt_seeds=[sample_repeat_seed(key.sample_index, key.repeat_index, stage=1)],
                    )[0]
                    parse_error: str | None = None
                    decoded_calls: list[dict[str, Any]] = []
                    try:
                        if _looks_like_template_leak(output.text):
                            raise ValueError("decision stage leaked internal template/control tokens")
                        decision_completion = _complete_bfcl_exec_forced_prefix(record, output.text)
                        decoded_calls = decode_simple_tool_call_response(decision_completion)
                    except Exception as exc:  # noqa: BLE001
                        parse_error = str(exc)
                        decision_completion = _complete_bfcl_exec_forced_prefix(record, output.text)
                    evaluation = evaluate_bfcl_exec_calls(record, decoded_calls, parse_error=parse_error, sandbox=sandbox)
                    stage = StageRecord(prompt=prompt, completion=decision_completion, stop_reason=output.finish_reason)
                    payload = SampleRecord(
                        benchmark_name=run.benchmark_name,
                        dataset_split=run.dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        stages=[stage],
                        sampling_config=sampling_payload,
                    ).as_payload()
                    payload["agent_result"] = {
                        "reward": float(evaluation.reward),
                        "num_turns": 1,
                        "cost": 0.0,
                        "is_passed": bool(evaluation.is_passed),
                        "error": evaluation.fail_reason or None,
                    }
                    payload["agent_info"] = {
                        **dict(evaluation.details),
                        "fail_reason": evaluation.fail_reason,
                        "cot_mode": CoTMode.COT.value,
                    }
                    payload["agent_trace"] = [
                        {
                            "decision_completion": output.text,
                            "decision_completion_for_eval": decision_completion,
                            "decision_stop_reason": output.finish_reason,
                            "decoded_calls": decoded_calls,
                            "decoded_executable_calls": evaluation.details.get("decoded_executable_calls", []),
                            "parse_error": parse_error or "",
                        }
                    ]
                    payload["task_id"] = record.task_id
                    payload["domain"] = "function_call"
                    payload["instruction"] = record.instruction
                    payload["metadata"] = dict(record.metadata)
                    writer.enqueue(payload)
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_bfcl_exec_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={"cot_mode": CoTMode.COT.value, "history_max_chars": history_max_chars},
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"bfcl_exec done: {len(completions_payloads)} samples")
    return 0


def _execution_result_matches(actual: BfclExecCallResult, expected: BfclExecCallResult, match_type: str) -> tuple[bool, str]:
    if not expected.success:
        return False, f"expected_execution_error({expected.error})"
    if not actual.success:
        return False, f"decoded_execution_error({actual.error})"
    normalized = str(match_type or "exact_match").strip().lower()
    if normalized == "structural_match":
        return (True, "ok") if _same_structure(actual.result, expected.result) else (False, "structure_mismatch")
    if normalized == "real_time_match":
        return (True, "ok") if _real_time_value_matches(actual.result, expected.result) else (False, "real_time_mismatch")
    return (True, "ok") if _value_matches(actual.result, expected.result) else (False, "exact_mismatch")


def _same_structure(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(key in actual and _same_structure(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if not expected or not actual:
            return True
        return _same_structure(actual[0], expected[0])
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return True
    return type(actual) is type(expected) or isinstance(actual, type(expected))


def _real_time_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (OverflowError, ValueError):
            return actual == expected
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        baseline = max(abs(expected_float), 1.0)
        return abs(actual_float - expected_float) / baseline <= 0.20
    return _value_matches(actual, expected) or _same_structure(actual, expected)


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (OverflowError, ValueError):
            return actual == expected
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        return math.isclose(actual_float, expected_float, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, str) and isinstance(expected, str):
        return normalize_rwkv_text(actual).strip() == normalize_rwkv_text(expected).strip()
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return dict(actual) == dict(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_value_matches(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _both_plain_ints(actual: Any, expected: Any) -> bool:
    return (
        isinstance(actual, int)
        and not isinstance(actual, bool)
        and isinstance(expected, int)
        and not isinstance(expected, bool)
    )


def bfcl_official_ast_checker_status(official_root: str | Path | None = None) -> BfclOfficialAstStatus:
    root = Path(official_root or _default_bfcl_official_root()).expanduser().resolve()
    missing = tuple(name for name in BFCL_OFFICIAL_AST_DEPENDENCIES if importlib.util.find_spec(name) is None)
    if missing:
        return BfclOfficialAstStatus(False, str(root), missing, "")
    if not (root / "bfcl_eval" / "eval_checker" / "ast_eval" / "ast_checker.py").exists():
        return BfclOfficialAstStatus(False, str(root), (), f"missing official BFCL checker under {root}")
    try:
        import sys

        added = False
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
            added = True
        try:
            __import__("bfcl_eval.eval_checker.ast_eval.ast_checker")
        finally:
            if added:
                try:
                    sys.path.remove(str(root))
                except ValueError:
                    pass
    except Exception as exc:  # noqa: BLE001
        return BfclOfficialAstStatus(False, str(root), (), str(exc))
    return BfclOfficialAstStatus(True, str(root), (), "")


def _default_bfcl_official_root() -> Path:
    for candidate in (
        Path(__file__).resolve().parents[3] / "references" / "gorilla" / "berkeley-function-call-leaderboard",
        Path("/tmp/gorilla-official/berkeley-function-call-leaderboard"),
        Path("/tmp/rwkv-official-refs/gorilla/berkeley-function-call-leaderboard"),
    ):
        if (candidate / "bfcl_eval" / "eval_checker").is_dir():
            return candidate
    return Path(__file__).resolve().parents[3] / "references" / "gorilla" / "berkeley-function-call-leaderboard"


def _truthy_exec_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _parse_call(text: str) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"not a function call: {text}")
    name = _call_name(parsed.body.func)
    args = tuple(_literal_from_ast(item) for item in parsed.body.args)
    kwargs = {keyword.arg: _literal_from_ast(keyword.value) for keyword in parsed.body.keywords if keyword.arg is not None}
    return name, args, kwargs


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_from_ast(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.BinOp):
        left = _literal_from_ast(node.left)
        right = _literal_from_ast(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
    return ast.literal_eval(node)


def _python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        rendered = ", ".join(_python_literal(item) for item in value)
        return f"({rendered}{',' if len(value) == 1 else ''})"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_python_literal(key)}: {_python_literal(item)}" for key, item in value.items()) + "}"
    return repr(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _result_payload(result: BfclExecCallResult) -> dict[str, Any]:
    return {"call": result.call, "success": result.success, "result": result.result, "error": result.error or ""}


def _ground_truth_to_exec_calls(raw: Any) -> list[str]:
    calls: list[str] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            text = item.strip()
            if text:
                calls.append(text)
            continue
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        name, raw_arguments = next(iter(item.items()))
        arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, Mapping):
            for key, value in raw_arguments.items():
                options = _coerce_list(value)
                arguments[str(key)] = options[0] if options else value
        calls.append(render_bfcl_exec_call({"name": str(name), "arguments": arguments}))
    return calls


def _expected_tool_calls_from_exec(calls: Sequence[str]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for call in calls:
        try:
            name, args, kwargs = _parse_call(call)
        except Exception:
            continue
        arguments = dict(kwargs)
        if args:
            arguments["_args"] = list(args)
        expected.append(
            {
                "name": name,
                "arguments": arguments,
                "argument_options": {key: [value] for key, value in arguments.items()},
            }
        )
    return expected


def _render_bfcl_question(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    turns = _coerce_list(raw)
    parts: list[str] = []
    for turn in turns:
        messages = _coerce_list(turn)
        for message in messages:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or "").strip()
                if content:
                    parts.append(f"{role.title()}: {content}")
            elif str(message or "").strip():
                parts.append(str(message).strip())
    return "\n".join(parts).strip()


def _normalize_tool_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "name": "unknown_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    parameters = source.get("parameters") or {"type": "object", "properties": {}, "required": []}
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = dict(parameters)
    if str(parameters.get("type") or "").lower() == "dict":
        parameters["type"] = "object"
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    return {
        "name": str(source.get("name") or raw.get("name") or "unknown_tool"),
        "description": str(source.get("description") or raw.get("description") or ""),
        "parameters": parameters,
    }


def _read_json_or_jsonl_items(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError(f"unsupported JSON payload: {path}")


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


__all__ = [
    "BfclExecCallResult",
    "BfclExecEvaluation",
    "BfclExecRecord",
    "BfclExecSandbox",
    "BfclOfficialAstStatus",
    "bfcl_official_ast_checker_status",
    "build_bfcl_exec_prompt",
    "evaluate_bfcl_exec_calls",
    "load_bfcl_exec_manifest_records",
    "load_bfcl_exec_rows_from_sources",
    "normalize_bfcl_exec_manifest_row",
    "render_bfcl_exec_call",
    "_run_bfcl_exec",
]
