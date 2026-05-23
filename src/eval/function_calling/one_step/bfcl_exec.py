from __future__ import annotations

"""BFCL executable scorer backed by the official legacy BFCL runtime logic.

The BFCL v4 `exec_*` datasets are now under the official repository's
`unused_datasets`, while the current upstream evaluator skips executable
categories. This module vendors the relevant legacy executable behavior from
ShishirPatil/gorilla commit 28a0f42:

- execute ground-truth Python function calls to produce expected results
- execute model-produced Python function calls
- compare with exact_match, structural_match, or real_time_match

No argument-identity fallback is used. Unsupported functions, malformed calls,
missing API credentials, or runtime API failures fail the corresponding item.
"""

import ast
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import requests

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord

from .simple_tool_call import SimpleToolCallEvaluation


OFFICIAL_BFCL_EXEC_SOURCE = "ShishirPatil/gorilla@28a0f42"
REAL_TIME_MATCH_ALLOWED_DIFFERENCE = 0.2
_REQUEST_TIMEOUT_SECONDS = 30
_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


class BFCLCredentialError(RuntimeError):
    pass


class BFCLExecutionError(RuntimeError):
    pass


def evaluate_bfcl_executable_calls(
    record: FunctionCallTaskRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    expected_exprs = _ground_truth_expressions(record)
    match_types = _execution_result_types(record, len(expected_exprs))
    details: dict[str, Any] = {
        "official_bfcl_exec_source": OFFICIAL_BFCL_EXEC_SOURCE,
        "expected_expressions": expected_exprs,
        "decoded_tool_calls": [
            {
                "name": str(item.get("name") or ""),
                "arguments": dict(item.get("arguments") or {}),
            }
            for item in decoded_calls
        ],
        "actual_expressions": [],
        "execution_result_type": match_types,
        "parse_error": parse_error or "",
    }
    if parse_error:
        return SimpleToolCallEvaluation(0.0, False, parse_error, details)
    if not expected_exprs:
        return SimpleToolCallEvaluation(0.0, False, "bfcl_exec:missing_ground_truth", details)

    try:
        actual_exprs = [_tool_call_to_expression(item) for item in decoded_calls]
    except Exception as exc:  # noqa: BLE001
        details["expression_error"] = str(exc)
        return SimpleToolCallEvaluation(0.0, False, f"bfcl_exec:invalid_model_call:{exc}", details)
    details["actual_expressions"] = actual_exprs

    expected_results: list[Any] = []
    for expr in expected_exprs:
        result = _execute_official_expression(expr)
        if not result["valid"]:
            details["expected_execution_error"] = result
            return SimpleToolCallEvaluation(
                0.0,
                False,
                "bfcl_exec:official_ground_truth_execution_failed",
                details,
            )
        expected_results.append(result["value"])
    details["expected_results"] = [_jsonable(item) for item in expected_results]

    if _is_parallel_record(record):
        check = _official_parallel_no_order(actual_exprs, expected_results, match_types)
    else:
        check = _official_ordered_wrapper(actual_exprs, expected_results, match_types)

    details["official_check"] = check
    passed = bool(check["valid"])
    return SimpleToolCallEvaluation(
        reward=1.0 if passed else 0.0,
        is_passed=passed,
        fail_reason="" if passed else str(check.get("error_type") or "bfcl_exec:failed"),
        details=details,
    )


def _ground_truth_expressions(record: FunctionCallTaskRecord) -> list[str]:
    raw = (
        record.scorer.get("ground_truth")
        or record.metadata.get("expected_executable_calls")
        or record.metadata.get("bfcl_ground_truth")
    )
    if isinstance(raw, str):
        values: Sequence[Any] = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        values = raw
    else:
        values = []
    expressions = [str(item).strip() for item in values if str(item).strip()]
    if expressions:
        return expressions

    reconstructed: list[str] = []
    for item in record.expected_tool_calls:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments") or {}
        if not name or not isinstance(arguments, Mapping):
            continue
        reconstructed.append(_tool_call_to_expression({"name": name, "arguments": arguments}))
    return reconstructed


def _execution_result_types(record: FunctionCallTaskRecord, expected_count: int) -> list[str]:
    raw = (
        record.scorer.get("execution_result_type")
        or record.metadata.get("bfcl_execution_result_type")
        or record.metadata.get("execution_result_type")
    )
    if isinstance(raw, str):
        result = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        result = [str(item) for item in raw]
    else:
        result = []
    if not result:
        result = ["exact_match"]
    while len(result) < expected_count:
        result.append(result[-1])
    return result


def _is_parallel_record(record: FunctionCallTaskRecord) -> bool:
    category = str(record.metadata.get("category") or record.task_id or "")
    return "parallel" in category


def _tool_call_to_expression(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or "").strip()
    if not _FUNCTION_NAME_RE.match(name):
        raise ValueError(f"invalid function name: {name!r}")
    args = call.get("arguments") or {}
    if not isinstance(args, Mapping):
        raise ValueError(f"arguments must be an object for {name!r}")
    rendered_args = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({rendered_args})"


def _execute_official_expression(function_call: str) -> dict[str, Any]:
    try:
        parsed = ast.parse(str(function_call), mode="eval")
        if not isinstance(parsed.body, ast.Call):
            raise ValueError("expression is not a function call")
        value = eval(compile(parsed, "<bfcl_exec>", "eval"), _official_exec_globals(), {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "error": [f"Error in execution: {function_call!r}. Error: {exc}"],
            "error_type": "executable_checker:execution_error",
            "exception_type": type(exc).__name__,
        }
    if isinstance(value, tuple):
        value = list(value)
    return {"valid": True, "value": value}


def _official_ordered_wrapper(
    actual_exprs: Sequence[str],
    expected_results: Sequence[Any],
    expected_result_types: Sequence[str],
) -> dict[str, Any]:
    if len(actual_exprs) != len(expected_results):
        return {
            "valid": False,
            "error": [
                f"Wrong number of functions provided. Expected {len(expected_results)}, but got {len(actual_exprs)}."
            ],
            "error_type": "value_error:exec_result_count",
        }
    sub_checks: list[dict[str, Any]] = []
    for index, (actual_expr, expected_result) in enumerate(zip(actual_exprs, expected_results)):
        result_type = expected_result_types[index] if index < len(expected_result_types) else "exact_match"
        result = _official_executable_checker_simple(actual_expr, expected_result, result_type)
        sub_checks.append({"index": index, **result})
        if not result["valid"]:
            return {
                "valid": False,
                "error": result.get("error", []),
                "error_type": result.get("error_type", "executable_checker:failed"),
                "sub_checks": sub_checks,
            }
    return {
        "valid": True,
        "error": [],
        "error_type": "executable_checker:unclear",
        "sub_checks": sub_checks,
    }


def _official_parallel_no_order(
    actual_exprs: Sequence[str],
    expected_results: Sequence[Any],
    expected_result_types: Sequence[str],
) -> dict[str, Any]:
    if len(actual_exprs) != len(expected_results):
        return {
            "valid": False,
            "error": [
                f"Wrong number of functions provided. Expected {len(expected_results)}, but got {len(actual_exprs)}."
            ],
            "error_type": "value_error:exec_result_count",
        }

    matched_indices: list[int] = []
    for expected_index in range(len(expected_results)):
        all_errors: list[Any] = []
        result = {
            "valid": False,
            "error": [],
            "error_type": "executable_checker:unclear",
        }
        for actual_index, actual_expr in enumerate(actual_exprs):
            if actual_index in matched_indices:
                continue
            result = _official_executable_checker_simple(
                actual_expr,
                expected_results[expected_index],
                expected_result_types[expected_index],
            )
            if result["valid"]:
                matched_indices.append(actual_index)
                break
            all_errors.append(
                {
                    f"Model Result Index {actual_index}": {
                        "sub_error": result["error"],
                        "sub_error_type": result["error_type"],
                        "model_executed_output": result.get("model_executed_output"),
                    }
                }
            )
        if not result["valid"]:
            considered_indices = [idx for idx in range(len(actual_exprs)) if idx not in matched_indices]
            all_errors.insert(
                0,
                (
                    "Could not find a matching function among index "
                    f"{considered_indices} of model output for index {expected_index} of possible answers."
                ),
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "executable_checker:cannot_find_match",
            }
    return {"valid": True, "error": [], "error_type": "executable_checker:unclear"}


def _official_executable_checker_simple(
    function_call: str,
    expected_result: Any,
    expected_result_type: str,
    is_sanity_check: bool = False,
) -> dict[str, Any]:
    result = {"valid": True, "error": [], "error_type": "executable_checker:unclear"}
    executed = _execute_official_expression(function_call)
    if not executed["valid"]:
        return executed
    exec_output = executed["value"]

    if expected_result_type == "exact_match":
        if exec_output != expected_result:
            result["valid"] = False
            result["error"].append(
                f"Wrong execution result for {function_call!r}. Expected: {expected_result}, but got: {exec_output}."
            )
            result["error_type"] = "executable_checker:wrong_result"
            result["model_executed_output"] = _jsonable(exec_output)
            return result
    elif expected_result_type == "real_time_match":
        if isinstance(expected_result, (float, int)) and isinstance(exec_output, (float, int)):
            lower = expected_result * (1 - REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
            upper = expected_result * (1 + REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
            if not lower <= exec_output <= upper:
                result["valid"] = False
                result["error"].append(
                    (
                        f"Wrong execution result for {function_call!r}. Expected: {expected_result}, "
                        f"but got: {exec_output}. {REAL_TIME_MATCH_ALLOWED_DIFFERENCE * 100}% difference allowed."
                    )
                )
                result["error_type"] = "executable_checker:wrong_result_real_time"
                result["model_executed_output"] = _jsonable(exec_output)
                return result
        else:
            result["valid"] = False
            result["error"].append(
                (
                    f"Wrong execution result for {function_call!r}. Expected: {expected_result}, "
                    f"but got: {exec_output}. Type needs to be float or int for real time match criteria."
                )
            )
            result["error_type"] = "executable_checker:wrong_result_real_time"
            result["model_executed_output"] = _jsonable(exec_output)
            return result
    else:
        pattern_result = _official_pattern_matcher(
            exec_output,
            expected_result,
            function_call,
            is_sanity_check,
        )
        if not pattern_result["valid"]:
            return pattern_result
    result["model_executed_output"] = _jsonable(exec_output)
    return result


def _official_pattern_matcher(
    exec_output: Any,
    expected_result: Any,
    function_call: str,
    is_sanity_check: bool,
) -> dict[str, Any]:
    result = {"valid": True, "error": [], "error_type": "executable_checker:unclear"}
    if type(exec_output) is not type(expected_result):
        return {
            "valid": False,
            "error": [
                (
                    f"Wrong execution result type for {function_call!r}. Expected type: "
                    f"{type(expected_result)}, but got: {type(exec_output)}."
                )
            ],
            "error_type": "executable_checker:wrong_result_type",
            "model_executed_output": _jsonable(exec_output),
        }
    if isinstance(exec_output, dict):
        if is_sanity_check:
            if len(exec_output) != len(expected_result):
                return {
                    "valid": False,
                    "error": [
                        (
                            f"Wrong execution result pattern for {function_call!r}. Expect type Dict, "
                            "but wrong number of elements in the output."
                        )
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_length",
                    "model_executed_output": _jsonable(exec_output),
                }
            return result
        for key in expected_result:
            if key not in exec_output:
                return {
                    "valid": False,
                    "error": [
                        (
                            f"Wrong execution result pattern for {function_call!r}. Expect type Dict, "
                            f"but key {key!r} not found in the model output."
                        )
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_key_not_found",
                    "model_executed_output": _jsonable(exec_output),
                }
        for key in exec_output:
            if key not in expected_result:
                return {
                    "valid": False,
                    "error": [
                        (
                            f"Wrong execution result pattern for {function_call!r}. Expect type Dict, "
                            f"but key {key!r} not expected in the model output."
                        )
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_extra_key",
                    "model_executed_output": _jsonable(exec_output),
                }
    if isinstance(exec_output, list) and len(exec_output) != len(expected_result):
        return {
            "valid": False,
            "error": [
                (
                    f"Wrong execution result pattern for {function_call!r}. Expect type list, "
                    f"but wrong number of elements in the output. Expected length: {len(expected_result)}, "
                    f"but got: {len(exec_output)}."
                )
            ],
            "error_type": "executable_checker:wrong_result_type:list_length",
            "model_executed_output": _jsonable(exec_output),
        }
    return result


def _official_exec_globals() -> dict[str, Any]:
    values: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "len": len,
            "max": max,
            "min": min,
            "sum": sum,
            "range": range,
        }
    }
    values.update(_OFFICIAL_FUNCTIONS)
    values["math"] = math
    return values


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


_CREDENTIALS_CACHE: dict[str, str] | None = None


def _load_credentials() -> dict[str, str]:
    global _CREDENTIALS_CACHE
    if _CREDENTIALS_CACHE is not None:
        return _CREDENTIALS_CACHE

    credentials: dict[str, str] = {}
    candidates = [
        Path(os.environ["BFCL_FUNCTION_CREDENTIAL_CONFIG"])
        for _ in [0]
        if os.environ.get("BFCL_FUNCTION_CREDENTIAL_CONFIG")
    ]
    candidates.extend(
        [
            Path("function_credential_config.json"),
            Path(__file__).resolve().parents[3] / "function_credential_config.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    for key, value in item.items():
                        if value:
                            credentials[str(key)] = str(value)
        elif isinstance(payload, Mapping):
            for key, value in payload.items():
                if value:
                    credentials[str(key)] = str(value)

    env_aliases = {
        "GEOCODE-API-KEY": ("BFCL_GEOCODE_API_KEY", "GEOCODE_API_KEY"),
        "EXCHANGERATE-API-KEY": ("BFCL_EXCHANGERATE_API_KEY", "EXCHANGERATE_API_KEY"),
        "RAPID-API-KEY": ("BFCL_RAPID_API_KEY", "RAPID_API_KEY"),
        "OMDB-API-KEY": ("BFCL_OMDB_API_KEY", "OMDB_API_KEY"),
    }
    for official_name, aliases in env_aliases.items():
        for alias in aliases:
            if os.environ.get(alias):
                credentials[official_name] = str(os.environ[alias])
                break
    _CREDENTIALS_CACHE = credentials
    return credentials


def _api_key(name: str) -> str:
    value = _load_credentials().get(name)
    if not value:
        raise BFCLCredentialError(
            f"Missing BFCL executable credential {name}. Provide function_credential_config.json or env alias."
        )
    return value


def _request_get(*args: Any, **kwargs: Any) -> requests.Response:
    kwargs.setdefault("timeout", _REQUEST_TIMEOUT_SECONDS)
    return requests.get(*args, **kwargs)


def calculate_triangle_area(base, height):
    return base * height / 2


def get_distance(pointA, pointB):
    return ((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2) ** 0.5


def math_factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def quadratic_roots(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant >= 0:
        root1 = (-b + discriminant**0.5) / (2 * a)
        root2 = (-b - discriminant**0.5) / (2 * a)
        return [root1, root2]
    real_part = -b / (2 * a)
    imaginary_part = (abs(discriminant) ** 0.5) / (2 * a)
    return [
        {"real": real_part, "imaginary": imaginary_part},
        {"real": real_part, "imaginary": -imaginary_part},
    ]


def geometry_area_circle(radius):
    return math.pi * radius**2


def get_prime_factors(number):
    factors = []
    divisor = 2
    while number > 1:
        while number % divisor == 0:
            factors.append(divisor)
            number /= divisor
        divisor += 1
    return factors


def math_gcd(a, b):
    if b == 0:
        return a
    return math_gcd(b, a % b)


def math_lcm(a, b):
    return a * b / math_gcd(a, b)


def calculate_final_velocity(initial_velocity, acceleration, time):
    return initial_velocity + acceleration * time


def calculate_displacement(initial_velocity, acceleration, time):
    return initial_velocity * time + 0.5 * acceleration * time**2


def calculate_electrostatic_potential_energy(charge, voltage):
    return charge * voltage


def calculate_density(mass, volume):
    return mass / volume


def mat_mul(matA, matB):
    result = [[0 for _ in range(len(matB[0]))] for _ in range(len(matA))]
    for i in range(len(matA)):
        for j in range(len(matB[0])):
            for k in range(len(matB)):
                result[i][j] += matA[i][k] * matB[k][j]
    return result


def calculate_mean(numbers):
    return sum(numbers) / len(numbers)


def calculate_standard_deviation(numbers):
    mean = calculate_mean(numbers)
    variance = sum((number - mean) ** 2 for number in numbers) / len(numbers)
    return variance**0.5


def calc_binomial_probability(n, k, p):
    return math_factorial(n) / (math_factorial(k) * math_factorial(n - k)) * (p**k * (1 - p) ** (n - k))


def calculate_permutations(n, k):
    return math_factorial(n) / math_factorial(n - k)


def get_fibonacci_sequence(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence


def estimate_derivative(function, x):
    func = eval(function)  # noqa: S307 - mirrors BFCL executable_python_function.py
    h = 0.0000000001
    return (func(x + h) - func(x)) / h


def calculate_cosine_similarity(vectorA, vectorB):
    dot_product = sum(vectorA[i] * vectorB[i] for i in range(len(vectorA)))
    magnitudeA = (sum(vectorA[i] ** 2 for i in range(len(vectorA)))) ** 0.5
    magnitudeB = (sum(vectorB[i] ** 2 for i in range(len(vectorB)))) ** 0.5
    return dot_product / (magnitudeA * magnitudeB)


def mortgage_calculator(loan_amount, interest_rate, loan_period):
    monthly_interest_rate = interest_rate / 12
    number_of_payments = loan_period * 12
    monthly_payment = (
        loan_amount
        * monthly_interest_rate
        * (1 + monthly_interest_rate) ** number_of_payments
        / ((1 + monthly_interest_rate) ** number_of_payments - 1)
    )
    return monthly_payment


def calculate_future_value(present_value, interest_rate, periods):
    return present_value * (1 + interest_rate) ** periods


def sort_array(array, reverse=False):
    return sorted(array, reverse=reverse)


def get_weather_data(coordinates):
    lat, long = coordinates
    response = _request_get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": long,
            "current": "temperature_2m",
            "temperature_unit": "fahrenheit",
        },
    )
    if response.status_code == 200:
        return response.json()["current"]["temperature_2m"]
    return f"Failed to fetch data with status code: {response.status_code}"


def get_coordinates_from_city(city_name):
    time.sleep(2)
    response = _request_get(
        "https://geocode.maps.co/search",
        params={"q": city_name, "api_key": _api_key("GEOCODE-API-KEY")},
    )
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]["lat"], data[0]["lon"]
        return "No data found for the given city name."
    return f"Failed to fetch data with status code: {response.status_code}"


def convert_currency(amount, from_currency, to_currency):
    key = _api_key("EXCHANGERATE-API-KEY")
    response = _request_get(f"https://v6.exchangerate-api.com/v6/{key}/latest/{from_currency}")
    if response.status_code == 200:
        data = response.json()
        rates = data.get("conversion_rates", {})
        if to_currency in rates:
            return amount * rates[to_currency]
        return "Target currency code not found."
    return f"Failed to fetch data with status code: {response.status_code}"


def find_term_on_urban_dictionary(term):
    response = _request_get(
        "https://mashape-community-urban-dictionary.p.rapidapi.com/define",
        headers={
            "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
            "X-RapidAPI-Host": "mashape-community-urban-dictionary.p.rapidapi.com",
        },
        params={"term": term},
    )
    return response.json()["list"][0]["definition"]


def get_coordinate_by_ip_address(ip_address):
    response = _request_get(f"http://ip-api.com/json/{ip_address}")
    try:
        return (response.json()["lat"], response.json()["lon"])
    except Exception:  # noqa: BLE001
        return response.json()["message"]


def get_zipcode_by_ip_address(ip_address):
    response = _request_get(f"http://ip-api.com/json/{ip_address}")
    try:
        return response.json()["zip"]
    except Exception:  # noqa: BLE001
        return response.json()["message"]


def get_covid_death_by_country(country):
    response = _request_get(
        "https://covid-193.p.rapidapi.com/statistics",
        headers={
            "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
            "X-RapidAPI-Host": "covid-193.p.rapidapi.com",
        },
        params={"country": country},
    )
    try:
        return response.json()["response"][0]["deaths"]["total"]
    except Exception:  # noqa: BLE001
        return response.json()


def get_active_covid_case_by_country(country):
    response = _request_get(
        "https://covid-193.p.rapidapi.com/statistics",
        headers={
            "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
            "X-RapidAPI-Host": "covid-193.p.rapidapi.com",
        },
        params={"country": country},
    )
    try:
        return response.json()["response"][0]["cases"]["active"]
    except Exception:  # noqa: BLE001
        return response.json()


def get_rating_by_amazon_ASIN(ASIN):
    return _amazon_product_details(ASIN, "product_star_rating")


def get_price_by_amazon_ASIN(ASIN):
    return _amazon_product_details(ASIN, "product_price")


def get_product_name_by_amazon_ASIN(ASIN):
    return _amazon_product_details(ASIN, "product_title")


def _amazon_product_details(ASIN, field):
    retries = 0
    while retries < 5:
        response = _request_get(
            "https://real-time-amazon-data.p.rapidapi.com/product-details",
            headers={
                "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
                "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com",
            },
            params={"asin": ASIN, "country": "US"},
        )
        try:
            return response.json()["data"][field]
        except KeyError:
            time.sleep(2**retries)
            retries += 1
    return None


def get_company_name_by_stock_name(stock_name):
    response = _request_get(
        "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/search",
        headers={
            "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com",
        },
        params={"search": stock_name},
    )
    try:
        return response.json()["body"][0]["name"]
    except Exception:  # noqa: BLE001
        return response.json()


def get_stock_price_by_stock_name(stock_name):
    response = _request_get(
        "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/quotes",
        headers={
            "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com",
        },
        params={"ticker": stock_name},
    )
    try:
        return float(response.json()["body"][0]["regularMarketPrice"])
    except Exception:  # noqa: BLE001
        return response.json()


def get_stock_history(stock_name, interval, diffandsplits="true"):
    response = _request_get(
        "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history",
        headers={
            "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com",
        },
        params={
            "symbol": stock_name,
            "interval": interval,
            "diffandsplits": diffandsplits,
        },
    )
    try:
        data = response.json()["body"]
        return {key: data[key] for key in list(data)[-10:]}
    except Exception:  # noqa: BLE001
        return response.json()


def retrieve_city_based_on_zipcode(zipcode):
    response = _request_get(f"http://ziptasticapi.com/{zipcode}")
    try:
        return response.json()["city"]
    except Exception:  # noqa: BLE001
        return response.json()


def retrieve_holiday_by_year(country, year):
    return _request_get(f"https://date.nager.at/api/v3/publicholidays/{year}/{country}").json()


def get_time_zone_by_coord(long, lat):
    response = _request_get(
        "https://timezone-by-location.p.rapidapi.com/timezone",
        headers={
            "X-RapidAPI-Key": _api_key("RAPID-API-KEY"),
            "X-RapidAPI-Host": "timezone-by-location.p.rapidapi.com",
        },
        params={"lat": lat, "lon": long, "c": "1", "s": "0"},
    )
    try:
        return response.json()["Zones"][0]["TimezoneId"]
    except Exception:  # noqa: BLE001
        return response.json()


def linear_regression(x, y, point):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(x_i**2 for x_i in x)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n
    return slope * point + intercept


def add_binary_numbers(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]


def maxPoints(points) -> int:
    counter = 1
    if len(points) < 2:
        return 1
    for i in range(len(points)):
        slopes = {}
        for j in range(i + 1, len(points)):
            y = points[j][1] - points[i][1]
            x = points[j][0] - points[i][0]
            if x != 0:
                slopes[y / x] = 1 + slopes.get(y / x, 0)
            else:
                slopes["inf"] = 1 + slopes.get("inf", 0)
        for value in slopes.values():
            counter = max(counter, value)
    return counter + 1


def calculate_investment_value(
    initial_investment,
    annual_contribution,
    years,
    annual_return,
    inflation_rate,
    adjust_for_inflation=True,
):
    current_value = initial_investment
    real_value = initial_investment
    for year in range(1, years + 1):
        current_value = current_value * (1 + annual_return) + annual_contribution
        if adjust_for_inflation:
            inflation_adjustment = (
                1 - inflation_rate[year - 1] if year <= len(inflation_rate) else 1 - inflation_rate[-1]
            )
            real_value = (
                real_value * (1 + annual_return - inflation_rate[year - 1]) + annual_contribution * inflation_adjustment
            )
        else:
            real_value = current_value
    return real_value if adjust_for_inflation else current_value


def calculate_nutritional_needs(weight, height, age, gender, activity_level, goal):
    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    activity_multipliers = [1.2, 1.375, 1.55, 1.725, 1.9]
    tdee = bmr * activity_multipliers[activity_level - 1]
    if goal == "lose":
        tdee -= 500
    elif goal == "gain":
        tdee += 500
    return {
        "calories": tdee,
        "proteins_g": (tdee * 0.30) / 4,
        "fats_g": (tdee * 0.25) / 9,
        "carbohydrates_g": (tdee * 0.45) / 4,
    }


def book_room(room_type, price, check_in_date, check_out_date, customer_id, discount_code=None):
    if discount_code and discount_code == "DISCOUNT10":
        price *= 0.9
    return {
        "customer_id": customer_id,
        "room_number": room_type,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "total_price": price,
    }


def order_food(item, quantity, price):
    return sum([quantity[i] * price[i] for i in range(len(item))])


def get_movie_rating(movie_name):
    response = _request_get(
        "http://www.omdbapi.com/",
        params={"t": movie_name, "apikey": _api_key("OMDB-API-KEY")},
    )
    return response.json()["Rated"]


def get_movie_director(movie_name):
    response = _request_get(
        "http://www.omdbapi.com/",
        params={"t": movie_name, "apikey": _api_key("OMDB-API-KEY")},
    )
    return response.json()["Director"]


def polygon_area(vertices):
    n = len(vertices)
    if n < 3:
        raise ValueError("A polygon must have at least 3 vertices.")
    vertices.append(vertices[0])
    area = 0
    for i in range(n):
        area += (vertices[i][0] * vertices[i + 1][1]) - (vertices[i + 1][0] * vertices[i][1])
    return abs(area) / 2.0


_OFFICIAL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    name: value
    for name, value in globals().items()
    if callable(value)
    and not name.startswith("_")
    and name
    not in {
        "Any",
        "BFCLCredentialError",
        "BFCLExecutionError",
        "Callable",
        "FunctionCallTaskRecord",
        "Mapping",
        "Path",
        "Sequence",
        "SimpleToolCallEvaluation",
    }
}


__all__ = ["evaluate_bfcl_executable_calls"]
