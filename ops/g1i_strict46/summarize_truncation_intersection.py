#!/usr/bin/env python3
"""Summarize truncation rates on an exact common benchmark-cell intersection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ops.g1i_strict46.analyze_truncation_history import _aggregate, _pct


def _cell(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["benchmark_name"]),
        str(row["benchmark_split"]),
        str(row["expected_mode"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--database", required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--sizes", nargs="+", required=True)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    rows = payload["databases"][args.database]["selected_rows"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["family"]), str(row["size"]))
        if key[0] in args.families and key[1] in args.sizes:
            groups[key].append(row)

    requested = [(family, size) for family in args.families for size in args.sizes]
    missing_groups = [key for key in requested if key not in groups]
    if missing_groups:
        raise SystemExit(f"missing groups: {missing_groups}")
    common = set.intersection(*({ _cell(row) for row in groups[key] } for key in requested))
    domains: dict[str, int] = defaultdict(int)
    exemplar = { _cell(row): row for row in groups[requested[0]] }
    for cell in common:
        domains[str(exemplar[cell]["domain"])] += 1

    print(f"common_cells={len(common)} domains={dict(sorted(domains.items()))}")
    print("family size initial final overall missing blank")
    for key in requested:
        selected = [row for row in groups[key] if _cell(row) in common]
        aggregate = _aggregate(selected)
        print(
            f"{key[0]:4s} {key[1]:5s} "
            f"{_pct(aggregate['macro_initial_truncation_rate'])} "
            f"{_pct(aggregate['macro_final_stage_truncation_rate'])} "
            f"{_pct(aggregate['macro_overall_truncation_rate'])} "
            f"{_pct(aggregate['macro_missing_prediction_rate'])} "
            f"{_pct(aggregate['macro_blank_primary_rate'])}"
        )

    print("\nby_domain")
    for key in requested:
        for domain in sorted(domains):
            selected = [
                row
                for row in groups[key]
                if _cell(row) in common and str(row["domain"]) == domain
            ]
            aggregate = _aggregate(selected)
            print(
                f"{key[0]:4s} {key[1]:5s} {domain:24s} "
                f"cells={aggregate['cells']:2d} "
                f"initial={_pct(aggregate['macro_initial_truncation_rate'])} "
                f"final={_pct(aggregate['macro_final_stage_truncation_rate'])}"
            )


if __name__ == "__main__":
    main()
