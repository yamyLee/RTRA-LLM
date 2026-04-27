"""Batch runners for reproducing multi-case RTRA-LLM experiments."""

from __future__ import annotations

import csv
import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from src.utils.imazu_cases import get_case_numbers


def _finite_mean(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    return float(np.mean(finite_values)) if finite_values.size > 0 else 0.0


def _summarize_single_case(results: Dict[str, Any], args: SimpleNamespace) -> Dict[str, Any]:
    risk = results["risk"]
    kdir = results["kdir"]
    final_distance = np.sqrt(results["x"][-1] ** 2 + results["y"][-1] ** 2) / 1852

    return {
        "case_number": args.case_number,
        "method": "llm" if args.llm == 1 else "baseline",
        "llm_provider": args.llm_provider or "",
        "trigger_mode": args.llm_trigger_mode,
        "risk_threshold": args.llm_risk_threshold,
        "memory_enabled": not args.disable_memory,
        "validator_enabled": not args.disable_rule_validator,
        "total_turns": int(np.sum(np.abs(kdir) > 0.1)),
        "max_risk": float(np.max(risk)),
        "avg_risk": _finite_mean(risk),
        "avg_positive_risk": _finite_mean(risk[risk > 0]),
        "final_distance_nm": float(final_distance),
        "llm_calls": int(results.get("llm_call_count", 0)),
        "trigger_events": int(len(results.get("llm_trigger_history", []))),
    }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_batch_simulation(args: SimpleNamespace) -> Dict[str, Any]:
    """Run the selected simulation mode over all supported Imazu cases."""
    from src.core.comparison_simulation import run_comparison_simulation
    from src.core.simulation import run_simulation

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []

    for case_number in get_case_numbers():
        case_args = copy(args)
        case_args.case_number = case_number
        case_args.all_cases = False
        case_args.no_animation = True

        print(f"\n=== Batch Case {case_number} / {get_case_numbers()[-1]} ===")
        if case_args.compare:
            comparison = run_comparison_simulation(case_args)
            baseline_stats = comparison["stats"]["baseline"]
            llm_stats = comparison["stats"]["llm"]
            compare_rows.append(
                {
                    "case_number": case_number,
                    "llm_provider": case_args.llm_provider or "",
                    "trigger_mode": llm_stats.get("trigger_mode", case_args.llm_trigger_mode),
                    "baseline_turns": int(baseline_stats["total_turns"]),
                    "llm_turns": int(llm_stats["total_turns"]),
                    "turn_agreement": baseline_stats["turn_agreement"],
                    "baseline_max_risk": float(baseline_stats["max_risk"]),
                    "llm_max_risk": float(llm_stats["max_risk"]),
                    "baseline_avg_risk": float(baseline_stats["avg_risk"]),
                    "llm_avg_risk": float(llm_stats["avg_risk"]),
                    "baseline_final_distance_nm": float(baseline_stats["final_distance"]),
                    "llm_final_distance_nm": float(llm_stats["final_distance"]),
                    "path_efficiency_diff_pct": float(baseline_stats["path_efficiency_diff"]),
                    "llm_calls": int(llm_stats["llm_calls"]),
                    "trigger_events": int(llm_stats["trigger_events"]),
                }
            )
        else:
            results = run_simulation(case_args, return_data=True)
            rows.append(_summarize_single_case(results, case_args))

    summary: Dict[str, Any]
    if args.compare:
        _write_csv(compare_rows, output_dir / "batch_comparison_summary.csv")
        summary = {"mode": "compare", "cases": compare_rows}
    else:
        _write_csv(rows, output_dir / "batch_summary.csv")
        summary = {"mode": "single", "cases": rows}

    json_path = output_dir / "batch_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBatch summary written to {json_path}")
    return summary
