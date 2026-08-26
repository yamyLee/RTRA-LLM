"""Fixed-interval LLM sensitivity experiment.

This runner evaluates how the invocation interval of a fixed-frequency LLM
baseline affects both communication cost and navigation behaviour.  It keeps
the simulation, case set, risk threshold and metric definitions consistent
with :mod:`run_llm_baseline` and adds an optional matched-module control in
which maneuver memory and the rule validator remain enabled.

Examples
--------
Run the legacy fixed-trigger baseline for all cases::

    python run_fixed_interval_sensitivity.py \
        --provider qwen --configuration legacy \
        --intervals 100 250 500 750

Run the matched fixed-trigger control and include the event-triggered RTRA
reference::

    python run_fixed_interval_sensitivity.py \
        --provider qwen --configuration matched --include-rtra

The script writes ``raw_results.csv``, ``summary.csv`` and
``experiment_metadata.json`` to a timestamped directory under
``output/experiments`` unless ``--output_dir`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from copy import copy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from src.config.paper_parameters import (
    LLM_RISK_THRESHOLD,
    PAPER_LOW_LEVEL_PLANNER,
    RANDOM_SEED,
    SIMULATION_DT_S,
    SIMULATION_TIME_S,
)
from src.core.experiment_metrics import count_turn_events

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METERS_TO_NMI = 1 / 1852
DEFAULT_INTERVALS = (100, 250, 500, 750)
NUMERIC_METRICS = (
    "R_max",
    "R_avg",
    "min_dcpa_nm",
    "A_turn_pct",
    "delta_D_pct",
    "final_dist_nm",
    "N_call",
    "trigger_events",
    "total_turns",
    "call_rate_per_1000_steps",
)


def _finite_mean(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else 0.0


def _compute_aturn(llm_kdir: Any, baseline_kdir: Any) -> float:
    llm = np.asarray(llm_kdir).reshape(-1)
    baseline = np.asarray(baseline_kdir).reshape(-1)
    n = min(len(llm), len(baseline))
    if n == 0:
        return 100.0
    return round(float(np.mean(np.sign(llm[:n]) == np.sign(baseline[:n])) * 100), 2)


def _distance_from_origin(results: Dict[str, Any]) -> float:
    x = np.asarray(results["x"], dtype=float).reshape(-1)
    y = np.asarray(results["y"], dtype=float).reshape(-1)
    return float(np.hypot(x[-1], y[-1]))


def _compute_delta_d(llm_final_dist_m: float, baseline_final_dist_m: float) -> float:
    if abs(baseline_final_dist_m) < 1e-12:
        return 0.0
    return round((llm_final_dist_m - baseline_final_dist_m) / baseline_final_dist_m * 100, 3)


def _case_numbers(values: Optional[Sequence[int]]) -> List[int]:
    if values:
        return [int(value) for value in values]
    from src.utils.imazu_cases import get_case_numbers

    return [int(value) for value in get_case_numbers()]


def _make_base_args(namespace: argparse.Namespace) -> SimpleNamespace:
    """Build the complete argument object expected by ``run_simulation``."""

    return SimpleNamespace(
        case_number=1,
        all_cases=False,
        sim_time=float(namespace.sim_time),
        dt=float(namespace.dt),
        seed=int(namespace.seed),
        no_animation=True,
        output_dir=str(namespace.output_dir),
        llm=1,
        llm_provider=namespace.provider,
        compare=False,
        llm_trigger_mode="fixed",
        llm_risk_threshold=float(namespace.risk_threshold),
        llm_fixed_interval=int(namespace.intervals[0]),
        disable_memory=namespace.configuration == "legacy",
        disable_rule_validator=namespace.configuration == "legacy",
    )


def _check_provider(provider: str) -> None:
    """Fail early instead of silently producing zero-call LLM results."""

    from src.core.simulation import load_env_file
    from src.decision_making.rtra_llm_supervisor import RiskTriggeredLLMSupervisor

    load_env_file()
    supervisor = RiskTriggeredLLMSupervisor(provider=provider)
    if not supervisor.available:
        raise RuntimeError(
            f"LLM provider '{provider}' is not configured. Set the provider API "
            "key/base URL in .env or config/api_keys.json before running the experiment."
        )


def _extract_metrics(
    results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    *,
    case_number: int,
    provider: str,
    method: str,
    schedule: str,
    interval_steps: Optional[int],
    configuration: str,
    sim_time: float,
    dt: float,
) -> Dict[str, Any]:
    """Extract metrics shared by all conditions in this sensitivity study."""

    llm_kdir = np.asarray(results["kdir"])
    baseline_kdir = np.asarray(baseline_results["kdir"])
    llm_final_dist_m = _distance_from_origin(results)
    baseline_final_dist_m = _distance_from_origin(baseline_results)
    interval_seconds = interval_steps * dt if interval_steps is not None else None
    steps = int(round(sim_time / dt))
    call_count = int(results.get("llm_call_count", 0))
    risk = np.asarray(results["risk"], dtype=float)

    return {
        "method": method,
        "schedule": schedule,
        "configuration": configuration,
        "backend_provider": provider,
        "case_number": int(case_number),
        "sim_time_s": float(sim_time),
        "dt_s": float(dt),
        "sim_steps": steps,
        "interval_steps": interval_steps,
        "interval_seconds": interval_seconds,
        "R_max": float(np.nanmax(risk)) if risk.size else 0.0,
        "R_avg": _finite_mean(risk),
        "min_dcpa_nm": float(np.nanmin(np.asarray(results["dcpa"], dtype=float)) * METERS_TO_NMI),
        "A_turn_pct": _compute_aturn(llm_kdir, baseline_kdir),
        "delta_D_pct": _compute_delta_d(llm_final_dist_m, baseline_final_dist_m),
        "final_dist_nm": llm_final_dist_m * METERS_TO_NMI,
        "baseline_dist_nm": baseline_final_dist_m * METERS_TO_NMI,
        "N_call": call_count,
        "trigger_events": int(len(results.get("llm_trigger_history", []))),
        "call_rate_per_1000_steps": call_count / steps * 1000 if steps else 0.0,
        "trigger_mode": str(results.get("llm_trigger_mode", schedule)),
        "risk_threshold": float(results.get("llm_risk_threshold", LLM_RISK_THRESHOLD)),
        "memory_enabled": bool(results.get("llm_memory_enabled", False)),
        "validator_enabled": bool(results.get("llm_validator_enabled", False)),
        "total_turns": count_turn_events(llm_kdir),
    }


def _run_condition(
    *,
    base_args: SimpleNamespace,
    case_number: int,
    baseline_results: Dict[int, Dict[str, Any]],
    method: str,
    schedule: str,
    interval_steps: Optional[int],
    configuration: str,
    provider: str,
) -> Optional[Dict[str, Any]]:
    from src.core.simulation import run_simulation

    if case_number not in baseline_results:
        baseline_args = copy(base_args)
        baseline_args.case_number = case_number
        baseline_args.llm = 0
        baseline_args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
        baseline_args.llm_provider = None
        baseline_results[case_number] = run_simulation(baseline_args, return_data=True)

    args = copy(base_args)
    args.case_number = case_number
    args.llm = 1
    args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
    args.llm_provider = provider
    args.llm_trigger_mode = schedule
    args.llm_fixed_interval = int(interval_steps or 1)
    args.disable_memory = configuration == "legacy"
    args.disable_rule_validator = configuration == "legacy"

    try:
        results = run_simulation(args, return_data=True)
        return _extract_metrics(
            results,
            baseline_results[case_number],
            case_number=case_number,
            provider=provider,
            method=method,
            schedule=schedule,
            interval_steps=interval_steps,
            configuration=configuration,
            sim_time=args.sim_time,
            dt=args.dt,
        )
    except Exception as exc:
        print(f"  [ERROR] {method}, case={case_number}: {exc}")
        traceback.print_exc()
        return None


def _summarize(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), row.get("interval_steps"))
        grouped.setdefault(key, []).append(row)

    summaries: List[Dict[str, Any]] = []
    for (method, interval_steps), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or 0)
    ):
        first = group[0]
        summary: Dict[str, Any] = {
            "method": method,
            "schedule": first["schedule"],
            "configuration": first["configuration"],
            "backend_provider": first["backend_provider"],
            "n_cases": len(group),
            "interval_steps": interval_steps,
            "interval_seconds": first["interval_seconds"],
            "sim_time_s": first["sim_time_s"],
            "dt_s": first["dt_s"],
        }
        for metric in NUMERIC_METRICS:
            values = np.asarray([row[metric] for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        summaries.append(summary)
    return summaries


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No successful experiment rows were produced.")
        return
    print("\nFixed-interval sensitivity (mean ± s.d. across cases)")
    print("method                 interval(s)   N_call       R_max       R_avg      A_turn%       ΔD%")
    print("-" * 94)
    for row in rows:
        interval = "RTRA" if row["interval_steps"] is None else f"{row['interval_seconds']:.1f}"
        print(
            f"{row['method']:<22} {interval:>10} "
            f"{row['N_call_mean']:>8.2f}±{row['N_call_std']:<5.2f} "
            f"{row['R_max_mean']:>8.3f}±{row['R_max_std']:<5.3f} "
            f"{row['R_avg_mean']:>8.3f}±{row['R_avg_std']:<5.3f} "
            f"{row['A_turn_pct_mean']:>8.2f}±{row['A_turn_pct_std']:<5.2f} "
            f"{row['delta_D_pct_mean']:>8.2f}±{row['delta_D_pct_std']:<5.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, help="Configured LLM provider, e.g. qwen or deepseek")
    parser.add_argument(
        "--configuration",
        choices=("legacy", "matched"),
        default="legacy",
        help="legacy disables memory/validator; matched keeps both enabled",
    )
    parser.add_argument("--intervals", nargs="+", type=int, default=list(DEFAULT_INTERVALS))
    parser.add_argument("--cases", nargs="+", type=int, default=None, help="Case numbers; default is all Imazu cases")
    parser.add_argument("--sim_time", type=float, default=SIMULATION_TIME_S)
    parser.add_argument("--dt", type=float, default=SIMULATION_DT_S)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="动力学随机扰动种子")
    parser.add_argument("--risk_threshold", type=float, default=LLM_RISK_THRESHOLD)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--include-rtra", action="store_true", help="Also run the event-triggered RTRA reference")
    parser.add_argument("--smoke", action="store_true", help="Run only case 1 with a short simulation")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned conditions without running simulations")
    return parser.parse_args()


def main() -> int:
    namespace = parse_args()
    if any(interval <= 0 for interval in namespace.intervals):
        raise SystemExit("All fixed intervals must be positive integers.")
    if namespace.dt <= 0 or namespace.sim_time <= 0:
        raise SystemExit("--dt and --sim_time must be positive.")

    if namespace.smoke:
        namespace.cases = [1]
        namespace.sim_time = min(namespace.sim_time, 30.0)

    cases = _case_numbers(namespace.cases)
    base_args = _make_base_args(namespace)
    conditions = [("Fixed-LLM", "fixed", interval) for interval in namespace.intervals]
    if namespace.include_rtra:
        conditions.append(("RTRA-LLM", "risk", None))

    print(f"Cases: {cases}")
    print(f"Conditions: {[(method, interval) for method, _, interval in conditions]}")
    print(f"Configuration: {namespace.configuration}; sim_time={namespace.sim_time}s; dt={namespace.dt}s")

    if namespace.dry_run:
        return 0

    _check_provider(namespace.provider)
    from src.core.simulation import load_env_file

    load_env_file()
    output_dir = namespace.output_dir or (
        ROOT / "output" / "experiments" / f"fixed_interval_sensitivity_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_results: Dict[int, Dict[str, Any]] = {}
    raw_rows: List[Dict[str, Any]] = []
    for method, schedule, interval_steps in conditions:
        print(f"\nRunning {method} ({'RTRA' if interval_steps is None else f'{interval_steps} steps'})")
        for case_number in cases:
            print(f"  case={case_number}")
            condition_configuration = "matched" if schedule == "risk" else namespace.configuration
            row = _run_condition(
                base_args=base_args,
                case_number=case_number,
                baseline_results=baseline_results,
                method=method,
                schedule=schedule,
                interval_steps=interval_steps,
                configuration=condition_configuration,
                provider=namespace.provider,
            )
            if row is not None:
                raw_rows.append(row)

    summary_rows = _summarize(raw_rows)
    _write_csv(raw_rows, output_dir / "raw_results.csv")
    _write_csv(summary_rows, output_dir / "summary.csv")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "provider": namespace.provider,
        "configuration": namespace.configuration,
        "cases": cases,
        "intervals_steps": namespace.intervals,
        "include_rtra": namespace.include_rtra,
        "sim_time_s": namespace.sim_time,
        "dt_s": namespace.dt,
        "low_level_planner": PAPER_LOW_LEVEL_PLANNER,
        "risk_threshold": namespace.risk_threshold,
        "expected_calls_by_interval": {
            str(interval): int(np.ceil((namespace.sim_time / namespace.dt) / interval))
            for interval in namespace.intervals
        },
        "metric_notes": {
            "N_call": "Observed successful LLM calls reported by RiskTriggeredLLMSupervisor.",
            "A_turn_pct": "Sign agreement of kdir with the no-LLM baseline, consistent with run_llm_baseline.py.",
            "delta_D_pct": "Final distance deviation from the no-LLM baseline; positive means farther from the origin.",
        },
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _print_summary(summary_rows)
    print(f"\nSaved results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
