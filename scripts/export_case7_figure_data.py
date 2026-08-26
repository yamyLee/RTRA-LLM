"""Run the paper's Case 7 and export trace data for Figures 3 and 4.

The exported NPZ keeps the numerical arrays separate from the JSON metadata so
the plotting script can be rerun without rerunning an LLM call.  The default
arguments are the paper settings: 450 s, 0.1 s sampling, q0=0.30 and VO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paper_parameters import (  # noqa: E402
    LLM_RISK_THRESHOLD,
    PAPER_LOW_LEVEL_PLANNER,
    RANDOM_SEED,
    SIMULATION_DT_S,
    SIMULATION_TIME_S,
)
from src.core.simulation import run_simulation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "output/figures/case7")
    parser.add_argument("--case_number", type=int, default=7)
    parser.add_argument("--sim_time", type=float, default=SIMULATION_TIME_S)
    parser.add_argument("--dt", type=float, default=SIMULATION_DT_S)
    parser.add_argument("--llm_provider", default="qwen")
    parser.add_argument("--llm", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rule_baseline", action="store_true")
    parser.add_argument("--low_level_planner", choices=["reactive", "vo"], default=PAPER_LOW_LEVEL_PLANNER)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--llm_trigger_mode", choices=["risk", "fixed", "always"], default="risk")
    parser.add_argument("--llm_risk_threshold", type=float, default=LLM_RISK_THRESHOLD)
    parser.add_argument("--disable_memory", action="store_true")
    parser.add_argument("--disable_rule_validator", action="store_true")
    return parser


def export_case7_data(args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_simulation(args, return_data=True)

    array_keys = [
        "time",
        "x",
        "y",
        "psi",
        "control_kdir",
        "kdir",
        "risk",
        "dcpa",
        "tcpa",
        "distance_ob",
        "obstacles_x_history",
        "obstacles_y_history",
        "obstacles_heading_rad",
        "target_speed_mps",
        "waypoints_x",
        "waypoints_y",
    ]
    missing = [key for key in array_keys if key not in result]
    if missing:
        raise KeyError(f"Simulation return_data is missing required arrays: {missing}")

    npz_path = args.output_dir / "case7_trace.npz"
    np.savez_compressed(
        npz_path,
        **{key: np.asarray(result[key]) for key in array_keys},
    )

    metadata = {
        "case_number": int(result.get("case_number", args.case_number)),
        "dt": float(result.get("dt", args.dt)),
        "sim_time": float(args.sim_time),
        "risk_threshold": float(result.get("llm_risk_threshold", args.llm_risk_threshold)),
        "llm_call_count": int(result.get("llm_call_count", 0)),
        "llm_trigger_mode": result.get("llm_trigger_mode", args.llm_trigger_mode),
        "llm_memory_enabled": bool(result.get("llm_memory_enabled", not args.disable_memory)),
        "llm_validator_enabled": bool(result.get("llm_validator_enabled", not args.disable_rule_validator)),
        "rule_baseline": bool(result.get("rule_baseline", args.rule_baseline)),
        "low_level_planner": result.get("low_level_planner", args.low_level_planner),
        "random_seed": result.get("random_seed", args.seed),
        "paper_roles": ["overtaking", "crossing"],
        "trigger_history": result.get("llm_trigger_history", []),
    }
    metadata_path = args.output_dir / "case7_trace.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, metadata_path


def main() -> None:
    args = build_parser().parse_args()
    npz_path, metadata_path = export_case7_data(args)
    print(f"Saved numerical trace: {npz_path}")
    print(f"Saved event metadata: {metadata_path}")


if __name__ == "__main__":
    main()
