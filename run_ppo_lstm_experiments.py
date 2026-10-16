#!/usr/bin/env python3
"""Run the 23-case x 5-seed PPO-LSTM paper evaluation and merge raw results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from src.config.paper_parameters import PAPER_RANDOM_SEEDS
from src.core.ppo_lstm_simulation import run_ppo_lstm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="output/ppo_lstm/paper_evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_RANDOM_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-traces", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_dir).resolve()
    all_rows = []
    for seed in args.seeds:
        run_args = SimpleNamespace(
            ppo_config=args.config,
            ppo_checkpoint=args.checkpoint,
            ppo_device=args.device,
            ppo_stochastic=False,
            ppo_save_traces=args.save_traces,
            compare=False,
            all_cases=True,
            case_number=1,
            seed=seed,
            output_dir=str(output_root / f"seed_{seed}"),
        )
        batch = run_ppo_lstm(run_args)
        all_rows.extend(batch["cases"])

    fields = [
        "metric_schema", "path_length_m", "route_excess_pct", "max_cross_track_m",
        "route_progress_m", "safety_domain_intrusion_count", "safety_domain_intrusion_duration_s",
        "reverse_switches", "execution_rule_violation_pct", "archive_path", "paired_vo_archive_path",
        "case_number", "seed", "R_max", "R_avg", "min_dcpa_nm",
        "min_actual_separation_nm", "A_turn_pct", "delta_D_pct", "collision",
        "success", "termination_reason", "policy_steps", "low_level_steps",
        "mean_inference_latency_s", "std_inference_latency_s", "runtime_s",
        "checkpoint",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "ppo_lstm_paper_runs.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field) for field in fields})
    print(f"Wrote {len(all_rows)} runs to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
