"""Evaluate a trained Sawada-style PPO-LSTM checkpoint in the unified simulator."""

from __future__ import annotations

import csv
import json
import time
from copy import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from src.config.ppo_lstm import PROJECT_ROOT, load_ppo_lstm_config
from src.core.ppo_lstm_environment import SawadaPpoLstmEnv
from src.core.experiment_metrics import trajectory_metrics
from src.core.run_archive import save_run
from src.decision_making.ppo_lstm_policy import SawadaPpoLstmAgent, require_torch


def _resolve_checkpoint(args) -> Path:
    value = getattr(args, "ppo_checkpoint", None)
    if not value:
        raise ValueError(
            "--ppo_checkpoint is required for PPO-LSTM evaluation. "
            "Train one with train_ppo_lstm.py first."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(f"PPO-LSTM checkpoint not found: {path}")
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_trace(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(
    args,
    config: Dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
    *,
    agent=None,
    checkpoint_metadata=None,
) -> Dict[str, Any]:
    if getattr(args, "compare", False):
        raise ValueError(
            "--compare is not supported for PPO-LSTM; run the same case/seed combinations "
            "and use scripts/analyze_paper_statistics.py for paired analysis"
        )
    case_number = int(args.case_number)
    if case_number not in config["case_numbers"]:
        raise ValueError(f"Unsupported PPO-LSTM case {case_number}")
    require_torch()
    if agent is None:
        agent, checkpoint_metadata = SawadaPpoLstmAgent.load(
            checkpoint, device=getattr(args, "ppo_device", "auto")
        )
    # The checkpoint is authoritative for network and observation dimensions;
    # common simulation invariants are checked against the requested config.
    checkpoint_config = agent.config
    for key in ("dt_s", "episode_time_s", "policy_interval_s"):
        if checkpoint_config["simulation"][key] != config["simulation"][key]:
            raise ValueError(f"Checkpoint/config mismatch for simulation.{key}")

    env = SawadaPpoLstmEnv(checkpoint_config, training=False)
    observation = env.reset(case_number=case_number, seed=int(args.seed))
    recurrent_state = agent.initial_state()
    wall_started = time.perf_counter()
    while not env.done:
        inference_started = time.perf_counter()
        action, _raw, _log_prob, _value, recurrent_state = agent.act(
            observation,
            recurrent_state,
            deterministic=not getattr(args, "ppo_stochastic", False),
        )
        env.inference_latency_s.append(time.perf_counter() - inference_started)
        observation, _reward, _done, _info = env.step(action)
    wall_time = time.perf_counter() - wall_started
    result = {
        "method": "ppo_lstm",
        "backend": "unified_rtra_environment",
        "case_number": case_number,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "policy_interval_steps": int(checkpoint_config["policy_interval_steps"]),
        "policy_interval_s": float(checkpoint_config["simulation"]["policy_interval_s"]),
        "dt_s": float(checkpoint_config["simulation"]["dt_s"]),
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "mean_inference_latency_s": float(np.mean(env.inference_latency_s)),
        "std_inference_latency_s": float(np.std(env.inference_latency_s)),
        "runtime_s": wall_time,
        **env.summary(),
    }
    # Build paired VO comparison using exactly the policy trajectory's time grid.
    from src.core.simulation import run_simulation
    from types import SimpleNamespace
    trajectory = env.evaluation_trajectory()
    reference_args = SimpleNamespace(case_number=case_number, seed=int(args.seed),
        sim_time=len(trajectory['time']) * env.dt, dt=env.dt, llm=0,
        no_animation=True, save_runs=True, output_dir=str(output_dir / "vo_reference"))
    reference = run_simulation(reference_args, return_data=True)
    result.update(trajectory_metrics(trajectory, reference))
    result['paired_vo_archive_path'] = reference['archive_path']
    result['evaluation_horizon_s'] = len(trajectory['time']) * env.dt
    output_dir.mkdir(parents=True, exist_ok=True)
    # Always archive numerical evaluation traces, even when the optional compact
    # per-policy CSV is disabled. Keep PPO weights tied to this run by hash.
    import hashlib
    result['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result['archive_path'] = save_run({**trajectory, **result}, args, root=output_dir)
    _write_json(output_dir / "ppo_lstm_result.json", result)
    if getattr(args, "ppo_save_traces", False):
        _write_trace(output_dir / "ppo_lstm_trace.csv", env.trace)
    return result


def _write_batch_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fields = [
        "metric_schema", "path_length_m", "route_excess_pct", "max_cross_track_m",
        "route_progress_m", "safety_domain_intrusion_count", "safety_domain_intrusion_duration_s",
        "reverse_switches", "execution_rule_violation_pct", "archive_path", "paired_vo_archive_path",
        "case_number", "seed", "R_max", "R_avg", "min_dcpa_nm",
        "min_actual_separation_nm", "A_turn_pct", "delta_D_pct", "final_dist_nm", "baseline_dist_nm", "collision",
        "success", "collision_free_episode", "policy_steps", "mean_inference_latency_s", "runtime_s",
        "checkpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _run_all(args, config: Dict[str, Any], checkpoint: Path, output_root: Path) -> Dict[str, Any]:
    require_torch()
    agent, checkpoint_metadata = SawadaPpoLstmAgent.load(
        checkpoint, device=getattr(args, "ppo_device", "auto")
    )
    results = []
    for case_number in config["case_numbers"]:
        case_args = copy(args)
        case_args.case_number = case_number
        case_args.all_cases = False
        print(f"\n=== PPO-LSTM evaluation: case {case_number}/{config['case_numbers'][-1]} ===")
        results.append(
            _run_one(
                case_args,
                config,
                checkpoint,
                output_root / f"case_{case_number:02d}",
                agent=agent,
                checkpoint_metadata=checkpoint_metadata,
            )
        )
    batch = {
        "method": "ppo_lstm",
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "cases": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "ppo_lstm_batch_summary.json", batch)
    _write_batch_csv(output_root / "ppo_lstm_batch_summary.csv", results)
    return batch


def run_ppo_lstm(args, return_data: bool = False) -> Dict[str, Any]:
    """Run one or all configured scenarios using a trained recurrent policy."""
    del return_data
    config = load_ppo_lstm_config(getattr(args, "ppo_config", None))
    checkpoint = _resolve_checkpoint(args)
    output_root = Path(getattr(args, "output_dir", "output/ppo_lstm"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    if getattr(args, "all_cases", False):
        return _run_all(args, config, checkpoint, output_root.resolve())
    return _run_one(args, config, checkpoint, output_root.resolve())
