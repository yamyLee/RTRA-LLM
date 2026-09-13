#!/usr/bin/env python3
"""Measure LLM calls made by the existing RTRA-LLM simulation.

This script does not change the simulator.  It temporarily replaces the
provider call with a tracing wrapper, runs the requested cases, and writes:

  - call_trace.csv       one row per LLM request;
  - case_summary.csv     one row per simulated case;
  - experiment_meta.json run configuration and notes.

Exact token counts are recorded when the OpenAI-compatible response exposes
usage metadata.  Otherwise the token columns are left blank; character
counts and wall-clock latency are still recorded.

Example:
    python scripts/measure_llm_calls.py \
        --provider qwen \
        --cases 1 7 \
        --trigger-mode risk \
        --risk-threshold 0.30 \
        --max-output-tokens 500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.paper_parameters import PAPER_LOW_LEVEL_PLANNER, PAPER_RANDOM_SEEDS
from src.core.experiment_metrics import count_turn_events as _count_turn_events

TRACE_ROWS: List[Dict[str, Any]] = []
CURRENT_CONTEXT: Dict[str, Any] = {}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _usage_from_response(response: Any) -> Dict[str, Optional[int]]:
    """Read common LangChain/OpenAI usage metadata layouts."""
    usage: Dict[str, Any] = {}

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage.update(usage_metadata)

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            usage.update(token_usage)
        usage.update({
            key: value
            for key, value in response_metadata.items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        })

    def first(*names: str) -> Optional[int]:
        for name in names:
            value = usage.get(name)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    return {
        "input_tokens": first("input_tokens", "prompt_tokens"),
        "output_tokens": first("output_tokens", "completion_tokens"),
        "total_tokens": first("total_tokens"),
    }


def _install_tracer() -> None:
    """Patch only the provider method used by the current simulation."""
    from src.decision_making.multi_llm_decision import ChatOpenAIProvider

    def traced_generate_response(provider_self: Any, prompt: str) -> str:
        if not provider_self.client:
            return f"{provider_self.provider_name} not available"

        started = time.perf_counter()
        status = "ok"
        error = ""
        response_text = ""
        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

        try:
            response = provider_self.client.invoke(prompt)
            response_text = _content_to_text(getattr(response, "content", response))
            usage = _usage_from_response(response)
        except Exception as exc:  # preserve current provider behaviour
            status = "error"
            error = str(exc)
            response_text = f"{provider_self.provider_name} error: {error}"

        elapsed = time.perf_counter() - started
        TRACE_ROWS.append({
            "case_number": CURRENT_CONTEXT.get("case_number"),
            "seed": CURRENT_CONTEXT.get("seed"),
            "condition": CURRENT_CONTEXT.get("condition", ""),
            "provider": provider_self.provider_name,
            "model": getattr(provider_self, "model", ""),
            "status": status,
            "latency_s": round(elapsed, 6),
            "prompt_chars": len(prompt),
            "response_chars": len(response_text),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "error": error,
        })
        return response_text

    ChatOpenAIProvider.generate_response = traced_generate_response


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 6) if values else None


def _case_summary(
    case_number: int,
    seed: int,
    run_elapsed_s: float,
    result: Dict[str, Any],
    configured_fixed_interval: int,
) -> Dict[str, Any]:
    rows = [
        row for row in TRACE_ROWS
        if row.get("case_number") == case_number and row.get("seed") == seed
    ]

    def values(name: str) -> List[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    input_tokens = values("input_tokens")
    output_tokens = values("output_tokens")
    total_tokens = values("total_tokens")
    risk = np.asarray(result.get("risk", []), dtype=float).reshape(-1).tolist()
    dcpa_m = np.asarray(result.get("dcpa", []), dtype=float).reshape(-1).tolist()
    kdir = [float(value) for value in result.get("kdir", [])]
    x_values = [float(value) for value in result.get("x", [])]
    y_values = [float(value) for value in result.get("y", [])]

    return {
        "case_number": case_number,
        "seed": seed,
        "reported_llm_calls": result.get("llm_call_count", 0),
        "traced_llm_calls": len(rows),
        "failed_calls": sum(row.get("status") != "ok" for row in rows),
        "input_tokens": sum(input_tokens) if input_tokens else None,
        "output_tokens": sum(output_tokens) if output_tokens else None,
        "total_tokens": sum(total_tokens) if total_tokens else None,
        "avg_latency_s": _mean(values("latency_s")),
        "llm_total_latency_s": round(sum(values("latency_s")), 6),
        "prompt_chars": sum(values("prompt_chars")),
        "response_chars": sum(values("response_chars")),
        "simulation_wall_time_s": round(run_elapsed_s, 6),
        "R_max": max(risk) if risk else None,
        "R_avg": _mean(risk),
        "min_dcpa_nm": min(abs(value) for value in dcpa_m) / 1852.0 if dcpa_m else None,
        "total_turns": _count_turn_events(kdir),
        "final_dist_nm": (
            math.hypot(x_values[-1], y_values[-1]) / 1852.0
            if x_values and y_values
            else None
        ),
        "trigger_mode": result.get("llm_trigger_mode", ""),
        "fixed_interval": configured_fixed_interval,
        "risk_threshold": result.get("llm_risk_threshold", 0.30),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, help="Provider name, e.g. qwen or deepseek")
    parser.add_argument("--cases", nargs="+", type=int, required=True, help="Case numbers to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_RANDOM_SEEDS))
    parser.add_argument("--trigger-mode", choices=("risk", "fixed", "always", "high_level_always"), default="risk")
    parser.add_argument("--risk-threshold", type=float, default=0.30)
    parser.add_argument("--fixed-interval", type=int, default=250)
    parser.add_argument("--sim-time", type=float, default=450.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--disable-rule-validator", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to output/experiments/llm_trace_<timestamp>",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from src.core.simulation import load_env_file, run_simulation
    from src.decision_making.rtra_llm_supervisor import RiskTriggeredLLMSupervisor

    load_env_file()
    provider_prefix = args.provider.upper()
    if args.provider.lower() == "openai":
        os.environ["OPENAI_MAX_TOKENS"] = str(args.max_output_tokens)
    else:
        os.environ[f"{provider_prefix}_MAX_TOKENS"] = str(args.max_output_tokens)

    _install_tracer()

    supervisor_probe = RiskTriggeredLLMSupervisor(provider=args.provider)
    if not supervisor_probe.available:
        raise RuntimeError(
            f"Provider '{args.provider}' is unavailable. Configure its API key, model, "
            "and base URL in .env or config/api_keys.json."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or ROOT / "output" / "experiments" / f"llm_trace_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for case_number in args.cases:
        for seed in args.seeds:
            CURRENT_CONTEXT.clear()
            CURRENT_CONTEXT.update({
                "case_number": case_number,
                "seed": seed,
                "condition": args.trigger_mode,
            })
            simulation_args = SimpleNamespace(
                case_number=case_number,
                all_cases=False,
                sim_time=args.sim_time,
                dt=args.dt,
                seed=seed,
                no_animation=True,
                output_dir=str(output_dir),
                llm=1,
                llm_provider=args.provider,
                compare=False,
                llm_trigger_mode=args.trigger_mode,
                llm_risk_threshold=args.risk_threshold,
                llm_fixed_interval=args.fixed_interval,
                disable_memory=args.disable_memory,
                disable_rule_validator=args.disable_rule_validator,
                low_level_planner=PAPER_LOW_LEVEL_PLANNER,
            )

            started = time.perf_counter()
            result = run_simulation(simulation_args, return_data=True)
            elapsed = time.perf_counter() - started
            summaries.append(_case_summary(case_number, seed, elapsed, result, args.fixed_interval))
            traced_calls = len([
                row for row in TRACE_ROWS
                if row.get("case_number") == case_number and row.get("seed") == seed
            ])
            print(
                f"case={case_number} seed={seed} reported_calls={result.get('llm_call_count', 0)} "
                f"traced_calls={traced_calls} wall_time={elapsed:.2f}s"
            )

    _write_csv(output_dir / "call_trace.csv", TRACE_ROWS)
    _write_csv(output_dir / "case_summary.csv", summaries)
    metadata = {
        "provider": args.provider,
        "cases": args.cases,
        "seeds": args.seeds,
        "trigger_mode": args.trigger_mode,
        "risk_threshold": args.risk_threshold,
        "fixed_interval": args.fixed_interval,
        "sim_time_s": args.sim_time,
        "dt_s": args.dt,
        "low_level_planner": PAPER_LOW_LEVEL_PLANNER,
        "max_output_tokens": args.max_output_tokens,
        "memory_enabled": not args.disable_memory,
        "validator_enabled": not args.disable_rule_validator,
        "note": "Exact token columns are blank when the provider response exposes no usage metadata.",
    }
    (output_dir / "experiment_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved_to={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
