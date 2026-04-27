"""
LLM-baseline 单独实验脚本
========================
仅运行一组“固定频率调用 LLM + 无机动记忆 + 无规则验证器”的对比实验，
用于生成单独的 LLM-baseline 结果。

实验设定
--------
  - 上层启用 LLM
  - trigger_mode = fixed
  - fixed_interval = 用户指定（默认 200 steps）
  - disable_memory = True
  - disable_rule_validator = True

对比指标
------------------------
  - R_max   : 场景内最大碰撞风险
  - R_avg   : 场景内平均碰撞风险
  - A_turn  : 相对 baseline 的转向一致率（%）
  - ΔD      : 相对 baseline 的路径效率偏差（%）
  - N_call  : LLM 调用次数/场景

用法
----
  # 运行完整 23 场实验
  python run_llm_baseline.py --provider deepseek

  # 指定固定调用间隔
  python run_llm_baseline.py --provider deepseek --fixed_interval 100

  # 冒烟测试（仅 case 1）
  python run_llm_baseline.py --provider deepseek --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from copy import copy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))

_mpl_dir = os.environ.get("MPLCONFIGDIR")
if not _mpl_dir:
    _mpl_fallback = Path("/tmp/claude/matplotlib_cache")
    _mpl_fallback.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_fallback)


def _ensure_socksio() -> None:
    if not os.environ.get("ALL_PROXY", "").startswith("socks"):
        return
    try:
        import socksio  # noqa: F401
    except ImportError:
        _pkg_dir = Path("/tmp/claude/pypackages")
        _pkg_dir.mkdir(parents=True, exist_ok=True)
        if str(_pkg_dir) not in sys.path:
            sys.path.insert(0, str(_pkg_dir))
        try:
            import socksio  # noqa: F401
        except ImportError:
            print("[setup] Installing socksio for SOCKS proxy support...")
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "socksio",
                 "--target", str(_pkg_dir), "-q"],
                timeout=60,
            )
            sys.path.insert(0, str(_pkg_dir))
            print("[setup] socksio installed OK")


_ensure_socksio()

METERS_TO_NMI = 1 / 1852


def _finite_mean(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _compute_aturn(llm_kdir: np.ndarray, baseline_kdir: np.ndarray) -> float:
    total = len(llm_kdir)
    if total == 0:
        return 100.0
    agree = int(np.sum(np.sign(llm_kdir) == np.sign(baseline_kdir)))
    return round(agree / total * 100, 2)


def _compute_delta_d(llm_final_dist_m: float, baseline_final_dist_m: float) -> float:
    if abs(baseline_final_dist_m) < 1e-6:
        return 0.0
    return round((llm_final_dist_m - baseline_final_dist_m) / baseline_final_dist_m * 100, 3)


def _extract_metrics(
    results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    case_number: int,
    provider: str,
    fixed_interval: int,
) -> Dict[str, Any]:
    risk = results["risk"]
    llm_kdir = np.asarray(results["kdir"])
    baseline_kdir = np.asarray(baseline_results["kdir"])

    x_arr = np.asarray(results["x"])
    y_arr = np.asarray(results["y"])
    bx_arr = np.asarray(baseline_results["x"])
    by_arr = np.asarray(baseline_results["y"])

    llm_dist_m = float(np.sqrt(x_arr[-1] ** 2 + y_arr[-1] ** 2))
    baseline_dist_m = float(np.sqrt(bx_arr[-1] ** 2 + by_arr[-1] ** 2))

    return {
        "provider": "LLM-baseline",
        "backend_provider": provider,
        "case_number": case_number,
        "R_max": float(np.max(risk)),
        "R_avg": _finite_mean(risk),
        "min_dcpa_nm": float(np.min(results["dcpa"]) * METERS_TO_NMI),
        "A_turn_pct": _compute_aturn(llm_kdir, baseline_kdir),
        "delta_D_pct": _compute_delta_d(llm_dist_m, baseline_dist_m),
        "final_dist_nm": llm_dist_m * METERS_TO_NMI,
        "baseline_dist_nm": baseline_dist_m * METERS_TO_NMI,
        "N_call": int(results.get("llm_call_count", 0)),
        "trigger_events": int(len(results.get("llm_trigger_history", []))),
        "trigger_mode": "fixed",
        "fixed_interval": fixed_interval,
        "risk_threshold": float(results.get("llm_risk_threshold", 0.30)),
        "memory_enabled": False,
        "validator_enabled": False,
        "total_turns": int(np.sum(np.abs(llm_kdir) > 0.1)),
    }


def run_one(
    provider: str,
    case_number: int,
    base_args: SimpleNamespace,
    baseline_results_cache: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    from src.core.simulation import load_env_file, run_simulation

    load_env_file()

    if case_number not in baseline_results_cache:
        b_args = copy(base_args)
        b_args.case_number = case_number
        b_args.llm = 0
        b_args.all_cases = False
        b_args.compare = False
        b_args.no_animation = True
        try:
            baseline_results_cache[case_number] = run_simulation(b_args, return_data=True)
        except Exception as exc:
            print(f"  [ERROR] baseline case={case_number}: {exc}")
            traceback.print_exc()
            return None

    baseline_res = baseline_results_cache[case_number]

    args = copy(base_args)
    args.case_number = case_number
    args.llm = 1
    args.llm_provider = provider
    args.llm_trigger_mode = "fixed"
    args.disable_memory = True
    args.disable_rule_validator = True
    args.all_cases = False
    args.compare = False
    args.no_animation = True

    try:
        results = run_simulation(args, return_data=True)
        return _extract_metrics(
            results,
            baseline_res,
            case_number,
            provider,
            args.llm_fixed_interval,
        )
    except Exception as exc:
        print(f"  [ERROR] provider={provider} case={case_number}: {exc}")
        traceback.print_exc()
        return None


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        print(f"  [WARN] No data to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Saved {len(rows)} rows to {path}")


NUMERIC_METRICS = [
    "R_max", "R_avg", "min_dcpa_nm", "A_turn_pct",
    "delta_D_pct", "final_dist_nm", "N_call", "trigger_events", "total_turns",
]


def _build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "provider": "LLM-baseline",
        "backend_provider": rows[0]["backend_provider"],
        "n_cases": len(rows),
        "trigger_mode": "fixed",
        "fixed_interval": rows[0]["fixed_interval"],
        "risk_threshold": rows[0]["risk_threshold"],
        "memory_enabled": False,
        "validator_enabled": False,
    }
    for metric in NUMERIC_METRICS:
        vals = np.array([r[metric] for r in rows if r.get(metric) is not None], dtype=float)
        row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size > 0 else float("nan")
        row[f"{metric}_std"] = float(np.std(vals)) if vals.size > 0 else float("nan")
    return row


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("LLM-BASELINE SUMMARY  (mean ± std across cases)")
    print("=" * 88)
    print(
        f"{'Method':<15} {'R_max':>7} {'R_avg':>7} {'A_turn%':>8} "
        f"{'ΔD%':>7} {'N_call':>7} {'MinDCPA':>8}"
    )
    print("-" * 88)
    print(
        f"{summary['provider']:<15} "
        f"{summary['R_max_mean']:>6.3f} "
        f"{summary['R_avg_mean']:>7.3f} "
        f"{summary['A_turn_pct_mean']:>7.1f}% "
        f"{summary['delta_D_pct_mean']:>6.2f}% "
        f"{summary['N_call_mean']:>7.1f} "
        f"{summary['min_dcpa_nm_mean']:>8.3f}"
    )
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTRA-LLM 单独 LLM-baseline 实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--provider", type=str, required=True,
                        help="底层 LLM provider，如 deepseek / qwen / zhipu / openai")
    parser.add_argument("--cases", type=int, nargs="+", default=None,
                        help="场景编号列表，默认全部23个")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认自动生成）")
    parser.add_argument("--sim_time", type=float, default=450.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--risk_threshold", type=float, default=0.30,
                        help="保留统一参数接口；fixed 模式下主要用于结果记录")
    parser.add_argument("--fixed_interval", type=int, default=200,
                        help="固定频率调用间隔（step 数）")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟测试：仅运行 case 1，sim_time=30s")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()

    if cli.smoke:
        cli.cases = [1]
        cli.sim_time = 30.0
        print("[SMOKE TEST] Running case 1 only, sim_time=30s")

    from src.utils.imazu_cases import get_case_numbers
    cases = cli.cases if cli.cases is not None else get_case_numbers()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cli.output_dir:
        out_dir = Path(cli.output_dir)
    else:
        out_dir = Path("output/experiments") / f"llm_baseline_{cli.provider}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = SimpleNamespace(
        llm_provider=cli.provider,
        llm_trigger_mode="fixed",
        llm_risk_threshold=cli.risk_threshold,
        llm_fixed_interval=cli.fixed_interval,
        disable_memory=True,
        disable_rule_validator=True,
        sim_time=cli.sim_time,
        dt=cli.dt,
        output_dir=str(out_dir),
    )

    print("\n" + "=" * 72)
    print("RTRA-LLM SINGLE METHOD EXPERIMENT — LLM-BASELINE")
    print("=" * 72)
    print(f"Provider       : {cli.provider}")
    print(f"Cases          : {cases}")
    print(f"Sim time       : {cli.sim_time}s  |  dt={cli.dt}s")
    print(f"Trigger mode   : fixed")
    print(f"Fixed interval : {cli.fixed_interval} steps")
    print("Memory         : off")
    print("Validator      : off")
    print(f"Output dir     : {out_dir}")
    print("=" * 72)

    baseline_cache: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    failed_cases: List[int] = []

    for idx, case_number in enumerate(cases, start=1):
        progress = f"[{idx:3d}/{len(cases)}]"
        print(f"{progress} Case {case_number:2d} | LLM-baseline ...", end=" ", flush=True)

        row = run_one(cli.provider, case_number, base_args, baseline_cache)
        if row is not None:
            rows.append(row)
            print(
                f"R_max={row['R_max']:.3f}  A_turn={row['A_turn_pct']:.1f}%  "
                f"ΔD={row['delta_D_pct']:+.2f}%  N_call={row['N_call']}"
            )
        else:
            failed_cases.append(case_number)
            print("FAILED")

    raw_csv_path = out_dir / "llm_baseline_raw.csv"
    summary_csv_path = out_dir / "llm_baseline_summary.csv"
    json_path = out_dir / "llm_baseline_results.json"

    _write_csv(rows, raw_csv_path)

    if rows:
        summary = _build_summary(rows)
        _write_csv([summary], summary_csv_path)
    else:
        summary = None
        print("  [WARN] No successful rows; summary will not be written.")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": ts,
                    "provider": cli.provider,
                    "cases": cases,
                    "sim_time": cli.sim_time,
                    "dt": cli.dt,
                    "trigger_mode": "fixed",
                    "fixed_interval": cli.fixed_interval,
                    "memory_enabled": False,
                    "validator_enabled": False,
                    "failed_cases": failed_cases,
                },
                "raw": rows,
                "summary": summary,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  -> JSON saved to {json_path}")

    if summary is not None:
        _print_summary(summary)

    print(f"\n{'=' * 72}")
    print("DONE")
    print(f"  Successful cases : {len(rows)}")
    print(f"  Failed cases     : {len(failed_cases)}")
    if failed_cases:
        print(f"  Failed list      : {failed_cases}")
    print(f"  Output dir       : {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
