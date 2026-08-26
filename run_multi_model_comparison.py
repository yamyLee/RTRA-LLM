"""
多模型对比实验脚本
===================================
在23个Imazu标准场景中对比多个LLM提供商的避碰性能。

所有提供商均以完整配置运行（LLM + Memory + Rule Validator），
触发模式均为 risk（q=0.30），与当前论文设定保持一致。

可选地附加一行 `LLM-baseline`：
  - 固定频率调用 LLM（trigger_mode=fixed）
  - 关闭机动记忆
  - 关闭规则验证器
  - 用于近似“在论文规定的 VO 底层规划器上仅叠加固定频率 LLM 决策”的对比设定

对比指标：
  - R_max  : 场景内最大碰撞风险（越低越好）
  - R_avg  : 场景内平均碰撞风险（越低越好）
  - A_turn : LLM与论文 VO 基准的转向一致率（越高越好）
  - ΔD     : 路径效率偏差（相对基准，越小越好）
  - N_call : LLM调用次数/场景（越少越好）

用法
----
  # 对比 deepseek 和 zhipu（最常用）
  python run_multi_model_comparison.py --providers deepseek zhipu

  # 完整多模型对比
  python run_multi_model_comparison.py --providers deepseek zhipu qwen openai

  # 冒烟测试（仅 case 1）
  python run_multi_model_comparison.py --providers deepseek --smoke

  # 指定场景子集
  python run_multi_model_comparison.py --providers deepseek zhipu --cases 1 2 7 8
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
from src.config.paper_parameters import (
    LLM_RISK_THRESHOLD,
    PAPER_LOW_LEVEL_PLANNER,
    RANDOM_SEED,
    SIMULATION_DT_S,
    SIMULATION_TIME_S,
)
from src.core.experiment_metrics import count_turn_events

# ── 项目根目录 ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))

# ── matplotlib 缓存 ───────────────────────────────────────────────────────────
_mpl_dir = os.environ.get("MPLCONFIGDIR")
if not _mpl_dir:
    _mpl_fallback = Path("/tmp/claude/matplotlib_cache")
    _mpl_fallback.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_fallback)


def _ensure_socksio() -> None:
    """若检测到 SOCKS 代理且 socksio 不可用，自动安装到临时目录。"""
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

# ── 共用工具函数（与其他脚本保持一致）────────────────────────────────────────

METERS_TO_NMI = 1 / 1852


def _finite_mean(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _compute_aturn(llm_kdir: np.ndarray, baseline_kdir: np.ndarray) -> float:
    """计算 LLM 与基准的转向一致率 A_turn（%）。

    定义：所有仿真时间步上两者方向符号一致的比例；双方均为 0
    （保向）时视为一致。
    """
    assert len(llm_kdir) == len(baseline_kdir), "kdir arrays must have equal length"
    total = len(llm_kdir)
    if total == 0:
        return 100.0
    agree = int(np.sum(np.sign(llm_kdir) == np.sign(baseline_kdir)))
    return round(agree / total * 100, 2)


def _compute_delta_d(llm_final_dist_m: float, baseline_final_dist_m: float) -> float:
    """计算路径效率偏差 ΔD（%）= (LLM - baseline) / baseline × 100。"""
    if abs(baseline_final_dist_m) < 1e-6:
        return 0.0
    return round((llm_final_dist_m - baseline_final_dist_m) / baseline_final_dist_m * 100, 3)


def _extract_metrics(
    results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    case_number: int,
    provider: str,
    backend_provider: str,
) -> Dict[str, Any]:
    """从 run_simulation(return_data=True) 的返回值提取多模型对比指标。"""
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
        "provider": provider,
        "backend_provider": backend_provider,
        "case_number": case_number,
        # 安全指标
        "R_max": float(np.max(risk)),
        "R_avg": _finite_mean(risk),
        "min_dcpa_nm": float(np.min(results["dcpa"]) * METERS_TO_NMI),
        # 机动一致性
        "A_turn_pct": _compute_aturn(llm_kdir, baseline_kdir),
        # 路径效率
        "delta_D_pct": _compute_delta_d(llm_dist_m, baseline_dist_m),
        "final_dist_nm": llm_dist_m * METERS_TO_NMI,
        "baseline_dist_nm": baseline_dist_m * METERS_TO_NMI,
        # LLM 调用
        "N_call": int(results.get("llm_call_count", 0)),
        "trigger_events": int(len(results.get("llm_trigger_history", []))),
        # 附加信息
        "trigger_mode": results.get("llm_trigger_mode", "risk"),
        "fixed_interval": results.get("llm_fixed_interval", 0),
        "risk_threshold": results.get("llm_risk_threshold", LLM_RISK_THRESHOLD),
        "total_turns": count_turn_events(llm_kdir),
    }


# ── 单次运行（一个 provider × 一个 case）────────────────────────────────────

def run_one(
    provider: str,
    case_number: int,
    base_args: SimpleNamespace,
    baseline_results_cache: Dict[int, Dict],
    *,
    display_name: Optional[str] = None,
    trigger_mode: str = "risk",
    fixed_interval: Optional[int] = None,
    disable_memory: bool = False,
    disable_rule_validator: bool = False,
    rule_baseline: bool = False,
) -> Optional[Dict[str, Any]]:
    """运行 (provider, case) 并与缓存的 baseline 结果计算对比指标。"""
    from src.core.simulation import run_simulation, load_env_file

    load_env_file()

    # ── 获取 baseline（仅首次运行时计算，之后复用缓存）──
    if case_number not in baseline_results_cache:
        b_args = copy(base_args)
        b_args.case_number = case_number
        b_args.llm = 0
        b_args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
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

    # ── 运行 LLM ──
    args = copy(base_args)
    args.case_number = case_number
    args.llm = 1
    args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
    args.llm_provider = None if rule_baseline else provider
    args.rule_baseline = rule_baseline
    args.llm_trigger_mode = trigger_mode
    if fixed_interval is not None:
        args.llm_fixed_interval = fixed_interval
    args.disable_memory = disable_memory
    args.disable_rule_validator = disable_rule_validator
    args.all_cases = False
    args.compare = False
    args.no_animation = True

    try:
        results = run_simulation(args, return_data=True)
        return _extract_metrics(
            results,
            baseline_res,
            case_number,
            display_name or provider,
            provider,
        )
    except Exception as exc:
        method_name = display_name or provider
        print(f"  [ERROR] provider={provider} method={method_name} case={case_number}: {exc}")
        traceback.print_exc()
        return None


# ── CSV / JSON 写入 ───────────────────────────────────────────────────────────

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


# ── 汇总统计（按 provider 聚合，跨 case 均值±std）────────────────────────────

NUMERIC_METRICS = [
    "R_max", "R_avg", "min_dcpa_nm", "A_turn_pct",
    "delta_D_pct", "final_dist_nm", "N_call", "trigger_events", "total_turns",
]


def _build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    groups: Dict[str, List] = defaultdict(list)
    for r in rows:
        groups[r["provider"]].append(r)

    summary = []
    for prov, group in groups.items():
        row: Dict[str, Any] = {
            "provider": prov,
            "n_cases": len(group),
            "backend_provider": group[0].get("backend_provider", prov),
            "trigger_mode": group[0]["trigger_mode"],
            "fixed_interval": group[0].get("fixed_interval", 0),
            "risk_threshold": group[0]["risk_threshold"],
        }
        for m in NUMERIC_METRICS:
            vals = np.array([g[m] for g in group if g.get(m) is not None], dtype=float)
            row[f"{m}_mean"] = float(np.mean(vals)) if vals.size > 0 else float("nan")
            row[f"{m}_std"] = float(np.std(vals)) if vals.size > 0 else float("nan")
        summary.append(row)

    return summary


def _print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print("MULTI-MODEL COMPARISON SUMMARY  (mean ± std across cases)")
    print("=" * 90)
    header = (
        f"{'Provider':<15} {'R_max':>7} {'R_avg':>7} {'A_turn%':>8} "
        f"{'ΔD%':>7} {'N_call':>7} {'MinDCPA':>8}"
    )
    print(header)
    print("-" * 90)
    for r in summary_rows:
        print(
            f"{r['provider']:<15} "
            f"{r['R_max_mean']:>6.3f} "
            f"{r['R_avg_mean']:>7.3f} "
            f"{r['A_turn_pct_mean']:>7.1f}% "
            f"{r['delta_D_pct_mean']:>6.2f}% "
            f"{r['N_call_mean']:>7.1f} "
            f"{r['min_dcpa_nm_mean']:>8.3f}"
        )
    print("=" * 90)


# ── 命令行 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTRA-LLM 多模型对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--providers", type=str, nargs="+", required=True,
                   help="LLM 提供商列表（空格分隔），如：deepseek zhipu qwen")
    p.add_argument("--cases", type=int, nargs="+", default=None,
                   help="场景编号列表，默认全部23个")
    p.add_argument("--output_dir", type=str, default=None,
                   help="输出目录（默认自动生成含时间戳目录）")
    p.add_argument("--sim_time", type=float, default=SIMULATION_TIME_S)
    p.add_argument("--dt", type=float, default=SIMULATION_DT_S)
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="动力学随机扰动种子")
    p.add_argument("--risk_threshold", type=float, default=LLM_RISK_THRESHOLD)
    p.add_argument("--include_llm_baseline", action="store_true",
                   help="附加一行 LLM-baseline：固定频率调用 LLM，关闭记忆与规则验证")
    p.add_argument("--include_rule_trigger", action="store_true",
                   help="附加 Rule-trigger baseline：无 LLM、保留风险触发/记忆/校验器")
    p.add_argument("--llm_baseline_provider", type=str, default=None,
                   help="LLM-baseline 使用的底层 provider；默认取 --providers 的第一个")
    p.add_argument("--llm_baseline_interval", type=int, default=250,
                   help="LLM-baseline 的固定调用步长（step 数）")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟测试：仅运行 case 1，sim_time=30s")
    return p.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = parse_args()

    if cli.smoke:
        cli.cases = [1]
        cli.sim_time = 30.0
        print("[SMOKE TEST] Running case 1 only, sim_time=30s")

    from src.utils.imazu_cases import get_case_numbers
    all_cases = get_case_numbers()
    cases = cli.cases if cli.cases is not None else all_cases

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    providers_tag = "_".join(cli.providers)
    if cli.output_dir:
        out_dir = Path(cli.output_dir)
    else:
        out_dir = Path("output/experiments") / f"multi_model_comparison_{providers_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = SimpleNamespace(
        llm_provider=None,
        llm_trigger_mode="risk",
        llm_risk_threshold=cli.risk_threshold,
        llm_fixed_interval=250,
        disable_memory=False,
        disable_rule_validator=False,
        sim_time=cli.sim_time,
        dt=cli.dt,
        seed=cli.seed,
        output_dir=str(out_dir),
    )

    include_llm_baseline = cli.include_llm_baseline
    llm_baseline_provider = cli.llm_baseline_provider or cli.providers[0]
    include_rule_trigger = cli.include_rule_trigger
    extra_methods = int(include_llm_baseline) + int(include_rule_trigger)
    total_runs = (len(cli.providers) + extra_methods) * len(cases)
    print("\n" + "=" * 70)
    print("RTRA-LLM MULTI-MODEL COMPARISON")
    print("=" * 70)
    print(f"Providers      : {cli.providers}")
    if include_llm_baseline:
        print(f"LLM-baseline   : on  ({llm_baseline_provider}, fixed every {cli.llm_baseline_interval} steps)")
    else:
        print("LLM-baseline   : off")
    print(f"Cases          : {cases}")
    print(f"Sim time       : {cli.sim_time}s  |  dt={cli.dt}s")
    print(f"Risk threshold : {cli.risk_threshold}")
    print(f"Total LLM runs : {total_runs}  (+ 1 baseline/case cached)")
    print(f"Output dir     : {out_dir}")
    print("=" * 70)

    # baseline 缓存：每个 case 只跑一次 baseline，所有 provider 复用
    baseline_cache: Dict[int, Dict] = {}
    all_rows: List[Dict[str, Any]] = []
    failed_runs: List[tuple] = []
    run_idx = 0

    for provider in cli.providers:
        print(f"\n{'─'*60}")
        print(f"[Provider] {provider}")
        print(f"{'─'*60}")
        prov_rows: List[Dict[str, Any]] = []

        for case_number in cases:
            run_idx += 1
            progress = f"[{run_idx:3d}/{total_runs}]"
            print(f"{progress} Case {case_number:2d} | {provider} ...", end=" ", flush=True)

            row = run_one(provider, case_number, base_args, baseline_cache)
            if row is not None:
                all_rows.append(row)
                prov_rows.append(row)
                print(
                    f"R_max={row['R_max']:.3f}  A_turn={row['A_turn_pct']:.1f}%  "
                    f"ΔD={row['delta_D_pct']:+.2f}%  N_call={row['N_call']}"
                )
            else:
                failed_runs.append((provider, case_number))
                print("FAILED")

        _write_csv(prov_rows, out_dir / f"multi_model_{provider}.csv")

    if include_llm_baseline:
        print(f"\n{'─'*60}")
        print(f"[Method] LLM-baseline ({llm_baseline_provider})")
        print(f"{'─'*60}")
        baseline_method_rows: List[Dict[str, Any]] = []
        for case_number in cases:
            run_idx += 1
            progress = f"[{run_idx:3d}/{total_runs}]"
            print(f"{progress} Case {case_number:2d} | LLM-baseline ...", end=" ", flush=True)

            row = run_one(
                llm_baseline_provider,
                case_number,
                base_args,
                baseline_cache,
                display_name="LLM-baseline",
                trigger_mode="fixed",
                fixed_interval=cli.llm_baseline_interval,
                disable_memory=True,
                disable_rule_validator=True,
            )
            if row is not None:
                all_rows.append(row)
                baseline_method_rows.append(row)
                print(
                    f"R_max={row['R_max']:.3f}  A_turn={row['A_turn_pct']:.1f}%  "
                    f"ΔD={row['delta_D_pct']:+.2f}%  N_call={row['N_call']}"
                )
            else:
                failed_runs.append(("LLM-baseline", case_number))
                print("FAILED")

        _write_csv(baseline_method_rows, out_dir / "multi_model_LLM-baseline.csv")

    if include_rule_trigger:
        print(f"\n{'─'*60}")
        print("[Method] Rule-trigger baseline (deterministic heuristic)")
        print(f"{'─'*60}")
        rule_rows: List[Dict[str, Any]] = []
        for case_number in cases:
            run_idx += 1
            progress = f"[{run_idx:3d}/{total_runs}]"
            print(f"{progress} Case {case_number:2d} | Rule-trigger ...", end=" ", flush=True)
            row = run_one(
                "rule",
                case_number,
                base_args,
                baseline_cache,
                display_name="Rule-trigger baseline",
                rule_baseline=True,
            )
            if row is not None:
                all_rows.append(row)
                rule_rows.append(row)
                print(
                    f"R_max={row['R_max']:.3f}  A_turn={row['A_turn_pct']:.1f}%  "
                    f"ΔD={row['delta_D_pct']:+.2f}%  N_call={row['N_call']}"
                )
            else:
                failed_runs.append(("Rule-trigger baseline", case_number))
                print("FAILED")
        _write_csv(rule_rows, out_dir / "multi_model_Rule-trigger-baseline.csv")

    # 写出 baseline 结果
    baseline_rows = []
    for case_number, br in baseline_cache.items():
        risk = br["risk"]
        kdir = np.asarray(br["kdir"])
        x_arr = np.asarray(br["x"])
        y_arr = np.asarray(br["y"])
        baseline_rows.append({
            "provider": "baseline",
            "backend_provider": "none",
            "case_number": case_number,
            "R_max": float(np.max(risk)),
            "R_avg": _finite_mean(risk),
            "min_dcpa_nm": float(np.min(br["dcpa"]) * METERS_TO_NMI),
            "A_turn_pct": 100.0,
            "delta_D_pct": 0.0,
            "final_dist_nm": float(np.sqrt(x_arr[-1] ** 2 + y_arr[-1] ** 2)) * METERS_TO_NMI,
            "baseline_dist_nm": float(np.sqrt(x_arr[-1] ** 2 + y_arr[-1] ** 2)) * METERS_TO_NMI,
            "N_call": 0,
            "trigger_events": 0,
            "trigger_mode": "none",
            "fixed_interval": 0,
            "risk_threshold": 0.0,
            "total_turns": count_turn_events(kdir),
        })
    all_rows_with_baseline = baseline_rows + all_rows
    _write_csv(all_rows_with_baseline, out_dir / "multi_model_all_raw.csv")

    summary_rows = _build_summary(all_rows)
    summary_csv_path = out_dir / "multi_model_summary.csv"
    _write_csv(summary_rows, summary_csv_path)

    json_path = out_dir / "multi_model_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": ts, "providers": cli.providers,
                    "cases": cases, "sim_time": cli.sim_time, "dt": cli.dt,
                    "low_level_planner": PAPER_LOW_LEVEL_PLANNER,
                    "risk_threshold": cli.risk_threshold,
                    "include_llm_baseline": include_llm_baseline,
                    "llm_baseline_provider": llm_baseline_provider if include_llm_baseline else None,
                    "llm_baseline_interval": cli.llm_baseline_interval if include_llm_baseline else None,
                    "failed_runs": [{"provider": p, "case": n} for p, n in failed_runs],
                },
                "raw": all_rows_with_baseline,
                "summary": summary_rows,
            },
            f, indent=2, default=str,
        )
    print(f"\n  -> JSON saved to {json_path}")

    _print_summary(summary_rows)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"  Total runs     : {total_runs}")
    print(f"  Successful     : {len(all_rows)}")
    print(f"  Failed         : {len(failed_runs)}")
    if failed_runs:
        for p, n in failed_runs:
            print(f"    - provider={p}, case={n}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
