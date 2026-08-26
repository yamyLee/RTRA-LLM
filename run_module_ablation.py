"""
模块消融实验脚本
================================
在23个Imazu标准场景中逐模块消融，验证各模块的独立贡献。

消融条件
--------------------------
  0. baseline          : 无 LLM（论文规定的 VO 底层规划器），作为参考基准
  1. full              : 完整 RTRA-LLM（LLM + Memory + Validator）
  2. wo_risk_trigger   : 无风险触发（trigger_mode=always，每步调用LLM）
                         ★ 等价于现有LLM避碰工作的朴素集成方式（[24][25]类）
  3. wo_memory         : 无机动记忆模块
  4. wo_validator      : 无规则验证器
  5. wo_memory_wo_val  : 同时关闭记忆与验证器

对比指标
--------------------------
  - N_call  : LLM调用次数/场景
  - A_turn  : 与基准的转向一致率（%）
  - R_max   : 最大碰撞风险
  - R_avg   : 平均碰撞风险
  - ΔD      : 路径效率偏差（%）
  - min_DCPA: 最小会遇距离（海里）

用法
----
  # 完整消融实验（deepseek）
  python run_module_ablation.py --provider deepseek

  # 冒烟测试（仅 case 1）
  python run_module_ablation.py --provider deepseek --smoke

  # 跳过某些条件
  python run_module_ablation.py --provider deepseek --skip_conditions 5

  # 指定场景子集
  python run_module_ablation.py --provider deepseek --cases 1 2 7 8
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

# ── 消融条件定义──────────────────────────────────────────────
ABLATION_CONDITIONS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "baseline",
        "label": "Baseline (no LLM)",
        "llm": 0,
        "trigger_mode": "risk",
        "disable_memory": False,
        "disable_rule_validator": False,
        "paper_row": "基准（无LLM）",
    },
    {
        "id": 1,
        "name": "full",
        "label": "Full RTRA-LLM",
        "llm": 1,
        "trigger_mode": "risk",
        "disable_memory": False,
        "disable_rule_validator": False,
        "paper_row": "完整方法",
    },
    {
        "id": 2,
        "name": "wo_risk_trigger",
        "label": "w/o Risk Trigger",
        "llm": 1,
        "trigger_mode": "always",        # ← 每步调用，等价于朴素LLM集成
        "disable_memory": False,
        "disable_rule_validator": False,
        "paper_row": "消去风险触发（≈现有LLM方法）",
    },
    {
        "id": 3,
        "name": "wo_memory",
        "label": "w/o Memory",
        "llm": 1,
        "trigger_mode": "risk",
        "disable_memory": True,
        "disable_rule_validator": False,
        "paper_row": "消去机动记忆",
    },
    {
        "id": 4,
        "name": "wo_validator",
        "label": "w/o Rule Validator",
        "llm": 1,
        "trigger_mode": "risk",
        "disable_memory": False,
        "disable_rule_validator": True,
        "paper_row": "消去规则验证器",
    },
    {
        "id": 5,
        "name": "wo_memory_wo_validator",
        "label": "w/o Memory & Validator",
        "llm": 1,
        "trigger_mode": "risk",
        "disable_memory": True,
        "disable_rule_validator": True,
        "paper_row": "消去记忆+验证器",
    },
]

METERS_TO_NMI = 1 / 1852


def _finite_mean(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _compute_aturn(llm_kdir: np.ndarray, baseline_kdir: np.ndarray) -> float:
    """A_turn：转向一致率（%）。"""
    total = len(llm_kdir)
    if total == 0:
        return 100.0
    agree = int(np.sum(np.sign(llm_kdir) == np.sign(baseline_kdir)))
    return round(agree / total * 100, 2)


def _compute_delta_d(llm_dist_m: float, baseline_dist_m: float) -> float:
    """ΔD：路径效率偏差（%）。"""
    if abs(baseline_dist_m) < 1e-6:
        return 0.0
    return round((llm_dist_m - baseline_dist_m) / baseline_dist_m * 100, 3)


def _extract_metrics(
    results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    case_number: int,
    condition: Dict[str, Any],
    provider: str,
) -> Dict[str, Any]:
    """提取论文所需指标。"""
    risk = results["risk"]
    llm_kdir = np.asarray(results["kdir"])
    baseline_kdir = np.asarray(baseline_results["kdir"])

    x_arr = np.asarray(results["x"])
    y_arr = np.asarray(results["y"])
    bx = np.asarray(baseline_results["x"])
    by = np.asarray(baseline_results["y"])

    llm_dist_m = float(np.sqrt(x_arr[-1] ** 2 + y_arr[-1] ** 2))
    baseline_dist_m = float(np.sqrt(bx[-1] ** 2 + by[-1] ** 2))

    return {
        "condition_id": condition["id"],
        "condition_name": condition["name"],
        "condition_label": condition["label"],
        "paper_row": condition["paper_row"],
        "case_number": case_number,
        "llm_provider": provider if condition["llm"] == 1 else "none",
        "llm_enabled": condition["llm"] == 1,
        "trigger_mode": condition["trigger_mode"],
        "memory_enabled": not condition["disable_memory"],
        "validator_enabled": not condition["disable_rule_validator"],
        "N_call": int(results.get("llm_call_count", 0)),
        "trigger_events": int(len(results.get("llm_trigger_history", []))),
        "A_turn_pct": _compute_aturn(llm_kdir, baseline_kdir),
        "R_max": float(np.max(risk)),
        "R_avg": _finite_mean(risk),
        "R_avg_positive": _finite_mean(risk[risk > 0]) if (risk > 0).any() else 0.0,
        "min_dcpa_nm": float(np.min(results["dcpa"]) * METERS_TO_NMI),
        "delta_D_pct": _compute_delta_d(llm_dist_m, baseline_dist_m),
        "final_dist_nm": llm_dist_m * METERS_TO_NMI,
        # ── 辅助指标 ──
        "total_turns": count_turn_events(llm_kdir),
        "starboard_turns": int(np.sum(llm_kdir > 0.5)),
        "port_turns": int(np.sum(llm_kdir < -0.5)),
    }


# ── 单次运行 ──────────────────────────────────────────────────────────────────

def run_one(
    condition: Dict[str, Any],
    case_number: int,
    base_args: SimpleNamespace,
    baseline_cache: Dict[int, Dict],
) -> Optional[Dict[str, Any]]:
    from src.core.simulation import run_simulation, load_env_file

    load_env_file()

    # ── 获取/缓存 baseline ──
    if case_number not in baseline_cache:
        b_args = copy(base_args)
        b_args.case_number = case_number
        b_args.llm = 0
        b_args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
        b_args.llm_trigger_mode = "risk"
        b_args.all_cases = False
        b_args.compare = False
        b_args.no_animation = True
        b_args.disable_memory = False
        b_args.disable_rule_validator = False
        try:
            baseline_cache[case_number] = run_simulation(b_args, return_data=True)
        except Exception as exc:
            print(f"  [ERROR] baseline case={case_number}: {exc}")
            traceback.print_exc()
            return None

    baseline_res = baseline_cache[case_number]

    # ── baseline 条件直接从缓存返回 ──
    if condition["llm"] == 0:
        return _extract_metrics(
            baseline_res, baseline_res, case_number, condition,
            base_args.llm_provider or "",
        )

    # ── LLM 条件 ──
    args = copy(base_args)
    args.case_number = case_number
    args.llm = 1
    args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
    args.llm_trigger_mode = condition["trigger_mode"]
    args.disable_memory = condition["disable_memory"]
    args.disable_rule_validator = condition["disable_rule_validator"]
    args.all_cases = False
    args.compare = False
    args.no_animation = True

    try:
        results = run_simulation(args, return_data=True)
        return _extract_metrics(
            results, baseline_res, case_number, condition,
            base_args.llm_provider or "",
        )
    except Exception as exc:
        print(f"  [ERROR] condition={condition['name']} case={case_number}: {exc}")
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


# ── 汇总统计 ──────────────────────────────────────────────────────────────────

NUMERIC_METRICS = [
    "N_call", "trigger_events", "A_turn_pct",
    "R_max", "R_avg", "R_avg_positive", "min_dcpa_nm",
    "delta_D_pct", "final_dist_nm", "total_turns",
]


def _build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    groups: Dict[str, List] = defaultdict(list)
    for r in rows:
        groups[r["condition_name"]].append(r)

    id_map = {c["name"]: c["id"] for c in ABLATION_CONDITIONS}
    summary = []
    for cond_name, group in groups.items():
        row: Dict[str, Any] = {
            "condition_name": cond_name,
            "condition_label": group[0]["condition_label"],
            "paper_row": group[0]["paper_row"],
            "llm_provider": group[0]["llm_provider"],
            "llm_enabled": group[0]["llm_enabled"],
            "trigger_mode": group[0]["trigger_mode"],
            "memory_enabled": group[0]["memory_enabled"],
            "validator_enabled": group[0]["validator_enabled"],
            "n_cases": len(group),
        }
        for m in NUMERIC_METRICS:
            vals = np.array([g[m] for g in group if g.get(m) is not None], dtype=float)
            row[f"{m}_mean"] = float(np.mean(vals)) if vals.size > 0 else float("nan")
            row[f"{m}_std"] = float(np.std(vals)) if vals.size > 0 else float("nan")
        summary.append(row)

    summary.sort(key=lambda r: id_map.get(r["condition_name"], 99))
    return summary


def _print_summary(summary_rows: List[Dict[str, Any]], provider: str) -> None:
    print("\n" + "=" * 100)
    print(f"MODULE ABLATION SUMMARY  [{provider}]  (mean ± std across cases)")
    print("=" * 100)
    header = (
        f"{'Condition':<30} {'N_call':>7} {'A_turn%':>8} "
        f"{'R_max':>7} {'R_avg':>7} {'ΔD%':>6} {'MinDCPA':>8}"
    )
    print(header)
    print("-" * 100)
    for r in summary_rows:
        ncall = f"{r['N_call_mean']:.1f}" if r["llm_enabled"] else "—"
        print(
            f"{r['condition_label']:<30} "
            f"{ncall:>7} "
            f"{r['A_turn_pct_mean']:>7.1f}% "
            f"{r['R_max_mean']:>7.3f} "
            f"{r['R_avg_mean']:>7.3f} "
            f"{r['delta_D_pct_mean']:>5.2f}% "
            f"{r['min_dcpa_nm_mean']:>8.3f}"
        )
    print("=" * 100)

    # 额外打印论文修订建议中的关键数字
    wo_rt = next((r for r in summary_rows if r["condition_name"] == "wo_risk_trigger"), None)
    full = next((r for r in summary_rows if r["condition_name"] == "full"), None)
    if wo_rt and full:
        wo_ncall = wo_rt["N_call_mean"]
        full_ncall = full["N_call_mean"]
        if wo_ncall > 0:
            reduction = (wo_ncall - full_ncall) / wo_ncall * 100
            print(f"\n  [实验注] w/o Risk Trigger N_call={wo_ncall:.1f} vs Full N_call={full_ncall:.1f}")
            print(f"  -> 风险触发机制减少调用 {reduction:.1f}%")


# ── 命令行 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTRA-LLM 模块消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--provider", type=str, required=True,
                   help="LLM 提供商（deepseek / zhipu / qwen / openai …）")
    p.add_argument("--cases", type=int, nargs="+", default=None,
                   help="场景编号列表，默认全部23个")
    p.add_argument("--skip_conditions", type=int, nargs="+", default=None,
                   help="跳过的条件 ID（0=baseline,1=full,2=wo_risk_trigger,"
                        "3=wo_memory,4=wo_validator,5=wo_memory_wo_validator）")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--sim_time", type=float, default=SIMULATION_TIME_S)
    p.add_argument("--dt", type=float, default=SIMULATION_DT_S)
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="动力学随机扰动种子")
    p.add_argument("--risk_threshold", type=float, default=LLM_RISK_THRESHOLD)
    p.add_argument("--smoke", action="store_true",
                   help="冒烟测试：仅 case 1，sim_time=30s")
    return p.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = parse_args()

    if cli.smoke:
        cli.cases = [1]
        cli.sim_time = 30.0
        print("[SMOKE TEST] Running case 1 only, sim_time=30s")

    from src.utils.imazu_cases import get_case_numbers
    cases = cli.cases if cli.cases is not None else get_case_numbers()

    skip_ids = set(cli.skip_conditions or [])
    conditions = [c for c in ABLATION_CONDITIONS if c["id"] not in skip_ids]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cli.output_dir:
        out_dir = Path(cli.output_dir)
    else:
        out_dir = Path("output/experiments") / f"module_ablation_{cli.provider}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = SimpleNamespace(
        llm_provider=cli.provider,
        llm_risk_threshold=cli.risk_threshold,
        llm_fixed_interval=250,
        sim_time=cli.sim_time,
        dt=cli.dt,
        seed=cli.seed,
        output_dir=str(out_dir),
    )

    # baseline 仅跑一次，所有条件复用
    baseline_cache: Dict[int, Dict] = {}

    # condition 0（baseline）不需要额外跑，直接从缓存取——
    # 但触发缓存填充需在第一个实际case时发生；
    # 我们在 run_one 内部处理这个逻辑。

    total_llm_runs = sum(1 for c in conditions if c["llm"] == 1) * len(cases)
    print("\n" + "=" * 70)
    print("RTRA-LLM MODULE ABLATION")
    print("=" * 70)
    print(f"Provider       : {cli.provider}")
    print(f"Cases          : {cases}")
    print(f"Conditions     : {[c['name'] for c in conditions]}")
    print(f"Sim time       : {cli.sim_time}s  |  dt={cli.dt}s")
    print(f"LLM runs       : {total_llm_runs}")
    print(f"Output dir     : {out_dir}")
    print("=" * 70)

    all_rows: List[Dict[str, Any]] = []
    failed_runs: List[tuple] = []
    run_idx = 0
    total_runs = len(conditions) * len(cases)

    for condition in conditions:
        print(f"\n{'─'*60}")
        print(f"[Condition {condition['id']}] {condition['label']}")
        print(f"{'─'*60}")
        cond_rows: List[Dict[str, Any]] = []

        for case_number in cases:
            run_idx += 1
            progress = f"[{run_idx:3d}/{total_runs}]"
            print(f"{progress} Case {case_number:2d} | {condition['name']} ...", end=" ", flush=True)

            row = run_one(condition, case_number, base_args, baseline_cache)
            if row is not None:
                all_rows.append(row)
                cond_rows.append(row)
                print(
                    f"N_call={row['N_call']:3d}  "
                    f"A_turn={row['A_turn_pct']:.1f}%  "
                    f"R_max={row['R_max']:.3f}  "
                    f"ΔD={row['delta_D_pct']:+.2f}%"
                )
            else:
                failed_runs.append((condition["name"], case_number))
                print("FAILED")

        _write_csv(cond_rows, out_dir / f"module_ablation_{condition['name']}.csv")

    raw_path = out_dir / "module_ablation_all_raw.csv"
    summary_path = out_dir / "module_ablation_summary.csv"
    json_path = out_dir / "module_ablation_results.json"

    _write_csv(all_rows, raw_path)
    summary_rows = _build_summary(all_rows)
    _write_csv(summary_rows, summary_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": ts, "provider": cli.provider,
                    "cases": cases, "sim_time": cli.sim_time,
                    "low_level_planner": PAPER_LOW_LEVEL_PLANNER,
                    "risk_threshold": cli.risk_threshold,
                    "conditions": [c["name"] for c in conditions],
                    "failed_runs": [{"condition": c, "case": n} for c, n in failed_runs],
                },
                "raw": all_rows,
                "summary": summary_rows,
            },
            f, indent=2, default=str,
        )
    print(f"\n  -> JSON saved to {json_path}")

    _print_summary(summary_rows, cli.provider)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"  Successful     : {len(all_rows)}")
    print(f"  Failed         : {len(failed_runs)}")
    if failed_runs:
        for c, n in failed_runs:
            print(f"    - condition={c}, case={n}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
