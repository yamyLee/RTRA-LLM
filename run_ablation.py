"""
RTRA-LLM 自动化消融实验脚本（历史版本）
=========================
运行 5 个消融条件 × 23 个 Imazu 场景，并将结果汇总为 CSV。

注意：
  本脚本为较早的通用消融入口。当前建议优先使用
  `run_module_ablation.py` 与 `scripts/run_module_ablation.sh`。

消融条件
--------
  0. baseline          : 无 LLM（论文规定的 VO 底层规划器）
  1. full              : LLM + memory + rule_validator（完整系统）
  2. no_memory         : LLM，关闭 memory
  3. no_validator      : LLM，关闭 rule_validator
  4. no_memory_no_val  : LLM，同时关闭 memory 与 rule_validator

用法
----
  # 快速冒烟测试（仅 case 1）
  python run_ablation.py --smoke

  # 完整实验
  python run_ablation.py --provider deepseek

  # 指定场景子集
  python run_ablation.py --provider deepseek --cases 1 2 7 8

  # 跳过某些消融条件（0=baseline,1=full,2=no_memory,3=no_val,4=no_mem_no_val）
  python run_ablation.py --provider deepseek --skip_conditions 2 3

  # 输出目录
  python run_ablation.py --provider deepseek --output_dir output/my_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import os
import traceback
from copy import copy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
from src.config.paper_parameters import LLM_RISK_THRESHOLD, PAPER_LOW_LEVEL_PLANNER, RANDOM_SEED, SIMULATION_DT_S, SIMULATION_TIME_S
from src.core.experiment_metrics import count_turn_events

# ── 将项目根目录加到路径，使 src.* 可以直接导入 ──────────────────────────────
_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))

# ── 自动处理常见环境问题 ──────────────────────────────────────────────────────
# 1. 确保 matplotlib 有可写缓存目录
_mpl_dir = os.environ.get("MPLCONFIGDIR")
if not _mpl_dir:
    _mpl_fallback = Path("/tmp/claude/matplotlib_cache")
    _mpl_fallback.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_fallback)

# 2. 自动安装 socksio（若系统设置了 SOCKS 代理但缺少该包）
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

# ── 消融条件定义 ─────────────────────────────────────────────────────────────
ABLATION_CONDITIONS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "baseline",
        "label": "Baseline (no LLM)",
        "llm": 0,
        "disable_memory": False,
        "disable_rule_validator": False,
    },
    {
        "id": 1,
        "name": "full",
        "label": "Full (LLM + Memory + Validator)",
        "llm": 1,
        "disable_memory": False,
        "disable_rule_validator": False,
    },
    {
        "id": 2,
        "name": "no_memory",
        "label": "No Memory",
        "llm": 1,
        "disable_memory": True,
        "disable_rule_validator": False,
    },
    {
        "id": 3,
        "name": "no_validator",
        "label": "No Rule Validator",
        "llm": 1,
        "disable_memory": False,
        "disable_rule_validator": True,
    },
    {
        "id": 4,
        "name": "no_memory_no_validator",
        "label": "No Memory & No Validator",
        "llm": 1,
        "disable_memory": True,
        "disable_rule_validator": True,
    },
]

# ── 统计指标提取 ──────────────────────────────────────────────────────────────

def _finite_mean(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _extract_metrics(results: Dict[str, Any], case_number: int, condition: Dict[str, Any],
                     provider: str) -> Dict[str, Any]:
    """从 run_simulation(return_data=True) 的返回值中提取标量指标。"""
    risk = results["risk"]           # shape (N, n_obstacles)
    kdir = results["kdir"]           # shape (N,)
    x_arr = np.asarray(results["x"])
    y_arr = np.asarray(results["y"])

    final_distance_nm = float(np.sqrt(x_arr[-1] ** 2 + y_arr[-1] ** 2) / 1852)

    # 转向次数：Kdir 从非 0 变为不同方向的次数（包括 ±1）
    total_turns = count_turn_events(kdir)
    starboard_turns = int(np.sum(kdir > 0.5))
    port_turns = int(np.sum(kdir < -0.5))

    max_risk = float(np.max(risk))
    avg_risk = _finite_mean(risk)
    positive_mask = risk > 0
    avg_positive_risk = _finite_mean(risk[positive_mask]) if positive_mask.any() else 0.0
    min_dcpa_nm = float(np.min(results["dcpa"]) / 1852)

    llm_calls = int(results.get("llm_call_count", 0))
    trigger_events = int(len(results.get("llm_trigger_history", [])))

    return {
        "condition_id": condition["id"],
        "condition_name": condition["name"],
        "condition_label": condition["label"],
        "case_number": case_number,
        "llm_provider": provider if condition["llm"] == 1 else "none",
        "llm_enabled": condition["llm"] == 1,
        "memory_enabled": not condition["disable_memory"],
        "validator_enabled": not condition["disable_rule_validator"],
        "trigger_mode": results.get("llm_trigger_mode", "risk"),
        "risk_threshold": results.get("llm_risk_threshold", LLM_RISK_THRESHOLD),
        # ── 安全指标 ──
        "max_risk": max_risk,
        "avg_risk": avg_risk,
        "avg_positive_risk": avg_positive_risk,
        "min_dcpa_nm": min_dcpa_nm,
        # ── 机动指标 ──
        "total_turns": total_turns,
        "starboard_turns": starboard_turns,
        "port_turns": port_turns,
        # ── 效率指标 ──
        "final_distance_nm": final_distance_nm,
        # ── LLM 调用统计 ──
        "llm_calls": llm_calls,
        "trigger_events": trigger_events,
    }


# ── 单个 (condition, case) 运行 ───────────────────────────────────────────────

def run_one(condition: Dict[str, Any], case_number: int,
            base_args: SimpleNamespace) -> Optional[Dict[str, Any]]:
    """运行一组 (消融条件 × case)，出错时返回 None。"""
    from src.core.simulation import run_simulation, load_env_file

    load_env_file()

    args = copy(base_args)
    args.case_number = case_number
    args.all_cases = False
    args.compare = False
    args.no_animation = True
    args.llm = condition["llm"]
    args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
    args.disable_memory = condition["disable_memory"]
    args.disable_rule_validator = condition["disable_rule_validator"]

    try:
        results = run_simulation(args, return_data=True)
        return _extract_metrics(results, case_number, condition, base_args.llm_provider or "")
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] condition={condition['name']} case={case_number}: {exc}")
        traceback.print_exc()
        return None


# ── CSV 写入 ──────────────────────────────────────────────────────────────────

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


# ── 汇总统计表 ────────────────────────────────────────────────────────────────

NUMERIC_METRICS = [
    "max_risk", "avg_risk", "avg_positive_risk", "min_dcpa_nm",
    "total_turns", "starboard_turns", "port_turns",
    "final_distance_nm", "llm_calls", "trigger_events",
]


def _build_summary_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按消融条件聚合，计算各指标的均值与标准差（跨 23 个 case）。"""
    from collections import defaultdict

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        groups[row["condition_name"]].append(row)

    summary_rows = []
    for cond_name, group in groups.items():
        row: Dict[str, Any] = {
            "condition_name": cond_name,
            "condition_label": group[0]["condition_label"],
            "llm_provider": group[0]["llm_provider"],
            "llm_enabled": group[0]["llm_enabled"],
            "memory_enabled": group[0]["memory_enabled"],
            "validator_enabled": group[0]["validator_enabled"],
            "n_cases": len(group),
        }
        for metric in NUMERIC_METRICS:
            vals = [g[metric] for g in group if g.get(metric) is not None]
            arr = np.array(vals, dtype=float)
            row[f"{metric}_mean"] = float(np.mean(arr)) if arr.size > 0 else float("nan")
            row[f"{metric}_std"] = float(np.std(arr)) if arr.size > 0 else float("nan")
        summary_rows.append(row)

    # 按 condition_id 排序
    id_map = {c["name"]: c["id"] for c in ABLATION_CONDITIONS}
    summary_rows.sort(key=lambda r: id_map.get(r["condition_name"], 99))
    return summary_rows


# ── 打印对比摘要 ──────────────────────────────────────────────────────────────

def _print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY  (mean ± std across cases)")
    print("=" * 80)
    header = f"{'Condition':<30} {'MaxRisk':>8} {'AvgRisk':>8} {'MinDCPA':>8} {'Turns':>7} {'LLMCalls':>9}"
    print(header)
    print("-" * 80)
    for r in summary_rows:
        print(
            f"{r['condition_label']:<30} "
            f"{r['max_risk_mean']:>7.3f} "
            f"{r['avg_risk_mean']:>8.3f} "
            f"{r['min_dcpa_nm_mean']:>8.3f} "
            f"{r['total_turns_mean']:>7.1f} "
            f"{r['llm_calls_mean']:>9.1f}"
        )
    print("=" * 80)


# ── 命令行解析 ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTRA-LLM 消融实验自动化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--provider", type=str, default=None,
                        help="LLM 提供商名称 (deepseek / zhipu / qwen / openai …)")
    parser.add_argument("--cases", type=int, nargs="+", default=None,
                        help="要运行的场景编号列表，默认全部 23 个")
    parser.add_argument("--skip_conditions", type=int, nargs="+", default=None,
                        help="跳过的消融条件 ID 列表 (0=baseline,1=full,2=no_memory,3=no_val,4=no_mem_no_val)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认自动生成含时间戳的目录）")
    parser.add_argument("--sim_time", type=float, default=SIMULATION_TIME_S,
                        help="每次仿真时长（秒），默认 450")
    parser.add_argument("--dt", type=float, default=SIMULATION_DT_S,
                        help="仿真时间步长（秒），默认 0.1")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="动力学随机扰动种子")
    parser.add_argument("--trigger_mode", choices=["risk", "fixed", "always"], default="risk",
                        help="LLM 触发模式，默认 risk")
    parser.add_argument("--risk_threshold", type=float, default=LLM_RISK_THRESHOLD,
                        help="LLM 触发风险阈值，默认 0.30")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟测试模式：仅运行 case 1，sim_time=30s")
    return parser.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = parse_args()

    # ── 冒烟测试覆盖 ──
    if cli.smoke:
        cli.cases = [1]
        cli.sim_time = 30.0
        print("[SMOKE TEST] Running case 1 only, sim_time=30s")

    # ── 确定场景列表 ──
    from src.utils.imazu_cases import get_case_numbers
    all_case_numbers = get_case_numbers()
    cases = cli.cases if cli.cases is not None else all_case_numbers

    # ── 确定消融条件 ──
    skip_ids = set(cli.skip_conditions or [])
    conditions = [c for c in ABLATION_CONDITIONS if c["id"] not in skip_ids]

    # ── 确定输出目录 ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_tag = cli.provider or "default"
    if cli.output_dir:
        out_dir = Path(cli.output_dir)
    else:
        out_dir = Path("output/experiments") / f"ablation_{provider_tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 构建基础 args ──
    base_args = SimpleNamespace(
        llm_provider=cli.provider,
        llm_trigger_mode=cli.trigger_mode,
        llm_risk_threshold=cli.risk_threshold,
        llm_fixed_interval=250,
        sim_time=cli.sim_time,
        dt=cli.dt,
        seed=cli.seed,
        output_dir=str(out_dir),
    )

    # ── 打印实验计划 ──
    total_runs = len(conditions) * len(cases)
    print("\n" + "=" * 70)
    print("RTRA-LLM ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"Provider       : {provider_tag}")
    print(f"Cases          : {cases}")
    print(f"Conditions     : {[c['name'] for c in conditions]}")
    print(f"Sim time       : {cli.sim_time}s  |  dt={cli.dt}s")
    print(f"Total runs     : {total_runs}")
    print(f"Output dir     : {out_dir}")
    print("=" * 70)

    # ── 主实验循环 ──
    all_rows: List[Dict[str, Any]] = []
    failed_runs: List[tuple] = []
    run_idx = 0

    for condition in conditions:
        print(f"\n{'─'*60}")
        print(f"[Condition {condition['id']}/{len(conditions)-1}] {condition['label']}")
        print(f"{'─'*60}")
        cond_rows: List[Dict[str, Any]] = []

        for case_number in cases:
            run_idx += 1
            progress = f"[{run_idx:3d}/{total_runs}]"
            print(f"{progress} Case {case_number:2d} | {condition['name']} ...", end=" ", flush=True)

            row = run_one(condition, case_number, base_args)
            if row is not None:
                all_rows.append(row)
                cond_rows.append(row)
                # 简洁打印关键指标
                print(
                    f"max_risk={row['max_risk']:.3f}  "
                    f"avg_risk={row['avg_risk']:.3f}  "
                    f"turns={row['total_turns']}  "
                    f"llm_calls={row['llm_calls']}"
                )
            else:
                failed_runs.append((condition["name"], case_number))
                print("FAILED")

        _write_csv(
            cond_rows,
            out_dir / f"ablation_{condition['name']}.csv",
        )

    # ── 保存完整原始数据 ──
    raw_csv_path = out_dir / "ablation_all_raw.csv"
    _write_csv(all_rows, raw_csv_path)

    # ── 汇总统计表 ──
    summary_rows = _build_summary_table(all_rows)
    summary_csv_path = out_dir / "ablation_summary.csv"
    _write_csv(summary_rows, summary_csv_path)

    # ── 保存 JSON（方便后续程序读取） ──
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": ts,
                    "provider": provider_tag,
                    "cases": cases,
                    "conditions": [c["name"] for c in conditions],
                    "sim_time": cli.sim_time,
                    "low_level_planner": PAPER_LOW_LEVEL_PLANNER,
                    "trigger_mode": cli.trigger_mode,
                    "risk_threshold": cli.risk_threshold,
                    "failed_runs": [{"condition": c, "case": n} for c, n in failed_runs],
                },
                "raw": all_rows,
                "summary": summary_rows,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  -> JSON saved to {json_path}")

    # ── 打印对比摘要 ──
    _print_summary(summary_rows)

    # ── 最终报告 ──
    print(f"\n{'='*70}")
    print("DONE")
    print(f"  Total runs     : {total_runs}")
    print(f"  Successful     : {len(all_rows)}")
    print(f"  Failed         : {len(failed_runs)}")
    if failed_runs:
        for cond_name, case_n in failed_runs:
            print(f"    - condition={cond_name}, case={case_n}")
    print(f"  Output dir     : {out_dir}")
    print(f"    raw CSV      : {raw_csv_path.name}")
    print(f"    summary CSV  : {summary_csv_path.name}")
    print(f"    JSON         : {json_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
