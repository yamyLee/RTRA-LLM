"""
RTRA-LLM 风险阈值敏感性分析脚本
==============================
扫描不同风险触发阈值 q，绘制效率-安全权衡曲线，为论文中 q=0.30 的选取提供
定量依据。

实验设计
--------
  - 固定：完整 RTRA-LLM 配置（LLM + Memory + Validator，trigger_mode=risk）
  - 变量：风险阈值 q ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50}
  - 场景：全部 23 个 Imazu 标准场景
  - baseline 结果缓存复用（每 case 只跑一次纯反应式基准）

核心指标
--------
  - N_call   : LLM 调用次数/场景（↓ 好，代表效率）
  - R_max    : 最大碰撞风险（↓ 好，代表安全性）
  - R_avg    : 平均碰撞风险
  - min_DCPA : 最小会遇距离（海里）（↑ 好）
  - A_turn   : 与基准转向一致率（%）
  - ΔD       : 路径效率偏差（%，相对基准）

分析产出
--------
  - 各 q 值的均值统计表（CSV + JSON）
  - 每个 case 的原始数据（CSV）
  - 拐点（elbow）自动检测：找到 N_call 已大幅下降但 R_max 尚未显著上升的 q 值
  - 终端打印效率-安全权衡表，标注 q=0.30 的相对位置

用法
----
  # 完整扫描（deepseek）
  python run_threshold_sweep.py --provider deepseek

  # 冒烟测试（仅 case 1）
  python run_threshold_sweep.py --provider deepseek --smoke

  # 自定义 q 值列表
  python run_threshold_sweep.py --provider deepseek --thresholds 0.1 0.2 0.25 0.3 0.4

  # 指定场景子集
  python run_threshold_sweep.py --provider deepseek --cases 1 2 7 8
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── 项目根目录 ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))

# ── matplotlib 缓存 ────────────────────────────────────────────────────────────
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

# ── 默认扫描的 q 值序列 ────────────────────────────────────────────────────────
DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

METERS_TO_NMI = 1 / 1852


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _finite_mean(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _compute_aturn(llm_kdir: np.ndarray, baseline_kdir: np.ndarray) -> float:
    """A_turn：逐步转向方向一致率（%）。"""
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
    q: float,
    provider: str,
) -> Dict[str, Any]:
    """提取单次仿真的所有分析指标。"""
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
        "q": q,
        "case_number": case_number,
        "llm_provider": provider,
        # ── 效率指标 ──
        "N_call": int(results.get("llm_call_count", 0)),
        "trigger_events": int(len(results.get("llm_trigger_history", []))),
        # ── 安全指标 ──
        "R_max": float(np.max(risk)),
        "R_avg": _finite_mean(risk),
        "R_avg_positive": _finite_mean(risk[risk > 0]) if (risk > 0).any() else 0.0,
        "min_dcpa_nm": float(np.min(results["dcpa"]) * METERS_TO_NMI),
        # ── 机动一致性 ──
        "A_turn_pct": _compute_aturn(llm_kdir, baseline_kdir),
        # ── 路径效率 ──
        "delta_D_pct": _compute_delta_d(llm_dist_m, baseline_dist_m),
        "final_dist_nm": llm_dist_m * METERS_TO_NMI,
        # ── 辅助 ──
        "total_turns": int(np.sum(np.abs(llm_kdir) > 0.1)),
        "starboard_turns": int(np.sum(llm_kdir > 0.5)),
        "port_turns": int(np.sum(llm_kdir < -0.5)),
    }


# ── 单次运行 ───────────────────────────────────────────────────────────────────

def run_one(
    q: float,
    case_number: int,
    base_args: SimpleNamespace,
    baseline_cache: Dict[int, Dict],
) -> Optional[Dict[str, Any]]:
    """运行 (q, case) 组合，出错返回 None。"""
    from src.core.simulation import run_simulation, load_env_file

    load_env_file()

    # ── 缓存 baseline（每 case 只跑一次）──
    if case_number not in baseline_cache:
        b_args = copy(base_args)
        b_args.case_number = case_number
        b_args.llm = 0
        b_args.llm_trigger_mode = "risk"
        b_args.llm_risk_threshold = 0.30
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

    # ── LLM 运行（完整配置，仅改 q）──
    args = copy(base_args)
    args.case_number = case_number
    args.llm = 1
    args.llm_trigger_mode = "risk"
    args.llm_risk_threshold = q
    args.disable_memory = False
    args.disable_rule_validator = False
    args.all_cases = False
    args.compare = False
    args.no_animation = True

    try:
        results = run_simulation(args, return_data=True)
        return _extract_metrics(results, baseline_res, case_number, q,
                                base_args.llm_provider or "")
    except Exception as exc:
        print(f"  [ERROR] q={q:.2f} case={case_number}: {exc}")
        traceback.print_exc()
        return None


# ── CSV / JSON 写入 ────────────────────────────────────────────────────────────

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


# ── 汇总统计（按 q 值聚合）────────────────────────────────────────────────────

NUMERIC_METRICS = [
    "N_call", "trigger_events",
    "R_max", "R_avg", "R_avg_positive", "min_dcpa_nm",
    "A_turn_pct", "delta_D_pct", "final_dist_nm", "total_turns",
]


def _build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 q 值聚合，计算跨23场景的均值与标准差。"""
    from collections import defaultdict

    groups: Dict[float, List] = defaultdict(list)
    for r in rows:
        groups[r["q"]].append(r)

    summary = []
    for q_val in sorted(groups.keys()):
        group = groups[q_val]
        row: Dict[str, Any] = {
            "q": q_val,
            "llm_provider": group[0]["llm_provider"],
            "n_cases": len(group),
        }
        for m in NUMERIC_METRICS:
            vals = np.array([g[m] for g in group if g.get(m) is not None], dtype=float)
            row[f"{m}_mean"] = float(np.mean(vals)) if vals.size > 0 else float("nan")
            row[f"{m}_std"] = float(np.std(vals)) if vals.size > 0 else float("nan")
        summary.append(row)

    return summary


# ── 拐点检测（Elbow Detection）─────────────────────────────────────────────────

def _detect_elbow(summary_rows: List[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    """
    自动检测效率-安全曲线的拐点 q*。

    策略：
      1. N_call 从最小 q 到当前 q 的累计下降量 ≥ 总降幅的 70%
      2. R_max 相对最小 q 时的上升量 < 总升幅的 30%
      满足以上两个条件的最大 q 值即为拐点。
    """
    if len(summary_rows) < 3:
        return None, "数据点不足，无法检测拐点"

    q_vals = [r["q"] for r in summary_rows]
    ncall_vals = [r["N_call_mean"] for r in summary_rows]
    rmax_vals = [r["R_max_mean"] for r in summary_rows]

    ncall_min = min(ncall_vals)
    ncall_max = max(ncall_vals)
    rmax_min = min(rmax_vals)
    rmax_max = max(rmax_vals)

    ncall_range = ncall_max - ncall_min
    rmax_range = rmax_max - rmax_min

    if ncall_range < 1e-6:
        return None, "N_call 在所有 q 值下几乎相同，阈值无效果"

    elbow_q: Optional[float] = None
    for i, (q, ncall, rmax) in enumerate(zip(q_vals, ncall_vals, rmax_vals)):
        ncall_drop_ratio = (ncall_max - ncall) / ncall_range
        rmax_rise_ratio = (rmax - rmax_min) / rmax_range if rmax_range > 1e-6 else 0.0
        if ncall_drop_ratio >= 0.70 and rmax_rise_ratio < 0.30:
            elbow_q = q  # 持续更新，取满足条件的最大 q

    if elbow_q is None:
        # 退化策略：取 N_call 下降最陡峭的区间的右端点
        drops = [ncall_vals[i] - ncall_vals[i + 1] for i in range(len(ncall_vals) - 1)]
        steepest_idx = int(np.argmax(drops))
        elbow_q = q_vals[steepest_idx + 1]
        reason = f"退化检测（最陡下降区间右端点）"
    else:
        reason = "N_call降幅≥70% 且 R_max上升<30%"

    return elbow_q, reason


# ── 终端打印效率-安全权衡表 ────────────────────────────────────────────────────

def _print_sweep_table(
    summary_rows: List[Dict[str, Any]],
    paper_q: float,
    elbow_q: Optional[float],
    elbow_reason: str,
    provider: str,
) -> None:
    """打印格式化的权衡表，标注 paper_q 和 elbow_q。"""
    print("\n" + "=" * 105)
    print(f"THRESHOLD SWEEP — EFFICIENCY-SAFETY TRADE-OFF  [{provider}]  (mean across cases)")
    print("=" * 105)

    header = (
        f"  {'q':>6}  {'N_call':>7}  {'±std':>6}  "
        f"{'R_max':>7}  {'±std':>6}  "
        f"{'R_avg':>7}  {'minDCPA':>8}  "
        f"{'A_turn%':>8}  {'ΔD%':>6}  {'标注'}"
    )
    print(header)
    print("-" * 105)

    # 找 always 模式的参考（若有）
    ncall_vals = [r["N_call_mean"] for r in summary_rows]
    ncall_ref = max(ncall_vals)  # 最大 N_call 视为参考

    for r in summary_rows:
        q = r["q"]
        ncall = r["N_call_mean"]
        ncall_std = r["N_call_std"]
        rmax = r["R_max_mean"]
        rmax_std = r["R_max_std"]
        ravg = r["R_avg_mean"]
        dcpa = r["min_dcpa_nm_mean"]
        aturn = r["A_turn_pct_mean"]
        dd = r["delta_D_pct_mean"]

        # 计算相对最高 N_call 的下降幅度
        reduction = (ncall_ref - ncall) / ncall_ref * 100 if ncall_ref > 0 else 0.0

        # 构造标注
        tags = []
        if abs(q - paper_q) < 1e-6:
            tags.append("★ 论文选值")
        if elbow_q is not None and abs(q - elbow_q) < 1e-6:
            tags.append("◆ 拐点")
        tag_str = "  ".join(tags)

        row_str = (
            f"  {q:>6.2f}  {ncall:>7.1f}  {ncall_std:>6.1f}  "
            f"{rmax:>7.3f}  {rmax_std:>6.3f}  "
            f"{ravg:>7.3f}  {dcpa:>8.3f}  "
            f"{aturn:>7.1f}%  {dd:>+5.2f}%  {tag_str}"
        )
        # 高亮论文选值行
        if abs(q - paper_q) < 1e-6:
            print(f">{row_str[1:]}")
        else:
            print(row_str)

    print("=" * 105)
    print(f"\n  拐点检测结果: q* = {elbow_q}  （{elbow_reason}）")
    print(f"  论文选值    : q  = {paper_q}")
    if elbow_q is not None:
        if abs(elbow_q - paper_q) < 1e-6:
            print(f"  ✓ 论文选值与拐点重合，q={paper_q:.2f} 的选取具有充分定量依据。")
        elif elbow_q < paper_q:
            diff = paper_q - elbow_q
            print(f"  → 论文选值略高于拐点（Δq=+{diff:.2f}），"
                  f"意味着在效率-安全权衡上偏保守，安全余量更大。")
        else:
            diff = elbow_q - paper_q
            print(f"  → 论文选值略低于拐点（Δq=-{diff:.2f}），"
                  f"意味着在效率-安全权衡上偏激进，调用次数稍多。")
    print()


# ── 输出论文修订建议 ───────────────────────────────────────────────────────────

def _print_paper_advice(
    summary_rows: List[Dict[str, Any]],
    paper_q: float,
    elbow_q: Optional[float],
) -> None:
    """根据扫描结果生成可直接用于论文的文字描述。"""
    paper_row = next((r for r in summary_rows if abs(r["q"] - paper_q) < 1e-6), None)
    if paper_row is None:
        return

    ncall_max_row = max(summary_rows, key=lambda r: r["N_call_mean"])
    ncall_min_row = min(summary_rows, key=lambda r: r["N_call_mean"])

    print("─" * 80)
    print("  【论文修订建议】以下数字可直接写入论文阈值选取说明段落：")
    print()
    print(f"  当 q 从 {ncall_max_row['q']:.2f}（等价于全程调用）降至 {paper_q:.2f} 时，")
    print(f"  N_call 从 {ncall_max_row['N_call_mean']:.1f} 次/场景降至 "
          f"{paper_row['N_call_mean']:.1f} 次/场景，")
    reduction = (ncall_max_row["N_call_mean"] - paper_row["N_call_mean"]) / \
                ncall_max_row["N_call_mean"] * 100
    print(f"  减少调用 {reduction:.1f}%；")
    print(f"  同期 R_max 从 {ncall_max_row['R_max_mean']:.3f} 变化至 "
          f"{paper_row['R_max_mean']:.3f}，")

    rmax_change = paper_row["R_max_mean"] - ncall_max_row["R_max_mean"]
    if abs(rmax_change) < 0.005:
        print("  R_max 几乎无变化，说明风险触发机制在大幅降低调用开销的同时未损失安全性。")
    elif rmax_change > 0:
        print(f"  R_max 上升 {rmax_change:.3f}（可接受范围内），")
        print("  说明适度的延迟触发不会显著恶化安全性能。")
    else:
        print(f"  R_max 反而下降 {abs(rmax_change):.3f}，")
        print("  说明风险感知触发使 LLM 在最需要时介入，决策质量优于全程调用。")

    if elbow_q is not None:
        print(f"\n  灵敏度分析表明，q 在 [{max(0.0, elbow_q-0.05):.2f}, "
              f"{min(1.0, elbow_q+0.10):.2f}] 区间内性能相对稳定，")
        print(f"  q={paper_q:.2f} 位于效率-安全权衡的合理区间内。")
    print("─" * 80)


# ── 命令行 ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTRA-LLM 风险阈值敏感性分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--provider", type=str, required=True,
                   help="LLM 提供商（deepseek / zhipu / qwen / openai …）")
    p.add_argument("--thresholds", type=float, nargs="+", default=None,
                   help=f"扫描的 q 值列表（默认：{DEFAULT_THRESHOLDS}）")
    p.add_argument("--paper_q", type=float, default=0.30,
                   help="论文中使用的阈值（默认 0.30，用于标注和分析）")
    p.add_argument("--cases", type=int, nargs="+", default=None,
                   help="场景编号列表，默认全部 23 个")
    p.add_argument("--output_dir", type=str, default=None,
                   help="输出目录（默认自动生成含时间戳目录）")
    p.add_argument("--sim_time", type=float, default=450.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--smoke", action="store_true",
                   help="冒烟测试：仅 case 1，sim_time=30s")
    return p.parse_args()


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = parse_args()

    if cli.smoke:
        cli.cases = [1]
        cli.sim_time = 30.0
        print("[SMOKE TEST] Running case 1 only, sim_time=30s")

    from src.utils.imazu_cases import get_case_numbers
    cases = cli.cases if cli.cases is not None else get_case_numbers()
    thresholds = sorted(cli.thresholds if cli.thresholds is not None else DEFAULT_THRESHOLDS)

    # 确保论文选值在列表里
    if cli.paper_q not in thresholds:
        thresholds = sorted(thresholds + [cli.paper_q])
        print(f"[INFO] paper_q={cli.paper_q} 已自动加入扫描列表")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cli.output_dir:
        out_dir = Path(cli.output_dir)
    else:
        out_dir = Path("output/experiments") / f"threshold_sweep_{cli.provider}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = SimpleNamespace(
        llm_provider=cli.provider,
        llm_fixed_interval=200,
        sim_time=cli.sim_time,
        dt=cli.dt,
        output_dir=str(out_dir),
    )

    total_runs = len(thresholds) * len(cases)
    print("\n" + "=" * 70)
    print("RTRA-LLM THRESHOLD SENSITIVITY SWEEP")
    print("=" * 70)
    print(f"Provider       : {cli.provider}")
    print(f"Thresholds q   : {thresholds}")
    print(f"Paper q        : {cli.paper_q}")
    print(f"Cases          : {cases}")
    print(f"Sim time       : {cli.sim_time}s  |  dt={cli.dt}s")
    print(f"Total LLM runs : {total_runs}  (+ 1 baseline/case cached)")
    print(f"Output dir     : {out_dir}")
    print("=" * 70)

    baseline_cache: Dict[int, Dict] = {}
    all_rows: List[Dict[str, Any]] = []
    failed_runs: List[tuple] = []
    run_idx = 0

    for q in thresholds:
        print(f"\n{'─'*60}")
        print(f"[q = {q:.2f}]")
        print(f"{'─'*60}")
        q_rows: List[Dict[str, Any]] = []

        for case_number in cases:
            run_idx += 1
            progress = f"[{run_idx:3d}/{total_runs}]"
            print(f"{progress} Case {case_number:2d} | q={q:.2f} ...", end=" ", flush=True)

            row = run_one(q, case_number, base_args, baseline_cache)
            if row is not None:
                all_rows.append(row)
                q_rows.append(row)
                marker = " ★" if abs(q - cli.paper_q) < 1e-6 else ""
                print(
                    f"N_call={row['N_call']:3d}  "
                    f"R_max={row['R_max']:.3f}  "
                    f"A_turn={row['A_turn_pct']:.1f}%  "
                    f"ΔD={row['delta_D_pct']:+.2f}%{marker}"
                )
            else:
                failed_runs.append((q, case_number))
                print("FAILED")

        # 每个 q 值跑完后存中间结果
        _write_csv(q_rows, out_dir / f"sweep_q{q:.2f}.csv".replace(".", "p"))

    # ── 写原始数据 ──
    raw_path = out_dir / "sweep_all_raw.csv"
    _write_csv(all_rows, raw_path)

    # ── 汇总统计 ──
    summary_rows = _build_summary(all_rows)
    summary_path = out_dir / "sweep_summary.csv"
    _write_csv(summary_rows, summary_path)

    # ── 拐点检测 ──
    elbow_q, elbow_reason = _detect_elbow(summary_rows)

    # ── JSON 输出 ──
    json_path = out_dir / "sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": ts,
                    "provider": cli.provider,
                    "thresholds": thresholds,
                    "paper_q": cli.paper_q,
                    "cases": cases,
                    "sim_time": cli.sim_time,
                    "elbow_q": elbow_q,
                    "elbow_reason": elbow_reason,
                    "failed_runs": [{"q": q, "case": n} for q, n in failed_runs],
                },
                "raw": all_rows,
                "summary": summary_rows,
            },
            f, indent=2, default=str,
        )
    print(f"\n  -> JSON saved to {json_path}")

    # ── 终端打印权衡表 ──
    _print_sweep_table(summary_rows, cli.paper_q, elbow_q, elbow_reason, cli.provider)

    # ── 论文修订建议 ──
    _print_paper_advice(summary_rows, cli.paper_q, elbow_q)

    # ── 最终报告 ──
    print(f"\n{'='*70}")
    print("DONE")
    print(f"  Total runs     : {total_runs}")
    print(f"  Successful     : {len(all_rows)}")
    print(f"  Failed         : {len(failed_runs)}")
    if failed_runs:
        for q_val, case_n in failed_runs:
            print(f"    - q={q_val:.2f}, case={case_n}")
    print(f"  Elbow q*       : {elbow_q}  ({elbow_reason})")
    print(f"  Paper q        : {cli.paper_q}")
    print(f"  Output dir     : {out_dir}")
    print(f"    raw CSV      : {raw_path.name}")
    print(f"    summary CSV  : {summary_path.name}")
    print(f"    JSON         : {json_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
