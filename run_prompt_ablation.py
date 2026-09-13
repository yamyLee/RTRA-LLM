"""
RTRA-LLM 提示词消融实验脚本
============================
在23个Imazu标准场景中对比不同提示词变体的避碰性能。

提示词变体（对应 prompt/ 目录下的文件）
-----------------------------------------
  none     : prompt_none.txt    ── 无COLREGs领域知识，仅输出格式约束
  weak     : prompt_weak.txt    ── 弱海事引导，无明确COLREG规则框架
  partial  : prompt_partial.txt ── 部分COLREGs引导，含会遇类型感知
  full     : prompt_full.txt    ── 完整COLREGs规则感知提示词（论文方法）

注意事项
--------
  - 脚本通过临时复制对应 .txt 到 prompt/prompt.txt 实现切换，实验结束后恢复原文件
  - 每次切换前自动备份当前 prompt/prompt.txt
  - 所有条件均使用完整配置（LLM + Memory + Validator，trigger_mode=risk）
  - 为避免信息泄漏，提示词消融时会屏蔽额外的显式会遇类型提示，仅保留原始态势要素与记忆上下文

用法
----
  # 完整提示词消融（deepseek）
  python run_prompt_ablation.py --provider deepseek

  # 冒烟测试
  python run_prompt_ablation.py --provider deepseek --smoke

  # 只测部分变体
  python run_prompt_ablation.py --provider deepseek --variants none full

  # 指定场景子集
  python run_prompt_ablation.py --provider deepseek --cases 1 2 7 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import traceback
from contextlib import contextmanager
from copy import copy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
from src.config.paper_parameters import LLM_RISK_THRESHOLD, PAPER_LOW_LEVEL_PLANNER, PAPER_RANDOM_SEEDS, RANDOM_SEED, SIMULATION_DT_S, SIMULATION_TIME_S
from src.core.experiment_metrics import count_turn_events
from src.core.paper_experiment_utils import add_scene_summary

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

# ── 提示词变体定义 ─────────────────────────────────────────────────────────────

PROMPT_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "none",
        "label": "No COLREGs (None)",
        "file": "prompt_none.txt",
        "paper_row": "无COLREGs知识",
    },
    {
        "id": 1,
        "name": "weak",
        "label": "Weak Guidance",
        "file": "prompt_weak.txt",
        "paper_row": "弱海事引导",
    },
    {
        "id": 2,
        "name": "partial",
        "label": "Partial COLREGs",
        "file": "prompt_partial.txt",
        "paper_row": "部分COLREGs引导",
    },
    {
        "id": 3,
        "name": "full",
        "label": "Full COLREGs (Ours)",
        "file": "prompt_full.txt",
        "paper_row": "完整COLREGs规则感知（论文方法）",
    },
]

PROMPT_DIR = _ROOT / "prompt"
ACTIVE_PROMPT = PROMPT_DIR / "prompt.txt"

METERS_TO_NMI = 1 / 1852


# ── 提示词切换上下文管理器 ────────────────────────────────────────────────────

@contextmanager
def _use_prompt(variant: Dict[str, Any]):
    """临时将指定变体文件复制为 prompt/prompt.txt，退出时恢复原文件。"""
    variant_file = PROMPT_DIR / variant["file"]
    if not variant_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {variant_file}\n"
            f"Available files: {list(PROMPT_DIR.glob('*.txt'))}"
        )

    # 备份
    backup_file: Optional[Path] = None
    if ACTIVE_PROMPT.exists():
        backup_file = PROMPT_DIR / f"_backup_prompt_{os.getpid()}.txt"
        shutil.copy2(ACTIVE_PROMPT, backup_file)
    previous_variant = os.environ.get("RTRA_LLM_PROMPT_ABLATION_VARIANT")

    try:
        shutil.copy2(variant_file, ACTIVE_PROMPT)
        os.environ["RTRA_LLM_PROMPT_ABLATION_VARIANT"] = variant["name"]
        print(f"  [prompt] Activated: {variant['file']}")
        yield
    finally:
        if previous_variant is None:
            os.environ.pop("RTRA_LLM_PROMPT_ABLATION_VARIANT", None)
        else:
            os.environ["RTRA_LLM_PROMPT_ABLATION_VARIANT"] = previous_variant
        if backup_file is not None and backup_file.exists():
            shutil.copy2(backup_file, ACTIVE_PROMPT)
            backup_file.unlink(missing_ok=True)
            print(f"  [prompt] Restored: prompt.txt")
        elif not ACTIVE_PROMPT.exists() and variant_file.exists():
            # 如果原来没有 prompt.txt，删除我们创建的
            ACTIVE_PROMPT.unlink(missing_ok=True)


# ── 指标提取 ──────────────────────────────────────────────────────────────────

def _finite_mean(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def _compute_aturn(llm_kdir: np.ndarray, baseline_kdir: np.ndarray) -> float:
    total = len(llm_kdir)
    if total == 0:
        return 100.0
    agree = int(np.sum(np.sign(llm_kdir) == np.sign(baseline_kdir)))
    return round(agree / total * 100, 2)


def _compute_delta_d(llm_dist_m: float, baseline_dist_m: float) -> float:
    if abs(baseline_dist_m) < 1e-6:
        return 0.0
    return round((llm_dist_m - baseline_dist_m) / baseline_dist_m * 100, 3)


def _extract_metrics(
    results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    case_number: int,
    variant: Dict[str, Any],
    provider: str,
) -> Dict[str, Any]:
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
        "variant_id": variant["id"],
        "variant_name": variant["name"],
        "variant_label": variant["label"],
        "paper_row": variant["paper_row"],
        "prompt_file": variant["file"],
        "case_number": case_number,
        "seed": int(results.get("random_seed", RANDOM_SEED)),
        "llm_provider": provider,
        "model": results.get("llm_model", ""),
        # ── 核心指标 ──
        "N_call": int(results.get("llm_call_count", 0)),
        "A_turn_pct": _compute_aturn(llm_kdir, baseline_kdir),
        "R_max": float(np.max(risk)),
        "R_avg": _finite_mean(risk),
        "R_avg_positive": _finite_mean(risk[risk > 0]) if (risk > 0).any() else 0.0,
        "min_dcpa_nm": float(np.min(np.abs(results["dcpa"])) * METERS_TO_NMI),
        "delta_D_pct": _compute_delta_d(llm_dist_m, baseline_dist_m),
        "final_dist_nm": llm_dist_m * METERS_TO_NMI,
        "total_turns": count_turn_events(llm_kdir),
        "trigger_events": int(len(results.get("llm_trigger_history", []))),
    }


# ── 单次运行 ──────────────────────────────────────────────────────────────────

def run_one(
    variant: Dict[str, Any],
    case_number: int,
    base_args: SimpleNamespace,
    baseline_cache: Dict[tuple, Dict],
) -> Optional[Dict[str, Any]]:
    from src.core.simulation import run_simulation, load_env_file

    load_env_file()

    # ── 缓存 baseline ──
    cache_key = (case_number, int(base_args.seed))
    if cache_key not in baseline_cache:
        b_args = copy(base_args)
        b_args.case_number = case_number
        b_args.llm = 0
        b_args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
        b_args.all_cases = False
        b_args.compare = False
        b_args.no_animation = True
        b_args.disable_memory = False
        b_args.disable_rule_validator = False
        b_args.llm_trigger_mode = "risk"
        try:
            baseline_cache[cache_key] = run_simulation(b_args, return_data=True)
        except Exception as exc:
            print(f"  [ERROR] baseline case={case_number}: {exc}")
            traceback.print_exc()
            return None

    baseline_res = baseline_cache[cache_key]

    args = copy(base_args)
    args.case_number = case_number
    args.llm = 1
    args.low_level_planner = PAPER_LOW_LEVEL_PLANNER
    args.llm_trigger_mode = "risk"
    args.disable_memory = False
    args.disable_rule_validator = False
    args.all_cases = False
    args.compare = False
    args.no_animation = True

    try:
        with _use_prompt(variant):
            results = run_simulation(args, return_data=True)
        return _extract_metrics(results, baseline_res, case_number, variant,
                                base_args.llm_provider or "")
    except Exception as exc:
        print(f"  [ERROR] variant={variant['name']} case={case_number}: {exc}")
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
    "N_call", "A_turn_pct", "R_max", "R_avg", "R_avg_positive",
    "min_dcpa_nm", "delta_D_pct", "final_dist_nm", "total_turns", "trigger_events",
]


def _build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    groups: Dict[str, List] = defaultdict(list)
    for r in rows:
        groups[r["variant_name"]].append(r)

    id_map = {v["name"]: v["id"] for v in PROMPT_VARIANTS}
    summary = []
    for vname, group in groups.items():
        row: Dict[str, Any] = {
            "variant_name": vname,
            "variant_label": group[0]["variant_label"],
            "paper_row": group[0]["paper_row"],
            "prompt_file": group[0]["prompt_file"],
            "llm_provider": group[0]["llm_provider"],
            "model": group[0]["model"],
            "n_cases": len({g["case_number"] for g in group}),
            "n_seeds": len({g["seed"] for g in group}),
        }
        add_scene_summary(row, group, NUMERIC_METRICS)
        summary.append(row)

    summary.sort(key=lambda r: id_map.get(r["variant_name"], 99))
    return summary


def _print_summary(summary_rows: List[Dict[str, Any]], provider: str) -> None:
    print("\n" + "=" * 100)
    print(f"PROMPT ABLATION SUMMARY  [{provider}]  (mean ± std across cases)")
    print("=" * 100)
    header = (
        f"{'Variant':<28} {'N_call':>7} {'A_turn%':>8} "
        f"{'R_max':>7} {'R_avg':>7} {'ΔD%':>6} {'MinDCPA':>8}"
    )
    print(header)
    print("-" * 100)
    for r in summary_rows:
        print(
            f"{r['variant_label']:<28} "
            f"{r['N_call_mean']:>7.1f} "
            f"{r['A_turn_pct_mean']:>7.1f}% "
            f"{r['R_max_mean']:>7.3f} "
            f"{r['R_avg_mean']:>7.3f} "
            f"{r['delta_D_pct_mean']:>5.2f}% "
            f"{r['min_dcpa_nm_mean']:>8.3f}"
        )
    print("=" * 100)


# ── 命令行 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTRA-LLM 提示词消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--provider", type=str, required=True,
                   help="LLM 提供商")
    p.add_argument("--variants", type=str, nargs="+", default=None,
                   choices=[v["name"] for v in PROMPT_VARIANTS],
                   help="要测试的提示词变体（默认全部4种）")
    p.add_argument("--cases", type=int, nargs="+", default=None,
                   help="场景编号，默认全部23个")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--sim_time", type=float, default=SIMULATION_TIME_S)
    p.add_argument("--dt", type=float, default=SIMULATION_DT_S)
    p.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_RANDOM_SEEDS), help="动力学随机扰动种子")
    p.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--risk_threshold", type=float, default=LLM_RISK_THRESHOLD)
    p.add_argument("--smoke", action="store_true",
                   help="冒烟测试：仅 case 1，sim_time=30s")
    return p.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = parse_args()
    seeds = [cli.seed] if cli.seed is not None else list(cli.seeds)

    if cli.smoke:
        cli.cases = [1]
        cli.sim_time = 30.0
        seeds = seeds[:1]
        print("[SMOKE TEST] Running case 1 only, sim_time=30s")

    from src.utils.imazu_cases import get_case_numbers
    cases = cli.cases if cli.cases is not None else get_case_numbers()

    variant_names = set(cli.variants) if cli.variants else {v["name"] for v in PROMPT_VARIANTS}
    variants = [v for v in PROMPT_VARIANTS if v["name"] in variant_names]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cli.output_dir:
        out_dir = Path(cli.output_dir)
    else:
        out_dir = Path("output/experiments") / f"prompt_ablation_{cli.provider}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = SimpleNamespace(
        llm_provider=cli.provider,
        llm_risk_threshold=cli.risk_threshold,
        llm_fixed_interval=250,
        sim_time=cli.sim_time,
        dt=cli.dt,
        seed=seeds[0],
        output_dir=str(out_dir),
    )

    # 检查所有 prompt 文件存在
    for v in variants:
        pf = PROMPT_DIR / v["file"]
        if not pf.exists():
            print(f"[WARN] Prompt file missing: {pf}  -- variant '{v['name']}' will be skipped")
    variants = [v for v in variants if (PROMPT_DIR / v["file"]).exists()]

    if not variants:
        print("[ERROR] No valid prompt variants found. Check prompt/ directory.")
        sys.exit(1)

    total_runs = len(variants) * len(cases) * len(seeds)
    print("\n" + "=" * 70)
    print("RTRA-LLM PROMPT ABLATION")
    print("=" * 70)
    print(f"Provider       : {cli.provider}")
    print(f"Variants       : {[v['name'] for v in variants]}")
    print(f"Cases          : {cases}")
    print(f"Seeds          : {seeds}")
    print(f"Sim time       : {cli.sim_time}s  |  dt={cli.dt}s")
    print(f"Total runs     : {total_runs}")
    print(f"Output dir     : {out_dir}")
    print("=" * 70)

    baseline_cache: Dict[tuple, Dict] = {}
    all_rows: List[Dict[str, Any]] = []
    failed_runs: List[tuple] = []
    run_idx = 0

    for variant in variants:
        print(f"\n{'─'*60}")
        print(f"[Variant] {variant['label']}  ({variant['file']})")
        print(f"{'─'*60}")
        var_rows: List[Dict[str, Any]] = []

        for case_number in cases:
            for seed in seeds:
                base_args.seed = seed
                run_idx += 1
                progress = f"[{run_idx:3d}/{total_runs}]"
                print(f"{progress} Case {case_number:2d} seed={seed} | {variant['name']} ...", end=" ", flush=True)

                row = run_one(variant, case_number, base_args, baseline_cache)
                if row is not None:
                    all_rows.append(row)
                    var_rows.append(row)
                    print(
                        f"N_call={row['N_call']:3d}  "
                        f"A_turn={row['A_turn_pct']:.1f}%  "
                        f"R_max={row['R_max']:.3f}  "
                        f"ΔD={row['delta_D_pct']:+.2f}%"
                    )
                else:
                    failed_runs.append((variant["name"], case_number, seed))
                    print("FAILED")

        _write_csv(var_rows, out_dir / f"prompt_ablation_{variant['name']}.csv")

    raw_path = out_dir / "prompt_ablation_all_raw.csv"
    summary_path = out_dir / "prompt_ablation_summary.csv"
    json_path = out_dir / "prompt_ablation_results.json"

    _write_csv(all_rows, raw_path)
    summary_rows = _build_summary(all_rows)
    _write_csv(summary_rows, summary_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "timestamp": ts, "provider": cli.provider,
                    "variants": [v["name"] for v in variants],
                    "cases": cases, "seeds": seeds, "sim_time": cli.sim_time,
                    "low_level_planner": PAPER_LOW_LEVEL_PLANNER,
                    "failed_runs": [{"variant": v, "case": n, "seed": s} for v, n, s in failed_runs],
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
        for v, n, seed in failed_runs:
            print(f"    - variant={v}, case={n}, seed={seed}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
