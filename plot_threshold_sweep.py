from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def _read_summary_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if key == "llm_provider":
                    parsed[key] = value  # type: ignore[assignment]
                else:
                    parsed[key] = float(value) if value not in (None, "") else float("nan")
            rows.append(parsed)
    rows.sort(key=lambda r: float(r["q"]))
    return rows


def _read_meta(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("meta", {})


def _rank(values: np.ndarray, higher_is_better: bool = False) -> np.ndarray:
    order = np.argsort(-values if higher_is_better else values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def _recommend_q(rows: List[Dict[str, float]]) -> float:
    q = np.array([float(r["q"]) for r in rows])
    n_call = np.array([float(r["N_call_mean"]) for r in rows])
    r_max = np.array([float(r["R_max_mean"]) for r in rows])
    r_avg = np.array([float(r["R_avg_mean"]) for r in rows])
    a_turn = np.array([float(r["A_turn_pct_mean"]) for r in rows])
    delta_d = np.abs(np.array([float(r["delta_D_pct_mean"]) for r in rows]))

    score = (
        _rank(n_call)
        + _rank(r_max)
        + _rank(r_avg)
        + _rank(a_turn, higher_is_better=True)
        + _rank(delta_d)
    )
    return float(q[int(np.argmin(score))])


def _find_row(rows: List[Dict[str, float]], q: Optional[float]) -> Optional[Dict[str, float]]:
    if q is None:
        return None
    for row in rows:
        if abs(float(row["q"]) - q) < 1e-9:
            return row
    return None


def _annotate_point(ax: plt.Axes, row: Optional[Dict[str, float]], y_key: str, label: str, color: str) -> None:
    if row is None:
        return
    x = float(row["q"])
    y = float(row[y_key])
    ax.scatter([x], [y], color=color, s=48, zorder=5)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color=color,
        fontweight="bold",
    )


def _draw_reference_lines(ax: plt.Axes, best_q: Optional[float]) -> None:
    if best_q is not None:
        ax.axvline(best_q, color="#2ca02c", linestyle="-.", linewidth=1.5, alpha=0.85)


def _configure_publication_style() -> None:
    rcParams["font.family"] = ["STHeiti", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["font.sans-serif"] = ["STHeiti", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["font.size"] = 10
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 10.5
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 9


def build_figure(
    rows: List[Dict[str, float]],
    provider: str,
    best_q: float,
    lang: str = "en",
) -> plt.Figure:
    if lang == "cn":
        _configure_publication_style()
    else:
        rcParams["font.family"] = ["DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        rcParams["pdf.fonttype"] = 42
        rcParams["ps.fonttype"] = 42
        rcParams["font.size"] = 10
        rcParams["axes.labelsize"] = 10
        rcParams["axes.titlesize"] = 10.5
        rcParams["xtick.labelsize"] = 9
        rcParams["ytick.labelsize"] = 9
        rcParams["legend.fontsize"] = 9

    q = np.array([float(r["q"]) for r in rows])
    n_call = np.array([float(r["N_call_mean"]) for r in rows])
    r_max = np.array([float(r["R_max_mean"]) for r in rows])
    r_avg = np.array([float(r["R_avg_mean"]) for r in rows])
    a_turn = np.array([float(r["A_turn_pct_mean"]) for r in rows])
    delta_d_abs = np.abs(np.array([float(r["delta_D_pct_mean"]) for r in rows]))

    best_row = _find_row(rows, best_q)

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.94, wspace=0.22, hspace=0.32)

    colors = {
        "n_call": "#4C78A8",
        "r_max": "#E45756",
        "r_avg": "#F58518",
        "a_turn": "#54A24B",
        "delta_d": "#B279A2",
    }

    if lang == "cn":
        labels = {
            "a_title": "(a) 单场景平均调用次数",
            "b_title": "(b) 风险指标",
            "c_title": "(c) 转向一致率",
            "d_title": "(d) 路径效率偏差",
            "x": "风险触发阈值 q",
            "risk_y": "风险值",
            "top_title": f"风险触发阈值灵敏度分析（{provider.lower()}）",
            "top_subtitle": "绿色虚线表示最终选定阈值 q=0.30",
            "q_label": "q=0.30",
        }
    else:
        labels = {
            "a_title": "(a) Mean LLM Calls per Case",
            "b_title": "(b) Risk Metrics",
            "c_title": "(c) Turn Agreement",
            "d_title": "(d) Path Efficiency Deviation",
            "x": "Risk Threshold q",
            "risk_y": "Risk",
            "top_title": f"Sensitivity Analysis of Risk Threshold q ({provider.lower()})",
            "top_subtitle": "Green dash-dot line indicates the selected threshold q=0.30",
            "q_label": "q=0.30",
        }

    ax = axes[0, 0]
    ax.plot(q, n_call, marker="o", linewidth=1.8, markersize=4.5, color=colors["n_call"])
    _draw_reference_lines(ax, best_q)
    _annotate_point(ax, best_row, "N_call_mean", labels["q_label"], "#2ca02c")
    ax.set_title(labels["a_title"], loc="left", pad=4, fontweight="bold")
    ax.set_xlabel(labels["x"])
    ax.set_ylabel("N_call")
    ax.grid(True, alpha=0.2, linewidth=0.6)

    ax = axes[0, 1]
    ax.plot(q, r_max, marker="o", linewidth=1.8, markersize=4.5, color=colors["r_max"], label="R_max")
    ax.plot(q, r_avg, marker="s", linewidth=1.7, markersize=4.2, color=colors["r_avg"], label="R_avg")
    _draw_reference_lines(ax, best_q)
    _annotate_point(ax, best_row, "R_avg_mean", labels["q_label"], "#2ca02c")
    ax.set_title(labels["b_title"], loc="left", pad=4, fontweight="bold")
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["risk_y"])
    ax.legend(frameon=False, loc="center right")
    ax.grid(True, alpha=0.2, linewidth=0.6)

    ax = axes[1, 0]
    ax.plot(q, a_turn, marker="o", linewidth=1.8, markersize=4.5, color=colors["a_turn"])
    _draw_reference_lines(ax, best_q)
    _annotate_point(ax, best_row, "A_turn_pct_mean", labels["q_label"], "#2ca02c")
    ax.set_title(labels["c_title"], loc="left", pad=4, fontweight="bold")
    ax.set_xlabel(labels["x"])
    ax.set_ylabel("A_turn (%)")
    ax.grid(True, alpha=0.2, linewidth=0.6)

    ax = axes[1, 1]
    ax.plot(q, delta_d_abs, marker="o", linewidth=1.8, markersize=4.5, color=colors["delta_d"])
    _draw_reference_lines(ax, best_q)
    if best_row is not None:
        temp = dict(best_row)
        temp["delta_D_pct_mean"] = abs(float(temp["delta_D_pct_mean"]))
        _annotate_point(ax, temp, "delta_D_pct_mean", labels["q_label"], "#2ca02c")
    ax.set_title(labels["d_title"], loc="left", pad=4, fontweight="bold")
    ax.set_xlabel(labels["x"])
    ax.set_ylabel("|ΔD| (%)")
    ax.grid(True, alpha=0.2, linewidth=0.6)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot threshold sweep results for RTRA-LLM.")
    parser.add_argument("--results_dir", required=True, help="Directory containing sweep_summary.csv and sweep_results.json")
    parser.add_argument("--recommended_q", type=float, default=None, help="Override recommended q to highlight")
    parser.add_argument("--lang", choices=["en", "cn"], default="en", help="Figure language")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    summary_path = results_dir / "sweep_summary.csv"
    results_json = results_dir / "sweep_results.json"

    rows = _read_summary_csv(summary_path)
    meta = _read_meta(results_json)

    provider = str(meta.get("provider", "unknown"))
    best_q = float(args.recommended_q) if args.recommended_q is not None else _recommend_q(rows)

    fig = build_figure(rows, provider=provider, best_q=best_q, lang=args.lang)

    suffix = f"publication_{args.lang}"
    png_path = results_dir / f"threshold_sweep_{suffix}.png"
    pdf_path = results_dir / f"threshold_sweep_{suffix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Recommended q: {best_q:.2f}")


if __name__ == "__main__":
    main()
