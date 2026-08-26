"""Plot evidence-linked versions of the paper's Case 7 Figures 3 and 4.

Figure 3 links the spatial trajectory to start/end points, headings, CPA and
trigger/action locations.  Figure 4 links the same numbered events to DCPA,
range, TCPA, risk q and the applied high-level direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NMI = 1852.0
COLORS = {"own": "#263238", "TS1": "#2F6DB2", "TS2": "#E47C2C"}
EVENT_COLOR = "#6A3D9A"
THRESHOLD_COLOR = "#B22222"
EVENT_REASON_LABELS = {
    "initial_trigger": "initial",
    "risk_threshold_crossing": "q crossing",
    "encounter_type_change": "encounter change",
    "previous_action_invalid": "action invalid",
    "fixed_interval": "fixed interval",
    "always": "always",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="NPZ exported by export_case7_figure_data.py")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON exported with the NPZ")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def _event_reason(event: dict) -> str:
    reasons = event.get("reasons", [])
    if not reasons:
        return "trigger"
    return "+".join(EVENT_REASON_LABELS.get(reason, reason) for reason in reasons)


def _load_trace(data_path: Path, metadata_path: Path) -> tuple[dict, dict]:
    with np.load(data_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data, metadata


def _event_step(event: dict, n_steps: int) -> int:
    return int(np.clip(event.get("step", 0), 0, max(0, n_steps - 1)))


def _plot_track_arrows(ax, x: np.ndarray, y: np.ndarray, color: str, count: int = 4) -> None:
    if len(x) < 3:
        return
    indices = np.linspace(1, len(x) - 2, count, dtype=int)
    for idx in indices:
        ax.annotate(
            "",
            xy=(x[idx + 1], y[idx + 1]),
            xytext=(x[idx - 1], y[idx - 1]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9, "alpha": 0.8},
        )


def _add_track(ax, x: np.ndarray, y: np.ndarray, label: str, color: str, linewidth: float = 1.6) -> None:
    ax.plot(x, y, color=color, lw=linewidth, label=label, zorder=2)
    _plot_track_arrows(ax, x, y, color)


def _plot_cpa(ax, data: dict, target_idx: int, target_label: str, color: str) -> None:
    dcpa = np.asarray(data["dcpa"])[:, target_idx]
    # Use the closest observed encounter distance rather than the global
    # minimum of |DCPA|.  The latter can select the initial collinear segment
    # before the encounter and is not the informative CPA location for Fig. 3.
    range_series = np.asarray(data["distance_ob"])[:, target_idx]
    valid_range = np.where(range_series > 0.0, range_series, np.nan)
    # The first simulation row contains the avoidance module's display value
    # in nmi; subsequent rows are stored in metres, so exclude that row.
    valid_range[0] = np.nan
    cpa_idx = int(np.nanargmin(valid_range))
    own_x = data["x"][cpa_idx] / NMI
    own_y = data["y"][cpa_idx] / NMI
    target_x = data["obstacles_x_history"][cpa_idx, target_idx] / NMI
    target_y = data["obstacles_y_history"][cpa_idx, target_idx] / NMI
    ax.plot([own_x, target_x], [own_y, target_y], ls="--", lw=0.8, color=color, alpha=0.7, zorder=3)
    ax.scatter([own_x], [own_y], marker="*", s=55, color=color, edgecolor="white", linewidth=0.5, zorder=5)
    ax.scatter([target_x], [target_y], marker="*", s=55, color=color, edgecolor="white", linewidth=0.5, zorder=5)
    cpa_nm = abs(float(dcpa[cpa_idx])) / NMI
    range_nm = float(valid_range[cpa_idx]) / NMI
    time_s = float(data["time"][cpa_idx])
    ax.annotate(
        f"Closest encounter {target_label}\nR={range_nm:.2f} nmi, |DCPA|={cpa_nm:.2f} nmi\nt={time_s:.1f} s",
        xy=((own_x + target_x) / 2, (own_y + target_y) / 2),
        xytext=(5, 16 if target_idx == 0 else -30),
        textcoords="offset points",
        fontsize=7,
        color=color,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.85},
    )


def plot_figure3(data: dict, metadata: dict, output_dir: Path) -> Path:
    x_own = data["x"] / NMI
    y_own = data["y"] / NMI
    target_x = data["obstacles_x_history"] / NMI
    target_y = data["obstacles_y_history"] / NMI
    way_x = data["waypoints_x"]
    way_y = data["waypoints_y"]

    # Use a wide overview above a larger detail panel.  This keeps the route
    # context visible while giving event/CPA annotations enough room below.
    fig = plt.figure(figsize=(7.4, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.0])
    ax_global = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[1, 0])

    ax_global.plot(way_x, way_y, color="#777777", ls=(0, (4, 2)), lw=1.0, label="Planned route", zorder=1)
    _add_track(ax_global, x_own, y_own, "Own ship", COLORS["own"], linewidth=1.8)
    for target_idx in range(target_x.shape[1]):
        target_label = f"TS{target_idx + 1}"
        role = metadata.get("paper_roles", ["target"] * target_x.shape[1])[target_idx]
        _add_track(ax_global, target_x[:, target_idx], target_y[:, target_idx], f"{target_label} ({role})", COLORS.get(target_label, "#555555"), linewidth=1.3)
    ax_global.scatter(x_own[0], y_own[0], marker="o", s=30, color=COLORS["own"], edgecolor="white", zorder=6)
    ax_global.text(x_own[0], y_own[0], " S", fontsize=8, va="bottom", color=COLORS["own"])
    ax_global.scatter(x_own[-1], y_own[-1], marker="s", s=30, color=COLORS["own"], edgecolor="white", zorder=6)
    ax_global.text(x_own[-1], y_own[-1], " E", fontsize=8, va="bottom", color=COLORS["own"])
    ax_global.set_title("Global trajectory")
    ax_global.set_xlabel("East x (nmi)")
    ax_global.set_ylabel("North y (nmi)")
    # The Case 7 trajectory data occupy the first ~11 nmi; crop the overview
    # to the informative region and use a taller y-range for readability.
    ax_global.set_xlim(0.0, 15.0)
    ax_global.set_ylim(-2.0, 2.0)
    ax_global.grid(True, alpha=0.25)
    ax_global.legend(loc="upper right", ncol=4, fontsize=6.5, frameon=True)

    ax_zoom.plot(way_x, way_y, color="#777777", ls=(0, (4, 2)), lw=1.0, label="Planned route", zorder=1)
    _add_track(ax_zoom, x_own, y_own, "Own ship", COLORS["own"], linewidth=2.0)
    for target_idx in range(target_x.shape[1]):
        target_label = f"TS{target_idx + 1}"
        role = metadata.get("paper_roles", ["target"] * target_x.shape[1])[target_idx]
        _add_track(ax_zoom, target_x[:, target_idx], target_y[:, target_idx], f"{target_label} ({role})", COLORS.get(target_label, "#555555"), linewidth=1.5)
    ax_zoom.set_xlabel("East x (nmi)")
    ax_zoom.set_ylabel("North y (nmi)")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.set_aspect("equal", adjustable="box")

    events = metadata.get("trigger_history", [])
    for event_idx, event in enumerate(events, start=1):
        step = _event_step(event, len(x_own))
        event_x, event_y = x_own[step], y_own[step]
        ax_zoom.scatter(event_x, event_y, marker="D", s=30, color=EVENT_COLOR, edgecolor="white", zorder=7)
        ax_zoom.annotate(
            str(event_idx),
            xy=(event_x, event_y),
            xytext=(-12, 10) if event_idx == 1 else (3, 3),
            textcoords="offset points",
            fontsize=7,
            color=EVENT_COLOR,
            bbox={"boxstyle": "circle,pad=0.12", "fc": "white", "ec": EVENT_COLOR, "alpha": 0.85},
        )

    cpa_x = np.concatenate([target_x[:, j] for j in range(target_x.shape[1])])
    cpa_y = np.concatenate([target_y[:, j] for j in range(target_y.shape[1])])
    zoom_x = np.concatenate([x_own, cpa_x])
    zoom_y = np.concatenate([y_own, cpa_y])
    x_min, x_max = np.nanmin(zoom_x), np.nanmax(zoom_x)
    y_min, y_max = np.nanpercentile(zoom_y, [1, 99])
    margin_x = max(0.8, 0.12 * (x_max - x_min))
    margin_y = max(0.8, 0.12 * (y_max - y_min))
    ax_zoom.set_xlim(x_min - margin_x, x_max + margin_x)
    ax_zoom.set_ylim(y_min - margin_y, y_max + margin_y)
    for target_idx in range(target_x.shape[1]):
        _plot_cpa(ax_zoom, data, target_idx, f"TS{target_idx + 1}", COLORS.get(f"TS{target_idx + 1}", "#555555"))
    ax_zoom.set_title("Encounter and trigger detail")
    ax_global.text(-0.06, 1.04, "a", transform=ax_global.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_zoom.text(-0.06, 1.04, "b", transform=ax_zoom.transAxes, fontsize=10, fontweight="bold", va="top")

    output_path = output_dir / "fig3_case7_trajectory_evidence"
    for suffix, dpi in (("svg", None), ("pdf", None), ("png", 600), ("tiff", 600)):
        fig.savefig(output_path.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path.with_suffix(".png")


def _plot_trigger_lines(axes: list, events: list, dt: float) -> None:
    for event_idx, event in enumerate(events, start=1):
        time_s = float(event.get("step", 0)) * dt
        for ax in axes:
            ax.axvline(time_s, color=EVENT_COLOR, lw=0.8, ls=(0, (3, 2)), alpha=0.48, zorder=1)
        # Keep the initial event marker at t=0, but move its numeral slightly
        # right so it remains legible next to the second event at 0.1 s.
        label_time = time_s + (5.0 if event_idx == 1 else 0.0)
        axes[-1].text(label_time, 0.98, str(event_idx), transform=axes[-1].get_xaxis_transform(), ha="center", va="top", fontsize=7, color=EVENT_COLOR)


def plot_figure4(data: dict, metadata: dict, output_dir: Path) -> Path:
    time = data["time"]
    target_count = data["risk"].shape[1]
    events = metadata.get("trigger_history", [])
    dt = float(metadata.get("dt", np.median(np.diff(time)) if len(time) > 1 else 0.1))
    fig = plt.figure(figsize=(8.1, 6.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.34])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    ax_dcpa, ax_range, ax_tcpa, ax_risk = axes
    series = [
        (ax_dcpa, np.abs(data["dcpa"]) / NMI, "|DCPA| (nmi)"),
        (ax_range, data["distance_ob"] / NMI, "Range R (nmi)"),
        (ax_tcpa, data["tcpa"], "TCPA (s)"),
        (ax_risk, data["risk"], "Risk q"),
    ]
    handles = []
    for target_idx in range(target_count):
        label = f"TS{target_idx + 1}"
        line_color = COLORS.get(label, "#555555")
        for ax, values, ylabel in series:
            line, = ax.plot(time, values[:, target_idx], color=line_color, lw=1.25, label=label)
            if ax is ax_dcpa:
                handles.append(line)

    ax_risk.axhline(metadata.get("risk_threshold", 0.30), color=THRESHOLD_COLOR, lw=1.0, ls=(0, (4, 2)), label="q0=0.30")
    _plot_trigger_lines(axes, events, dt)

    action_ax = fig.add_subplot(grid[2, :], sharex=ax_dcpa)
    action = data.get("control_kdir", data.get("kdir"))
    action_ax.step(time, action, where="post", color=COLORS["own"], lw=1.1)
    action_ax.set_yticks([-1, 0, 1])
    action_ax.set_yticklabels(["port", "stand on", "starboard"], fontsize=7)
    action_ax.set_ylabel("Action", fontsize=8)
    action_ax.set_xlabel("Time (s)")
    action_ax.grid(True, axis="x", alpha=0.25)
    for event_idx, event in enumerate(events, start=1):
        effective_time = float(event.get("effective_step", event.get("step", 0))) * dt
        action_ax.scatter(effective_time, event.get("kdir", 0), marker="D", s=23, color=EVENT_COLOR, zorder=4)
        label_time = effective_time + (5.0 if event_idx == 1 else 0.0)
        action_ax.text(label_time, 1.08, str(event_idx), ha="center", va="bottom", fontsize=7, color=EVENT_COLOR, clip_on=False)

    for ax, _, ylabel in series:
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.tick_params(labelsize=7)
    ax_tcpa.set_xlabel("Time (s)")
    ax_risk.set_xlabel("Time (s)")
    ax_risk.set_ylim(bottom=0, top=max(1.0, float(np.nanmax(data["risk"])) * 1.05))
    ax_risk.legend(handles=handles + [Line2D([0], [0], color=THRESHOLD_COLOR, ls=(0, (4, 2)), lw=1.0, label="q0=0.30")], loc="upper right", fontsize=7)
    for label, ax in zip(("a", "b", "c", "d"), axes):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    action_ax.text(-0.035, 1.15, "e", transform=action_ax.transAxes, fontsize=10, fontweight="bold", va="top")

    output_path = output_dir / "fig4_case7_risk_evidence"
    for suffix, dpi in (("svg", None), ("pdf", None), ("png", 600), ("tiff", 600)):
        fig.savefig(output_path.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path.with_suffix(".png")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = _load_trace(args.data, args.metadata)
    fig3 = plot_figure3(data, metadata, args.output_dir)
    fig4 = plot_figure4(data, metadata, args.output_dir)
    print(f"Saved Figure 3: {fig3}")
    print(f"Saved Figure 4: {fig4}")


if __name__ == "__main__":
    main()
