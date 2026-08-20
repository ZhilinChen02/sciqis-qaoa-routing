#!/usr/bin/env python3
"""Export a presentation-safe fallback animation from validated saved dynamics.

The browser demo is primary.  This exporter produces a compact GIF and three
PNG storyboard frames for slide software that cannot host the live local app.
It reconstructs no optimization trajectory and refuses to overwrite a non-empty
target unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "qaoa_dynamics_visualizer" / "v1"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.qaoa_dynamics_visualization import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    QAOADynamicsVisualizationRepository,
)


COLORS = {
    "background": "#071015",
    "panel": "#0d1b23",
    "text": "#eef8f7",
    "muted": "#96adb2",
    "teal": "#3ee0c1",
    "gold": "#ffc857",
    "red": "#ff6680",
    "green": "#7be495",
    "cyan": "#57c7ff",
}


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["panel"])
    axis.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#29414b")
    axis.xaxis.label.set_color(COLORS["muted"])
    axis.yaxis.label.set_color(COLORS["muted"])
    axis.title.set_color(COLORS["text"])
    axis.grid(color="#29414b", alpha=0.35, linewidth=0.6)


def build_storyboard(run: dict, checkpoint_index: int) -> plt.Figure:
    """Render one faithful, compact view of a selected saved checkpoint."""

    frames = run["frames"]
    frame = frames[checkpoint_index]
    figure = plt.figure(figsize=(14, 8), facecolor=COLORS["background"])
    grid = figure.add_gridspec(2, 3, height_ratios=(0.72, 2.2), hspace=0.38, wspace=0.30)
    circuit_axis = figure.add_subplot(grid[0, :])
    energy_axis = figure.add_subplot(grid[1, 0])
    metric_axis = figure.add_subplot(grid[1, 1])
    complex_axis = figure.add_subplot(grid[1, 2])
    for axis in (circuit_axis, energy_axis, metric_axis, complex_axis):
        _style_axis(axis)

    circuit_axis.set_xlim(-0.6, len(run["circuit"]["gates"]) - 0.4)
    circuit_axis.set_ylim(-0.7, 0.8)
    circuit_axis.axis("off")
    for index, gate in enumerate(run["circuit"]["gates"]):
        active = gate["checkpoint_index"] == checkpoint_index
        edge = COLORS["teal"] if active else "#29414b"
        circuit_axis.text(
            index,
            0.18,
            gate["label"],
            ha="center",
            va="center",
            color=COLORS["text"],
            fontsize=10,
            weight="bold" if active else "normal",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": COLORS["panel"], "edgecolor": edge, "linewidth": 2 if active else 1},
        )
        if index < len(run["circuit"]["gates"]) - 1:
            circuit_axis.text(index + 0.5, 0.18, "→", ha="center", va="center", color=COLORS["muted"], fontsize=14)
    circuit_axis.text(
        0,
        -0.48,
        f"{frame['checkpoint']}: {frame['teaching']}",
        color=COLORS["gold"],
        fontsize=10,
        ha="left",
    )

    x_values = np.arange(len(frames))
    energies = [item["metrics"]["expected_hc"] for item in frames]
    energy_axis.plot(x_values, energies, "-o", color=COLORS["teal"], lw=2)
    energy_axis.scatter([checkpoint_index], [energies[checkpoint_index]], s=100, facecolor=COLORS["text"], edgecolor=COLORS["teal"], zorder=4)
    energy_axis.set_xticks(x_values, [item["checkpoint"] for item in frames], rotation=25, ha="right")
    energy_axis.set_title("Expected cost energy")
    energy_axis.set_ylabel("⟨H_C⟩")
    energy_axis.text(0.02, 0.98, "Cost steps preserve energy", transform=energy_axis.transAxes, va="top", color=COLORS["muted"], fontsize=8)

    for field, label, color in (
        ("p_feas", "p_feas", COLORS["green"]),
        ("p_opt", "p_opt", COLORS["gold"]),
        ("invalid_mass", "invalid", COLORS["red"]),
    ):
        values = [item["metrics"][field] for item in frames]
        metric_axis.plot(x_values, values, "-o", color=color, label=label, lw=2)
        metric_axis.scatter([checkpoint_index], [values[checkpoint_index]], s=65, facecolor=COLORS["text"], edgecolor=color, zorder=4)
    metric_axis.set_ylim(-0.025, 1.025)
    metric_axis.set_xticks(x_values, [item["checkpoint"] for item in frames], rotation=25, ha="right")
    metric_axis.set_title("Exact probability mass")
    metric_axis.legend(frameon=False, labelcolor=COLORS["text"], fontsize=8)

    complex_axis.set_aspect("equal", adjustable="box")
    complex_axis.axhline(0, color="#29414b", lw=0.8)
    complex_axis.axvline(0, color="#29414b", lw=0.8)
    positions = run["complex_amplitudes"]["display_positions"][:12]
    values = np.asarray([frame["state_values"][position] for position in positions])
    maximum = max(float(values[:, 2].max()), 1e-15)
    state_colors = []
    for position in positions:
        state = run["display_states"][position]
        state_colors.append(COLORS["gold"] if state["optimal"] else (COLORS["green"] if state["feasible"] else COLORS["red"]))
    for value, color in zip(values, state_colors):
        complex_axis.arrow(0, 0, value[0], value[1], width=maximum * 0.012, head_width=maximum * 0.07, length_includes_head=True, color=color, alpha=0.78)
    limit = maximum * 1.2
    complex_axis.set_xlim(-limit, limit)
    complex_axis.set_ylim(-limit, limit)
    complex_axis.set_title("Representative complex amplitudes")
    complex_axis.set_xlabel("Re(a)")
    complex_axis.set_ylabel("Im(a)")

    figure.suptitle(
        f"Global Grover-Mixer QAOA · p=2 · {frame['checkpoint']}",
        color=COLORS["text"],
        fontsize=18,
        weight="bold",
        y=0.98,
    )
    figure.text(0.99, 0.01, "Validated saved DTU SCIQIS dynamics · no optimizer rerun", ha="right", color=COLORS["muted"], fontsize=8)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists() and any(output.iterdir()) and not arguments.force:
        raise SystemExit(f"Refusing to overwrite non-empty export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repository = QAOADynamicsVisualizationRepository(arguments.result_root)
    if not repository.validation_report["all_checks_passed"]:
        raise SystemExit("visualization data validation failed")
    run = repository.load_run("grover_global", 2)

    images: list[Image.Image] = []
    for index in range(len(run["frames"])):
        figure = build_storyboard(run, index)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=105, facecolor=COLORS["background"])
        plt.close(figure)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("RGB").copy())
        buffer.close()
    images[0].save(
        output / "global_grover_p2_cost_mixer_dynamics.gif",
        save_all=True,
        append_images=images[1:],
        duration=1400,
        loop=0,
        optimize=False,
    )

    for index, filename in ((0, "fallback_initial.png"), (1, "fallback_cost_1.png"), (4, "fallback_mixer_2.png")):
        figure = build_storyboard(run, index)
        figure.savefig(output / filename, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
        plt.close(figure)
    print(f"wrote GIF and 3 fallback frames to {output}")


if __name__ == "__main__":
    main()
