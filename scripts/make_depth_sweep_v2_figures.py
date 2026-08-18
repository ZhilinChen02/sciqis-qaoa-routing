"""Create the three main figures from the saved Depth-110 v2 rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "depth110_ext" / "v2"
DEFAULT_FIGURE_ROOT = PROJECT_ROOT / "figures" / "depth110_ext" / "v2"
PENALTY_X = "penalty_x"
GROVER_COST_PHASE = "grover_cost_phase"
ALGORITHMS = (PENALTY_X, GROVER_COST_PHASE)
NAMES = {
    PENALTY_X: "Penalty-X QAOA",
    GROVER_COST_PHASE: "Grover-Mixer QAOA (route-cost phase)",
}
COLORS = {PENALTY_X: "#c0392b", GROVER_COST_PHASE: "#1f77b4"}
MARKERS = {PENALTY_X: "o", GROVER_COST_PHASE: "s"}


def load_rows(result_root: Path) -> dict[str, list[dict[str, float | int | str]]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {name: [] for name in ALGORITHMS}
    with (result_root / "canonical_rows.csv").open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | int | str] = dict(raw)
            row["depth"] = int(raw["depth"])
            for key in (
                "p_feas", "p_opt", "invalid_mass", "expected_cost_full",
                "expected_cost_feasible", "normalized_regret", "bsp",
                "optimizer_wall_time_seconds", "evaluation_budget",
                "evaluation_budget_used",
            ):
                value = raw.get(key, "")
                row[key] = float(value) if value not in ("", "nan") else float("nan")
            grouped[raw["algorithm"]].append(row)
    for algorithm in ALGORITHMS:
        grouped[algorithm].sort(key=lambda row: int(row["depth"]))
    return grouped


def load_status(result_root: Path) -> dict[tuple[str, int], str]:
    payload = json.loads((result_root / "failure_taxonomy.json").read_text(encoding="utf-8"))
    return {(item["algorithm"], int(item["depth"])): item["status"] for item in payload}


def series(rows: list[dict[str, float | int | str]], key: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([int(row["depth"]) for row in rows]),
        np.asarray([float(row[key]) for row in rows]),
    )


def draw_curve(ax, rows, status, algorithm: str, key: str) -> None:
    x, y = series(rows, key)
    ax.plot(x, y, color=COLORS[algorithm], linewidth=1.4, alpha=0.9, label=NAMES[algorithm])
    completed = np.asarray([status[(algorithm, int(depth))] == "COMPLETED" for depth in x])
    limited = ~completed
    if completed.any():
        ax.scatter(x[completed], y[completed], s=24, marker=MARKERS[algorithm],
                   color=COLORS[algorithm], zorder=3)
    if limited.any():
        ax.scatter(x[limited], y[limited], s=15, marker=MARKERS[algorithm],
                   facecolors="none", edgecolors=COLORS[algorithm], linewidths=0.7,
                   alpha=0.72, zorder=2)


def threshold_depth(rows, threshold: float) -> int | None:
    for row in rows:
        if float(row["p_opt"]) >= threshold:
            return int(row["depth"])
    return None


def saturation_depth(rows, tolerance: float = 1e-3, window: int = 10) -> int | None:
    """Return the first depth in a stable window of ``window`` changes.

    A change at index ``start`` is the difference between depths ``start`` and
    ``start + 1``.  Therefore the first depth belonging to that change window
    is ``depths[start + 1]``, not ``depths[start]``.
    """

    values = np.asarray([float(row["p_opt"]) for row in rows])
    depths = [int(row["depth"]) for row in rows]
    for start in range(len(values) - window):
        if np.max(np.abs(np.diff(values[start:start + window + 1]))) < tolerance:
            return depths[start + 1]
    return None


def first_grover_above_penalty(penalty, grover) -> int | None:
    """Return the plan-defined crossover: first p with Grover p_opt > Penalty."""

    for p_row, g_row in zip(penalty, grover, strict=True):
        if float(g_row["p_opt"]) > float(p_row["p_opt"]):
            return int(g_row["depth"])
    return None


def first_penalty_above_grover(penalty, grover) -> int | None:
    """Return the first later observation where Penalty-X is above Grover."""

    for p_row, g_row in zip(penalty, grover, strict=True):
        if float(p_row["p_opt"]) > float(g_row["p_opt"]):
            return int(p_row["depth"])
    return None


def make_figures(result_root: Path, figure_root: Path) -> dict[str, object]:
    grouped = load_rows(result_root)
    status = load_status(result_root)
    figure_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algorithm in ALGORITHMS:
        draw_curve(ax, grouped[algorithm], status, algorithm, "p_opt")
        best = max(grouped[algorithm], key=lambda row: float(row["p_opt"]))
        ax.scatter([int(best["depth"])], [float(best["p_opt"])], s=120,
                   facecolors="none", edgecolors=COLORS[algorithm], linewidths=1.7)
        ax.annotate(f"best p={int(best['depth'])}, {float(best['p_opt']):.4f}",
                    (int(best["depth"]), float(best["p_opt"])),
                    xytext=(6, 8), textcoords="offset points", fontsize=8,
                    color=COLORS[algorithm])

    grover_rows = grouped[GROVER_COST_PHASE]
    for threshold, label in ((0.50, "p50"), (0.90, "p90"), (0.99, "p99")):
        depth = threshold_depth(grover_rows, threshold)
        if depth is not None:
            row = next(item for item in grover_rows if int(item["depth"]) == depth)
            value = float(row["p_opt"])
            ax.scatter([depth], [value], marker="^", s=48,
                       color=COLORS[GROVER_COST_PHASE], zorder=4)
            ax.annotate(f"{label}={depth}", (depth, value),
                        xytext=(5, -15), textcoords="offset points",
                        fontsize=8, color=COLORS[GROVER_COST_PHASE])

    for algorithm in ALGORITHMS:
        depth = saturation_depth(grouped[algorithm])
        if depth is not None:
            ax.axvline(depth, color=COLORS[algorithm], linestyle=":",
                       linewidth=1.0, alpha=0.7)
            ax.text(depth + 1, 0.08 if algorithm == PENALTY_X else 0.22,
                    f"{NAMES[algorithm]} saturation: p={depth}",
                    rotation=90, va="bottom", fontsize=7,
                    color=COLORS[algorithm])

    plan_crossover = first_grover_above_penalty(
        grouped[PENALTY_X], grouped[GROVER_COST_PHASE]
    )
    penalty_overtake = first_penalty_above_grover(
        grouped[PENALTY_X], grouped[GROVER_COST_PHASE]
    )
    if plan_crossover is not None:
        ax.axvline(plan_crossover, color="#2e8b57", linestyle="--",
                   linewidth=1.0, alpha=0.8)
        ax.text(plan_crossover + 1, 0.48,
                f"plan crossover: p={plan_crossover}",
                rotation=90, va="bottom", fontsize=7, color="#2e8b57")
    ax.set(title="Optimal-route probability versus QAOA depth",
           xlabel="QAOA depth p", ylabel="Probability of the optimal route")
    ax.set_xlim(0.5, 110)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plan_crossover_text = (
        "Plan-defined Grover > Penalty crossover: NOT REACHED"
        if plan_crossover is None
        else f"Plan-defined Grover > Penalty crossover: p={plan_crossover}"
    )
    penalty_overtake_text = (
        "Penalty-X later overtake: NONE"
        if penalty_overtake is None
        else f"Penalty-X later overtake: p={penalty_overtake}"
    )
    fig.text(0.5, 0.01,
             "Hollow markers are budget-limited observations. "
             f"{plan_crossover_text}; {penalty_overtake_text}.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(figure_root / "figure_1_p_opt_vs_depth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algorithm in ALGORITHMS:
        draw_curve(ax, grouped[algorithm], status, algorithm, "p_feas")
    ax.set(title="Feasible-route probability versus QAOA depth",
           xlabel="QAOA depth p", ylabel="Feasible probability")
    ax.set_xlim(1, 110)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.text(0.5, 0.01,
             "Grover-Mixer is structurally feasible; Penalty-X remains dominated by invalid mass.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(figure_root / "figure_2_p_feas_vs_depth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algorithm in ALGORITHMS:
        draw_curve(ax, grouped[algorithm], status, algorithm, "normalized_regret")
    ax.set(title="Normalized regret versus QAOA depth",
           xlabel="QAOA depth p", ylabel="(expected cost - optimal cost) / optimal cost")
    ax.set_xlim(1, 110)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.text(0.5, 0.01,
             "Penalty-X uses the full-space expected cost; Grover-Mixer uses the feasible-route expected cost.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(figure_root / "figure_3_regret_vs_depth.png", dpi=180)
    plt.close(fig)

    summary: dict[str, object] = {"row_count": sum(map(len, grouped.values()))}
    for algorithm in ALGORITHMS:
        rows = grouped[algorithm]
        best = max(rows, key=lambda row: float(row["p_opt"]))
        summary[algorithm] = {
            "best_depth": int(best["depth"]),
            "best_p_opt": float(best["p_opt"]),
            "p50_depth": threshold_depth(rows, 0.50),
            "p90_depth": threshold_depth(rows, 0.90),
            "p99_depth": threshold_depth(rows, 0.99),
            "saturation_depth": saturation_depth(rows),
            "total_optimizer_wall_time_seconds": float(sum(float(row["optimizer_wall_time_seconds"]) for row in rows)),
            "status_counts": dict(Counter(status[(algorithm, int(row["depth"]))] for row in rows)),
        }
    summary["first_grover_above_penalty_depth"] = plan_crossover
    summary["first_penalty_x_above_grover_depth"] = penalty_overtake
    summary["figure_files"] = [
        "figure_1_p_opt_vs_depth.png",
        "figure_2_p_feas_vs_depth.png",
        "figure_3_regret_vs_depth.png",
    ]
    (result_root / "depth_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    args = parser.parse_args()
    print(json.dumps(make_figures(args.result_root, args.figure_root), indent=2))


if __name__ == "__main__":
    main()
