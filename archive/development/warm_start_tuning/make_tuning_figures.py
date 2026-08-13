#!/usr/bin/env python3
"""Archived plotting controller for the superseded tuning study."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "results" / "tuning"
FINAL = ROOT / "results" / "final"
FIGURES = ROOT / "figures"
COLORS = {1: "#4c78a8", 2: "#f28e2b", 3: "#59a14f"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    svg = FIGURES / f"{name}.svg"
    figure.savefig(svg, bbox_inches="tight")
    normalized = "\n".join(
        line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
    )
    svg.write_text(normalized + "\n", encoding="utf-8")
    plt.close(figure)


def epsilon_metric(rows: list[dict[str, str]], key: str, name: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for depth in (1, 2):
        selected = sorted(
            (row for row in rows if int(row["p"]) == depth),
            key=lambda row: float(row["epsilon"]),
        )
        axis.plot(
            [float(row["epsilon"]) for row in selected],
            [float(row[key]) for row in selected],
            marker="o",
            linewidth=2,
            color=COLORS[depth],
            label=f"Q2 p={depth}",
        )
    axis.axvline(0.1, color="#777", linestyle="--", linewidth=1, label="original ε=0.10")
    axis.set_xlabel("warm-start epsilon")
    axis.set_ylabel(ylabel)
    axis.set_title(name)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    save(figure, name.lower().replace(" ", "_"))


def incumbent_probability(rows: list[dict[str, str]]) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    for depth in (1, 2):
        selected = sorted(
            (row for row in rows if int(row["p"]) == depth),
            key=lambda row: float(row["epsilon"]),
        )
        axis.plot(
            [float(row["epsilon"]) for row in selected],
            [float(row["incumbent_probability"]) for row in selected],
            marker="o", linewidth=2, color=COLORS[depth], label=f"final p={depth}",
        )
    p1 = sorted(
        (row for row in rows if int(row["p"]) == 1),
        key=lambda row: float(row["epsilon"]),
    )
    axis.plot(
        [float(row["epsilon"]) for row in p1],
        [float(row["initial_incumbent_probability"]) for row in p1],
        color="#555", linestyle=":", linewidth=2.2, label="initial product state",
    )
    axis.set_yscale("log")
    axis.set_xlabel("warm-start epsilon")
    axis.set_ylabel("incumbent probability (log scale)")
    axis.set_title("Epsilon vs incumbent probability\nstrong prior → exploration trade-off")
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False)
    save(figure, "epsilon_vs_incumbent_probability")


def initial_final_optimum(rows: list[dict[str, str]]) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    for depth in (1, 2):
        selected = sorted(
            (row for row in rows if int(row["p"]) == depth),
            key=lambda row: float(row["epsilon"]),
        )
        epsilon = [float(row["epsilon"]) for row in selected]
        axis.plot(
            epsilon,
            [float(row["optimal_state_probability"]) for row in selected],
            marker="o", linewidth=2, color=COLORS[depth], label=f"final p={depth}",
        )
    initial = sorted(
        (row for row in rows if int(row["p"]) == 1),
        key=lambda row: float(row["epsilon"]),
    )
    axis.plot(
        [float(row["epsilon"]) for row in initial],
        [float(row["initial_optimal_probability"]) for row in initial],
        color="#555", linestyle=":", linewidth=2.2, label="initial product state",
    )
    axis.set_yscale("log")
    axis.set_xlabel("warm-start epsilon")
    axis.set_ylabel("global-optimum probability (log scale)")
    axis.set_title("Initial vs optimized optimum probability")
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False)
    save(figure, "initial_vs_final_optimum_probability")


def multistart_variability(runs: list[dict[str, str]]) -> None:
    primary = [
        row for row in runs
        if row["p"] == "2" and row["optimizer_budget"] == "100"
        and row["initialization_strategy"] == "source_uniform"
    ]
    epsilons = sorted({float(row["epsilon"]) for row in primary})
    observations = [
        [float(row["p_opt"]) for row in primary if float(row["epsilon"]) == epsilon]
        for epsilon in epsilons
    ]
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.boxplot(observations, tick_labels=[f"{epsilon:.2f}" for epsilon in epsilons], showmeans=True)
    axis.set_yscale("log")
    axis.set_xlabel("epsilon (Q2 p=2, seeds 0–9)")
    axis.set_ylabel("p_opt (log scale)")
    axis.set_title("Multi-start variability\nflat boxes indicate convergence to the same boundary solution")
    axis.grid(axis="y", alpha=0.22, which="both")
    save(figure, "multistart_variability")


def final_comparison(rows: list[dict[str, str]]) -> None:
    labels = []
    for row in rows:
        if row["experiment_id"] == "q1_penalty_x_p1":
            labels.append("Q1\np=1")
        elif row["experiment_id"] == "q2_warm_start_p1":
            labels.append("Q2 original\np=1")
        elif row["experiment_id"] == "q2_warm_start_p2":
            labels.append("Q2 original\np=2")
        else:
            labels.append(f"Q2 tuned\np={row['p']}, ε={row['epsilon']}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#777" if row["solver"].startswith("Q1") else "#4c78a8" if "tuned" not in row["solver"] else "#f28e2b" for row in rows]
    for axis, key, title in zip(
        axes,
        ("p_feas", "p_opt"),
        ("Feasibility probability", "Optimal-solution probability"),
    ):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, labels=[f"{value:.4g}" for value in values], padding=3, fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("probability")
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(0, max(values) * 1.25)
    figure.suptitle("Original baselines and tuned energy-selected configurations")
    figure.tight_layout()
    save(figure, "final_solver_comparison")


def main() -> None:
    epsilon_rows = read_csv(TUNING / "epsilon_sweep.csv")
    runs = read_csv(TUNING / "multistart_runs.csv")
    final_rows = read_csv(FINAL / "metrics.csv")
    epsilon_metric(epsilon_rows, "p_feas", "Epsilon vs pfeas", "$p_{feas}$")
    epsilon_metric(epsilon_rows, "p_opt", "Epsilon vs popt", "$p_{opt}$")
    incumbent_probability(epsilon_rows)
    initial_final_optimum(epsilon_rows)
    multistart_variability(runs)
    final_comparison(final_rows)
    print(f"Wrote 6 tuning figures (PNG and SVG) to {FIGURES}")


if __name__ == "__main__":
    main()
