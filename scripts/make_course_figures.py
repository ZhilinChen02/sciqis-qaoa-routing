#!/usr/bin/env python3
"""Rebuild five course figures from the graph and immutable result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from graph import load_graph, path_edges  # noqa: E402


Q2R = PROJECT / "results/q2_revision_formal" / (
    "q2r-9c98b90049545d0508fb20bb488019608a4356d55fe91ff809bce8dc5c5e0c69"
)
Q2F = PROJECT / "results/q2f_course_extension" / (
    "q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7"
)
Q2F_FINAL = PROJECT / "results/q2f_final_improvement" / (
    "q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1"
)


METHOD_LABELS = {
    "bsp_path_exchange": "BSP path-exchange",
    "gm_qaoa_expectation": "GM-QAOA expectation",
    "gm_th_qaoa": "GM-Th-QAOA",
}
METHOD_COLORS = {
    "bsp_path_exchange": "#e45756",
    "gm_qaoa_expectation": "#4c78a8",
    "gm_th_qaoa": "#54a24b",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(figure: plt.Figure, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return path


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.5),
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def routing_graph_figure(output: Path, basis_payload: dict[str, object]) -> Path:
    graph = load_graph()
    basis = basis_payload["basis"]
    optimum = tuple(basis["exact_optimal_route"])
    incumbent = tuple(basis["incumbent_route"])
    positions = {
        0: (0.0, 0.3), 1: (1.2, 1.1), 2: (2.3, 0.8), 3: (2.6, -0.4),
        4: (3.8, 0.8), 5: (4.8, -0.1), 6: (6.0, 0.3),
    }
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    nx.draw_networkx_nodes(graph, positions, node_color="#f4f6f8", edgecolors="#263238", node_size=820, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(graph, positions, edge_color="#b7c0c8", arrows=True, arrowsize=18, width=1.4, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=list(path_edges(incumbent)), edge_color="#f2a541", arrows=True, arrowsize=18, width=3.0, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=list(path_edges(optimum)), edge_color="#c23b3b", arrows=True, arrowsize=18, width=3.2, ax=ax)
    weights = {(u, v): graph.edges[u, v]["weight"] for u, v in graph.edges}
    nx.draw_networkx_edge_labels(graph, positions, weights, font_size=9, rotate=False, ax=ax)
    ax.set_title("Weighted routing instance: optimum (red) and cost-11 incumbent (orange)")
    ax.set_axis_off()
    return _save(fig, output, "01_routing_graph_and_routes")


def popt_depth_figure(
    output: Path,
    old_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> Path:
    fig, ax = plt.subplots()
    old_expectation = [row for row in old_rows if row["objective"] == "expectation"]
    for depth in (1, 2, 3):
        values = [float(row["final_p_opt"]) for row in old_expectation if int(row["depth"]) == depth]
        ax.scatter([depth - 0.08] * len(values), values, color="#777777", alpha=0.42, s=26)
    old_medians = [
        float(np.median([float(row["final_p_opt"]) for row in old_expectation if int(row["depth"]) == depth]))
        for depth in (1, 2, 3)
    ]
    ax.plot((1, 2, 3), old_medians, marker="o", color="#555555", linewidth=2,
            label="Q2-F path-exchange expectation")
    offsets = {"bsp_path_exchange": -0.03, "gm_qaoa_expectation": 0.03, "gm_th_qaoa": 0.09}
    for method in METHOD_LABELS:
        rows = [row for row in final_rows if row["method"] == method]
        ax.scatter(
            [int(row["depth"]) + offsets[method] for row in rows],
            [float(row["p_opt"]) for row in rows],
            color=METHOD_COLORS[method], alpha=0.42, s=26,
        )
        medians = [
            float(np.median([float(row["p_opt"]) for row in rows if int(row["depth"]) == depth]))
            for depth in (1, 2, 3, 4)
        ]
        ax.plot((1, 2, 3, 4), medians, marker="o", linewidth=2,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set(title="Optimal-route probability versus logical QAOA depth",
           xlabel="Depth p", ylabel="p_opt", xticks=(1, 2, 3, 4), ylim=(0, 1.04))
    ax.legend(fontsize=9)
    return _save(fig, output, "02_popt_versus_depth")


def pfeas_method_figure(
    output: Path,
    q2r_rows: list[dict[str, str]],
    old_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> Path:
    q2r_a0 = next(float(row["p_feas"]) for row in q2r_rows if row["run_id"] == "A0")
    old_p3 = [float(row["p_feas"]) for row in old_rows if row["objective"] == "expectation" and int(row["depth"]) == 3]
    final_p3 = [float(row["p_feas"]) for row in final_rows if row["method"] == "gm_th_qaoa" and int(row["depth"]) == 3]
    labels = ["Q2-R edge-space\nexpectation p=3", "Q2-F feasible\npath-exchange p=3", "Final feasible\nGM-Th p=3"]
    values = [q2r_a0, float(np.median(old_p3)), float(np.median(final_p3))]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(labels, values, color=["#7f8c8d", "#d17c36", "#54a24b"], width=0.62)
    ax.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=4)
    ax.set(title="Feasible probability by representation and method", ylabel="p_feas", ylim=(0, 1.08))
    ax.text(
        0.5,
        0.035,
        "Logical feasible-space p_feas=1 is structural.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    return _save(fig, output, "03_pfeas_versus_method")


def final_distribution_figure(
    output: Path,
    basis_payload: dict[str, object],
    final_rows: list[dict[str, str]],
) -> Path:
    p3 = [row for row in final_rows if row["method"] == "gm_th_qaoa" and int(row["depth"]) == 3]
    median_value = float(np.median([float(row["p_opt"]) for row in p3]))
    selected = min(p3, key=lambda row: (abs(float(row["p_opt"]) - median_value), int(row["seed"])))
    raw = json.loads((Q2F_FINAL / "raw" / f"{selected['run_id']}.json").read_text(encoding="utf-8"))
    probabilities = raw["result"]["final_probabilities"]
    route_count = int(basis_payload["basis"]["basis_size"])
    route_ids = np.arange(route_count)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.bar(route_ids, probabilities, color="#54a24b", alpha=0.84, label=f"GM-Th p=3, seed {selected['seed']}")
    ax.axhline(1 / route_count, color="#4c78a8", linestyle="--", label="Uniform feasible initialization (5%)")
    ax.set(title="Final probability distribution over 20 feasible routes",
           xlabel="Logical route ID (cost, then lexicographic order)", ylabel="Probability",
           xticks=route_ids)
    ax.legend()
    return _save(fig, output, "04_final_probability_distribution")


def grover_amplification_figure(output: Path, final_rows: list[dict[str, str]]) -> Path:
    depths = (0, 1, 2, 3)
    values = [0.05] + [
        float(np.median([
            float(row["p_opt"]) for row in final_rows
            if row["method"] == "gm_th_qaoa" and int(row["depth"]) == depth
        ]))
        for depth in (1, 2, 3)
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(depths, values, color=["#9ecae9", "#74c476", "#41ab5d", "#238b45"], width=0.68)
    ax.bar_label(bars, labels=[f"{100 * value:.2f}%" for value in values], padding=4)
    ax.plot(depths, values, color="#1b6e3b", linewidth=1.5, marker="o")
    ax.set(title="Grover-style amplification of the incumbent-threshold route",
           xlabel="GM-Th-QAOA depth p (p=0 is initialization)", ylabel="p_opt", xticks=depths,
           ylim=(0, 1.08))
    return _save(fig, output, "05_grover_amplification")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build course figures without rerunning optimization.")
    parser.add_argument("--output", type=Path, default=PROJECT / "figures" / "course")
    arguments = parser.parse_args()
    basis_payload = json.loads((Q2F_FINAL / "basis.json").read_text(encoding="utf-8"))
    q2r_rows = _csv(Q2R / "summary" / "formal_results.csv")
    old_rows = _csv(Q2F / "summary" / "q2f_results.csv")
    final_rows = _csv(Q2F_FINAL / "summary" / "q2f_final_improvement_results.csv")
    _style()
    paths = [
        routing_graph_figure(arguments.output, basis_payload),
        popt_depth_figure(arguments.output, old_rows, final_rows),
        pfeas_method_figure(arguments.output, q2r_rows, old_rows, final_rows),
        final_distribution_figure(arguments.output, basis_payload, final_rows),
        grover_amplification_figure(arguments.output, final_rows),
    ]
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
