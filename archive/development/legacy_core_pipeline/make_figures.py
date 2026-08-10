#!/usr/bin/env python3
"""Archived plotter for the superseded comparison outputs."""
"""Generate the five presentation figures from freshly written result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

from graph import get_edge_order, load_graph


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
COLORS = {
    "optimal": "#2a9d8f",
    "feasible": "#457b9d",
    "invalid": "#d16d5b",
    "q1": "#6c757d",
    "q2p1": "#4c78a8",
    "q2p2": "#f28e2b",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    svg_path = FIGURES / f"{name}.svg"
    figure.savefig(svg_path, bbox_inches="tight")
    # Matplotlib emits path commands with trailing spaces. Normalize the
    # generated text so repository-wide ``git diff --check`` stays clean.
    normalized = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(normalized + "\n", encoding="utf-8")
    plt.close(figure)


def routing_graph() -> None:
    graph = load_graph()
    exact = json.loads((RESULTS / "exact_reference.json").read_text(encoding="utf-8"))
    route = exact["exact_reference"]["node_path"]
    optimal_edges = set(zip(route, route[1:]))
    positions = {
        0: (0.0, 0.0), 1: (1.0, 0.8), 2: (2.0, 0.25),
        3: (2.0, -0.8), 4: (3.2, 0.45), 5: (4.2, -0.25), 6: (5.2, 0.25),
    }
    figure, axis = plt.subplots(figsize=(9.2, 4.4))
    regular = [edge for edge in graph.edges if edge not in optimal_edges]
    nx.draw_networkx_edges(
        graph, positions, edgelist=regular, edge_color="#a8adb3", width=1.4,
        arrows=True, arrowsize=16, connectionstyle="arc3,rad=0.04", ax=axis,
    )
    nx.draw_networkx_edges(
        graph, positions, edgelist=list(optimal_edges), edge_color="#e63946", width=3.4,
        arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.04", ax=axis,
    )
    node_colors = ["#2a9d8f" if node == 0 else "#f4a261" if node == 6 else "#edf2f4" for node in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, edgecolors="#264653", node_size=900, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_weight="bold", ax=axis)
    labels = {edge: graph.edges[edge]["weight"] for edge in graph.edges}
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=labels, font_size=9, label_pos=0.48, ax=axis)
    axis.set_title("DTU weighted routing instance\nExact optimum highlighted: 0→1→2→4→5→6 (cost 10)")
    axis.text(*positions[0], "\n\nsource", ha="center", va="top", fontsize=9)
    axis.text(*positions[6], "\n\ntarget", ha="center", va="top", fontsize=9)
    axis.axis("off")
    _save(figure, "routing_graph")


def warm_start_values() -> None:
    rows = _csv(RESULTS / "warm_start_values.csv")
    labels = [row["variable"].split()[0] for row in rows]
    values = [float(row["clipped_value"]) for row in rows]
    selected = [row["incumbent_bit"] == "1" for row in rows]
    figure, axis = plt.subplots(figsize=(10, 4.4))
    bars = axis.bar(labels, values, color=["#f28e2b" if value else "#9ecae1" for value in selected])
    axis.axhline(0.5, color="#555", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("clipped warm-start value $c_i$")
    axis.set_xlabel("edge variable (q0 → q13 order)")
    axis.set_title("Incumbent-product relaxation (ε = 0.1)")
    axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.legend(handles=[Patch(color="#f28e2b", label="edge in incumbent route"), Patch(color="#9ecae1", label="other edge")], frameon=False)
    axis.grid(axis="y", alpha=0.2)
    _save(figure, "warm_start_values")


def probability_distribution() -> None:
    metrics = {row["experiment_id"]: row for row in _csv(RESULTS / "metrics.csv")}
    rows = _csv(RESULTS / "top_bitstrings.csv")
    exact = json.loads((RESULTS / "exact_reference.json").read_text(encoding="utf-8"))
    optimal_bitstring = exact["exact_reference"]["edge_bitstring"]
    experiment_ids = ["q1_penalty_x_p1", "q2_warm_start_p1", "q2_warm_start_p2"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
    for axis, experiment_id in zip(axes, experiment_ids):
        selected = [row for row in rows if row["experiment_id"] == experiment_id][:6]
        if not any(row["optimal"] == "True" for row in selected):
            selected.append({
                "bitstring": optimal_bitstring,
                "probability": metrics[experiment_id]["p_opt"],
                "valid": "True",
                "optimal": "True",
            })
        labels = [row["bitstring"] for row in selected]
        probabilities = [float(row["probability"]) for row in selected]
        categories = ["optimal" if row["optimal"] == "True" else "feasible" if row["valid"] == "True" else "invalid" for row in selected]
        axis.barh(range(len(selected)), probabilities, color=[COLORS[value] for value in categories])
        axis.set_yticks(range(len(selected)), labels=labels, fontsize=8, family="monospace")
        axis.invert_yaxis()
        row = metrics[experiment_id]
        axis.set_title(f"{row['solver']} p={row['p']}\n$p_{{feas}}$={float(row['p_feas']):.3f}, $p_{{opt}}$={float(row['p_opt']):.4f}")
        axis.set_xlabel("probability")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Highest-probability bitstrings (plus exact optimum when outside top 6)", y=1.02)
    figure.legend(handles=[Patch(color=COLORS["optimal"], label="globally optimal"), Patch(color=COLORS["feasible"], label="feasible non-optimal"), Patch(color=COLORS["invalid"], label="invalid")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    figure.tight_layout()
    _save(figure, "probability_distribution")


def solver_comparison() -> None:
    rows = _csv(RESULTS / "metrics.csv")
    labels = [f"{row['solver'].replace(' Penalty-X','').replace(' Warm-Start','')}\np={row['p']}" for row in rows]
    colors = [COLORS["q1"], COLORS["q2p1"], COLORS["q2p2"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for axis, key, title in zip(axes, ["p_feas", "p_opt"], ["Feasibility probability", "Optimal-solution probability"]):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=9)
        axis.set_ylim(0, max(values) * 1.2 + 1e-4)
        axis.set_title(title)
        axis.set_ylabel("probability")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Q1 baseline versus Q2 Warm-Start QAOA")
    figure.tight_layout()
    _save(figure, "solver_comparison")


def resource_comparison() -> None:
    rows = _csv(RESULTS / "metrics.csv")
    labels = [f"Q{1 if row['solver'].startswith('Q1') else 2}\np={row['p']}" for row in rows]
    colors = [COLORS["q1"], COLORS["q2p1"], COLORS["q2p2"]]
    specifications = [
        ("circuit_depth", "transpiled depth", ""),
        ("two_qubit_gate_count", "two-qubit gates", ""),
        ("total_runtime_s", "total runtime", "seconds"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for axis, (key, title, unit) in zip(axes, specifications):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(labels, values, color=colors)
        labels_on_bars = [f"{value:.3f}" if unit else f"{value:.0f}" for value in values]
        axis.bar_label(bars, labels=labels_on_bars, padding=3, fontsize=9)
        axis.set_ylim(0, max(values) * 1.22)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Circuit and runtime cost")
    figure.tight_layout()
    _save(figure, "resource_comparison")


def main() -> None:
    required = [RESULTS / "metrics.csv", RESULTS / "top_bitstrings.csv", RESULTS / "warm_start_values.csv", RESULTS / "exact_reference.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"run scripts/run_experiment.py first; missing: {missing}")
    routing_graph()
    warm_start_values()
    probability_distribution()
    solver_comparison()
    resource_comparison()
    print(f"Wrote 5 figures (PNG and SVG) to {FIGURES}")


if __name__ == "__main__":
    main()
