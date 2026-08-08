"""Build only the authorized Day-1 reference tables and scientific figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

from exact_reference import compute_exact_reference, write_reference_files
from graph import DEFAULT_GRAPH_PATH, PROJECT_ROOT, get_edge_order, load_graph


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"

NODE_POSITIONS = {
    0: (0.0, 1.50),
    1: (1.75, 2.85),
    2: (1.75, 1.50),
    3: (1.75, 0.15),
    4: (3.85, 2.25),
    5: (3.85, 0.75),
    6: (5.80, 1.50),
}

EDGE_CURVATURE = {
    (1, 3): 0.22,
    (2, 5): 0.08,
    (3, 4): -0.08,
}

LABEL_POSITIONS = {
    (0, 1): (0.72, 2.30),
    (0, 2): (0.86, 1.66),
    (0, 3): (0.72, 0.69),
    (1, 2): (1.57, 2.19),
    (1, 3): (2.20, 1.55),
    (1, 4): (2.82, 2.70),
    (2, 3): (1.57, 0.82),
    (2, 4): (2.76, 2.02),
    (2, 5): (2.72, 0.95),
    (3, 4): (2.90, 1.38),
    (3, 5): (2.78, 0.29),
    (4, 5): (3.65, 1.50),
    (4, 6): (4.88, 2.05),
    (5, 6): (4.88, 0.94),
}


def _draw_arrow(
    axis: plt.Axes,
    edge: tuple[int, int],
    *,
    highlighted: bool,
) -> None:
    color = "#e4572e" if highlighted else "#9aa5b1"
    arrow = FancyArrowPatch(
        NODE_POSITIONS[edge[0]],
        NODE_POSITIONS[edge[1]],
        arrowstyle="-|>",
        mutation_scale=18 if highlighted else 15,
        linewidth=3.4 if highlighted else 1.55,
        color=color,
        alpha=1.0 if highlighted else 0.72,
        connectionstyle=f"arc3,rad={EDGE_CURVATURE.get(edge, 0.0)}",
        shrinkA=23,
        shrinkB=23,
        zorder=1 if not highlighted else 2,
    )
    axis.add_patch(arrow)


def generate_graph_figure(
    graph,
    reference: dict[str, Any],
    *,
    png_path: str | Path,
    svg_path: str | Path,
) -> None:
    """Generate the deterministic presentation-quality frozen graph figure."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day1",
        }
    )
    optimal_edges = {
        tuple(edge) for edge in reference["exact_reference"]["edge_path"]
    }
    figure, axis = plt.subplots(figsize=(12.0, 6.4))
    figure.subplots_adjust(left=0.035, right=0.965, top=0.84, bottom=0.08)
    axis.set_xlim(-0.55, 6.35)
    axis.set_ylim(-0.45, 3.35)
    axis.set_aspect("equal")
    axis.axis("off")

    for edge in get_edge_order(graph):
        _draw_arrow(axis, edge, highlighted=edge in optimal_edges)

    for edge in get_edge_order(graph):
        x, y = LABEL_POSITIONS[edge]
        weight = graph.edges[edge]["weight"]
        highlighted = edge in optimal_edges
        axis.text(
            x,
            y,
            f"{weight}",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold" if highlighted else "normal",
            color="#9c2f16" if highlighted else "#536170",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "#fff6ed" if highlighted else "white",
                "edgecolor": "none",
                "alpha": 0.95,
            },
            zorder=5,
        )

    for node, (x, y) in NODE_POSITIONS.items():
        if node == graph.graph["source"]:
            face, edge_color = "#d9f3e4", "#18794e"
        elif node == graph.graph["target"]:
            face, edge_color = "#fde2dd", "#b42318"
        else:
            face, edge_color = "#e8f1fb", "#2f5d8a"
        axis.scatter(
            [x],
            [y],
            s=1200,
            c=[face],
            edgecolors=[edge_color],
            linewidths=2.2,
            zorder=4,
        )
        axis.text(x, y, str(node), ha="center", va="center", fontsize=17, fontweight="bold", color="#17212b", zorder=6)
        if node == graph.graph["source"]:
            axis.text(x, y - 0.48, "SOURCE", ha="center", color="#18794e", fontsize=10, fontweight="bold")
        elif node == graph.graph["target"]:
            axis.text(x, y - 0.48, "TARGET", ha="center", color="#b42318", fontsize=10, fontweight="bold")

    figure.suptitle(
        "Frozen 7-Node Weighted Routing Instance",
        x=0.50,
        y=0.955,
        fontsize=23,
        fontweight="bold",
        color="#17212b",
    )
    figure.text(
        0.50,
        0.885,
        "Exact shortest route highlighted",
        ha="center",
        fontsize=13.5,
        color="#536170",
    )
    axis.text(
        6.18,
        3.16,
        f"C* = {reference['exact_reference']['cost']}",
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#9c2f16",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff6ed", "edgecolor": "#f2b8a5"},
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color="#e4572e", linewidth=3.4, label="Exact shortest route"),
            Line2D([0], [0], color="#9aa5b1", linewidth=1.55, label="Other directed edges"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )

    png_output = Path(png_path)
    svg_output = Path(svg_path)
    png_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_output,
        dpi=240,
        facecolor="white",
        metadata={"Software": "sciqis-qaoa-routing Day-1 builder"},
    )
    figure.savefig(
        svg_output,
        facecolor="white",
        metadata={"Creator": "sciqis-qaoa-routing Day-1 builder", "Date": None},
    )
    plt.close(figure)


def generate_route_cost_spectrum(
    all_routes,
    reference: dict[str, Any],
    *,
    output_path: str | Path,
) -> None:
    """Generate the optional diagnostic spectrum of every simple route cost."""

    costs = np.asarray([route.cost for route in all_routes], dtype=int)
    ranks = np.arange(1, len(costs) + 1)
    optimum = int(reference["exact_reference"]["cost"])
    second_best = int(reference["second_best_cost"])
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    axis.plot(ranks, costs, color="#8a99a8", linewidth=1.4, marker="o", markersize=4.5, label="Simple route")
    axis.scatter(ranks[costs == optimum], costs[costs == optimum], s=150, marker="*", color="#e4572e", edgecolor="#8f2f17", zorder=4, label=f"Optimum: C*={optimum}")
    axis.scatter(ranks[costs == second_best], costs[costs == second_best], s=62, marker="D", color="#2f6f9f", zorder=3, label=f"Second-best cost: {second_best}")
    axis.set_title("Simple Source-to-Target Route Cost Spectrum", fontsize=16, fontweight="bold", pad=12)
    axis.set_xlabel("Route rank (sorted by cost, then node path)")
    axis.set_ylabel("Total route cost")
    axis.set_xticks(ranks)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.text(0.99, 0.05, f"Optimality gap = {second_best - optimum}", transform=axis.transAxes, ha="right", color="#536170")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white", metadata={"Software": "sciqis-qaoa-routing Day-1 builder"})
    plt.close(figure)


def build_day1_artifacts(
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """Regenerate all authorized Day-1 files from the canonical graph JSON."""

    graph_file = Path(graph_path).resolve()
    results = Path(results_dir).resolve()
    figures = Path(figures_dir).resolve()
    graph = load_graph(graph_file)
    reference, all_routes = compute_exact_reference(graph, graph_path=graph_file)
    write_reference_files(
        reference,
        all_routes,
        json_path=results / "exact_reference.json",
        csv_path=results / "all_simple_paths.csv",
    )
    generate_graph_figure(
        graph,
        reference,
        png_path=figures / "01_frozen_weighted_graph.png",
        svg_path=figures / "01_frozen_weighted_graph.svg",
    )
    generate_route_cost_spectrum(
        all_routes,
        reference,
        output_path=figures / "02_route_cost_spectrum.png",
    )
    return {
        "exact_node_path": reference["exact_reference"]["node_path"],
        "exact_edge_bitstring": reference["exact_reference"]["edge_bitstring"],
        "exact_cost": reference["exact_reference"]["cost"],
        "simple_path_count": reference["simple_path_count"],
        "second_best_cost": reference["second_best_cost"],
        "optimality_gap": reference["optimality_gap"],
        "method_agreement": reference["agreement"]["valid"],
    }
