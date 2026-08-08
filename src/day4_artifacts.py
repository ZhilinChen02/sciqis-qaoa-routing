"""Cheap Day-4 table/figure rebuild from the saved frozen experiment."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

from day1_artifacts import LABEL_POSITIONS, NODE_POSITIONS
from graph import PROJECT_ROOT, get_edge_order, load_graph
from optimization import (
    EXPECTED_OPTIMIZATION_CONTRACT_SHA256,
    expected_qubo_energy,
    file_sha256,
    split_interleaved_parameters,
    statevector_for_parameters,
)
from qubo import enumerate_state_space
from statevector_reference import penalty_energies, probabilities


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "graph.json"
EXPECTED_GRAPH_SHA256 = "451a6e4cffd01552c1cb6e4290b5ee8d50fc97514aab8ed47a6b33a111f5e54e"
EXPECTED_PENALTY_SHA256 = "3e31dc31434dcd06c430e90e75884536cbef8cdfefec195afeb1acc1b77acd57"


def _normalize_svg_text(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _save_figure(figure, png_path: Path, svg_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, facecolor="white", metadata={"Software": "sciqis-qaoa-routing Day-4 builder"})
    figure.savefig(svg_path, facecolor="white", metadata={"Creator": "sciqis-qaoa-routing Day-4 builder", "Date": None})
    plt.close(figure)
    _normalize_svg_text(svg_path)


def load_core_results(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> dict[str, Any]:
    results = Path(results_dir).resolve()
    payload = json.loads((results / "core_experiment_results.json").read_text(encoding="utf-8"))
    if payload["optimization_contract"]["sha256"] != EXPECTED_OPTIMIZATION_CONTRACT_SHA256:
        raise RuntimeError("saved result optimization-contract identity changed")
    identity = payload["scientific_identity"]
    if identity["graph_sha256"] != EXPECTED_GRAPH_SHA256:
        raise RuntimeError("saved result graph identity changed")
    if identity["penalty_contract_sha256"] != EXPECTED_PENALTY_SHA256:
        raise RuntimeError("saved result penalty identity changed")
    if len(payload["optimization_runs"]) != 24 or len(payload["selected_cells"]) != 8:
        raise RuntimeError("saved Day-4 result matrix is incomplete")
    return payload


def _cell_lookup(payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (cell["identity"]["A"], cell["identity"]["p"]): cell
        for cell in payload["selected_cells"]
    }


def _selected_distribution(states, cell: dict[str, Any]) -> np.ndarray:
    penalty = cell["identity"]["A"]
    p = cell["identity"]["p"]
    energies = penalty_energies(states, penalty)
    statevector = statevector_for_parameters(
        energies,
        p=p,
        parameters=cell["optimization"]["final_parameters"],
    )
    return probabilities(statevector)


def _load_or_compute_landscape(
    states,
    *,
    output_path: Path,
    gamma_points: int = 121,
    beta_points: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output_path.exists():
        data = np.loadtxt(output_path, delimiter=",", skiprows=1)
        expected_rows = gamma_points * beta_points
        if data.shape == (expected_rows, 3):
            gammas = np.unique(data[:, 0])
            betas = np.unique(data[:, 1])
            # CSV rows are written with gamma outermost and beta innermost,
            # while Matplotlib expects Z[beta_index, gamma_index].
            energy = data[:, 2].reshape(gamma_points, beta_points).T
            return gammas, betas, energy

    gammas = np.linspace(0.0, 2 * np.pi, gamma_points)
    betas = np.linspace(0.0, np.pi, beta_points)
    energies = penalty_energies(states, 6)
    landscape = np.empty((beta_points, gamma_points), dtype=float)
    rows = []
    for gamma_index, gamma in enumerate(gammas):
        for beta_index, beta in enumerate(betas):
            distribution = probabilities(
                statevector_for_parameters(
                    energies, p=1, parameters=(gamma, beta)
                )
            )
            value = expected_qubo_energy(distribution, energies)
            landscape[beta_index, gamma_index] = value
            rows.append((gamma, beta, value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("gamma", "beta", "expected_qubo_energy"))
        writer.writerows(rows)
    return gammas, betas, landscape


def generate_landscape_figure(
    payload: dict[str, Any],
    states,
    *,
    landscape_path: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "dtu-sciqis-routing-v3-day4-figure8"})
    gammas, betas, landscape = _load_or_compute_landscape(states, output_path=landscape_path)
    figure, axis = plt.subplots(figsize=(11.8, 7.2), facecolor="white")
    image = axis.contourf(gammas, betas, landscape, levels=36, cmap="viridis")
    contours = axis.contour(gammas, betas, landscape, levels=12, colors="white", linewidths=0.35, alpha=0.45)
    axis.clabel(contours, inline=True, fontsize=6, fmt="%.0f")
    colorbar = figure.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label(r"expected QUBO energy $\langle Q_{A=6}\rangle$")

    colors = ("#e4572e", "#2f6f9f", "#7a3db8")
    runs = sorted(
        (run for run in payload["optimization_runs"] if run["A"] == 6 and run["p"] == 1),
        key=lambda run: run["start_id"],
    )
    selected_id = _cell_lookup(payload)[6, 1]["optimization"]["selected_start_id"]
    for run, color in zip(runs, colors):
        trajectory = np.asarray([point["parameters"] for point in run["evaluation_trace"]])
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.1, alpha=0.72, label=f"start {run['start_id']} trajectory")
        axis.scatter([run["initial_parameters"][0]], [run["initial_parameters"][1]], marker="X", s=95, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        final_marker = "*" if run["start_id"] == selected_id else "o"
        final_size = 220 if final_marker == "*" else 70
        axis.scatter([run["final_parameters"][0]], [run["final_parameters"][1]], marker=final_marker, s=final_size, color=color, edgecolor="white", linewidth=0.8, zorder=6)

    grid_minimum = np.unravel_index(np.argmin(landscape), landscape.shape)
    axis.scatter([gammas[grid_minimum[1]]], [betas[grid_minimum[0]]], marker="D", s=75, facecolor="none", edgecolor="white", linewidth=1.5, zorder=6, label="lowest sampled grid point")
    axis.set_xlim(0, 2 * np.pi)
    axis.set_ylim(0, np.pi)
    axis.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    axis.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], ["0", "π/4", "π/2", "3π/4", "π"])
    axis.set_xlabel(r"$\gamma_1$")
    axis.set_ylabel(r"$\beta_1$")
    axis.set_title("p=1 Variational Energy Landscape with Frozen COBYLA Traces", fontsize=17, fontweight="bold", pad=14)
    axis.legend(frameon=False, fontsize=8.5, loc="upper right")
    figure.text(0.5, 0.02, f"A=6 was fixed by the Day-2 just-supercritical rule. The {len(gammas)}×{len(betas)} grid is descriptive and is not claimed to prove a global variational optimum.", ha="center", fontsize=9.5, color="#536170")
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.98))
    _save_figure(figure, png_path, svg_path)


def generate_core_results_figure(payload: dict[str, Any], *, png_path: Path, svg_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "dtu-sciqis-routing-v3-day4-figure9"})
    cells = _cell_lookup(payload)
    penalties = (2, 5, 6, 12)
    uniform = payload["uniform_state_reference"]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor="white")
    styles = {1: ("#e4572e", "o"), 2: ("#2f6f9f", "s")}
    for axis, metric, ylabel in (
        (axes[0], "p_feas", r"valid-route probability $p_{feas}$"),
        (axes[1], "p_opt", r"exact-route probability $p_{opt}$"),
    ):
        for p, (color, marker) in styles.items():
            values = [cells[A, p]["quality"][metric] for A in penalties]
            axis.plot(penalties, values, color=color, marker=marker, markersize=7, linewidth=2, label=f"p={p}")
        reference = uniform[metric]
        axis.axhline(reference, color="#65727e", linestyle="--", linewidth=1.3, label=r"uniform $|+\rangle^{\otimes 14}$ reference")
        axis.axvline(5, color="#7a3db8", linestyle=":", linewidth=1.7)
        axis.text(5.08, 0.95, r"$A_{crit}=5$", transform=axis.get_xaxis_transform(), color="#7a3db8", va="top", fontsize=9)
        axis.set_xticks(penalties)
        axis.set_xlabel("frozen flow-penalty A")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.20)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].legend(frameon=False, loc="upper right")
    axes[0].set_title("Probability of any valid route", fontsize=13, fontweight="bold")
    axes[1].set_title("Probability of the unique shortest route", fontsize=13, fontweight="bold")
    figure.suptitle("Core Penalty × Depth QAOA Results", fontsize=20, fontweight="bold", y=0.98, color="#17212b")
    figure.text(0.5, 0.02, "Exact selected-run data under the frozen minimum-energy rule; lines only connect the four prescribed A values and are not smoothed fits.", ha="center", fontsize=9.5, color="#536170")
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.92))
    _save_figure(figure, png_path, svg_path)


def _draw_marginal_panel(axis, graph, marginals: np.ndarray, *, p: int, common_max: float) -> None:
    exact_edges = {(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)}
    cmap = plt.get_cmap("Blues")
    norm = Normalize(0, common_max)
    for index, edge in enumerate(get_edge_order(graph)):
        marginal = float(marginals[index])
        width = 0.9 + 7.5 * marginal / common_max
        if edge in exact_edges:
            halo = FancyArrowPatch(NODE_POSITIONS[edge[0]], NODE_POSITIONS[edge[1]], arrowstyle="-|>", mutation_scale=13, linewidth=width + 2.6, color="#e4572e", alpha=0.88, shrinkA=20, shrinkB=20, zorder=1)
            axis.add_patch(halo)
        arrow = FancyArrowPatch(NODE_POSITIONS[edge[0]], NODE_POSITIONS[edge[1]], arrowstyle="-|>", mutation_scale=12, linewidth=width, color=cmap(norm(marginal)), alpha=0.95, shrinkA=20, shrinkB=20, zorder=2)
        axis.add_patch(arrow)
        x, y = LABEL_POSITIONS[edge]
        axis.text(x, y, f"q{index}\n{marginal:.3f}", ha="center", va="center", fontsize=6.2, color="#283746", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.45, "alpha": 0.90}, zorder=5)
    for node, (x, y) in NODE_POSITIONS.items():
        face = "#d9f3e4" if node == 0 else "#fde2dd" if node == 6 else "#e8f1fb"
        border = "#18794e" if node == 0 else "#b42318" if node == 6 else "#2f5d8a"
        axis.scatter([x], [y], s=600, color=face, edgecolor=border, linewidth=1.8, zorder=4)
        axis.text(x, y, str(node), ha="center", va="center", fontsize=12, fontweight="bold", zorder=6)
    axis.set_xlim(-0.55, 6.35)
    axis.set_ylim(-0.4, 3.35)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(f"A=6, selected p={p}", fontsize=13, fontweight="bold")


def generate_route_marginal_figure(payload: dict[str, Any], graph, *, png_path: Path, svg_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "dtu-sciqis-routing-v3-day4-figure10"})
    cells = _cell_lookup(payload)
    marginals = {
        p: np.asarray(cells[6, p]["quality"]["edge_marginals_q0_to_q13"])
        for p in (1, 2)
    }
    common_max = max(float(np.max(value)) for value in marginals.values())
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.8), facecolor="white")
    for p, axis in zip((1, 2), axes):
        _draw_marginal_panel(axis, graph, marginals[p], p=p, common_max=common_max)
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, common_max), cmap="Blues")
    colorbar = figure.colorbar(scalar, ax=axes, orientation="horizontal", fraction=0.04, pad=0.055, aspect=45)
    colorbar.set_label(r"edge marginal $m_e=P(x_e=1)$ (common scale)")
    figure.suptitle("Optimized Quantum Edge-Selection Marginals Return to the Routing Graph", fontsize=19, fontweight="bold", y=0.97, color="#17212b")
    figure.text(0.5, 0.018, "Orange outlines mark exact shortest-route edges. Edge marginals are individual edge-selection probabilities; they do not constitute a single sampled valid route.", ha="center", fontsize=9.5, color="#536170")
    figure.subplots_adjust(left=0.025, right=0.98, top=0.87, bottom=0.20, wspace=0.08)
    _save_figure(figure, png_path, svg_path)


def generate_state_space_probability_figure(payload: dict[str, Any], states, *, png_path: Path, svg_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "dtu-sciqis-routing-v3-day4-figure11"})
    cells = _cell_lookup(payload)
    panel_keys = ((2, 1), (2, 2), (6, 1), (6, 2))
    aggregates = {}
    all_probabilities = []
    for key in panel_keys:
        distribution = _selected_distribution(states, cells[key])
        aggregate = defaultdict(float)
        for state, probability in zip(states, distribution):
            aggregate[state.flow_penalty, state.routing_cost] += float(probability)
        aggregates[key] = dict(aggregate)
        all_probabilities.extend(value for value in aggregate.values() if value > 0)
    norm = LogNorm(vmin=max(min(all_probabilities), 1e-8), vmax=max(all_probabilities))
    cmap = plt.get_cmap("viridis")
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), sharex=True, sharey=True, facecolor="white")
    for axis, key in zip(axes.flat, panel_keys):
        aggregate = aggregates[key]
        coordinates = sorted(aggregate)
        values = np.asarray([aggregate[coordinate] for coordinate in coordinates])
        edgecolors = ["#2f6f9f" if coordinate[0] == 0 else "#8a99a8" for coordinate in coordinates]
        axis.scatter(
            [coordinate[0] for coordinate in coordinates],
            [coordinate[1] for coordinate in coordinates],
            s=8 + 1500 * np.sqrt(values),
            c=values,
            cmap=cmap,
            norm=norm,
            edgecolors=edgecolors,
            linewidths=0.6,
            alpha=0.82,
        )
        axis.axvline(0, color="#2f6f9f", linewidth=1.7, alpha=0.65)
        optimum_probability = aggregate.get((0, 10), 0.0)
        critical_probability = aggregate.get((2, 0), 0.0)
        axis.scatter([0], [10], marker="*", s=90 + 1500 * np.sqrt(optimum_probability), color="#e4572e", edgecolor="#8f2f17", linewidth=0.9, zorder=6)
        axis.scatter([2], [0], marker="X", s=65 + 1500 * np.sqrt(critical_probability), color="#7a3db8", edgecolor="white", linewidth=0.7, zorder=6)
        axis.set_title(f"A={key[0]}, p={key[1]}", fontsize=12.5, fontweight="bold")
        axis.grid(alpha=0.15)
        axis.set_xlabel(r"flow penalty $P_{flow}(x)$")
        axis.set_ylabel(r"routing cost $C(x)$")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08, aspect=50)
    colorbar.set_label("aggregate optimized quantum probability at (P_flow, C), common log color scale")
    figure.legend(
        handles=[
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#e4572e", markeredgecolor="#8f2f17", markersize=12, label="exact optimum (0,10)"),
            Line2D([0], [0], color="#2f6f9f", linewidth=2, label="feasible line P_flow=0"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=8.5,
    )
    figure.suptitle("Optimized Quantum Probability on the Frozen State-Space Geometry", fontsize=19, fontweight="bold", y=0.98, color="#17212b")
    figure.text(0.5, 0.055, "Marker area and color use common mappings across panels; probability is aggregated when states share the same (P_flow, C) coordinate.", ha="center", fontsize=9.2, color="#536170")
    figure.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.20, hspace=0.25, wspace=0.15)
    _save_figure(figure, png_path, svg_path)


def build_day4_artifacts(
    *,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """Rebuild Day-4 figures cheaply from saved optimization results."""

    graph_file = Path(graph_path).resolve()
    results = Path(results_dir).resolve()
    figures = Path(figures_dir).resolve()
    if file_sha256(graph_file) != EXPECTED_GRAPH_SHA256:
        raise RuntimeError("frozen graph changed")
    if file_sha256(PROJECT_ROOT / "data" / "penalty_contract.json") != EXPECTED_PENALTY_SHA256:
        raise RuntimeError("frozen penalty contract changed")
    payload = load_core_results(results)
    graph = load_graph(graph_file)
    states = enumerate_state_space(graph)
    generate_landscape_figure(
        payload,
        states,
        landscape_path=results / "p1_landscape_A6.csv",
        png_path=figures / "08_p1_variational_landscape.png",
        svg_path=figures / "08_p1_variational_landscape.svg",
    )
    generate_core_results_figure(
        payload,
        png_path=figures / "09_penalty_depth_core_results.png",
        svg_path=figures / "09_penalty_depth_core_results.svg",
    )
    generate_route_marginal_figure(
        payload,
        graph,
        png_path=figures / "10_quantum_route_marginals.png",
        svg_path=figures / "10_quantum_route_marginals.svg",
    )
    generate_state_space_probability_figure(
        payload,
        states,
        png_path=figures / "11_quantum_probability_state_space.png",
        svg_path=figures / "11_quantum_probability_state_space.svg",
    )
    return {
        "optimization_contract_sha256": payload["optimization_contract"]["sha256"],
        "run_count": len(payload["optimization_runs"]),
        "selected_cell_count": len(payload["selected_cells"]),
        "landscape_grid": "121x81",
        "figures": [8, 9, 10, 11],
    }
