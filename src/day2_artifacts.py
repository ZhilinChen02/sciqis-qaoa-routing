"""Build the authorized Day-2 QUBO, Ising, and scientific figure artifacts."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np

from day1_artifacts import LABEL_POSITIONS, NODE_POSITIONS
from exact_reference import compute_exact_reference
from graph import (
    DEFAULT_GRAPH_PATH,
    PROJECT_ROOT,
    canonical_bitstring_to_edge_vector,
    edge_order_records,
    edge_vector_to_canonical_bitstring,
    edge_vector_to_qiskit_display_bitstring,
    get_edge_order,
    load_graph,
)
from ising import (
    BIT_TO_Z_CONVENTION,
    ISING_CONVENTION,
    IsingHamiltonian,
    max_qubo_ising_error,
    qubo_to_ising,
)
from qubo import (
    FLOW_DEFINITION,
    QUBO_CONVENTION,
    PenaltyChoice,
    QuboPolynomial,
    StateRecord,
    build_qubo,
    derive_critical_penalty,
    enumerate_state_space,
    flow_equation_strings,
    flow_feasibility_verdict,
    flow_penalty_coefficients,
    frozen_penalty_grid,
    incidence_matrix,
    max_qubo_expansion_error,
    minimum_energy_states,
    node_supplies,
)


EXPECTED_GRAPH_SHA256 = "451a6e4cffd01552c1cb6e4290b5ee8d50fc97514aab8ed47a6b33a111f5e54e"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _fraction_record(value: Fraction) -> dict[str, int | float | str]:
    value = Fraction(value)
    return {
        "exact": str(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": int(value) if value.denominator == 1 else float(value),
    }


def _json_number(value: Fraction) -> int | float:
    value = Fraction(value)
    return int(value) if value.denominator == 1 else float(value)


def bit_order_metadata(exact_vector: Iterable[int] | None = None) -> dict[str, Any]:
    """Return the machine-readable three-representation bit-order contract."""

    metadata: dict[str, Any] = {
        "edge_vector": {
            "description": "[x0, x1, ..., x13] in frozen edge/qubit order",
            "order": "q0 -> q13",
        },
        "canonical_bitstring": {
            "description": "textual edge vector",
            "order": "q0 -> q13",
        },
        "qiskit_display_bitstring": {
            "description": "Qiskit-style classical display text when needed",
            "order": "q13 -> q0",
        },
        "state_index": {
            "range": "0..16383",
            "rule": "q0 is the least-significant integer bit",
        },
        "no_implicit_reversal": True,
    }
    if exact_vector is not None:
        vector = tuple(exact_vector)
        metadata["exact_route_example"] = {
            "edge_vector": list(vector),
            "canonical_bitstring": edge_vector_to_canonical_bitstring(vector),
            "qiskit_display_bitstring": edge_vector_to_qiskit_display_bitstring(vector),
        }
    return metadata


def _polynomial_payload(polynomial: QuboPolynomial) -> dict[str, Any]:
    return {
        "constant": _fraction_record(polynomial.constant),
        "linear": [
            {"qubit_index": index, "coefficient": _fraction_record(coefficient)}
            for index, coefficient in enumerate(polynomial.linear)
        ],
        "pair": [
            {
                "i": i,
                "j": j,
                "coefficient": _fraction_record(coefficient),
            }
            for (i, j), coefficient in sorted(polynomial.pair.items())
        ],
    }


def _ising_payload(hamiltonian: IsingHamiltonian) -> dict[str, Any]:
    return {
        "constant": _fraction_record(hamiltonian.constant),
        "h": [
            {"qubit_index": index, "coefficient": _fraction_record(coefficient)}
            for index, coefficient in enumerate(hamiltonian.h)
        ],
        "J": [
            {
                "i": i,
                "j": j,
                "coefficient": _fraction_record(coefficient),
            }
            for (i, j), coefficient in sorted(hamiltonian.coupling.items())
        ],
    }


def _state_payload(graph: nx.DiGraph, state: StateRecord) -> dict[str, Any]:
    ordered_edges = get_edge_order(graph)
    return {
        "state_index": state.state_index,
        "edge_vector": list(state.edge_vector),
        "canonical_bitstring": state.canonical_bitstring,
        "qiskit_display_bitstring": state.qiskit_display_bitstring,
        "selected_edges": [
            {
                "qubit_index": index,
                "u": ordered_edges[index][0],
                "v": ordered_edges[index][1],
                "weight": graph.edges[ordered_edges[index]]["weight"],
            }
            for index, bit in enumerate(state.edge_vector)
            if bit
        ],
        "routing_cost": state.routing_cost,
        "flow_residuals": {
            str(node): residual
            for node, residual in zip(sorted(graph.nodes), state.flow_residuals)
        },
        "flow_penalty": state.flow_penalty,
        "decoder_valid": state.is_decoder_valid,
        "decoded_route": list(state.decoded_route) if state.decoded_route else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalize_svg_text(path: Path) -> None:
    """Remove backend whitespace noise while preserving deterministic SVG XML."""

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _draw_routing_panel(axis: plt.Axes, graph: nx.DiGraph) -> None:
    for index, edge in enumerate(get_edge_order(graph)):
        arrow = FancyArrowPatch(
            NODE_POSITIONS[edge[0]],
            NODE_POSITIONS[edge[1]],
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.05,
            color="#8190a0",
            alpha=0.72,
            connectionstyle="arc3,rad=0.0",
            shrinkA=12,
            shrinkB=12,
            zorder=1,
        )
        axis.add_patch(arrow)
        label_x, label_y = LABEL_POSITIONS[edge]
        axis.text(
            label_x,
            label_y,
            f"q{index} ({graph.edges[edge]['weight']})",
            fontsize=6.2,
            ha="center",
            va="center",
            color="#405263",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9},
            zorder=4,
        )
    for node, (x, y) in NODE_POSITIONS.items():
        if node == graph.graph["source"]:
            face, border = "#d9f3e4", "#18794e"
        elif node == graph.graph["target"]:
            face, border = "#fde2dd", "#b42318"
        else:
            face, border = "#e8f1fb", "#2f5d8a"
        axis.scatter([x], [y], s=430, color=face, edgecolor=border, linewidth=1.5, zorder=3)
        axis.text(x, y, str(node), ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)
    axis.set_xlim(-0.5, 6.25)
    axis.set_ylim(-0.35, 3.25)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("Frozen routing graph\nedge qubits q0–q13 (weight)", fontsize=12, fontweight="bold", pad=8)


def generate_structure_figure(
    graph: nx.DiGraph,
    flow_qubo: QuboPolynomial,
    *,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Show graph topology, flow-QUBO couplings, and Ising interaction topology."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day2-figure3",
        }
    )
    figure = plt.figure(figsize=(16.0, 6.3), facecolor="white")
    grid = figure.add_gridspec(1, 3, width_ratios=(1.25, 1.0, 1.05), wspace=0.28)
    graph_axis = figure.add_subplot(grid[0, 0])
    heat_axis = figure.add_subplot(grid[0, 1])
    interaction_axis = figure.add_subplot(grid[0, 2])

    _draw_routing_panel(graph_axis, graph)

    matrix = np.zeros((14, 14), dtype=float)
    for index, coefficient in enumerate(flow_qubo.linear):
        matrix[index, index] = float(coefficient)
    for (i, j), coefficient in flow_qubo.pair.items():
        matrix[i, j] = matrix[j, i] = float(coefficient)
    image = heat_axis.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
    heat_axis.set_xticks(range(14), [f"q{i}" for i in range(14)], rotation=90, fontsize=7)
    heat_axis.set_yticks(range(14), [f"q{i}" for i in range(14)], fontsize=7)
    heat_axis.set_title("Flow-penalty coefficients\nmirrored interaction view", fontsize=12, fontweight="bold", pad=8)
    heat_axis.set_xlabel("edge qubit")
    heat_axis.set_ylabel("edge qubit")
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.035, ticks=(-2, 0, 2))
    colorbar.set_label("coefficient in $P_{flow}$", fontsize=8)

    interaction = nx.Graph()
    interaction.add_nodes_from(range(14))
    interaction.add_edges_from(flow_qubo.pair)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 14, endpoint=False)
    positions = {index: (np.cos(angle), np.sin(angle)) for index, angle in enumerate(angles)}
    positive = [pair for pair, value in flow_qubo.pair.items() if value > 0]
    negative = [pair for pair, value in flow_qubo.pair.items() if value < 0]
    nx.draw_networkx_edges(interaction, positions, edgelist=positive, edge_color="#4477aa", width=1.0, alpha=0.35, ax=interaction_axis)
    nx.draw_networkx_edges(interaction, positions, edgelist=negative, edge_color="#d65f5f", width=1.15, alpha=0.42, ax=interaction_axis)
    nx.draw_networkx_nodes(interaction, positions, node_size=490, node_color="#f4f7fb", edgecolors="#31475e", linewidths=1.25, ax=interaction_axis)
    nx.draw_networkx_labels(interaction, positions, labels={i: f"q{i}" for i in range(14)}, font_size=8, font_weight="bold", ax=interaction_axis)
    interaction_axis.set_aspect("equal")
    interaction_axis.axis("off")
    interaction_axis.set_title("Nonzero Ising $J_{ij}$ topology\n(sign inherited from flow coupling)", fontsize=12, fontweight="bold", pad=8)
    interaction_axis.legend(
        handles=[
            Line2D([0], [0], color="#4477aa", lw=2, label="positive $J_{ij}$"),
            Line2D([0], [0], color="#d65f5f", lw=2, label="negative $J_{ij}$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    figure.suptitle("Frozen Routing Model → Flow-QUBO → Ising Interactions", fontsize=21, fontweight="bold", y=0.98, color="#17212b")
    figure.text(0.5, 0.02, "The flow-interaction pattern is A-independent; A scales coupling magnitudes while routing weights shift QUBO linear terms.", ha="center", fontsize=10, color="#536170")
    figure.subplots_adjust(top=0.84, bottom=0.12, left=0.025, right=0.985)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, facecolor="white", metadata={"Software": "sciqis-qaoa-routing Day-2 builder"})
    figure.savefig(svg_path, facecolor="white", metadata={"Creator": "sciqis-qaoa-routing Day-2 builder", "Date": None})
    plt.close(figure)
    _normalize_svg_text(svg_path)


def _bubble_sizes(counts: Iterable[int], base: float = 12.0) -> np.ndarray:
    return np.asarray([base + 16.0 * np.log1p(count) for count in counts])


def generate_penalty_geometry_figure(
    states: tuple[StateRecord, ...],
    critical_states: tuple[StateRecord, ...],
    grid: tuple[PenaltyChoice, ...],
    *,
    optimal_cost: int,
    critical_penalty: Fraction,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Visualize all 16384 states in ``(P_flow,C)`` and penalized-energy space."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day2-figure4",
        }
    )
    figure = plt.figure(figsize=(16.0, 8.7), facecolor="white")
    layout = figure.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0), wspace=0.28, hspace=0.35)
    geometry_axis = figure.add_subplot(layout[:, 0])
    energy_axes = [
        figure.add_subplot(layout[0, 1]),
        figure.add_subplot(layout[0, 2]),
        figure.add_subplot(layout[1, 1]),
        figure.add_subplot(layout[1, 2]),
    ]

    occupancy = Counter((state.flow_penalty, state.routing_cost) for state in states)
    points = sorted(occupancy)
    p_values = np.asarray([point[0] for point in points])
    costs = np.asarray([point[1] for point in points])
    multiplicities = np.asarray([occupancy[point] for point in points])
    scatter = geometry_axis.scatter(
        p_values,
        costs,
        s=_bubble_sizes(multiplicities, base=10),
        c=np.log10(multiplicities),
        cmap="viridis",
        alpha=0.70,
        edgecolors="white",
        linewidths=0.35,
    )
    geometry_axis.axvline(0, color="#2f6f9f", lw=2.0, alpha=0.65, label="$P_{flow}=0$ feasible line")
    geometry_axis.scatter([0], [optimal_cost], marker="*", s=270, color="#e4572e", edgecolor="#8f2f17", zorder=6, label=f"exact route, $C^*={optimal_cost}$")
    for index, state in enumerate(critical_states):
        geometry_axis.scatter([state.flow_penalty], [state.routing_cost], marker="X", s=170, color="#7a3db8", edgecolor="white", linewidth=0.8, zorder=7, label="critical infeasible state" if index == 0 else None)
    line_p = np.linspace(0, float(Fraction(optimal_cost, 1) / critical_penalty), 80)
    geometry_axis.plot(line_p, optimal_cost - float(critical_penalty) * line_p, color="#7a3db8", linestyle="--", linewidth=1.7, label=f"tie boundary: $C=C^*-{_json_number(critical_penalty)}P$")
    geometry_axis.set_title("All 16,384 edge selections", fontsize=13, fontweight="bold", pad=10)
    geometry_axis.set_xlabel("flow penalty $P_{flow}(x)$")
    geometry_axis.set_ylabel("raw routing cost $C(x)$")
    geometry_axis.grid(alpha=0.18)
    geometry_axis.legend(loc="upper left", frameon=False, fontsize=8)
    colorbar = figure.colorbar(scatter, ax=geometry_axis, fraction=0.045, pad=0.03)
    colorbar.set_label("log10 multiplicity", fontsize=8)

    for axis, choice in zip(energy_axes, grid):
        penalty = choice.value
        grouped = Counter(
            (
                state.flow_penalty,
                Fraction(state.routing_cost) + penalty * state.flow_penalty,
                state.is_decoder_valid,
            )
            for state in states
        )
        infeasible = sorted(key for key in grouped if not key[2])
        feasible = sorted(key for key in grouped if key[2])
        axis.scatter(
            [key[0] for key in infeasible],
            [float(key[1]) for key in infeasible],
            s=_bubble_sizes([grouped[key] for key in infeasible], base=5),
            color="#aab4bf",
            alpha=0.48,
            edgecolors="none",
            label="infeasible",
        )
        axis.scatter(
            [key[0] for key in feasible],
            [float(key[1]) for key in feasible],
            s=_bubble_sizes([grouped[key] for key in feasible], base=11),
            color="#2f6f9f",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            label="valid route",
        )
        for state in critical_states:
            energy = Fraction(state.routing_cost) + penalty * state.flow_penalty
            axis.scatter([state.flow_penalty], [float(energy)], marker="X", s=105, color="#7a3db8", edgecolor="white", linewidth=0.6, zorder=6)
        axis.scatter([0], [optimal_cost], marker="*", s=160, color="#e4572e", edgecolor="#8f2f17", zorder=7)
        axis.axhline(optimal_cost, color="#e4572e", linestyle="--", linewidth=1.1, alpha=0.75)
        minimum, minimizers = minimum_energy_states(states, penalty)
        if all(state.is_decoder_valid for state in minimizers):
            validity = "valid"
        elif any(state.is_decoder_valid for state in minimizers):
            validity = "valid/invalid tie"
        else:
            validity = "invalid"
        axis.text(0.97, 0.92, f"min Q = {_json_number(minimum)} ({validity})", transform=axis.transAxes, ha="right", va="top", fontsize=8.5, color="#384a5a", bbox={"facecolor": "white", "edgecolor": "#d9e0e6", "boxstyle": "round,pad=0.25", "alpha": 0.92})
        axis.set_title(f"{choice.label}: A = {_json_number(penalty)}", fontsize=11.5, fontweight="bold")
        axis.set_xlabel("$P_{flow}(x)$")
        axis.set_ylabel("$Q_A(x)=C(x)+A P_{flow}(x)$")
        axis.set_ylim(-1, 36)
        axis.grid(alpha=0.16)

    energy_axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)
    figure.suptitle("Penalty Geometry of the Complete Classical State Space", fontsize=21, fontweight="bold", y=0.975, color="#17212b")
    figure.text(0.5, 0.02, f"At Acrit = {_json_number(critical_penalty)}, the empty selection ties C*. Any A > Acrit lifts every infeasible state above the unique shortest route.", ha="center", fontsize=10.5, color="#536170")
    figure.subplots_adjust(top=0.89, bottom=0.10, left=0.055, right=0.975)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, facecolor="white", metadata={"Software": "sciqis-qaoa-routing Day-2 builder"})
    figure.savefig(svg_path, facecolor="white", metadata={"Creator": "sciqis-qaoa-routing Day-2 builder", "Date": None})
    plt.close(figure)
    _normalize_svg_text(svg_path)


def build_day2_artifacts(
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Regenerate every authorized Day-2 artifact from the immutable graph."""

    graph_file = Path(graph_path).resolve()
    results = Path(results_dir).resolve()
    figures = Path(figures_dir).resolve()
    data = Path(data_dir).resolve()
    graph_sha256 = hashlib.sha256(graph_file.read_bytes()).hexdigest()
    if graph_sha256 != EXPECTED_GRAPH_SHA256:
        raise RuntimeError(
            "immutable Day-1 graph SHA-256 changed: "
            f"expected {EXPECTED_GRAPH_SHA256}, got {graph_sha256}"
        )

    graph = load_graph(graph_file)
    reference, all_routes = compute_exact_reference(graph, graph_path=graph_file)
    exact_cost = int(reference["exact_reference"]["cost"])
    exact_vector = canonical_bitstring_to_edge_vector(
        str(reference["exact_reference"]["edge_bitstring"])
    )
    states = enumerate_state_space(graph)
    verdict = flow_feasibility_verdict(states)
    if not verdict["sets_identical"]:
        raise RuntimeError("P_flow=0 and decoder-valid state sets are not identical")
    if verdict["decoder_valid_route_count"] != len(all_routes):
        raise RuntimeError("Day-2 decoder route count disagrees with Day-1 census")

    critical_penalty, critical_states = derive_critical_penalty(states, exact_cost)
    grid = frozen_penalty_grid(critical_penalty)
    penalties = tuple(choice.value for choice in grid)
    qubos = tuple(build_qubo(graph, penalty) for penalty in penalties)
    flow_qubo = flow_penalty_coefficients(graph)
    qubo_error = max_qubo_expansion_error(graph, states, penalties)
    ising_error = max_qubo_ising_error(states, qubos)
    if qubo_error != 0:
        raise RuntimeError(f"direct and coefficient QUBO values differ by {qubo_error}")
    if ising_error != 0:
        raise RuntimeError(f"QUBO and Ising basis energies differ by {ising_error}")

    minima = []
    for choice in grid:
        minimum, minimizers = minimum_energy_states(states, choice.value)
        minima.append(
            {
                "label": choice.label,
                "A": _json_number(choice.value),
                "minimum_energy": _fraction_record(minimum),
                "minimizer_count": len(minimizers),
                "all_minimizers_decoder_valid": all(state.is_decoder_valid for state in minimizers),
                "minimizers": [_state_payload(graph, state) for state in minimizers],
            }
        )
    exact_state = next(state for state in states if state.edge_vector == exact_vector)
    if grid[0].value >= critical_penalty or minima[0]["minimum_energy"]["value"] >= exact_cost:
        raise RuntimeError("weak grid point does not expose a cheaper infeasible state")
    if minima[1]["minimum_energy"]["value"] != exact_cost or minima[1]["minimizer_count"] < 2:
        raise RuntimeError("critical grid point does not exhibit the expected tie")
    for record in minima[2:]:
        minimizers = record["minimizers"]
        if len(minimizers) != 1 or minimizers[0]["state_index"] != exact_state.state_index:
            raise RuntimeError("supercritical penalty does not uniquely recover the exact route")

    common_graph = {
        "path": "data/graph.json",
        "sha256": graph_sha256,
        "source": graph.graph["source"],
        "target": graph.graph["target"],
        "edge_order": edge_order_records(graph),
    }
    bit_order = bit_order_metadata(exact_vector)
    relevant_count = sum(
        state.flow_penalty > 0 and state.routing_cost < exact_cost for state in states
    )
    threshold_payload = {
        "schema": "dtu-sciqis-routing-penalty-threshold-analysis",
        "version": "1.0",
        "graph": common_graph,
        "bit_order_convention": bit_order,
        "state_space_size": len(states),
        "exact_feasible_optimum": {
            "cost": exact_cost,
            "state": _state_payload(graph, exact_state),
        },
        "flow_feasibility_equivalence": verdict,
        "day1_simple_route_count": len(all_routes),
        "relevant_infeasible_state_count": relevant_count,
        "crossing_formula": "A_x = (C* - C(x)) / P_flow(x) for infeasible states with C(x) < C*",
        "A_crit": _fraction_record(critical_penalty),
        "critical_infeasible_state_count": len(critical_states),
        "critical_infeasible_states": [
            {
                **_state_payload(graph, state),
                "crossing_A": _fraction_record(
                    Fraction(exact_cost - state.routing_cost, state.flow_penalty)
                ),
            }
            for state in critical_states
        ],
        "frozen_grid_minimum_checks": minima,
        "interpretation": {
            "below": "For A < A_crit, at least one infeasible state has energy below C*.",
            "at": "At A = A_crit, at least one infeasible state ties the true optimal route.",
            "above": "For A > A_crit, exhaustive verification gives the true route as unique global minimizer.",
        },
    }
    _write_json(results / "penalty_threshold_analysis.json", threshold_payload)

    derivation_rule = (
        "weak=floor(A_crit/2); critical=A_crit; "
        "just-supercritical=floor(A_crit)+1; strong=2*(just-supercritical)"
    )
    contract_payload = {
        "schema": "dtu-sciqis-routing-penalty-contract",
        "version": "1.0",
        "graph_sha256": graph_sha256,
        "exact_C_star": exact_cost,
        "A_crit": _json_number(critical_penalty),
        "A_crit_exact": str(critical_penalty),
        "derivation_rule": derivation_rule,
        "penalty_values": [
            {"label": choice.label, "A": _json_number(choice.value), "A_exact": str(choice.value)}
            for choice in grid
        ],
        "bit_order_convention": bit_order,
        "statement": "Frozen before any QAOA result was observed.",
    }
    _write_json(data / "penalty_contract.json", contract_payload)

    flow_payload = _polynomial_payload(flow_qubo)
    qubo_payload = {
        "schema": "dtu-sciqis-routing-qubo-coefficients",
        "version": "1.0",
        "graph": common_graph,
        "bit_order_convention": bit_order,
        "coefficient_convention": QUBO_CONVENTION,
        "flow_definition": FLOW_DEFINITION,
        "flow_equations": list(flow_equation_strings(graph)),
        "node_order": list(sorted(graph.nodes)),
        "node_supplies_b": {str(node): value for node, value in node_supplies(graph).items()},
        "incidence_matrix_a": [list(row) for row in incidence_matrix(graph)],
        "expanded_flow_penalty": flow_payload,
        "records": [
            {
                "label": choice.label,
                "penalty_A": _fraction_record(choice.value),
                "Q_A": _polynomial_payload(qubo),
            }
            for choice, qubo in zip(grid, qubos)
        ],
        "exhaustive_validation": {
            "state_count": len(states),
            "penalty_count": len(grid),
            "comparisons": len(states) * len(grid),
            "direct_equals_coefficient_polynomial": True,
            "maximum_absolute_error": _fraction_record(qubo_error),
        },
    }
    _write_json(results / "qubo_coefficients.json", qubo_payload)

    ising_payload = {
        "schema": "dtu-sciqis-routing-ising-coefficients",
        "version": "1.0",
        "graph": common_graph,
        "bit_order_convention": bit_order,
        "qubo_coefficient_convention": QUBO_CONVENTION,
        "ising_coefficient_convention": ISING_CONVENTION,
        "bit_to_z_convention": BIT_TO_Z_CONVENTION,
        "records": [
            {
                "label": choice.label,
                "penalty_A": _fraction_record(choice.value),
                "H_C": _ising_payload(qubo_to_ising(qubo)),
            }
            for choice, qubo in zip(grid, qubos)
        ],
        "exhaustive_validation": {
            "state_count": len(states),
            "penalty_count": len(grid),
            "comparisons": len(states) * len(grid),
            "qubo_equals_computational_basis_ising_energy": True,
            "maximum_absolute_error": _fraction_record(ising_error),
        },
    }
    _write_json(results / "ising_coefficients.json", ising_payload)

    generate_structure_figure(
        graph,
        flow_qubo,
        png_path=figures / "03_graph_qubo_ising_structure.png",
        svg_path=figures / "03_graph_qubo_ising_structure.svg",
    )
    generate_penalty_geometry_figure(
        states,
        critical_states,
        grid,
        optimal_cost=exact_cost,
        critical_penalty=critical_penalty,
        png_path=figures / "04_penalty_state_space_geometry.png",
        svg_path=figures / "04_penalty_state_space_geometry.svg",
    )

    return {
        "graph_sha256": graph_sha256,
        "state_count": len(states),
        "flow_penalty_zero_count": verdict["flow_penalty_zero_count"],
        "decoder_valid_route_count": verdict["decoder_valid_route_count"],
        "flow_decoder_sets_identical": verdict["sets_identical"],
        "A_crit": _json_number(critical_penalty),
        "critical_infeasible_state_count": len(critical_states),
        "penalty_grid": [
            {"label": choice.label, "A": _json_number(choice.value)} for choice in grid
        ],
        "qubo_max_absolute_error": float(qubo_error),
        "ising_max_absolute_error": float(ising_error),
    }
