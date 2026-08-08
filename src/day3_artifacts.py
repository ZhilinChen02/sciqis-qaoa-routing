"""Build Day-3 explicit p=1 circuit diagnostics and teaching figures."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import pi
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import qiskit

from circuit import (
    BETA_PARAMETER_NAME,
    GAMMA_PARAMETER_NAME,
    build_p1_qaoa_circuit,
    primitive_gate_counts,
    statevector_at_checkpoints,
)
from exact_reference import compute_exact_reference
from graph import (
    DEFAULT_GRAPH_PATH,
    PROJECT_ROOT,
    edge_order_records,
    edge_vector_to_canonical_bitstring,
    edge_vector_to_qiskit_display_bitstring,
    load_graph,
)
from ising import BIT_TO_Z_CONVENTION, ISING_CONVENTION, qubo_to_ising
from qubo import QUBO_CONVENTION, StateRecord, build_qubo, edge_vector_to_state_index, enumerate_state_space
from statevector_reference import (
    StatevectorComparison,
    compare_statevectors,
    penalty_energies,
    probabilities,
    reference_statevector_at_checkpoints,
    total_variation_distance,
)


EXPECTED_GRAPH_SHA256 = "451a6e4cffd01552c1cb6e4290b5ee8d50fc97514aab8ed47a6b33a111f5e54e"
EXPECTED_PENALTY_CONTRACT_SHA256 = "3e31dc31434dcd06c430e90e75884536cbef8cdfefec195afeb1acc1b77acd57"
EXPECTED_EXACT_PATH = (0, 1, 2, 4, 5, 6)
EXPECTED_EXACT_COST = 10
EXPECTED_CANONICAL_BITSTRING = "10010001000101"
EXPECTED_QISKIT_DISPLAY_BITSTRING = "10100010001001"
EXPECTED_EXACT_STATE_INDEX = 10377
EXPECTED_A_CRIT = 5
EXPECTED_PENALTIES = (
    ("weak", 2),
    ("critical", 5),
    ("just-supercritical", 6),
    ("strong", 12),
)

DIAGNOSTIC_GAMMA = pi / 7
DIAGNOSTIC_BETA = pi / 11
DIAGNOSTIC_LABEL = "DIAGNOSTIC_ONLY_NOT_OPTIMIZED"
VALIDATION_PARAMETER_PAIRS = (
    ("diagnostic_pi_rule", DIAGNOSTIC_GAMMA, DIAGNOSTIC_BETA),
    ("fixed_decimal_pair", 0.173, 0.271),
    ("fixed_signed_pi_rule", -pi / 13, pi / 17),
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _normalize_svg_text(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _comparison_payload(comparison: StatevectorComparison) -> dict[str, float]:
    return {
        "fidelity": comparison.fidelity,
        "max_amplitude_absolute_error": comparison.max_amplitude_absolute_error,
        "reference_norm_error": comparison.reference_norm_error,
        "qiskit_norm_error": comparison.candidate_norm_error,
        "probability_vector_max_error": comparison.probability_vector_max_error,
    }


def _bit_order_metadata(exact_vector: Sequence[int]) -> dict[str, Any]:
    return {
        "edge_vector": {"order": "[x0, x1, ..., x13]", "direction": "q0 -> q13"},
        "canonical_bitstring": {"direction": "q0 -> q13"},
        "qiskit_display_bitstring": {"direction": "q13 -> q0"},
        "basis_state_index": "q0 is the least-significant bit",
        "no_implicit_string_reversal": True,
        "exact_route_example": {
            "edge_vector": list(exact_vector),
            "canonical_bitstring": edge_vector_to_canonical_bitstring(exact_vector),
            "qiskit_display_bitstring": edge_vector_to_qiskit_display_bitstring(exact_vector),
            "state_index": edge_vector_to_state_index(exact_vector),
        },
    }


def _reconcile_identity(graph_path: Path, penalty_path: Path):
    graph_hash = _sha256(graph_path)
    penalty_hash = _sha256(penalty_path)
    if graph_hash != EXPECTED_GRAPH_SHA256:
        raise RuntimeError(f"frozen graph identity changed: {graph_hash}")
    if penalty_hash != EXPECTED_PENALTY_CONTRACT_SHA256:
        raise RuntimeError(f"frozen penalty contract identity changed: {penalty_hash}")

    graph = load_graph(graph_path)
    reference, routes = compute_exact_reference(graph, graph_path=graph_path)
    exact = reference["exact_reference"]
    penalty_contract = json.loads(penalty_path.read_text(encoding="utf-8"))
    penalties = tuple((record["label"], record["A"]) for record in penalty_contract["penalty_values"])
    exact_vector = tuple(int(bit) for bit in exact["edge_bitstring"])
    observed = {
        "path": tuple(exact["node_path"]),
        "cost": exact["cost"],
        "canonical": exact["edge_bitstring"],
        "qiskit_display": edge_vector_to_qiskit_display_bitstring(exact_vector),
        "state_index": edge_vector_to_state_index(exact_vector),
        "A_crit": penalty_contract["A_crit"],
        "penalties": penalties,
    }
    expected = {
        "path": EXPECTED_EXACT_PATH,
        "cost": EXPECTED_EXACT_COST,
        "canonical": EXPECTED_CANONICAL_BITSTRING,
        "qiskit_display": EXPECTED_QISKIT_DISPLAY_BITSTRING,
        "state_index": EXPECTED_EXACT_STATE_INDEX,
        "A_crit": EXPECTED_A_CRIT,
        "penalties": EXPECTED_PENALTIES,
    }
    if observed != expected:
        raise RuntimeError(f"frozen scientific identity mismatch: {observed!r}")
    return graph, reference, routes, penalty_contract, exact_vector, graph_hash, penalty_hash


def _state_payload(
    state: StateRecord,
    probability: float,
    phase: float,
    *,
    penalty: int,
    optimal_index: int,
) -> dict[str, Any]:
    return {
        "state_index": state.state_index,
        "edge_vector": list(state.edge_vector),
        "canonical_bitstring": state.canonical_bitstring,
        "qiskit_display_bitstring": state.qiskit_display_bitstring,
        "routing_cost": state.routing_cost,
        "flow_penalty": state.flow_penalty,
        "qubo_energy": state.routing_cost + penalty * state.flow_penalty,
        "feasible": state.flow_penalty == 0,
        "optimal": state.state_index == optimal_index,
        "probability": float(probability),
        "phase_radians": float(phase),
    }


def _representative_indices(
    states: Sequence[StateRecord],
    post_mixer_probabilities: np.ndarray,
    optimal_index: int,
    *,
    limit: int = 12,
) -> tuple[int, ...]:
    feasible = sorted(
        (state for state in states if state.flow_penalty == 0),
        key=lambda state: (state.routing_cost, state.state_index),
    )
    selected = [optimal_index]
    selected.extend(state.state_index for state in feasible if state.state_index != optimal_index)
    selected = selected[:5]
    if 0 not in selected:
        selected.append(0)
    top_final = sorted(range(len(states)), key=lambda index: (-post_mixer_probabilities[index], index))
    for index in top_final:
        if index not in selected:
            selected.append(index)
        if len(selected) >= limit:
            break
    return tuple(selected)


def circuit_contract_payload(
    graph,
    *,
    graph_hash: str,
    penalty_hash: str,
    exact_vector: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema": "dtu-sciqis-routing-p1-circuit-contract",
        "version": "1.0",
        "graph": {"path": "data/graph.json", "sha256": graph_hash},
        "penalty_contract": {
            "path": "data/penalty_contract.json",
            "sha256": penalty_hash,
            "A_crit": EXPECTED_A_CRIT,
            "frozen_grid": [
                {"label": label, "A": penalty} for label, penalty in EXPECTED_PENALTIES
            ],
        },
        "qubit_count": 14,
        "qubit_mapping": edge_order_records(graph),
        "bit_order_convention": _bit_order_metadata(exact_vector),
        "p": 1,
        "parameters": [GAMMA_PARAMETER_NAME, BETA_PARAMETER_NAME],
        "initial_state": "|+>^14 prepared by H on q0,...,q13",
        "cost_hamiltonian": {
            "convention": ISING_CONVENTION,
            "source_qubo_convention": QUBO_CONVENTION,
            "bit_to_z": BIT_TO_Z_CONVENTION,
        },
        "cost_gate_angle_convention": {
            "qiskit_RZ": "RZ(theta) = exp(-i theta Z / 2)",
            "single_Z_mapping": "exp(-i gamma_1 h_i Z_i) -> RZ_i(2 gamma_1 h_i)",
            "qiskit_RZZ": "RZZ(theta) = exp(-i theta Z⊗Z / 2)",
            "ZZ_mapping": "exp(-i gamma_1 J_ij Z_i Z_j) -> RZZ_ij(2 gamma_1 J_ij)",
            "deterministic_order": "RZ by q_i, then RZZ lexicographically by (i,j)",
        },
        "mixer_hamiltonian": "H_M = sum_i X_i",
        "mixer_gate_angle_convention": {
            "qiskit_RX": "RX(theta) = exp(-i theta X / 2)",
            "mapping": "exp(-i beta_1 X_i) -> RX_i(2 beta_1)",
            "applied_to": "every q_i for i=0,...,13",
        },
        "layer_order": ["H_initialization", "U_C(gamma_1)", "U_M(beta_1)"],
        "identity_global_phase_rule": {
            "physical_identity_gate": "omitted",
            "effect": "c0 I contributes only exp(-i gamma_1 c0), a global phase",
            "validation": "Qiskit and full-energy NumPy states compared up to global phase",
        },
        "diagnostic_parameters": {
            "gamma_1": DIAGNOSTIC_GAMMA,
            "gamma_rule": "pi/7",
            "beta_1": DIAGNOSTIC_BETA,
            "beta_rule": "pi/11",
            "label": DIAGNOSTIC_LABEL,
            "selection_rule": "predetermined constants used identically for all frozen A values",
        },
        "optimization_performed": False,
    }


def _checkpoint_probabilities(
    reference: dict[str, np.ndarray],
    feasible_mask: np.ndarray,
    optimal_index: int,
) -> tuple[dict[str, float], dict[str, float]]:
    feasible = {}
    optimal = {}
    for checkpoint, vector in reference.items():
        distribution = probabilities(vector)
        feasible[checkpoint] = float(distribution[feasible_mask].sum())
        optimal[checkpoint] = float(distribution[optimal_index])
    return feasible, optimal


def build_diagnostic_payload(
    graph,
    states: tuple[StateRecord, ...],
    exact_vector: Sequence[int],
    *,
    graph_hash: str,
    penalty_hash: str,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    optimal_index = edge_vector_to_state_index(exact_vector)
    feasible_mask = np.asarray([state.flow_penalty == 0 for state in states], dtype=bool)
    all_validation = []
    diagnostic_references: dict[int, dict[str, np.ndarray]] = {}
    minimum_fidelity = 1.0
    maximum_amplitude_error = 0.0
    maximum_probability_error = 0.0
    maximum_norm_error = 0.0

    for label, penalty in EXPECTED_PENALTIES:
        hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
        for pair_label, gamma, beta in VALIDATION_PARAMETER_PAIRS:
            qiskit_checkpoints = statevector_at_checkpoints(
                graph, hamiltonian, gamma=gamma, beta=beta
            )
            numpy_checkpoints = reference_statevector_at_checkpoints(
                states, penalty=penalty, gamma=gamma, beta=beta
            )
            checkpoint_payload = {}
            for checkpoint in ("initial", "post_cost", "post_mixer"):
                comparison = compare_statevectors(
                    numpy_checkpoints[checkpoint], qiskit_checkpoints[checkpoint]
                )
                checkpoint_payload[checkpoint] = _comparison_payload(comparison)
                minimum_fidelity = min(minimum_fidelity, comparison.fidelity)
                maximum_amplitude_error = max(
                    maximum_amplitude_error,
                    comparison.max_amplitude_absolute_error,
                )
                maximum_probability_error = max(
                    maximum_probability_error,
                    comparison.probability_vector_max_error,
                )
                maximum_norm_error = max(
                    maximum_norm_error,
                    comparison.reference_norm_error,
                    comparison.candidate_norm_error,
                )
            all_validation.append(
                {
                    "penalty_label": label,
                    "A": penalty,
                    "parameter_pair": pair_label,
                    "gamma": gamma,
                    "beta": beta,
                    "checkpoints": checkpoint_payload,
                }
            )
            if pair_label == "diagnostic_pi_rule":
                diagnostic_references[penalty] = numpy_checkpoints

    if minimum_fidelity < 1.0 - 1e-12:
        raise RuntimeError(f"Qiskit/NumPy fidelity gate failed: {minimum_fidelity}")
    if maximum_amplitude_error > 1e-12 or maximum_probability_error > 1e-14:
        raise RuntimeError("Qiskit/NumPy amplitude or probability gate failed")
    if maximum_norm_error > 1e-12:
        raise RuntimeError(f"statevector normalization gate failed: {maximum_norm_error}")

    diagnostic_records = []
    for label, penalty in EXPECTED_PENALTIES:
        hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
        circuit = build_p1_qaoa_circuit(graph, hamiltonian)
        qiskit_checkpoints = statevector_at_checkpoints(
            graph,
            hamiltonian,
            gamma=DIAGNOSTIC_GAMMA,
            beta=DIAGNOSTIC_BETA,
        )
        reference = diagnostic_references[penalty]
        distributions = {name: probabilities(vector) for name, vector in reference.items()}
        max_cost_change = float(
            np.max(np.abs(distributions["post_cost"] - distributions["initial"]))
        )
        mixer_tv = total_variation_distance(
            distributions["post_cost"], distributions["post_mixer"]
        )
        mixer_max_change = float(
            np.max(np.abs(distributions["post_mixer"] - distributions["post_cost"]))
        )
        cost_phases = np.angle(reference["post_cost"] / reference["initial"])
        unique_cost_phases = len(np.unique(np.round(cost_phases, decimals=12)))
        if max_cost_change > 1e-15:
            raise RuntimeError("cost layer changed computational-basis probabilities")
        if unique_cost_phases <= 1:
            raise RuntimeError("nonzero gamma failed to produce relative cost phases")
        if mixer_tv <= 1e-8:
            raise RuntimeError("mixer failed to redistribute probability")

        p_feas, p_opt = _checkpoint_probabilities(reference, feasible_mask, optimal_index)
        top_indices = sorted(
            range(len(states)),
            key=lambda index: (-distributions["post_mixer"][index], index),
        )[:10]
        representative_indices = _representative_indices(
            states, distributions["post_mixer"], optimal_index, limit=12
        )
        phase_information = []
        for index in representative_indices:
            state = states[index]
            phase_information.append(
                {
                    "state_index": index,
                    "canonical_bitstring": state.canonical_bitstring,
                    "feasible": state.flow_penalty == 0,
                    "optimal": index == optimal_index,
                    "qubo_energy": state.routing_cost + penalty * state.flow_penalty,
                    "checkpoints": {
                        checkpoint: {
                            "probability": float(distributions[checkpoint][index]),
                            "phase_radians": float(np.angle(reference[checkpoint][index])),
                        }
                        for checkpoint in ("initial", "post_cost", "post_mixer")
                    },
                }
            )
        diagnostic_records.append(
            {
                "penalty_label": label,
                "A": penalty,
                "diagnostic_label": DIAGNOSTIC_LABEL,
                "gamma": DIAGNOSTIC_GAMMA,
                "beta": DIAGNOSTIC_BETA,
                "gate_counts": primitive_gate_counts(circuit),
                "norms": {
                    checkpoint: {
                        "numpy": float(np.linalg.norm(reference[checkpoint])),
                        "qiskit": float(np.linalg.norm(qiskit_checkpoints[checkpoint])),
                    }
                    for checkpoint in ("initial", "post_cost", "post_mixer")
                },
                "qiskit_vs_numpy": {
                    checkpoint: _comparison_payload(
                        compare_statevectors(reference[checkpoint], qiskit_checkpoints[checkpoint])
                    )
                    for checkpoint in ("initial", "post_cost", "post_mixer")
                },
                "max_probability_change_after_cost": max_cost_change,
                "cost_relative_phase_unique_count_rounded_12dp": unique_cost_phases,
                "probability_distribution_change_after_mixer": {
                    "total_variation_distance": mixer_tv,
                    "maximum_state_probability_change": mixer_max_change,
                    "changed_state_count_above_1e-15": int(
                        np.count_nonzero(
                            np.abs(distributions["post_mixer"] - distributions["post_cost"])
                            > 1e-15
                        )
                    ),
                },
                "p_feas": p_feas,
                "p_opt": p_opt,
                "top_representative_states_after_mixer": [
                    _state_payload(
                        states[index],
                        distributions["post_mixer"][index],
                        np.angle(reference["post_mixer"][index]),
                        penalty=penalty,
                        optimal_index=optimal_index,
                    )
                    for index in top_indices
                ],
                "selected_state_phase_information": phase_information,
            }
        )

    payload = {
        "schema": "dtu-sciqis-routing-p1-state-evolution-diagnostic",
        "version": "1.0",
        "label": DIAGNOSTIC_LABEL,
        "optimization_performed": False,
        "interpretation_guardrail": "Mechanism visualization only; not a performance or optimization result.",
        "backend": {
            "qiskit_version": qiskit.__version__,
            "qiskit_method": "qiskit.quantum_info.Statevector.from_instruction",
            "independent_reference": "NumPy diagonal energy phase plus pairwise exp(-i beta X) updates",
        },
        "graph_sha256": graph_hash,
        "penalty_contract_sha256": penalty_hash,
        "bit_order_convention": _bit_order_metadata(exact_vector),
        "diagnostic_parameters": {
            "gamma": DIAGNOSTIC_GAMMA,
            "gamma_rule": "pi/7",
            "beta": DIAGNOSTIC_BETA,
            "beta_rule": "pi/11",
            "used_for_every_frozen_penalty": True,
        },
        "hard_equivalence_gate": {
            "penalty_values": [penalty for _label, penalty in EXPECTED_PENALTIES],
            "parameter_pair_count": len(VALIDATION_PARAMETER_PAIRS),
            "checkpoint_count_per_case": 3,
            "comparison_count": len(EXPECTED_PENALTIES) * len(VALIDATION_PARAMETER_PAIRS) * 3,
            "minimum_fidelity": minimum_fidelity,
            "maximum_amplitude_absolute_error": maximum_amplitude_error,
            "maximum_probability_vector_error": maximum_probability_error,
            "maximum_norm_error": maximum_norm_error,
            "passed": True,
            "cases": all_validation,
        },
        "frozen_penalty_diagnostics": diagnostic_records,
        "teaching_message": {
            "cost_layer": "encodes basis-state energy into relative phase without changing basis probabilities",
            "mixer_layer": "converts phase differences into amplitude/probability redistribution by interference",
        },
    }
    return payload, diagnostic_references


def generate_circuit_figure(graph, *, png_path: Path, svg_path: Path) -> None:
    """Draw a compact presentation circuit with explicit gate-angle inset."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day3-figure5",
        }
    )
    figure, axis = plt.subplots(figsize=(16.0, 10.0), facecolor="white")
    axis.set_xlim(-1.4, 9.4)
    axis.set_ylim(-0.8, 15.2)
    axis.axis("off")
    records = edge_order_records(graph)
    y_positions = {index: 14 - index for index in range(14)}

    cost_background = FancyBboxPatch(
        (1.9, 0.45), 3.85, 14.1,
        boxstyle="round,pad=0.12,rounding_size=0.12",
        facecolor="#fff3df", edgecolor="#d58b20", linewidth=2.0, alpha=0.85, zorder=0,
    )
    mixer_background = FancyBboxPatch(
        (6.05, 0.45), 1.35, 14.1,
        boxstyle="round,pad=0.12,rounding_size=0.12",
        facecolor="#e9f2fc", edgecolor="#2f6f9f", linewidth=2.0, alpha=0.88, zorder=0,
    )
    axis.add_patch(cost_background)
    axis.add_patch(mixer_background)

    for record in records:
        index = int(record["qubit_index"])
        y = y_positions[index]
        axis.plot([0.1, 8.8], [y, y], color="#778899", linewidth=1.05, zorder=1)
        axis.text(
            -0.05, y,
            f"q{index} / {record['edge_id']}  {record['u']}→{record['v']}",
            ha="right", va="center", fontsize=8.2, color="#283746",
        )
        h_box = Rectangle((0.35, y - 0.24), 0.48, 0.48, facecolor="#e8f6ee", edgecolor="#18794e", linewidth=1.1, zorder=3)
        axis.add_patch(h_box)
        axis.text(0.59, y, "H", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
        rx_box = Rectangle((6.42, y - 0.24), 0.62, 0.48, facecolor="white", edgecolor="#2f6f9f", linewidth=1.1, zorder=3)
        axis.add_patch(rx_box)
        axis.text(6.73, y, r"$R_X$", ha="center", va="center", fontsize=7.4, fontweight="bold", zorder=4)

    axis.text(0.59, 14.75, r"$|+\rangle^{\otimes 14}$", ha="center", fontsize=11, fontweight="bold", color="#18794e")
    axis.text(3.82, 14.75, r"COST  $U_C(\gamma_1)$", ha="center", fontsize=13, fontweight="bold", color="#9b5d00")
    axis.text(6.73, 14.75, r"MIXER  $U_M(\beta_1)$", ha="center", fontsize=12, fontweight="bold", color="#245b83")

    inset = FancyBboxPatch(
        (2.35, 5.55), 2.95, 3.9,
        boxstyle="round,pad=0.20", facecolor="white", edgecolor="#d9a24f", linewidth=1.4, zorder=5,
    )
    axis.add_patch(inset)
    axis.text(3.825, 8.95, "Explicit commuting Ising gates", ha="center", fontsize=10.5, fontweight="bold", color="#6e4300", zorder=6)
    axis.text(2.62, 8.15, r"$h_i Z_i$", ha="left", fontsize=11, zorder=6)
    axis.text(3.50, 8.15, r"$\longrightarrow\ R_Z(2\gamma_1 h_i)$", ha="left", fontsize=11, zorder=6)
    axis.text(2.62, 7.15, r"$J_{ij} Z_iZ_j$", ha="left", fontsize=11, zorder=6)
    axis.text(3.50, 7.15, r"$\longrightarrow\ R_{ZZ}(2\gamma_1 J_{ij})$", ha="left", fontsize=11, zorder=6)
    axis.text(3.825, 6.25, r"$c_0 I$: omitted gate (global phase only)", ha="center", fontsize=9.2, color="#536170", zorder=6)
    axis.text(6.73, 0.12, r"each gate: $R_X(2\beta_1)$", ha="center", va="center", fontsize=9.2, color="#245b83", zorder=4)

    axis.axvline(8.15, ymin=0.055, ymax=0.94, color="#8a99a8", linestyle="--", linewidth=1.5)
    axis.text(8.48, 7.5, "Statevector /\nmeasurement\nboundary", ha="center", va="center", fontsize=10, color="#536170")
    axis.annotate("", xy=(8.05, 7.5), xytext=(7.55, 7.5), arrowprops={"arrowstyle": "-|>", "color": "#536170", "lw": 1.5})

    figure.suptitle("Explicit p=1 Penalty-X QAOA Circuit", fontsize=23, fontweight="bold", y=0.975, color="#17212b")
    figure.text(0.5, 0.025, "Each qᵢ is the frozen routing edge eᵢ.  Primitive gates implement H → exp(−iγ₁H_C) → exp(−iβ₁ΣXᵢ); no ansatz class or optimizer is used.", ha="center", fontsize=11, color="#536170")
    figure.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.07)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, facecolor="white", metadata={"Software": "sciqis-qaoa-routing Day-3 builder"})
    figure.savefig(svg_path, facecolor="white", metadata={"Creator": "sciqis-qaoa-routing Day-3 builder", "Date": None})
    plt.close(figure)
    _normalize_svg_text(svg_path)


def generate_state_evolution_figure(
    states: tuple[StateRecord, ...],
    reference: dict[str, np.ndarray],
    *,
    penalty: int,
    optimal_index: int,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Show probability as bubble area and phase as cyclic color at three checkpoints."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day3-figure6",
        }
    )
    distributions = {name: probabilities(vector) for name, vector in reference.items()}
    selected = _representative_indices(states, distributions["post_mixer"], optimal_index)
    y = np.arange(len(selected))[::-1]
    uniform_probability = 1.0 / len(states)
    checkpoint_titles = (
        ("initial", "1  Initial $|+\\rangle^{\\otimes14}$\nequal probability, equal phase"),
        ("post_cost", "2  After $U_C(\\gamma)$\nequal probability, energy-dependent phase"),
        ("post_mixer", "3  After $U_M(\\beta)$\ninterference redistributes probability"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.8), sharey=True, facecolor="white")
    cmap = plt.get_cmap("twilight_shifted")
    norm = Normalize(-np.pi, np.pi)

    labels = []
    for index in selected:
        state = states[index]
        if index == optimal_index:
            kind = "OPTIMUM"
        elif index == 0:
            kind = "CRITICAL INVALID"
        elif state.flow_penalty == 0:
            kind = "VALID"
        else:
            kind = "INFEASIBLE"
        labels.append(f"{kind}  |  index {index}  |  E={state.routing_cost + penalty * state.flow_penalty}")

    for axis, (checkpoint, title) in zip(axes, checkpoint_titles):
        distribution = distributions[checkpoint]
        phases = np.angle(reference[checkpoint])
        for row, index in zip(y, selected):
            state = states[index]
            probability = distribution[index]
            size = float(np.clip(150.0 * np.sqrt(probability / uniform_probability), 18.0, 1100.0))
            if index == optimal_index:
                border, linewidth = "#e4572e", 3.0
            elif index == 0:
                border, linewidth = "#7a3db8", 2.5
            elif state.flow_penalty == 0:
                border, linewidth = "#2f6f9f", 2.0
            else:
                border, linewidth = "#65727e", 1.2
            axis.scatter([0], [row], s=size, c=[cmap(norm(phases[index]))], edgecolors=border, linewidths=linewidth, zorder=4)
            axis.text(0.22, row, f"p={probability:.2e}", va="center", fontsize=7.6, color="#405263")
        axis.set_xlim(-0.36, 0.75)
        axis.set_xticks([])
        axis.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right", "bottom"]].set_visible(False)
    axes[0].set_yticks(y, labels, fontsize=7.7)
    axes[0].set_ylabel("selected computational-basis states")
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_axis = figure.add_axes([0.30, 0.145, 0.60, 0.025])
    colorbar = figure.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("amplitude phase angle (radians): cyclic hue", fontsize=9)
    colorbar.set_ticks([-np.pi, 0, np.pi])
    colorbar.set_ticklabels(["−π", "0", "+π"])
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markeredgecolor="#e4572e", markeredgewidth=2.5, label="exact optimum"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="#2f6f9f", markeredgewidth=2.0, label="valid route"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="#7a3db8", markeredgewidth=2.2, label="critical empty state"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="#65727e", label="other infeasible state"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False, fontsize=8.5,
    )
    figure.suptitle("p=1 Quantum-State Evolution: Phase Encoding and Interference", fontsize=21, fontweight="bold", y=0.975, color="#17212b")
    figure.text(0.5, 0.925, f"A={penalty} (just-supercritical from the frozen classical threshold), γ=π/7, β=π/11 — {DIAGNOSTIC_LABEL}", ha="center", fontsize=10.5, color="#536170")
    figure.text(0.5, 0.078, "Bubble area encodes probability; fill hue encodes phase. A=6 was chosen from Day-2 classical threshold logic, not from QAOA performance.", ha="center", fontsize=9.5, color="#536170")
    figure.subplots_adjust(left=0.28, right=0.98, top=0.86, bottom=0.23, wspace=0.18)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, facecolor="white", metadata={"Software": "sciqis-qaoa-routing Day-3 builder"})
    figure.savefig(svg_path, facecolor="white", metadata={"Creator": "sciqis-qaoa-routing Day-3 builder", "Date": None})
    plt.close(figure)
    _normalize_svg_text(svg_path)


def generate_cost_phase_figure(
    states: tuple[StateRecord, ...],
    *,
    penalty: int,
    gamma: float,
    optimal_index: int,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Show the deterministic basis-energy to cost-phase map."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day3-figure7",
        }
    )
    energies = penalty_energies(states, penalty)
    phases = np.angle(np.exp(-1j * gamma * energies))
    occupancy = Counter(
        (float(energies[state.state_index]), state.flow_penalty == 0)
        for state in states
    )
    figure, axis = plt.subplots(figsize=(12.5, 6.3), facecolor="white")
    infeasible = sorted(key for key in occupancy if not key[1])
    feasible = sorted(key for key in occupancy if key[1])
    for keys, color, marker, label, alpha in (
        (infeasible, "#9aa5b1", "o", "infeasible selections", 0.50),
        (feasible, "#2f6f9f", "D", "valid routes", 0.90),
    ):
        axis.scatter(
            [key[0] for key in keys],
            [float(np.angle(np.exp(-1j * gamma * key[0]))) for key in keys],
            s=[18 + 18 * np.log1p(occupancy[key]) for key in keys],
            color=color,
            marker=marker,
            alpha=alpha,
            edgecolors="white" if marker == "D" else "none",
            linewidths=0.5,
            label=label,
        )
    optimum_energy = float(energies[optimal_index])
    optimum_phase = float(phases[optimal_index])
    axis.scatter([optimum_energy], [optimum_phase], marker="*", s=260, color="#e4572e", edgecolor="#8f2f17", linewidth=1.0, zorder=5, label="exact optimum")
    axis.annotate(f"C*=E={optimum_energy:g}\nphase={optimum_phase:.3f}", xy=(optimum_energy, optimum_phase), xytext=(optimum_energy + 18, optimum_phase + 0.55), arrowprops={"arrowstyle": "->", "color": "#8f2f17"}, fontsize=9, color="#8f2f17")
    axis.set_xlabel(r"QUBO basis energy $Q_A(x)$")
    axis.set_ylabel(r"cost-layer phase $\phi_x=\arg[e^{-i\gamma Q_A(x)}]$")
    axis.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "+π/2", "+π"])
    axis.set_ylim(-np.pi - 0.15, np.pi + 0.15)
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, loc="upper right")
    axis.set_title("Cost Layer Encodes QUBO Energy into Basis-State Phase", fontsize=17, fontweight="bold", pad=15)
    figure.text(0.5, 0.02, f"A={penalty}, γ=π/7 — {DIAGNOSTIC_LABEL}. Marker area indicates energy-level multiplicity, not measurement probability.", ha="center", fontsize=9.5, color="#536170")
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, facecolor="white", metadata={"Software": "sciqis-qaoa-routing Day-3 builder"})
    figure.savefig(svg_path, facecolor="white", metadata={"Creator": "sciqis-qaoa-routing Day-3 builder", "Date": None})
    plt.close(figure)
    _normalize_svg_text(svg_path)


def build_day3_artifacts(
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """Regenerate all authorized Day-3 p=1 diagnostic artifacts."""

    graph_file = Path(graph_path).resolve()
    data = Path(data_dir).resolve()
    results = Path(results_dir).resolve()
    figures = Path(figures_dir).resolve()
    penalty_path = data / "penalty_contract.json"
    (
        graph,
        _reference,
        _routes,
        _penalty_contract,
        exact_vector,
        graph_hash,
        penalty_hash,
    ) = _reconcile_identity(graph_file, penalty_path)
    states = enumerate_state_space(graph)

    contract = circuit_contract_payload(
        graph,
        graph_hash=graph_hash,
        penalty_hash=penalty_hash,
        exact_vector=exact_vector,
    )
    diagnostic, diagnostic_references = build_diagnostic_payload(
        graph,
        states,
        exact_vector,
        graph_hash=graph_hash,
        penalty_hash=penalty_hash,
    )
    _write_json(data / "circuit_contract.json", contract)
    _write_json(results / "p1_state_evolution_diagnostic.json", diagnostic)

    generate_circuit_figure(
        graph,
        png_path=figures / "05_explicit_p1_qaoa_circuit.png",
        svg_path=figures / "05_explicit_p1_qaoa_circuit.svg",
    )
    generate_state_evolution_figure(
        states,
        diagnostic_references[6],
        penalty=6,
        optimal_index=EXPECTED_EXACT_STATE_INDEX,
        png_path=figures / "06_p1_quantum_state_evolution.png",
        svg_path=figures / "06_p1_quantum_state_evolution.svg",
    )
    generate_cost_phase_figure(
        states,
        penalty=6,
        gamma=DIAGNOSTIC_GAMMA,
        optimal_index=EXPECTED_EXACT_STATE_INDEX,
        png_path=figures / "07_cost_phase_encoding.png",
        svg_path=figures / "07_cost_phase_encoding.svg",
    )
    hard_gate = diagnostic["hard_equivalence_gate"]
    return {
        "label": DIAGNOSTIC_LABEL,
        "optimization_performed": False,
        "qiskit_version": qiskit.__version__,
        "graph_sha256": graph_hash,
        "penalty_contract_sha256": penalty_hash,
        "gamma": DIAGNOSTIC_GAMMA,
        "beta": DIAGNOSTIC_BETA,
        "minimum_fidelity": hard_gate["minimum_fidelity"],
        "maximum_amplitude_absolute_error": hard_gate["maximum_amplitude_absolute_error"],
        "maximum_probability_vector_error": hard_gate["maximum_probability_vector_error"],
        "penalty_values": [penalty for _label, penalty in EXPECTED_PENALTIES],
    }
