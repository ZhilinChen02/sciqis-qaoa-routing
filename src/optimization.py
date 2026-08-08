"""Frozen Day-4 exact-statevector optimization and primary route metrics."""

from __future__ import annotations

import csv
import hashlib
import json
from math import pi
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, minimize

from circuit import build_qaoa_circuit, primitive_gate_counts, qaoa_statevector
from graph import PROJECT_ROOT, get_edge_order
from ising import qubo_to_ising
from qubo import StateRecord, build_qubo, enumerate_state_space
from statevector_reference import (
    compare_statevectors,
    penalty_energies,
    probabilities,
    reference_qaoa_from_energies,
    reference_qaoa_statevector,
)


EXPECTED_OPTIMIZATION_CONTRACT_SHA256 = "12d44320e83ce06ae9172c6d68bd8b0f617bc4fa2b9273418fc0e8bec2ecba3d"
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "data" / "optimization_contract.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_optimization_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[str, Any], str]:
    """Load and strictly verify the protocol frozen before final runs."""

    contract_path = Path(path).resolve()
    contract_hash = file_sha256(contract_path)
    if contract_hash != EXPECTED_OPTIMIZATION_CONTRACT_SHA256:
        raise RuntimeError(
            "optimization contract changed after freeze: "
            f"expected {EXPECTED_OPTIMIZATION_CONTRACT_SHA256}, got {contract_hash}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["optimizer"]["scipy_version"] != scipy.__version__:
        raise RuntimeError("active SciPy version differs from frozen optimization contract")
    if contract["statement"] != (
        "The optimization protocol was frozen before the final core experiment was executed."
    ):
        raise RuntimeError("optimization freeze statement is missing")
    if contract["experimental_matrix"]["total_optimization_runs"] != 24:
        raise RuntimeError("frozen experiment must contain exactly 24 runs")
    return contract, contract_hash


def split_interleaved_parameters(
    parameters: Sequence[float], p: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Split ``[gamma_1,beta_1,...]`` without changing the frozen order."""

    values = tuple(float(value) for value in parameters)
    if p not in (1, 2) or len(values) != 2 * p:
        raise ValueError("parameter vector must have length 2*p for p in {1,2}")
    return values[0::2], values[1::2]


def parameter_bounds(p: int) -> Bounds:
    """Return frozen per-layer gamma/beta bounds in interleaved order."""

    if p not in (1, 2):
        raise ValueError("p must be 1 or 2")
    lower = []
    upper = []
    for _layer in range(p):
        lower.extend((0.0, 0.0))
        upper.extend((2 * pi, pi))
    return Bounds(np.asarray(lower), np.asarray(upper))


def expected_qubo_energy(probability: np.ndarray, energies: np.ndarray) -> float:
    """Evaluate the frozen objective ``sum_x P(x) Q_A(x)``."""

    probability = np.asarray(probability, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if probability.shape != energies.shape or probability.shape != (16384,):
        raise ValueError("probability and energy vectors must both have length 16384")
    return float(np.dot(probability, energies))


def statevector_for_parameters(
    energies: np.ndarray,
    *,
    p: int,
    parameters: Sequence[float],
) -> np.ndarray:
    gammas, betas = split_interleaved_parameters(parameters, p)
    return reference_qaoa_from_energies(energies, gammas=gammas, betas=betas)


def _state_summary(
    state: StateRecord,
    probability: float,
    *,
    optimal_index: int,
) -> dict[str, Any]:
    return {
        "state_index": state.state_index,
        "canonical_bitstring": state.canonical_bitstring,
        "qiskit_display_bitstring": state.qiskit_display_bitstring,
        "probability": float(probability),
        "feasible": state.flow_penalty == 0,
        "optimal": state.state_index == optimal_index,
        "routing_cost": state.routing_cost,
        "flow_penalty": state.flow_penalty,
        "decoded_route": list(state.decoded_route) if state.decoded_route else None,
    }


def compute_core_metrics(
    states: Sequence[StateRecord],
    probability: np.ndarray,
    *,
    optimal_index: int,
) -> dict[str, Any]:
    """Compute the final route metrics directly from all 16384 probabilities."""

    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(states),) or len(states) != 16384:
        raise ValueError("complete frozen state space is required")
    feasible_indices = np.asarray(
        [state.state_index for state in states if state.flow_penalty == 0], dtype=int
    )
    if len(feasible_indices) != 20:
        raise RuntimeError("frozen feasible-state identity must contain exactly 20 routes")
    costs = np.asarray([state.routing_cost for state in states], dtype=float)
    p_feas = float(probability[feasible_indices].sum())
    p_opt = float(probability[optimal_index])
    conditional_cost = float(
        np.dot(probability[feasible_indices], costs[feasible_indices]) / p_feas
    )
    most_probable_index = int(np.argmax(probability))
    most_probable_feasible_index = min(
        feasible_indices,
        key=lambda index: (-probability[index], int(index)),
    )
    ranked_all = sorted(range(len(states)), key=lambda index: (-probability[index], index))
    ranked_feasible = sorted(
        (int(index) for index in feasible_indices),
        key=lambda index: (-probability[index], index),
    )
    bit_matrix = np.asarray([state.edge_vector for state in states], dtype=float)
    edge_marginals = probability @ bit_matrix
    return {
        "p_feas": p_feas,
        "p_opt": p_opt,
        "invalid_mass": float(1.0 - p_feas),
        "conditional_expected_route_cost": conditional_cost,
        "expected_routing_cost_all_states": float(np.dot(probability, costs)),
        "all_zero_critical_state_probability": float(probability[0]),
        "probability_sum": float(probability.sum()),
        "most_probable_computational_state": _state_summary(
            states[most_probable_index],
            probability[most_probable_index],
            optimal_index=optimal_index,
        ),
        "most_probable_feasible_route": _state_summary(
            states[int(most_probable_feasible_index)],
            probability[int(most_probable_feasible_index)],
            optimal_index=optimal_index,
        ),
        "top_10_computational_states": [
            _state_summary(states[index], probability[index], optimal_index=optimal_index)
            for index in ranked_all[:10]
        ],
        "top_valid_routes": [
            _state_summary(states[index], probability[index], optimal_index=optimal_index)
            for index in ranked_feasible[:10]
        ],
        "edge_marginals_q0_to_q13": [float(value) for value in edge_marginals],
    }


def validate_p2_equivalence_gate(graph, states: tuple[StateRecord, ...]) -> dict[str, Any]:
    """Hard gate: all A and two predetermined p=2 vectors, Qiskit vs NumPy."""

    vectors = (
        ("fixed_pair_0", (pi / 7, pi / 11, pi / 13, pi / 17)),
        ("fixed_pair_1", (0.17, 0.29, 1.13, 0.73)),
    )
    cases = []
    minimum_fidelity = 1.0
    maximum_amplitude_error = 0.0
    maximum_probability_error = 0.0
    maximum_norm_error = 0.0
    for penalty in (2, 5, 6, 12):
        hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
        for label, parameters in vectors:
            gammas, betas = split_interleaved_parameters(parameters, 2)
            numpy_state = reference_qaoa_statevector(
                states, penalty=penalty, gammas=gammas, betas=betas
            )
            qiskit_state = qaoa_statevector(
                graph, hamiltonian, gammas=gammas, betas=betas
            )
            comparison = compare_statevectors(numpy_state, qiskit_state)
            minimum_fidelity = min(minimum_fidelity, comparison.fidelity)
            maximum_amplitude_error = max(
                maximum_amplitude_error, comparison.max_amplitude_absolute_error
            )
            maximum_probability_error = max(
                maximum_probability_error, comparison.probability_vector_max_error
            )
            maximum_norm_error = max(
                maximum_norm_error,
                comparison.reference_norm_error,
                comparison.candidate_norm_error,
            )
            cases.append(
                {
                    "A": penalty,
                    "parameter_vector_label": label,
                    "parameters": list(parameters),
                    "fidelity": comparison.fidelity,
                    "max_amplitude_absolute_error": comparison.max_amplitude_absolute_error,
                    "maximum_probability_error": comparison.probability_vector_max_error,
                    "maximum_norm_error": max(
                        comparison.reference_norm_error,
                        comparison.candidate_norm_error,
                    ),
                }
            )
    passed = (
        minimum_fidelity >= 1 - 1e-12
        and maximum_amplitude_error < 1e-12
        and maximum_probability_error < 1e-14
        and maximum_norm_error < 1e-12
    )
    if not passed:
        raise RuntimeError("p=2 Qiskit/NumPy equivalence gate failed")
    return {
        "passed": True,
        "case_count": len(cases),
        "minimum_fidelity": minimum_fidelity,
        "maximum_amplitude_absolute_error": maximum_amplitude_error,
        "maximum_probability_error": maximum_probability_error,
        "maximum_norm_error": maximum_norm_error,
        "global_phase_handling": "comparison after overlap-phase alignment",
        "cases": cases,
    }


def run_optimization_start(
    energies: np.ndarray,
    *,
    penalty: int,
    p: int,
    start_id: int,
    initial_parameters: Sequence[float],
    optimizer_options: dict[str, Any],
    contract_hash: str,
) -> dict[str, Any]:
    """Run one frozen COBYLA start and preserve its complete evaluation trace."""

    initial = np.asarray(initial_parameters, dtype=float)
    trace = []

    def evaluate(parameters: np.ndarray, *, record: bool) -> float:
        statevector = statevector_for_parameters(energies, p=p, parameters=parameters)
        value = expected_qubo_energy(probabilities(statevector), energies)
        if record:
            trace.append(
                {
                    "evaluation": len(trace) + 1,
                    "parameters": [float(item) for item in parameters],
                    "expected_qubo_energy": value,
                }
            )
        return value

    initial_energy = evaluate(initial, record=False)
    started = perf_counter()
    try:
        result = minimize(
            lambda parameters: evaluate(parameters, record=True),
            initial,
            method="COBYLA",
            bounds=parameter_bounds(p),
            options=dict(optimizer_options),
        )
        elapsed = perf_counter() - started
        final_parameters = np.asarray(result.x, dtype=float)
        final_energy = evaluate(final_parameters, record=False)
        return {
            "run_id": f"A{penalty}_p{p}_s{start_id}",
            "optimization_contract_sha256": contract_hash,
            "A": penalty,
            "p": p,
            "start_id": start_id,
            "initial_parameters": [float(item) for item in initial],
            "final_parameters": [float(item) for item in final_parameters],
            "initial_expected_qubo_energy": initial_energy,
            "final_expected_qubo_energy": final_energy,
            "optimizer_reported_fun": float(result.fun),
            "objective_evaluation_count": int(result.nfev),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "converged": bool(result.success),
            "elapsed_seconds": elapsed,
            "evaluation_trace": trace,
        }
    except Exception as exc:  # preserve and continue every frozen start
        elapsed = perf_counter() - started
        return {
            "run_id": f"A{penalty}_p{p}_s{start_id}",
            "optimization_contract_sha256": contract_hash,
            "A": penalty,
            "p": p,
            "start_id": start_id,
            "initial_parameters": [float(item) for item in initial],
            "final_parameters": [float(item) for item in initial],
            "initial_expected_qubo_energy": initial_energy,
            "final_expected_qubo_energy": initial_energy,
            "optimizer_reported_fun": None,
            "objective_evaluation_count": len(trace),
            "optimizer_status": -999,
            "optimizer_message": f"{type(exc).__name__}: {exc}",
            "converged": False,
            "elapsed_seconds": elapsed,
            "evaluation_trace": trace,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_runs_csv(path: Path, runs: Sequence[dict[str, Any]]) -> None:
    fields = (
        "run_id", "optimization_contract_sha256", "A", "p", "start_id",
        "initial_parameters", "final_parameters", "initial_expected_qubo_energy",
        "final_expected_qubo_energy", "objective_evaluation_count", "optimizer_status",
        "optimizer_message", "converged", "elapsed_seconds", "p_feas", "p_opt",
        "invalid_mass", "conditional_expected_route_cost", "trace_point_count",
        "trace_json_pointer",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, run in enumerate(runs):
            quality = run["quality"]
            writer.writerow(
                {
                    **{field: run.get(field) for field in fields},
                    "initial_parameters": json.dumps(run["initial_parameters"], separators=(",", ":")),
                    "final_parameters": json.dumps(run["final_parameters"], separators=(",", ":")),
                    "p_feas": quality["p_feas"],
                    "p_opt": quality["p_opt"],
                    "invalid_mass": quality["invalid_mass"],
                    "conditional_expected_route_cost": quality["conditional_expected_route_cost"],
                    "trace_point_count": len(run["evaluation_trace"]),
                    "trace_json_pointer": f"core_experiment_results.json#/optimization_runs/{index}/evaluation_trace",
                }
            )


def _write_summary_csv(path: Path, cells: Sequence[dict[str, Any]]) -> None:
    fields = (
        "graph_sha256", "penalty_contract_sha256", "optimization_contract_sha256",
        "A", "penalty_label", "p", "selected_start_id", "final_parameters",
        "objective_evaluation_count", "optimizer_status", "converged",
        "final_expected_qubo_energy", "p_feas", "p_opt", "invalid_mass",
        "conditional_expected_route_cost", "all_zero_critical_state_probability",
        "most_probable_state_index", "most_probable_state_feasible",
        "most_probable_feasible_route", "most_probable_feasible_route_probability",
        "qiskit_numpy_fidelity", "qiskit_numpy_max_probability_discrepancy",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            identity = cell["identity"]
            optimization = cell["optimization"]
            quality = cell["quality"]
            verification = cell["verification"]
            most_state = quality["most_probable_computational_state"]
            most_route = quality["most_probable_feasible_route"]
            writer.writerow(
                {
                    "graph_sha256": identity["graph_sha256"],
                    "penalty_contract_sha256": identity["penalty_contract_sha256"],
                    "optimization_contract_sha256": identity["optimization_contract_sha256"],
                    "A": identity["A"],
                    "penalty_label": identity["penalty_label"],
                    "p": identity["p"],
                    "selected_start_id": optimization["selected_start_id"],
                    "final_parameters": json.dumps(optimization["final_parameters"], separators=(",", ":")),
                    "objective_evaluation_count": optimization["objective_evaluation_count"],
                    "optimizer_status": optimization["optimizer_status"],
                    "converged": optimization["converged"],
                    "final_expected_qubo_energy": optimization["final_expected_qubo_energy"],
                    "p_feas": quality["p_feas"],
                    "p_opt": quality["p_opt"],
                    "invalid_mass": quality["invalid_mass"],
                    "conditional_expected_route_cost": quality["conditional_expected_route_cost"],
                    "all_zero_critical_state_probability": quality["all_zero_critical_state_probability"],
                    "most_probable_state_index": most_state["state_index"],
                    "most_probable_state_feasible": most_state["feasible"],
                    "most_probable_feasible_route": "->".join(map(str, most_route["decoded_route"])),
                    "most_probable_feasible_route_probability": most_route["probability"],
                    "qiskit_numpy_fidelity": verification["fidelity"],
                    "qiskit_numpy_max_probability_discrepancy": verification["maximum_probability_discrepancy"],
                }
            )


def run_core_experiment(
    graph,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Execute all 24 frozen starts and write the three required result files."""

    contract, contract_hash = load_optimization_contract(contract_path)
    results = Path(results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    states = enumerate_state_space(graph)
    optimal_index = contract["scientific_identity"]["exact_state_index"]

    # This gate occurs before the first call to SciPy minimize.
    p2_gate = validate_p2_equivalence_gate(graph, states)

    labels = {
        row["A"]: row["label"]
        for row in contract["experimental_matrix"]["penalties"]
    }
    options = contract["optimizer"]["options"]
    runs = []
    for penalty in labels:
        energies = penalty_energies(states, penalty)
        for p in contract["experimental_matrix"]["depths"]:
            starts = contract["start_generation"]["starts_by_depth"][str(p)]
            for start in starts:
                run = run_optimization_start(
                    energies,
                    penalty=penalty,
                    p=p,
                    start_id=start["start_id"],
                    initial_parameters=start["parameters"],
                    optimizer_options=options,
                    contract_hash=contract_hash,
                )
                final_state = statevector_for_parameters(
                    energies, p=p, parameters=run["final_parameters"]
                )
                run["quality"] = compute_core_metrics(
                    states, probabilities(final_state), optimal_index=optimal_index
                )
                runs.append(run)

    selected_cells = []
    for penalty, penalty_label in labels.items():
        for p in contract["experimental_matrix"]["depths"]:
            cell_runs = [run for run in runs if run["A"] == penalty and run["p"] == p]
            selected = min(
                cell_runs,
                key=lambda run: (run["final_expected_qubo_energy"], run["start_id"]),
            )
            gammas, betas = split_interleaved_parameters(selected["final_parameters"], p)
            energies = penalty_energies(states, penalty)
            numpy_state = reference_qaoa_from_energies(
                energies, gammas=gammas, betas=betas
            )
            hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
            qiskit_state = qaoa_statevector(
                graph, hamiltonian, gammas=gammas, betas=betas
            )
            comparison = compare_statevectors(numpy_state, qiskit_state)
            if comparison.fidelity < 1 - 1e-12 or comparison.probability_vector_max_error > 1e-14:
                raise RuntimeError(f"selected explicit-circuit verification failed for A={penalty}, p={p}")
            objectives = sorted(run["final_expected_qubo_energy"] for run in cell_runs)
            selected_cells.append(
                {
                    "identity": {
                        "graph_sha256": contract["scientific_identity"]["graph_sha256"],
                        "penalty_contract_sha256": contract["scientific_identity"]["penalty_contract_sha256"],
                        "optimization_contract_sha256": contract_hash,
                        "A": penalty,
                        "penalty_label": penalty_label,
                        "p": p,
                    },
                    "optimization": {
                        "selected_run_id": selected["run_id"],
                        "selected_start_id": selected["start_id"],
                        "selection_rule": contract["cell_selection_rule"],
                        "final_parameters": selected["final_parameters"],
                        "gammas": list(gammas),
                        "betas": list(betas),
                        "objective_evaluation_count": selected["objective_evaluation_count"],
                        "optimizer_status": selected["optimizer_status"],
                        "optimizer_message": selected["optimizer_message"],
                        "converged": selected["converged"],
                        "elapsed_seconds": selected["elapsed_seconds"],
                        "final_expected_qubo_energy": selected["final_expected_qubo_energy"],
                    },
                    "quality": selected["quality"],
                    "all_start_summary": {
                        "minimum_final_objective": objectives[0],
                        "median_final_objective": median(objectives),
                        "maximum_final_objective": objectives[-1],
                        "starts": [
                            {
                                "start_id": run["start_id"],
                                "final_expected_qubo_energy": run["final_expected_qubo_energy"],
                                "p_feas": run["quality"]["p_feas"],
                                "p_opt": run["quality"]["p_opt"],
                                "converged": run["converged"],
                            }
                            for run in sorted(cell_runs, key=lambda run: run["start_id"])
                        ],
                    },
                    "verification": {
                        "backend": "explicit Qiskit primitive-gate statevector vs independent NumPy",
                        "fidelity": comparison.fidelity,
                        "maximum_amplitude_absolute_error": comparison.max_amplitude_absolute_error,
                        "maximum_probability_discrepancy": comparison.probability_vector_max_error,
                        "maximum_norm_error": max(
                            comparison.reference_norm_error,
                            comparison.candidate_norm_error,
                        ),
                    },
                    "circuit_gate_counts": primitive_gate_counts(
                        build_qaoa_circuit(graph, hamiltonian, p=p)
                    ),
                }
            )

    feasible_count = sum(state.flow_penalty == 0 for state in states)
    uniform = {
        "state_count": len(states),
        "feasible_state_count": feasible_count,
        "p_feas": feasible_count / len(states),
        "p_opt": 1 / len(states),
        "role": "unoptimized uniform-state reference; not a classical baseline",
    }
    payload = {
        "schema": "dtu-sciqis-routing-core-penalty-depth-results",
        "version": "1.0",
        "optimization_contract": {
            "path": "data/optimization_contract.json",
            "sha256": contract_hash,
            "statement": contract["statement"],
        },
        "scientific_identity": contract["scientific_identity"],
        "experiment_completeness": {
            "expected_run_count": 24,
            "actual_run_count": len(runs),
            "expected_selected_cell_count": 8,
            "actual_selected_cell_count": len(selected_cells),
            "all_failed_or_nonconverged_runs_preserved": True,
        },
        "p2_preoptimization_equivalence_gate": p2_gate,
        "uniform_state_reference": uniform,
        "optimization_runs": runs,
        "selected_cells": selected_cells,
    }
    if len(runs) != 24 or len(selected_cells) != 8:
        raise RuntimeError("frozen experiment completeness gate failed")
    _write_json(results / "core_experiment_results.json", payload)
    _write_runs_csv(results / "optimization_runs.csv", runs)
    _write_summary_csv(results / "core_experiment_summary.csv", selected_cells)
    return payload
