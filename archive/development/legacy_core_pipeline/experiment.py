"""Archived orchestration for the superseded Q1-versus-Q2 comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Callable

import numpy as np
from qiskit.quantum_info import Statevector

from support.exact_reference import compute_exact_reference
from graph import DEFAULT_GRAPH_PATH, load_graph, path_cost
from qubo import qubo_to_ising
from metrics import distribution_metrics, top_state_rows
from qaoa import (
    Q1_PENALTY_X,
    Q2_WARM_START,
    build_qaoa_circuit,
    circuit_statistics,
    normalized_diagonal,
    optimize_cobyla,
    seeded_initial_parameters,
    simulate_qaoa_state,
    standard_plus_state,
    state_probabilities,
)
from qubo import (
    build_qubo,
    derive_critical_penalty,
    enumerate_state_space,
    flow_feasibility_verdict,
    minimum_energy_states,
)
from support.warm_start import greedy_incumbent_route, incumbent_relaxation, product_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "experiment_config.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def _measure(function: Callable[[], Any]) -> tuple[Any, float, float]:
    wall_started, cpu_started = perf_counter(), process_time()
    result = function()
    return result, perf_counter() - wall_started, process_time() - cpu_started


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _route_text(route: tuple[int, ...] | None) -> str:
    return "" if route is None else "->".join(map(str, route))


def load_experiment_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "fixed_penalty",
        "warm_start_epsilon",
        "optimizer_budget",
        "optimizer_rhobeg",
        "optimizer_tolerance",
        "seed",
        "top_k",
        "experiments",
    }
    if not required <= payload.keys():
        raise ValueError(f"experiment config missing keys: {sorted(required - payload.keys())}")
    if payload.get("optimizer") != "COBYLA":
        raise ValueError("only the validated COBYLA policy is supported")
    return payload


def run_experiment(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, object]:
    """Run exact reference, Q1 p=1, and Q2 p=1/p=2 and write results."""

    config = load_experiment_config(config_path)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime_rows: list[dict[str, object]] = []

    graph, wall, cpu = _measure(load_graph)
    runtime_rows.append({
        "experiment_id": "shared", "solver": "shared", "p": "",
        "stage": "graph construction", "wall_time_s": wall, "cpu_time_s": cpu,
    })

    (exact_payload, all_routes), wall, cpu = _measure(
        lambda: compute_exact_reference(graph, graph_path=DEFAULT_GRAPH_PATH)
    )
    runtime_rows.append({
        "experiment_id": "shared", "solver": "exact classical", "p": 0,
        "stage": "exact solve", "wall_time_s": wall, "cpu_time_s": cpu,
    })
    exact_route = tuple(exact_payload["exact_reference"]["node_path"])
    exact_cost = int(exact_payload["exact_reference"]["cost"])
    (output / "exact_reference.json").write_text(
        json.dumps(exact_payload, indent=2) + "\n", encoding="utf-8"
    )

    penalty = float(config["fixed_penalty"])

    def construct_encoding():
        states_value = enumerate_state_space(graph)
        qubo_value = build_qubo(graph, penalty)
        ising_value = qubo_to_ising(qubo_value)
        critical, critical_states = derive_critical_penalty(states_value, exact_cost)
        ground_energy, ground_states = minimum_energy_states(states_value, penalty)
        return states_value, qubo_value, ising_value, critical, critical_states, ground_energy, ground_states

    (
        (states, qubo, ising, critical_penalty, _critical_states, ground_energy, ground_states),
        wall,
        cpu,
    ) = _measure(construct_encoding)
    runtime_rows.append({
        "experiment_id": "shared", "solver": "shared", "p": "",
        "stage": "QUBO/Ising construction", "wall_time_s": wall, "cpu_time_s": cpu,
    })
    if penalty <= float(critical_penalty):
        raise RuntimeError(
            f"fixed penalty {penalty} must exceed exact crossing threshold {critical_penalty}"
        )
    if len(ground_states) != 1 or ground_states[0].decoded_route != exact_route:
        raise RuntimeError("fixed-penalty QUBO ground state is not the unique exact route")
    verdict = flow_feasibility_verdict(states)
    if not verdict["sets_identical"]:
        raise RuntimeError("flow feasibility and independent route decoder disagree")

    raw_diagonal = np.asarray([float(qubo.evaluate(state.edge_vector)) for state in states])
    normalized_cost, cost_shift, cost_scale = normalized_diagonal(raw_diagonal)
    if not np.allclose(
        raw_diagonal,
        [state.routing_cost + penalty * state.flow_penalty for state in states],
    ):
        raise RuntimeError("explicit QUBO does not match direct penalized energies")

    warm_values, wall, cpu = _measure(
        lambda: incumbent_relaxation(graph, float(config["warm_start_epsilon"]))
    )
    runtime_rows.append({
        "experiment_id": "shared", "solver": Q2_WARM_START, "p": "",
        "stage": "continuous relaxation", "wall_time_s": wall, "cpu_time_s": cpu,
    })
    clipped_values = tuple(row.clipped_value for row in warm_values)
    incumbent_route = greedy_incumbent_route(graph)
    incumbent_cost = path_cost(graph, incumbent_route)
    if incumbent_route == exact_route:
        raise RuntimeError("warm-start incumbent unexpectedly equals the exact optimum")
    warm_state = product_state(clipped_values)
    for qubit, c_value in enumerate(clipped_values):
        indices = np.arange(len(warm_state), dtype=np.uint64)
        selected = ((indices >> np.uint64(qubit)) & np.uint64(1)).astype(bool)
        measured = float(np.sum(np.abs(warm_state[selected]) ** 2))
        if not np.isclose(measured, c_value, atol=1e-12):
            raise RuntimeError("prepared warm-start marginal does not match clipped value")

    _write_csv(
        output / "warm_start_values.csv",
        [
            "variable", "incumbent_bit", "relaxed_value", "clipped_value",
            "preparation_angle", "mixer_x", "mixer_z",
        ],
        [row.__dict__ for row in warm_values],
    )

    metrics_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for specification in config["experiments"]:
        experiment_started, experiment_cpu_started = perf_counter(), process_time()
        experiment_id = str(specification["experiment_id"])
        solver = str(specification["solver"])
        depth = int(specification["p"])
        if (solver, depth) not in {
            (Q1_PENALTY_X, 1),
            (Q2_WARM_START, 1),
            (Q2_WARM_START, 2),
        }:
            raise ValueError(f"experiment outside declared course scope: {specification}")
        warm = clipped_values if solver == Q2_WARM_START else None
        simulation_seconds = 0.0
        simulation_cpu_seconds = 0.0

        _, preparation_wall, preparation_cpu = _measure(
            (lambda: product_state(warm))
            if solver == Q2_WARM_START
            else (lambda: standard_plus_state(len(ising.h)))
        )

        def simulate(parameters: np.ndarray) -> np.ndarray:
            nonlocal simulation_seconds, simulation_cpu_seconds
            started, cpu_started = perf_counter(), process_time()
            state = simulate_qaoa_state(
                normalized_cost,
                parameters,
                depth=depth,
                solver=solver,
                warm_start_values=warm,
            )
            simulation_seconds += perf_counter() - started
            simulation_cpu_seconds += process_time() - cpu_started
            return state

        def objective(parameters: np.ndarray) -> float:
            probabilities = state_probabilities(simulate(parameters))
            return float(probabilities @ normalized_cost)

        optimizer_result, optimizer_wall, optimizer_cpu = _measure(
            lambda: optimize_cobyla(
                objective,
                depth=depth,
                seed=int(config["seed"]),
                evaluation_budget=int(config["optimizer_budget"]),
                tolerance=float(config["optimizer_tolerance"]),
                rhobeg=float(config["optimizer_rhobeg"]),
            )
        )
        runtime_rows.append({
            "experiment_id": experiment_id, "solver": solver, "p": depth,
            "stage": "classical optimization/statevector evaluations",
            "wall_time_s": optimizer_wall, "cpu_time_s": optimizer_cpu,
        })

        final_state = simulate(np.asarray(optimizer_result.parameters, dtype=float))
        probabilities = state_probabilities(final_state)
        probability_sum = float(np.sum(probabilities))

        circuit, circuit_wall, circuit_cpu = _measure(
            lambda: build_qaoa_circuit(
                ising,
                optimizer_result.parameters,
                depth=depth,
                solver=solver,
                normalization_scale=cost_scale,
                warm_start_values=warm,
            )
        )
        statistics, stats_wall, stats_cpu = _measure(lambda: circuit_statistics(circuit))
        runtime_rows.append({
            "experiment_id": experiment_id, "solver": solver, "p": depth,
            "stage": "warm-start preparation" if solver == Q2_WARM_START else "plus-state preparation",
            "wall_time_s": preparation_wall, "cpu_time_s": preparation_cpu,
        })
        runtime_rows.append({
            "experiment_id": experiment_id, "solver": solver, "p": depth,
            "stage": "circuit construction", "wall_time_s": circuit_wall + stats_wall,
            "cpu_time_s": circuit_cpu + stats_cpu,
        })

        qiskit_state, validation_wall, validation_cpu = _measure(
            lambda: Statevector.from_instruction(circuit)
        )
        qiskit_probabilities = np.abs(np.asarray(qiskit_state.data)) ** 2
        circuit_error = float(np.max(np.abs(qiskit_probabilities - probabilities)))
        if circuit_error > 1e-9:
            raise RuntimeError(
                f"matrix/circuit probability mismatch for {experiment_id}: {circuit_error}"
            )
        runtime_rows.append({
            "experiment_id": experiment_id, "solver": solver, "p": depth,
            "stage": "circuit/statevector validation", "wall_time_s": validation_wall,
            "cpu_time_s": validation_cpu,
        })

        summary, metric_wall, metric_cpu = _measure(
            lambda: distribution_metrics(probabilities, states, optimal_cost=exact_cost)
        )
        experiment_top_rows, decode_wall, decode_cpu = _measure(
            lambda: top_state_rows(
                probabilities,
                states,
                optimal_cost=exact_cost,
                top_k=int(config["top_k"]),
            )
        )
        runtime_rows.append({
            "experiment_id": experiment_id, "solver": solver, "p": depth,
            "stage": "decoding", "wall_time_s": decode_wall, "cpu_time_s": decode_cpu,
        })
        runtime_rows.append({
            "experiment_id": experiment_id, "solver": solver, "p": depth,
            "stage": "metric calculation", "wall_time_s": metric_wall,
            "cpu_time_s": metric_cpu,
        })

        optimized_energy = float(probabilities @ raw_diagonal)
        optimized_normalized = float(probabilities @ normalized_cost)
        total_wall = perf_counter() - experiment_started
        total_cpu = process_time() - experiment_cpu_started
        initial_parameters = seeded_initial_parameters(depth, int(config["seed"]))
        metrics_rows.append({
            "experiment_id": experiment_id,
            "solver": solver,
            "p": depth,
            "num_qubits": len(ising.h),
            "num_basis_states": len(states),
            "epsilon": config["warm_start_epsilon"] if solver == Q2_WARM_START else "",
            "penalty": penalty,
            "optimizer": config["optimizer"],
            "optimizer_budget": config["optimizer_budget"],
            "optimizer_evaluations": optimizer_result.evaluations,
            "optimizer_success": optimizer_result.success,
            "optimizer_reason": optimizer_result.reason,
            "seed": config["seed"],
            "initial_parameters": json.dumps(list(map(float, initial_parameters))),
            "optimized_parameters": json.dumps(list(optimizer_result.parameters)),
            "optimized_energy": optimized_energy,
            "optimized_normalized_energy": optimized_normalized,
            "p_feas": summary.p_feas,
            "p_opt": summary.p_opt,
            "feasible_conditional_cost": summary.feasible_conditional_cost,
            "best_decoded_route": _route_text(summary.best_decoded_route),
            "best_decoded_cost": summary.best_decoded_cost,
            "best_state_probability": summary.best_state_probability,
            "best_state_valid": summary.best_state_valid,
            "best_state_optimal": summary.best_state_optimal,
            "optimal_state_rank": summary.optimal_state_rank,
            "circuit_depth": statistics.circuit_depth,
            "total_gate_count": statistics.total_gate_count,
            "two_qubit_gate_count": statistics.two_qubit_gate_count,
            "optimization_time_s": optimizer_result.wall_time,
            "optimization_cpu_time_s": optimizer_cpu,
            "statevector_evaluation_time_s": simulation_seconds,
            "statevector_evaluation_cpu_s": simulation_cpu_seconds,
            "total_runtime_s": total_wall,
            "total_cpu_time_s": total_cpu,
            "probability_sum": probability_sum,
            "circuit_probability_max_error": circuit_error,
        })
        for row in experiment_top_rows:
            top_rows.append({
                "experiment_id": experiment_id,
                "solver": solver,
                "p": depth,
                **row,
            })
        runtime_rows.append({
            "experiment_id": experiment_id, "solver": solver, "p": depth,
            "stage": "total", "wall_time_s": total_wall, "cpu_time_s": total_cpu,
        })

    metrics_fields = list(metrics_rows[0].keys())
    _write_csv(output / "metrics.csv", metrics_fields, metrics_rows)
    _write_csv(
        output / "top_bitstrings.csv",
        [
            "experiment_id", "solver", "p", "rank", "bitstring", "probability",
            "valid", "optimal", "decoded_route", "decoded_cost", "invalid_reason",
        ],
        top_rows,
    )
    _write_csv(
        output / "runtime_profile.csv",
        ["experiment_id", "solver", "p", "stage", "wall_time_s", "cpu_time_s"],
        runtime_rows,
    )
    summary_payload: dict[str, object] = {
        "schema": "dtu-sciqis-warm-start-results",
        "version": "1.0",
        "exact_optimal_route": list(exact_route),
        "exact_optimal_cost": exact_cost,
        "exact_simple_path_count": len(all_routes),
        "fixed_penalty": penalty,
        "critical_penalty": float(critical_penalty),
        "qubo_ground_energy": float(ground_energy),
        "incumbent_route": list(incumbent_route),
        "incumbent_cost": incumbent_cost,
        "warm_start_kind": "incumbent-route binary relaxation with epsilon clipping",
        "warm_start_epsilon": config["warm_start_epsilon"],
        "source_q2_semantics": {
            "initial_state": "incumbent_product_warm_start_state",
            "mixer": "separable -sin(theta_i)X-cos(theta_i)Z",
            "layer_order": "cost then mixer",
            "parameter_order": "all gammas then all betas",
            "mixer_scale": "1 / num_qubits",
        },
        "flow_decoder_feasibility": verdict,
        "cost_normalization_shift": cost_shift,
        "cost_normalization_scale": cost_scale,
        "results": metrics_rows,
    }
    (output / "experiment_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    return summary_payload
