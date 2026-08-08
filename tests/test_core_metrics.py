from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from circuit import qaoa_statevector
from day4_artifacts import _load_or_compute_landscape
from graph import load_graph
from ising import qubo_to_ising
from optimization import compute_core_metrics, split_interleaved_parameters
from qubo import build_qubo, enumerate_state_space
from statevector_reference import (
    compare_statevectors,
    penalty_energies,
    probabilities,
    reference_qaoa_from_energies,
)


RESULT_PATH = Path("results/core_experiment_results.json")
PENALTIES = (2, 5, 6, 12)


def _load():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_core_metrics_sum_only_the_twenty_frozen_routes():
    states = enumerate_state_space(load_graph())
    raw = np.arange(1, 16385, dtype=float)
    distribution = raw / raw.sum()
    metrics = compute_core_metrics(states, distribution, optimal_index=10377)
    feasible = np.asarray(
        [state.state_index for state in states if state.flow_penalty == 0], dtype=int
    )
    costs = np.asarray([state.routing_cost for state in states], dtype=float)
    expected_p_feas = distribution[feasible].sum()
    assert len(feasible) == 20
    assert metrics["p_feas"] == expected_p_feas
    assert metrics["p_opt"] == distribution[10377]
    assert metrics["invalid_mass"] == 1.0 - expected_p_feas
    assert np.isclose(
        metrics["conditional_expected_route_cost"],
        np.dot(distribution[feasible], costs[feasible]) / expected_p_feas,
    )


def test_saved_result_matrix_is_complete_and_preserves_nonconvergence():
    payload = _load()
    runs = payload["optimization_runs"]
    cells = payload["selected_cells"]
    assert len(runs) == 24
    assert len(cells) == 8
    assert {(run["A"], run["p"], run["start_id"]) for run in runs} == {
        (A, p, start) for A in PENALTIES for p in (1, 2) for start in range(3)
    }
    assert {(cell["identity"]["A"], cell["identity"]["p"]) for cell in cells} == {
        (A, p) for A in PENALTIES for p in (1, 2)
    }
    assert all("converged" in run and "optimizer_status" in run for run in runs)
    assert any(not run["converged"] for run in runs)
    assert all(
        len(run["evaluation_trace"]) == run["objective_evaluation_count"]
        for run in runs
    )


def test_selected_start_is_exact_frozen_minimum_energy_choice():
    payload = _load()
    for cell in payload["selected_cells"]:
        A, p = cell["identity"]["A"], cell["identity"]["p"]
        candidates = [
            run for run in payload["optimization_runs"]
            if run["A"] == A and run["p"] == p
        ]
        expected = min(
            candidates,
            key=lambda run: (run["final_expected_qubo_energy"], run["start_id"]),
        )
        assert cell["optimization"]["selected_run_id"] == expected["run_id"]
        assert cell["optimization"]["selected_start_id"] == expected["start_id"]
        assert (
            cell["optimization"]["final_expected_qubo_energy"]
            == expected["final_expected_qubo_energy"]
        )


def test_uniform_reference_is_computed_from_frozen_state_identity():
    payload = _load()
    states = enumerate_state_space(load_graph())
    feasible = sum(state.flow_penalty == 0 for state in states)
    assert feasible == 20
    assert payload["uniform_state_reference"]["p_feas"] == feasible / len(states)
    assert payload["uniform_state_reference"]["p_opt"] == 1 / len(states)


def test_saved_landscape_axis_order_matches_direct_objective():
    states = enumerate_state_space(load_graph())
    gammas, betas, landscape = _load_or_compute_landscape(
        states, output_path=Path("results/p1_landscape_A6.csv")
    )
    assert landscape.shape == (81, 121)
    for beta_index, gamma_index in ((0, 0), (19, 61), (68, 1), (80, 120)):
        vector = reference_qaoa_from_energies(
            penalty_energies(states, 6),
            gammas=(gammas[gamma_index],),
            betas=(betas[beta_index],),
        )
        direct = float(np.dot(probabilities(vector), penalty_energies(states, 6)))
        assert np.isclose(landscape[beta_index, gamma_index], direct, atol=1e-12)


def test_all_selected_states_reproduce_with_explicit_qiskit_circuit():
    payload = _load()
    graph = load_graph()
    states = enumerate_state_space(graph)
    for cell in payload["selected_cells"]:
        A, p = cell["identity"]["A"], cell["identity"]["p"]
        parameters = cell["optimization"]["final_parameters"]
        gammas, betas = split_interleaved_parameters(parameters, p)
        numpy_state = reference_qaoa_from_energies(
            penalty_energies(states, A), gammas=gammas, betas=betas
        )
        qiskit_state = qaoa_statevector(
            graph,
            qubo_to_ising(build_qubo(graph, A)),
            gammas=gammas,
            betas=betas,
        )
        comparison = compare_statevectors(numpy_state, qiskit_state)
        assert comparison.fidelity >= 1 - 1e-12
        assert comparison.probability_vector_max_error < 1e-14
        assert np.allclose(probabilities(numpy_state), probabilities(qiskit_state), atol=1e-14)
