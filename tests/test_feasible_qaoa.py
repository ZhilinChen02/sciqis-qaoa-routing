from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest

from exact_reference import enumerate_simple_paths_independent
from feasible_qaoa import (
    ASCENDING_CVAR_OBJECTIVE,
    EXPECTATION_OBJECTIVE,
    FIXED_CVAR_OBJECTIVE,
    INCUMBENT_BIASED_FEASIBLE,
    PROBABILITY_TOLERANCE,
    UNIFORM_FEASIBLE,
    ascending_cvar_alpha,
    build_feasible_initial_state,
    build_feasible_route_basis,
    build_logical_cost_hamiltonian,
    build_logical_path_exchange_mixer,
    evaluate_q2f_objective,
    feasible_route_metrics,
    incumbent_biased_probabilities,
    run_q2f_cell,
    simulate_logical_qaoa,
)
from graph import path_cost, path_to_edge_bitstring
from objectives import probability_weighted_cvar, probability_weighted_expectation
from qubo import decode_valid_route


@pytest.fixture(scope="module")
def q2f(graph):
    basis = build_feasible_route_basis(graph)
    costs = build_logical_cost_hamiltonian(basis)
    mixer = build_logical_path_exchange_mixer(basis)
    uniform = build_feasible_initial_state(basis, costs, mode=UNIFORM_FEASIBLE)
    biased = build_feasible_initial_state(
        basis, costs, mode=INCUMBENT_BIASED_FEASIBLE, bias_lambda=1.0
    )
    return basis, costs, mixer, uniform, biased


def test_feasible_basis_matches_both_validated_enumerations(graph, q2f):
    basis, *_ = q2f
    independent = {record.node_path for record in enumerate_simple_paths_independent(graph)}
    networkx_routes = {
        tuple(path)
        for path in nx.all_simple_paths(
            graph, graph.graph["source"], graph.graph["target"]
        )
    }
    assert independent == networkx_routes
    assert {route.node_sequence for route in basis.routes} == independent
    assert basis.size == len(independent) == 20


def test_feasible_basis_is_unique_deterministic_and_cost_ordered(graph, q2f):
    basis, *_ = q2f
    rebuilt = build_feasible_route_basis(graph)
    assert basis == rebuilt
    assert [route.route_id for route in basis.routes] == list(range(basis.size))
    assert len({route.node_sequence for route in basis.routes}) == basis.size
    assert len({route.edge_bitstring for route in basis.routes}) == basis.size
    assert [(route.routing_cost, route.node_sequence) for route in basis.routes] == sorted(
        (route.routing_cost, route.node_sequence) for route in basis.routes
    )


def test_every_basis_route_is_valid_and_round_trips(graph, q2f):
    basis, *_ = q2f
    for route in basis.routes:
        assert nx.is_simple_path(graph, route.node_sequence)
        assert route.node_sequence[0] == graph.graph["source"]
        assert route.node_sequence[-1] == graph.graph["target"]
        assert route.edge_bitstring == path_to_edge_bitstring(graph, route.node_sequence)
        assert decode_valid_route(graph, route.edge_bitstring) == route.node_sequence
        assert path_cost(graph, route.node_sequence) == route.routing_cost


def test_feasible_basis_exact_optimum_agrees_with_reference(q2f, exact):
    basis, *_ = q2f
    optimum = basis.routes[basis.exact_optimal_route_id]
    assert optimum.exact_optimal
    assert optimum.node_sequence == tuple(exact["exact_reference"]["node_path"])
    assert optimum.routing_cost == exact["exact_reference"]["cost"] == 10
    assert sum(route.exact_optimal for route in basis.routes) == 1


def test_logical_cost_diagonals_equal_raw_routes_and_minimum(q2f):
    basis, costs, *_ = q2f
    assert np.array_equal(np.diag(costs.raw_matrix), basis.raw_costs)
    assert np.array_equal(
        np.diag(costs.normalized_matrix), np.asarray(costs.normalized_energies)
    )
    assert costs.normalization_shift == 10.0
    assert costs.normalization_scale == 4.0
    assert int(np.argmin(costs.raw_energies)) == basis.exact_optimal_route_id
    assert costs.raw_energies[basis.exact_optimal_route_id] == min(costs.raw_energies)


def test_path_exchange_mixer_is_hermitian_connected_and_expected_dimension(q2f):
    basis, _, mixer, *_ = q2f
    assert mixer.dimension == basis.size
    assert mixer.hamiltonian.shape == (basis.size, basis.size)
    assert np.array_equal(mixer.hamiltonian, mixer.hamiltonian.conj().T)
    assert np.allclose(np.diag(mixer.hamiltonian), 0.0)
    assert mixer.connected
    assert len(mixer.exchanges) == 106


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.7, -0.91])
def test_path_exchange_unitary_and_norm_preservation(q2f, beta):
    basis, _, mixer, *_ = q2f
    unitary = mixer.unitary(beta)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(basis.size), atol=1e-12)
    for route_id in range(basis.size):
        state = np.zeros(basis.size, dtype=np.complex128)
        state[route_id] = 1.0
        evolved = mixer.evolve(state, beta)
        assert abs(np.linalg.norm(evolved) - 1.0) <= 1e-12
        assert abs(np.sum(np.abs(evolved) ** 2) - 1.0) <= 1e-12


def test_path_exchange_beta_zero_is_identity(q2f):
    basis, _, mixer, *_ = q2f
    assert np.allclose(mixer.unitary(0.0), np.eye(basis.size), atol=1e-12)


def test_mixer_support_is_only_the_verified_feasible_basis(q2f):
    basis, _, mixer, *_ = q2f
    start = np.zeros(basis.size, dtype=np.complex128)
    start[basis.incumbent_route_id] = 1.0
    probabilities = np.abs(mixer.evolve(start, 0.43)) ** 2
    assert abs(float(np.sum(probabilities)) - 1.0) <= 1e-12
    assert all(route.node_sequence for route in basis.routes)


def test_uniform_feasible_initialization(q2f):
    basis, _, _, uniform, _ = q2f
    assert np.allclose(uniform.probabilities, np.full(basis.size, 1.0 / basis.size))
    assert abs(sum(uniform.probabilities) - 1.0) <= 1e-12
    assert abs(sum(abs(value) ** 2 for value in uniform.amplitudes) - 1.0) <= 1e-12
    assert uniform.p_feas == pytest.approx(1.0)
    assert uniform.p_opt == pytest.approx(1.0 / basis.size)


def test_incumbent_biased_distribution_is_deterministic_and_incumbent_highest(q2f):
    basis, costs, _, _, biased = q2f
    rebuilt = build_feasible_initial_state(
        basis, costs, mode=INCUMBENT_BIASED_FEASIBLE, bias_lambda=1.0
    )
    assert biased == rebuilt
    assert abs(sum(biased.probabilities) - 1.0) <= 1e-12
    assert biased.route_distances[basis.incumbent_route_id] == 0
    assert biased.probabilities[basis.incumbent_route_id] == max(biased.probabilities)
    assert biased.bias_lambda == 1.0


def test_incumbent_bias_constructor_has_no_optimum_or_cost_input(q2f):
    basis, *_ = q2f
    assert "optimal" not in inspect.signature(incumbent_biased_probabilities).parameters
    assert "cost" not in inspect.signature(incumbent_biased_probabilities).parameters
    probs, distances = incumbent_biased_probabilities(
        basis.edge_bitstrings,
        basis.routes[basis.incumbent_route_id].edge_bitstring,
        bias_lambda=1.0,
    )
    assert probs[basis.incumbent_route_id] == max(probs)
    assert distances[basis.incumbent_route_id] == 0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_zero_angles_return_initial_state_and_pfeas_one(q2f, depth):
    basis, costs, mixer, _, biased = q2f
    simulation = simulate_logical_qaoa(
        biased.amplitudes,
        costs.normalized_energies,
        mixer,
        np.zeros(2 * depth),
        depth=depth,
    )
    assert np.allclose(simulation.probabilities, biased.probabilities, atol=1e-12)
    assert len(simulation.layer_norms) == depth
    assert all(abs(record.after_cost_norm - 1.0) <= 1e-12 for record in simulation.layer_norms)
    assert all(abs(record.after_mixer_norm - 1.0) <= 1e-12 for record in simulation.layer_norms)
    metrics = feasible_route_metrics(
        simulation.probabilities, basis, costs, initial_p_opt=biased.p_opt
    )
    assert abs(metrics.p_feas - 1.0) <= PROBABILITY_TOLERANCE


def test_qaoa_simulation_is_deterministically_reproducible(q2f):
    _, costs, mixer, _, biased = q2f
    parameters = [0.2, 0.4, 0.6, 0.8]
    left = simulate_logical_qaoa(
        biased.amplitudes, costs.normalized_energies, mixer, parameters, depth=2
    )
    right = simulate_logical_qaoa(
        biased.amplitudes, costs.normalized_energies, mixer, parameters, depth=2
    )
    assert left == right


def test_expectation_and_fixed_cvar_objectives_reuse_validated_definitions(q2f):
    _, costs, _, uniform, _ = q2f
    expectation, alpha = evaluate_q2f_objective(
        uniform.probabilities,
        costs.normalized_energies,
        EXPECTATION_OBJECTIVE,
        evaluation_index=0,
        total_budget=100,
    )
    fixed, fixed_alpha = evaluate_q2f_objective(
        uniform.probabilities,
        costs.normalized_energies,
        FIXED_CVAR_OBJECTIVE,
        evaluation_index=0,
        total_budget=100,
    )
    assert alpha is None
    assert fixed_alpha == 0.25
    assert expectation == probability_weighted_expectation(
        uniform.probabilities, costs.normalized_energies
    )
    assert fixed == probability_weighted_cvar(
        uniform.probabilities, costs.normalized_energies, 0.25
    )


def test_ascending_cvar_schedule_boundaries_monotonicity_and_outcome_independence():
    schedule = [ascending_cvar_alpha(index, 100) for index in range(100)]
    assert schedule[0] == 0.25
    assert schedule[-1] == 1.0
    assert all(left <= right for left, right in zip(schedule, schedule[1:]))
    assert ascending_cvar_alpha(0, 1) == 0.25
    signature = inspect.signature(ascending_cvar_alpha)
    assert set(signature.parameters) == {
        "evaluation_index", "total_budget", "alpha_start", "alpha_end"
    }


def test_ascending_alpha_one_equals_expectation(q2f):
    _, costs, _, _, biased = q2f
    value, alpha = evaluate_q2f_objective(
        biased.probabilities,
        costs.normalized_energies,
        ASCENDING_CVAR_OBJECTIVE,
        evaluation_index=99,
        total_budget=100,
    )
    expectation = probability_weighted_expectation(
        biased.probabilities, costs.normalized_energies
    )
    assert alpha == 1.0
    assert value == pytest.approx(expectation, abs=1e-15)


def test_feasible_metrics_topk_entropy_rank_and_amplification(q2f):
    basis, costs, *_ = q2f
    probabilities = np.zeros(basis.size)
    probabilities[0] = 0.5
    probabilities[1] = 0.3
    probabilities[5] = 0.2
    metrics = feasible_route_metrics(probabilities, basis, costs, initial_p_opt=0.25)
    assert metrics.p_feas == 1.0
    assert metrics.p_opt == 0.5
    assert metrics.optimal_state_rank == 1
    assert metrics.top3_lowest_cost_mass == pytest.approx(0.8)
    assert metrics.top5_lowest_cost_mass == pytest.approx(0.8)
    assert metrics.probability_entropy == pytest.approx(
        -(0.5 * np.log(0.5) + 0.3 * np.log(0.3) + 0.2 * np.log(0.2))
    )
    assert metrics.p_opt_amplification == 2.0
    assert metrics.most_probable_route_id == 0


def test_feasible_metrics_reject_loss_of_normalization(q2f):
    basis, costs, *_ = q2f
    with pytest.raises(RuntimeError, match="feasibility_invariant"):
        feasible_route_metrics(
            np.full(basis.size, 0.9 / basis.size), basis, costs, initial_p_opt=0.1
        )


def test_small_optimizer_run_is_reproducible_and_traces_alpha(q2f):
    basis, costs, mixer, _, biased = q2f
    kwargs = dict(
        objective_mode=ASCENDING_CVAR_OBJECTIVE,
        depth=1,
        seed=2601,
        evaluation_budget=8,
    )
    left = run_q2f_cell(basis, costs, mixer, biased, **kwargs)
    right = run_q2f_cell(basis, costs, mixer, biased, **kwargs)
    assert left.optimizer.final_parameters == right.optimizer.final_parameters
    assert [record.objective_value for record in left.optimizer.evaluation_trace] == pytest.approx(
        [record.objective_value for record in right.optimizer.evaluation_trace], abs=1e-14
    )
    assert [record.alpha for record in left.optimizer.evaluation_trace] == [
        ascending_cvar_alpha(index, 8) for index in range(left.optimizer.evaluations)
    ]
    assert left.optimizer.evaluations <= 8
    assert abs(left.metrics.p_feas - 1.0) <= 1e-12
