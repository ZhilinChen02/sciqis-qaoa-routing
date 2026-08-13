from __future__ import annotations

import inspect

import numpy as np
import pytest

from feasible_qaoa import (
    INCUMBENT_BIASED_FEASIBLE,
    UNIFORM_FEASIBLE,
    build_feasible_initial_state,
    build_feasible_route_basis,
    build_logical_cost_hamiltonian,
    build_logical_path_exchange_mixer,
)
from feasible_experiments import (
    BSP_LOSS,
    BSP_PATH_EXCHANGE,
    EXPECTATION_LOSS,
    GM_QAOA_EXPECTATION,
    GM_TH_QAOA,
    better_solution_probability,
    better_than_incumbent_mask,
    build_grover_feasible_mixer,
    build_incumbent_threshold,
    evaluate_final_loss,
    final_improvement_metrics,
    optimize_final_variant,
    run_final_improvement_cell,
    simulate_final_improvement,
    threshold_phase_values,
)


@pytest.fixture(scope="module")
def final_study(graph):
    basis = build_feasible_route_basis(graph)
    costs = build_logical_cost_hamiltonian(basis)
    path_mixer = build_logical_path_exchange_mixer(basis)
    grover = build_grover_feasible_mixer(basis.size)
    uniform = build_feasible_initial_state(basis, costs, mode=UNIFORM_FEASIBLE)
    biased = build_feasible_initial_state(
        basis, costs, mode=INCUMBENT_BIASED_FEASIBLE, bias_lambda=1.0
    )
    incumbent_cost = costs.raw_energies[basis.incumbent_route_id]
    threshold = build_incumbent_threshold(costs.raw_energies, incumbent_cost)
    return basis, costs, path_mixer, grover, uniform, biased, threshold


def test_bsp_threshold_is_strict_and_derived_only_from_incumbent_cost(final_study):
    basis, costs, *_, threshold = final_study
    signature = inspect.signature(better_than_incumbent_mask)
    assert set(signature.parameters) == {"raw_route_costs", "incumbent_raw_cost"}
    assert all("opt" not in name for name in signature.parameters)
    incumbent_cost = costs.raw_energies[basis.incumbent_route_id]
    expected = np.asarray(costs.raw_energies) < incumbent_cost
    assert np.array_equal(threshold.better_mask, expected)
    assert not expected[basis.incumbent_route_id]
    assert threshold.incumbent_raw_cost == 11.0


def test_bsp_is_exact_probability_sum_and_deterministic(final_study):
    basis, _, *_, threshold = final_study
    probabilities = np.arange(1, basis.size + 1, dtype=float)
    probabilities /= probabilities.sum()
    expected = float(np.sum(probabilities[np.asarray(threshold.better_mask)]))
    assert better_solution_probability(probabilities, threshold.better_mask) == expected
    assert better_solution_probability(probabilities, threshold.better_mask) == expected
    assert evaluate_final_loss(
        probabilities,
        loss_kind=BSP_LOSS,
        normalized_costs=np.zeros(basis.size),
        better_mask=threshold.better_mask,
    ) == -expected


def test_bsp_constructor_and_optimizer_have_no_optimum_input():
    for function in (
        better_than_incumbent_mask,
        build_incumbent_threshold,
        threshold_phase_values,
        optimize_final_variant,
    ):
        assert all("opt" not in name for name in inspect.signature(function).parameters)


def test_grover_uniform_state_hamiltonian_and_dimensions(final_study):
    basis, _, _, grover, uniform, *_ = final_study
    vector = np.asarray(grover.uniform_state)
    assert grover.dimension == basis.size == 20
    assert grover.hamiltonian.shape == (20, 20)
    assert np.allclose(vector, np.full(20, 1 / np.sqrt(20)))
    assert np.allclose(vector, uniform.amplitudes)
    assert np.allclose(grover.hamiltonian, np.outer(vector, vector.conj()))
    assert np.allclose(grover.hamiltonian, grover.hamiltonian.conj().T)
    assert np.allclose(grover.hamiltonian @ grover.hamiltonian, grover.hamiltonian)


@pytest.mark.parametrize("beta", [0.0, 0.3, np.pi, -0.8])
def test_grover_mixer_unitary_norm_and_beta_zero(final_study, beta):
    basis, _, _, grover, *_ = final_study
    unitary = grover.unitary(beta)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(basis.size), atol=1e-12)
    state = np.zeros(basis.size, dtype=np.complex128)
    state[7] = 1.0
    evolved = grover.evolve(state, beta)
    assert abs(np.linalg.norm(evolved) - 1.0) <= 1e-12
    if beta == 0.0:
        assert np.allclose(unitary, np.eye(basis.size), atol=1e-14)
        assert np.allclose(evolved, state, atol=1e-14)


def test_threshold_phase_boolean_strict_and_no_optimum_leakage(final_study):
    basis, costs, *_, threshold = final_study
    phase = threshold_phase_values(costs.raw_energies, threshold.incumbent_raw_cost)
    assert set(phase) <= {0.0, 1.0}
    assert np.array_equal(phase.astype(bool), threshold.better_mask)
    assert phase[basis.incumbent_route_id] == 0.0
    assert all(
        value == float(cost < threshold.incumbent_raw_cost)
        for value, cost in zip(phase, costs.raw_energies)
    )


def test_threshold_phase_gamma_zero_identity_and_unitary(final_study):
    basis, costs, *_, threshold = final_study
    phase = threshold_phase_values(costs.raw_energies, threshold.incumbent_raw_cost)
    assert np.allclose(np.exp(-1j * 0.0 * phase), np.ones(basis.size))
    for gamma in (0.0, 0.4, np.pi):
        diagonal = np.diag(np.exp(-1j * gamma * phase))
        assert np.allclose(diagonal.conj().T @ diagonal, np.eye(basis.size))


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_gm_th_normalization_and_feasible_support(final_study, depth):
    basis, costs, _, grover, uniform, _, threshold = final_study
    simulation = simulate_final_improvement(
        uniform.amplitudes,
        threshold_phase_values(costs.raw_energies, threshold.incumbent_raw_cost),
        grover,
        np.linspace(0.1, 0.8, 2 * depth),
        depth=depth,
    )
    assert len(simulation.probabilities) == basis.size
    assert abs(sum(simulation.probabilities) - 1.0) <= 1e-12
    assert abs(simulation.final_norm - 1.0) <= 1e-12
    assert len(simulation.layer_norms) == depth


@pytest.mark.parametrize(
    "method", [BSP_PATH_EXCHANGE, GM_QAOA_EXPECTATION, GM_TH_QAOA]
)
def test_all_zero_angles_return_the_method_initial_state(final_study, method):
    _, costs, path_mixer, grover, uniform, biased, threshold = final_study
    if method == BSP_PATH_EXCHANGE:
        initial, mixer, phase = biased, path_mixer, costs.normalized_energies
    elif method == GM_QAOA_EXPECTATION:
        initial, mixer, phase = uniform, grover, costs.normalized_energies
    else:
        initial, mixer = uniform, grover
        phase = threshold_phase_values(costs.raw_energies, threshold.incumbent_raw_cost)
    simulation = simulate_final_improvement(
        initial.amplitudes, phase, mixer, np.zeros(8), depth=4
    )
    assert np.allclose(simulation.probabilities, initial.probabilities, atol=1e-12)


def test_expectation_loss_is_normalized_cost_expectation(final_study):
    basis, costs, *_, threshold = final_study
    probabilities = np.full(basis.size, 1 / basis.size)
    actual = evaluate_final_loss(
        probabilities,
        loss_kind=EXPECTATION_LOSS,
        normalized_costs=costs.normalized_energies,
        better_mask=threshold.better_mask,
    )
    assert actual == pytest.approx(probabilities @ np.asarray(costs.normalized_energies))


def test_metrics_keep_feasibility_and_exact_bsp(final_study):
    basis, costs, *_, uniform, _, threshold = final_study
    metrics = final_improvement_metrics(
        uniform.probabilities,
        basis,
        costs,
        threshold.better_mask,
        initial_p_opt=uniform.p_opt,
    )
    assert metrics.p_feas == pytest.approx(1.0, abs=1e-12)
    assert metrics.bsp == pytest.approx(
        better_solution_probability(uniform.probabilities, threshold.better_mask)
    )
    assert metrics.p_opt == pytest.approx(uniform.p_opt)


@pytest.mark.parametrize("method", [BSP_PATH_EXCHANGE, GM_QAOA_EXPECTATION, GM_TH_QAOA])
def test_small_final_cells_are_seed_deterministic_and_feasible(final_study, method):
    basis, costs, path_mixer, grover, uniform, biased, threshold = final_study
    kwargs = dict(
        method=method,
        depth=1,
        seed=2601,
        evaluation_budget=8,
    )
    left = run_final_improvement_cell(
        basis, costs, path_mixer, grover, biased, uniform, threshold, **kwargs
    )
    right = run_final_improvement_cell(
        basis, costs, path_mixer, grover, biased, uniform, threshold, **kwargs
    )
    assert left.optimizer.final_parameters == right.optimizer.final_parameters
    assert [item.loss for item in left.optimizer.evaluation_trace] == pytest.approx(
        [item.loss for item in right.optimizer.evaluation_trace], abs=1e-14
    )
    assert left.optimizer.evaluations <= 8
    assert abs(left.metrics.p_feas - 1.0) <= 1e-12
