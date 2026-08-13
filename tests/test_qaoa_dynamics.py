from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.linalg import expm

from experiments.dynamics_study import (
    GROVER_FEASIBLE,
    GROVER_GLOBAL,
    PENALTY_X,
    _x_mixer_evolve,
    prepare_study_context,
    regression_comparison,
)
from qaoa import build_global_grover_mixer, simulate_global_grover_state
from qubo import qubo_to_ising
from feasible_experiments import simulate_final_improvement
from qaoa import Q1_PENALTY_X, simulate_qaoa_state, standard_plus_state
from experiments.dynamics_trace import BasisMetadata, apply_cost_layer, trace_qaoa_evolution
from qubo import decode_valid_route, state_index_to_edge_vector


@pytest.fixture(scope="module")
def dynamics_context():
    return prepare_study_context(penalty=6.0)


def tiny_metadata() -> BasisMetadata:
    return BasisMetadata(
        basis_labels=("00", "10", "01", "11"),
        decoded_routes=((0, 1), (0, 1), None, None),
        route_costs=(0.0, 1.0, 2.0, 3.0),
        flow_penalties=(0.0, 0.0, 1.0, 1.0),
        total_energies=(0.0, 1.0, 4.0, 5.0),
        feasible_mask=(True, True, False, False),
        optimal_mask=(True, False, False, False),
        penalty_coefficient=2.0,
        representation="tiny_fixture",
    )


def test_global_grover_dimension_uniform_initialization_and_no_dense_storage():
    mixer = build_global_grover_mixer(14)
    state = mixer.initial_state()
    assert mixer.dimension == 2**14
    assert state.shape == (2**14,)
    assert np.allclose(np.abs(state) ** 2, 1.0 / 2**14, atol=0.0, rtol=1e-15)
    assert not hasattr(mixer, "hamiltonian")
    source = inspect.getsource(type(mixer).evolve)
    assert "np.eye" not in source and "outer" not in source and "expm" not in source


@pytest.mark.parametrize("beta", [0.0, 0.37, -0.6, np.pi])
def test_efficient_global_projector_matches_tiny_dense_exponential(beta):
    mixer = build_global_grover_mixer(3)
    rng = np.random.default_rng(2601)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state /= np.linalg.norm(state)
    uniform = np.full(8, 1.0 / np.sqrt(8), dtype=np.complex128)
    dense = expm(-1j * beta * np.outer(uniform, uniform.conj())) @ state
    assert np.allclose(mixer.evolve(state, beta), dense, atol=1e-12, rtol=0.0)
    if beta == 0.0:
        assert np.array_equal(mixer.evolve(state, beta), state)


def test_global_cost_diagonal_is_penalty_x_diagonal_bit_identical(dynamics_context):
    validation = dynamics_context.validations
    assert validation["penalty_x_global_hc_byte_identical"] is True
    assert validation["penalty_x_hc_sha256"] == validation["grover_global_hc_sha256"]
    assert dynamics_context.raw_full_diagonal.shape == (2**14,)


def test_qubo_and_ising_basis_energies_match_exactly(dynamics_context):
    qubo = __import__("qubo").build_qubo(dynamics_context.graph, 6.0)
    ising = qubo_to_ising(qubo)
    for index in (0, 1, 17, 413, 4096, 16383):
        bits = state_index_to_edge_vector(index)
        assert qubo.evaluate(bits) == ising.basis_energy(bits)
    assert dynamics_context.validations["qubo_ising_max_error"] == 0.0


def test_cost_identity_probability_energy_norm_and_nontrivial_phase():
    initial = standard_plus_state(2)
    diagonal = np.asarray([0.0, 0.2, 0.7, 1.0])
    after = apply_cost_layer(initial, diagonal, 0.9)
    assert np.allclose(np.abs(after) ** 2, np.abs(initial) ** 2, atol=1e-15)
    assert float(np.abs(after) ** 2 @ diagonal) == pytest.approx(
        float(np.abs(initial) ** 2 @ diagonal), abs=1e-15
    )
    assert np.linalg.norm(after) == pytest.approx(1.0, abs=1e-15)
    assert not np.allclose(after, initial)
    assert np.ptp(np.angle(after / initial)) > 0.1
    assert np.array_equal(apply_cost_layer(initial, diagonal, 0.0), initial)


@pytest.mark.parametrize("depth", [1, 2])
def test_trace_count_normalization_and_probability_identities(depth):
    mixer = build_global_grover_mixer(2)
    parameters = np.linspace(0.3, 1.1, 2 * depth)
    trace = trace_qaoa_evolution(
        mixer.initial_state(),
        np.asarray([0.0, 0.2, 0.7, 1.0]),
        mixer.evolve,
        parameters,
        depth=depth,
        metadata=tiny_metadata(),
    )
    assert len(trace.checkpoints) == 1 + 2 * depth
    assert all(item.norm == pytest.approx(1.0, abs=1e-12) for item in trace.checkpoints)
    assert all(item.probability_sum == pytest.approx(1.0, abs=1e-12) for item in trace.checkpoints)
    assert all(item.p_feas + item.invalid_mass == pytest.approx(1.0, abs=1e-12) for item in trace.checkpoints)
    assert all(item.p_opt <= item.p_feas + 1e-12 for item in trace.checkpoints)
    assert all(item.cost_probability_max_delta < 1e-14 for item in trace.physics)
    assert all(abs(item.cost_energy_delta) < 1e-14 for item in trace.physics)
    assert all(item.cost_changed_phase for item in trace.physics)
    assert any(item.mixer_changed_probability for item in trace.physics)
    assert any(abs(item.mixer_energy_delta) > 1e-8 for item in trace.physics)


def test_global_gamma_zero_and_mixer_beta_zero_identities():
    diagonal = np.asarray([0.0, 0.2, 0.7, 1.0])
    initial = standard_plus_state(2)
    mixer = build_global_grover_mixer(2)
    assert np.array_equal(apply_cost_layer(initial, diagonal, 0.0), initial)
    phased = apply_cost_layer(initial, diagonal, 0.8)
    assert np.array_equal(mixer.evolve(phased, 0.0), phased)


def test_existing_penalty_x_state_evolution_regression(dynamics_context):
    parameters = np.asarray([0.41, 1.07, 0.28, 0.63])
    reference = simulate_qaoa_state(
        dynamics_context.normalized_full_diagonal,
        parameters,
        depth=2,
        solver=Q1_PENALTY_X,
    )
    trace = trace_qaoa_evolution(
        dynamics_context.full_initial_state,
        dynamics_context.normalized_full_diagonal,
        _x_mixer_evolve,
        parameters,
        depth=2,
        metadata=dynamics_context.full_metadata,
    )
    assert np.allclose(trace.statevectors[-1], reference, atol=1e-13, rtol=0.0)


def test_existing_feasible_grover_state_evolution_regression(dynamics_context):
    parameters = np.asarray([0.41, 1.07, 0.28, 0.63])
    reference = np.asarray(
        simulate_final_improvement(
            dynamics_context.feasible_initial_state,
            dynamics_context.normalized_feasible_diagonal,
            dynamics_context.feasible_mixer,
            parameters,
            depth=2,
        ).state
    )
    trace = trace_qaoa_evolution(
        dynamics_context.feasible_initial_state,
        dynamics_context.normalized_feasible_diagonal,
        dynamics_context.feasible_mixer.evolve,
        parameters,
        depth=2,
        metadata=dynamics_context.feasible_metadata,
    )
    assert trace.statevectors[-1].shape == (dynamics_context.basis.size,)
    assert np.allclose(trace.statevectors[-1], reference, atol=1e-13, rtol=0.0)
    assert all(item.p_feas == pytest.approx(1.0, abs=1e-12) for item in trace.checkpoints)


def test_decoder_separates_full_space_feasible_and_infeasible(dynamics_context):
    feasible_indices = [
        state.state_index for state in dynamics_context.states if state.is_decoder_valid
    ]
    infeasible_indices = [
        state.state_index for state in dynamics_context.states if not state.is_decoder_valid
    ]
    assert len(feasible_indices) == 20
    assert decode_valid_route(
        dynamics_context.graph, state_index_to_edge_vector(feasible_indices[0])
    ) is not None
    assert decode_valid_route(
        dynamics_context.graph, state_index_to_edge_vector(infeasible_indices[0])
    ) is None


def test_global_simulation_distributions_are_normalized_for_primary_depths(dynamics_context):
    for depth in (1, 2):
        parameters = np.linspace(0.2, 0.9, 2 * depth)
        state = simulate_global_grover_state(
            dynamics_context.normalized_full_diagonal, parameters, depth=depth
        )
        probabilities = np.abs(state) ** 2
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)


def test_regression_helper_rejects_new_algorithm_without_historical_oracle(
    dynamics_context,
):
    class FakeRun:
        algorithm = GROVER_GLOBAL

    with pytest.raises(ValueError, match="regression_oracle"):
        regression_comparison(dynamics_context, FakeRun())
