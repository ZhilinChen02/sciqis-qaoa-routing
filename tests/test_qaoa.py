from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from ising import qubo_to_ising
from optimization import seeded_initial_parameters
from qaoa import (
    Q1_PENALTY_X,
    Q2_WARM_START,
    build_qaoa_circuit,
    normalized_diagonal,
    simulate_qaoa_state,
    state_probabilities,
)
from warm_start import incumbent_relaxation


def test_parameter_seed_and_statevector_normalization(graph, cost_diagonal):
    assert np.allclose(
        seeded_initial_parameters(1, 2601),
        [1.3789715463539463, 1.4497639358789494],
    )
    warm = [row.clipped_value for row in incumbent_relaxation(graph, 0.1)]
    cases = [
        (Q1_PENALTY_X, 1, None),
        (Q2_WARM_START, 1, warm),
        (Q2_WARM_START, 2, warm),
    ]
    for solver, depth, values in cases:
        parameters = seeded_initial_parameters(depth, 2601)
        state = simulate_qaoa_state(
            cost_diagonal, parameters, depth=depth, solver=solver,
            warm_start_values=values,
        )
        probabilities = state_probabilities(state)
        assert state.shape == (2**14,)
        assert np.isclose(np.sum(probabilities), 1.0, atol=1e-12)


def test_matrix_and_generated_circuit_probabilities_agree(graph, states, qubo):
    raw = np.asarray([float(qubo.evaluate(state.edge_vector)) for state in states])
    diagonal, _, scale = normalized_diagonal(raw)
    ising = qubo_to_ising(qubo)
    warm = [row.clipped_value for row in incumbent_relaxation(graph, 0.1)]
    for solver, values in ((Q1_PENALTY_X, None), (Q2_WARM_START, warm)):
        parameters = seeded_initial_parameters(1, 2601)
        expected = state_probabilities(
            simulate_qaoa_state(
                diagonal, parameters, depth=1, solver=solver,
                warm_start_values=values,
            )
        )
        circuit = build_qaoa_circuit(
            ising, parameters, depth=1, solver=solver,
            normalization_scale=scale, warm_start_values=values,
        )
        actual = np.abs(Statevector.from_instruction(circuit).data) ** 2
        assert np.max(np.abs(actual - expected)) < 1e-10
