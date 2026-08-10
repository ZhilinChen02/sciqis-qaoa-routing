from __future__ import annotations

from math import sqrt

import numpy as np

from graph import path_cost
from warm_start import (
    apply_single_qubit_hamiltonian_rotation,
    apply_warm_start_mixer,
    clip_relaxed_values,
    greedy_incumbent_route,
    incumbent_product_warm_start_state,
    incumbent_relaxation,
    mixer_hamiltonian,
    product_state,
    warm_start_single_qubit_mixer,
)


def test_incumbent_relaxation_bounds_and_clipping(graph):
    assert np.allclose(clip_relaxed_values([0.0, 0.4, 1.0], 0.1), [0.1, 0.4, 0.9])
    route = greedy_incumbent_route(graph)
    assert route == (0, 1, 2, 3, 4, 5, 6)
    assert path_cost(graph, route) == 11
    rows = incumbent_relaxation(graph, 0.1)
    assert len(rows) == 14
    assert all(0.0 <= row.relaxed_value <= 1.0 for row in rows)
    assert all(0.0 < row.clipped_value < 1.0 for row in rows)
    assert {row.clipped_value for row in rows} == {0.1, 0.9}


def test_prepared_single_qubit_probabilities_match_clipped_values(graph):
    values = [row.clipped_value for row in incumbent_relaxation(graph, 0.1)]
    state = product_state(values)
    indices = np.arange(len(state), dtype=np.uint64)
    assert np.isclose(np.linalg.norm(state), 1.0)
    for qubit, expected in enumerate(values):
        selected = ((indices >> np.uint64(qubit)) & np.uint64(1)).astype(bool)
        assert np.isclose(np.sum(np.abs(state[selected]) ** 2), expected, atol=1e-12)


def test_warm_start_mixer_has_prepared_state_as_ground_state():
    for probability_one in (0.1, 0.37, 0.9):
        hamiltonian = mixer_hamiltonian(probability_one)
        state = np.asarray([sqrt(1.0 - probability_one), sqrt(probability_one)])
        assert np.allclose(hamiltonian, hamiltonian.conj().T)
        assert np.allclose(hamiltonian @ hamiltonian, np.eye(2))
        assert np.allclose(hamiltonian @ state, -state)


def _frozen_source_state(bits, epsilon):
    """Literal tiny reference from source simulation.py SHA 902188..."""
    dimension = 1 << len(bits)
    amplitudes = np.ones(dimension)
    indices = np.arange(dimension, dtype=np.uint64)
    for qubit, bit in enumerate(bits):
        c_value = 1.0 - epsilon if bit else epsilon
        selected = ((indices >> np.uint64(qubit)) & np.uint64(1)).astype(bool)
        amplitudes *= np.where(selected, sqrt(c_value), sqrt(1.0 - c_value))
    return amplitudes.astype(complex)


def _frozen_source_mixer(bit, epsilon):
    c_value = 1.0 - epsilon if bit else epsilon
    x_value = 2.0 * sqrt(c_value * (1.0 - c_value))
    z_value = 1.0 - 2.0 * c_value
    return np.asarray([[-z_value, -x_value], [-x_value, z_value]], dtype=complex)


def test_migrated_q2_primitive_matches_frozen_source_on_synthetic_input():
    """Self-contained source/target equivalence; no external repo import."""
    bits = (1, 0, 1)
    epsilon = 0.1
    assert np.array_equal(
        incumbent_product_warm_start_state(bits, epsilon),
        _frozen_source_state(bits, epsilon),
    )
    for bit in (0, 1):
        assert np.array_equal(
            warm_start_single_qubit_mixer(bit, epsilon),
            _frozen_source_mixer(bit, epsilon),
        )
    initial = _frozen_source_state(bits, epsilon)
    target = apply_warm_start_mixer(initial, 0.37, [0.9, 0.1, 0.9])
    reference = initial.copy()
    for qubit, bit in enumerate(bits):
        reference = apply_single_qubit_hamiltonian_rotation(
            reference, qubit, 0.37, _frozen_source_mixer(bit, epsilon)
        )
    assert np.array_equal(target, reference)
