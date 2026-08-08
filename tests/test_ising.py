from __future__ import annotations

from fractions import Fraction

import pytest

from exact_reference import networkx_shortest_reference
from graph import load_graph, path_to_edge_bitstring
from ising import max_qubo_ising_error, qubo_to_ising, z_eigenvalues
from qubo import (
    build_qubo,
    derive_critical_penalty,
    enumerate_state_space,
    frozen_penalty_grid,
)


@pytest.fixture(scope="module")
def frozen_model():
    graph = load_graph()
    states = enumerate_state_space(graph)
    exact = networkx_shortest_reference(graph)
    critical, _critical_states = derive_critical_penalty(states, exact.cost)
    grid = frozen_penalty_grid(critical)
    qubos = tuple(build_qubo(graph, choice.value) for choice in grid)
    return graph, states, exact, grid, qubos


def test_explicit_bit_to_z_eigenvalue_convention():
    vector = (1, 0) + (0,) * 12
    assert z_eigenvalues(vector) == (-1, 1) + (1,) * 12


def test_qubo_to_ising_formula_includes_constant_and_couplings(frozen_model):
    _graph, _states, _exact, _grid, qubos = frozen_model
    qubo = qubos[0]
    ising = qubo_to_ising(qubo)

    expected_constant = qubo.constant + sum(qubo.linear, Fraction(0)) / 2 + sum(
        qubo.pair.values(), Fraction(0)
    ) / 4
    assert ising.constant == expected_constant
    assert all(ising.coupling[pair] == coefficient / 4 for pair, coefficient in qubo.pair.items())


def test_full_16384_state_qubo_ising_equality_for_every_frozen_penalty(frozen_model):
    _graph, states, _exact, grid, qubos = frozen_model
    assert tuple(choice.value for choice in grid) == (2, 5, 6, 12)
    assert max_qubo_ising_error(states, qubos) == 0
    for qubo in qubos:
        ising = qubo_to_ising(qubo)
        assert all(
            qubo.evaluate(state.edge_vector) == ising.basis_energy(state.edge_vector)
            for state in states
        )


def test_exact_route_basis_energy_is_exact_cost_for_all_penalties(frozen_model):
    graph, _states, exact, _grid, qubos = frozen_model
    vector = path_to_edge_bitstring(graph, exact.node_path)
    for qubo in qubos:
        assert qubo.evaluate(vector) == exact.cost
        assert qubo_to_ising(qubo).basis_energy(vector) == exact.cost
