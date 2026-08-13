from __future__ import annotations

from fractions import Fraction

from qubo import max_qubo_ising_error, qubo_to_ising
from qubo import (
    derive_critical_penalty,
    flow_feasibility_verdict,
    max_qubo_expansion_error,
    minimum_energy_states,
)


def test_qubo_and_ising_match_every_basis_state(graph, states, qubo):
    assert max_qubo_expansion_error(graph, states, [6]) == 0
    assert max_qubo_ising_error(states, [qubo]) == 0
    ising = qubo_to_ising(qubo)
    assert all(qubo.evaluate(state.edge_vector) == ising.basis_energy(state.edge_vector) for state in states)


def test_fixed_penalty_is_derived_and_sufficient(states):
    critical, critical_states = derive_critical_penalty(states, 10)
    assert critical == Fraction(5)
    assert critical_states
    energy, minimizers = minimum_energy_states(states, 6)
    assert energy == 10
    assert len(minimizers) == 1
    assert minimizers[0].decoded_route == (0, 1, 2, 4, 5, 6)


def test_flow_feasibility_equals_independent_decoder(states):
    verdict = flow_feasibility_verdict(states)
    assert verdict["sets_identical"] is True
    assert verdict["flow_penalty_zero_count"] == 20
    assert verdict["decoder_valid_route_count"] == 20
