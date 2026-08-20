from __future__ import annotations

from fractions import Fraction

from qubo import max_qubo_ising_error, minimum_energy_states, qubo_to_ising


def test_qubo_and_ising_match_every_basis_state(graph, states, qubo):
    assert all(
        qubo.evaluate(state.edge_vector)
        == state.routing_cost + 6 * state.flow_penalty
        for state in states
    )
    assert max_qubo_ising_error(states, [qubo]) == 0
    ising = qubo_to_ising(qubo)
    assert all(qubo.evaluate(state.edge_vector) == ising.basis_energy(state.edge_vector) for state in states)


def test_fixed_penalty_is_derived_and_sufficient(states):
    crossings = [
        Fraction(10 - state.routing_cost, state.flow_penalty)
        for state in states
        if state.flow_penalty > 0 and state.routing_cost < 10
    ]
    critical = max(crossings)
    assert critical == Fraction(5)
    energy, minimizers = minimum_energy_states(states, 6)
    assert energy == 10
    assert len(minimizers) == 1
    assert minimizers[0].decoded_route == (0, 1, 2, 4, 5, 6)


def test_flow_feasibility_equals_independent_decoder(states):
    zero_penalty = {state.state_index for state in states if state.flow_penalty == 0}
    decoded = {state.state_index for state in states if state.is_decoder_valid}
    assert zero_penalty == decoded
    assert len(zero_penalty) == 20
