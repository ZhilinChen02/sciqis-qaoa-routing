from __future__ import annotations

from fractions import Fraction

import pytest

from exact_reference import enumerate_simple_paths_independent, networkx_shortest_reference
from graph import get_edge_order, load_graph, path_to_edge_bitstring
from qubo import (
    build_qubo,
    decode_valid_route,
    derive_critical_penalty,
    enumerate_state_space,
    flow_feasibility_verdict,
    flow_penalty,
    flow_penalty_coefficients,
    frozen_penalty_grid,
    incidence_matrix,
    max_qubo_expansion_error,
    minimum_energy_states,
    node_flow_residuals,
    node_supplies,
    routing_cost,
)


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture(scope="module")
def states(graph):
    records = enumerate_state_space(graph)
    assert len(records) == 16384
    return records


def test_node_incidence_and_supply_conventions(graph):
    edges = get_edge_order(graph)
    matrix = incidence_matrix(graph)
    supplies = node_supplies(graph)
    nodes = tuple(sorted(graph.nodes))

    assert supplies == {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: -1}
    assert len(matrix) == 7
    assert all(len(row) == 14 for row in matrix)
    for index, (u, v) in enumerate(edges):
        column = {node: matrix[row_index][index] for row_index, node in enumerate(nodes)}
        assert column[u] == 1
        assert column[v] == -1
        assert sum(column.values()) == 0
        assert all(value == 0 for node, value in column.items() if node not in (u, v))


def test_every_day1_simple_route_has_zero_flow_penalty(graph):
    routes = enumerate_simple_paths_independent(graph)
    assert len(routes) == 20
    for route in routes:
        vector = route.edge_bitstring
        assert node_flow_residuals(graph, vector) == {node: 0 for node in graph.nodes}
        assert flow_penalty(graph, vector) == 0
        assert decode_valid_route(graph, vector) == route.node_path
        assert routing_cost(graph, vector) == route.cost


def test_intentionally_invalid_selections_have_positive_penalty(graph):
    empty = (0,) * 14
    disconnected_edge = (0,) * 11 + (1, 0, 0)
    assert flow_penalty(graph, empty) == 2
    assert flow_penalty(graph, disconnected_edge) > 0
    assert decode_valid_route(graph, empty) is None
    assert decode_valid_route(graph, disconnected_edge) is None


def test_expanded_flow_penalty_matches_direct_formula_for_all_states(graph, states):
    expansion = flow_penalty_coefficients(graph)
    assert expansion.constant == 2
    assert all(expansion.evaluate(state.edge_vector) == state.flow_penalty for state in states)


def test_full_state_qubo_direct_equals_coefficient_polynomial(graph, states):
    representative_penalties = (Fraction(1, 2), Fraction(2), Fraction(5), Fraction(6), Fraction(12))
    assert max_qubo_expansion_error(graph, states, representative_penalties) == 0


def test_zero_flow_penalty_iff_independent_decoder_valid_over_all_states(graph, states):
    verdict = flow_feasibility_verdict(states)
    day1_route_count = len(enumerate_simple_paths_independent(graph))
    assert verdict == {
        "flow_penalty_zero_count": day1_route_count,
        "decoder_valid_route_count": day1_route_count,
        "sets_identical": True,
        "zero_only_state_indices": [],
        "decoder_only_state_indices": [],
    }


def test_critical_penalty_and_frozen_grid_are_derived_consistently(graph, states):
    exact = networkx_shortest_reference(graph)
    critical, critical_states = derive_critical_penalty(states, exact.cost)
    crossings = [
        Fraction(exact.cost - state.routing_cost, state.flow_penalty)
        for state in states
        if state.flow_penalty > 0 and state.routing_cost < exact.cost
    ]

    assert critical == max(crossings) == Fraction(5)
    assert len(critical_states) == 1
    assert critical_states[0].edge_vector == (0,) * 14
    assert critical_states[0].routing_cost == 0
    assert critical_states[0].flow_penalty == 2

    grid = frozen_penalty_grid(critical)
    assert tuple(choice.label for choice in grid) == (
        "weak",
        "critical",
        "just-supercritical",
        "strong",
    )
    assert tuple(choice.value for choice in grid) == (2, 5, 6, 12)
    assert grid[0].value < critical == grid[1].value < grid[2].value < grid[3].value


def test_penalty_minima_cross_boundary_and_preserve_day1_optimum(graph, states):
    exact = networkx_shortest_reference(graph)
    exact_vector = path_to_edge_bitstring(graph, exact.node_path)
    critical, _critical_states = derive_critical_penalty(states, exact.cost)
    grid = frozen_penalty_grid(critical)

    weak_energy, weak_minimizers = minimum_energy_states(states, grid[0].value)
    assert weak_energy < exact.cost
    assert all(not state.is_decoder_valid for state in weak_minimizers)

    boundary_energy, boundary_minimizers = minimum_energy_states(states, grid[1].value)
    assert boundary_energy == exact.cost
    assert {state.is_decoder_valid for state in boundary_minimizers} == {False, True}

    for choice in grid[2:]:
        energy, minimizers = minimum_energy_states(states, choice.value)
        assert energy == exact.cost
        assert len(minimizers) == 1
        assert minimizers[0].edge_vector == exact_vector
