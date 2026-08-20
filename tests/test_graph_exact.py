from __future__ import annotations

from graph import (
    canonical_bitstring_to_edge_vector,
    edge_vector_to_qiskit_display_bitstring,
    exact_route,
    get_edge_order,
    path_cost,
    path_to_edge_bitstring,
    validate_graph,
)
from qubo import decode_valid_route, edge_vector_to_state_index, state_index_to_edge_vector


def test_graph_integrity_and_variable_mapping(graph):
    validate_graph(graph)
    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 14
    assert graph.graph["source"] == 0
    assert graph.graph["target"] == 6
    assert get_edge_order(graph)[0] == (0, 1)
    assert get_edge_order(graph)[-1] == (5, 6)
    assert [graph.edges[edge]["qubit_index"] for edge in get_edge_order(graph)] == list(range(14))


def test_exact_optimum_is_verified_at_runtime(graph):
    route, cost = exact_route(graph)
    assert route == (0, 1, 2, 4, 5, 6)
    assert cost == 10
    assert path_cost(graph, route) == 10


def test_bit_order_and_route_decoding_round_trip(graph):
    route, _ = exact_route(graph)
    vector = path_to_edge_bitstring(graph, route)
    assert "".join(map(str, vector)) == "10010001000101"
    assert canonical_bitstring_to_edge_vector("10010001000101") == vector
    assert edge_vector_to_qiskit_display_bitstring(vector) == "10100010001001"
    index = edge_vector_to_state_index(vector)
    assert state_index_to_edge_vector(index) == vector
    assert decode_valid_route(graph, vector) == route
