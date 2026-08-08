from __future__ import annotations

import pytest

from exact_reference import networkx_shortest_reference
from graph import (
    canonical_bitstring_to_edge_vector,
    canonical_bitstring_to_qiskit_display_bitstring,
    edge_vector_to_canonical_bitstring,
    edge_vector_to_qiskit_display_bitstring,
    load_graph,
    path_to_edge_bitstring,
    qiskit_display_bitstring_to_canonical_bitstring,
    qiskit_display_bitstring_to_edge_vector,
)
from qubo import edge_vector_to_state_index, state_index_to_edge_vector


def test_exact_route_three_bit_representations_round_trip():
    graph = load_graph()
    exact = networkx_shortest_reference(graph)
    edge_vector = path_to_edge_bitstring(graph, exact.node_path)

    canonical = edge_vector_to_canonical_bitstring(edge_vector)
    qiskit_display = edge_vector_to_qiskit_display_bitstring(edge_vector)

    assert canonical == "10010001000101"
    assert qiskit_display == "10100010001001"
    assert canonical_bitstring_to_edge_vector(canonical) == edge_vector
    assert qiskit_display_bitstring_to_edge_vector(qiskit_display) == edge_vector
    assert canonical_bitstring_to_qiskit_display_bitstring(canonical) == qiskit_display
    assert qiskit_display_bitstring_to_canonical_bitstring(qiskit_display) == canonical


def test_asymmetric_vector_proves_named_text_direction():
    vector = (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert edge_vector_to_canonical_bitstring(vector) == "10100000000000"
    assert edge_vector_to_qiskit_display_bitstring(vector) == "00000000000101"


@pytest.mark.parametrize("state_index", [0, 1, 7, 10377, 16383])
def test_state_index_round_trip_has_q0_as_least_significant_bit(state_index):
    vector = state_index_to_edge_vector(state_index)
    assert edge_vector_to_state_index(vector) == state_index


@pytest.mark.parametrize(
    "function,value",
    [
        (canonical_bitstring_to_edge_vector, "0101"),
        (canonical_bitstring_to_edge_vector, "0000000000000x"),
        (qiskit_display_bitstring_to_edge_vector, "101"),
        (qiskit_display_bitstring_to_edge_vector, "20000000000000"),
        (edge_vector_to_canonical_bitstring, (False,) + (0,) * 13),
    ],
)
def test_invalid_named_bit_representation_fails(function, value):
    with pytest.raises((TypeError, ValueError)):
        function(value)
