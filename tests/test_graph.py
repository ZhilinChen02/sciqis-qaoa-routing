from __future__ import annotations

from copy import deepcopy
import json

import networkx as nx
import pytest

from graph import (
    EXPECTED_EDGE_COUNT,
    GraphValidationError,
    edge_bitstring_to_edges,
    get_edge_order,
    load_graph,
    path_cost,
    path_edges,
    path_to_edge_bitstring,
)


EXPECTED_EDGES = (
    (0, 1, 2),
    (0, 2, 4),
    (0, 3, 7),
    (1, 2, 1),
    (1, 3, 4),
    (1, 4, 7),
    (2, 3, 2),
    (2, 4, 3),
    (2, 5, 7),
    (3, 4, 2),
    (3, 5, 4),
    (4, 5, 2),
    (4, 6, 5),
    (5, 6, 2),
)


def test_graph_identity_and_dag():
    graph = load_graph()
    assert tuple(sorted(graph.nodes)) == tuple(range(7))
    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == EXPECTED_EDGE_COUNT
    assert graph.graph["source"] == 0
    assert graph.graph["target"] == 6
    actual = tuple(
        (u, v, graph.edges[u, v]["weight"])
        for u, v in get_edge_order(graph)
    )
    assert actual == EXPECTED_EDGES
    assert nx.is_directed_acyclic_graph(graph)


def test_edge_order_and_indices_are_deterministic_across_loads():
    first = load_graph()
    second = load_graph()
    assert get_edge_order(first) == get_edge_order(second)
    assert get_edge_order(first) == tuple((u, v) for u, v, _weight in EXPECTED_EDGES)
    assert tuple(first.edges[edge]["qubit_index"] for edge in get_edge_order(first)) == tuple(range(14))
    assert tuple(first.edges[edge]["edge_id"] for edge in get_edge_order(first)) == tuple(f"e{index:02d}" for index in range(14))


def test_exact_route_bitstring_round_trip_is_exact():
    from exact_reference import networkx_shortest_reference

    graph = load_graph()
    exact = networkx_shortest_reference(graph)
    bits = path_to_edge_bitstring(graph, exact.node_path)
    assert len(bits) == 14
    assert edge_bitstring_to_edges(graph, bits) == path_edges(exact.node_path)
    assert edge_bitstring_to_edges(graph, "".join(map(str, bits))) == exact.edge_path
    assert path_cost(graph, exact.node_path) == exact.cost


def _payload():
    with open("data/graph.json", encoding="utf-8") as handle:
        return json.load(handle)


def _write_payload(tmp_path, payload):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reverse_edge_order_consistently(payload):
    payload["edges"].reverse()
    for position, edge in enumerate(payload["edges"]):
        edge["edge_id"] = f"e{position:02d}"
        edge["qubit_index"] = position


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["nodes"].pop(), "node count must be 7"),
        (lambda payload: payload["edges"].pop(), "edge count must be 14"),
        (lambda payload: payload["edges"][-1].update(u=0, v=1), "duplicate directed edges"),
        (lambda payload: payload["edges"][0].update(weight=0), "positive integer"),
        (lambda payload: payload.update(source=99), "source is missing"),
        (lambda payload: payload["edges"][-1].update(qubit_index=12), "qubit indices must be exactly"),
        (_reverse_edge_order_consistently, "lexicographically ordered"),
    ],
)
def test_invalid_graph_contract_fails_clearly(tmp_path, mutation, message):
    payload = deepcopy(_payload())
    mutation(payload)
    with pytest.raises(GraphValidationError, match=message):
        load_graph(_write_payload(tmp_path, payload))
