from __future__ import annotations

from exact_reference import (
    compute_exact_reference,
    enumerate_simple_paths_independent,
    networkx_shortest_reference,
)
from graph import DEFAULT_GRAPH_PATH, edge_bitstring_to_edges, load_graph, path_cost


def test_networkx_and_independent_enumeration_agree_exactly():
    graph = load_graph()
    method_a = networkx_shortest_reference(graph)
    routes = enumerate_simple_paths_independent(graph)
    minimum = min(route.cost for route in routes)
    optimal_routes = [route for route in routes if route.cost == minimum]

    assert len(optimal_routes) == 1
    method_b = optimal_routes[0]
    assert method_a.node_path == method_b.node_path
    assert method_a.edge_path == method_b.edge_path
    assert method_a.edge_bitstring == method_b.edge_bitstring
    assert method_a.cost == method_b.cost


def test_exact_reference_is_recomputed_from_graph_json():
    graph = load_graph()
    payload, routes = compute_exact_reference(graph, graph_path=DEFAULT_GRAPH_PATH)
    exact = payload["exact_reference"]

    assert payload["agreement"]["valid"] is True
    assert payload["simple_path_count"] == len(routes)
    assert payload["unique_optimum"] is True
    assert payload["second_best_cost"] > exact["cost"]
    assert payload["optimality_gap"] == payload["second_best_cost"] - exact["cost"]
    assert path_cost(graph, exact["node_path"]) == exact["cost"]
    assert edge_bitstring_to_edges(graph, exact["edge_bitstring"]) == tuple(
        tuple(edge) for edge in exact["edge_path"]
    )


def test_simple_path_table_is_deterministically_sorted():
    routes = enumerate_simple_paths_independent(load_graph())
    ordering = tuple((route.cost, route.node_path) for route in routes)
    assert ordering == tuple(sorted(ordering))
    assert len({route.node_path for route in routes}) == len(routes)
