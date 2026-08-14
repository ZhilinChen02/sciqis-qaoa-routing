"""Find the exact shortest route in two independent classical ways."""

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import networkx as nx

from graph import (
    Edge,
    get_edge_order,
    path_cost,
    path_edges,
    path_to_edge_bitstring,
    validate_graph,
)


REFERENCE_SCHEMA = "dtu-sciqis-routing-exact-reference"
REFERENCE_VERSION = "1.0"


# A small container for one complete source-to-target route.


@dataclass(frozen=True)
class RouteRecord:
    node_path: tuple[int, ...]
    edge_path: tuple[Edge, ...]
    edge_bitstring: tuple[int, ...]
    cost: int

    @property
    def bitstring_text(self) -> str:
        return "".join(str(bit) for bit in self.edge_bitstring)

    def as_dict(self) -> dict[str, object]:
        return {
            "node_path": list(self.node_path),
            "edge_path": [list(edge) for edge in self.edge_path],
            "edge_bitstring": self.bitstring_text,
            "cost": self.cost,
        }


def _route_record(graph: nx.DiGraph, node_path: tuple[int, ...], cost: int) -> RouteRecord:
    return RouteRecord(
        node_path=node_path,
        edge_path=path_edges(node_path),
        edge_bitstring=path_to_edge_bitstring(graph, node_path),
        cost=int(cost),
    )


def networkx_shortest_reference(graph: nx.DiGraph) -> RouteRecord:
    """Method A: NetworkX weighted shortest-path implementation."""

    validate_graph(graph)
    source = graph.graph["source"]
    target = graph.graph["target"]
    node_path = tuple(
        nx.shortest_path(graph, source=source, target=target, weight="weight")
    )
    reported_cost = nx.shortest_path_length(
        graph,
        source=source,
        target=target,
        weight="weight",
    )
    independently_summed_cost = path_cost(graph, node_path)
    if reported_cost != independently_summed_cost:
        raise RuntimeError(
            "NetworkX shortest-path length disagrees with direct edge-weight sum"
        )
    return _route_record(graph, node_path, independently_summed_cost)


# The second method uses an ordinary recursive depth-first search.  Agreement
# between this result and NetworkX helps catch mistakes in the graph data.


def enumerate_simple_paths_independent(graph: nx.DiGraph) -> tuple[RouteRecord, ...]:
    """Method B: deterministic DFS enumeration without NetworkX path routines."""

    validate_graph(graph)
    source = graph.graph["source"]
    target = graph.graph["target"]
    records: list[RouteRecord] = []

    def visit(node: int, node_path: tuple[int, ...], running_cost: int) -> None:
        if node == target:
            records.append(_route_record(graph, node_path, running_cost))
            return
        for successor in sorted(graph.successors(node)):
            if successor in node_path:
                continue
            weight = graph.edges[node, successor]["weight"]
            visit(successor, node_path + (successor,), running_cost + weight)

    visit(source, (source,), 0)
    if not records:
        raise RuntimeError("the frozen graph has no directed source-to-target path")
    records.sort(key=lambda record: (record.cost, record.node_path))
    return tuple(records)


def compute_exact_reference(
    graph: nx.DiGraph,
    *,
    graph_path: str | Path,
) -> tuple[dict[str, object], tuple[RouteRecord, ...]]:
    """Require exact agreement and return the frozen machine-readable record."""

    method_a = networkx_shortest_reference(graph)
    all_routes = enumerate_simple_paths_independent(graph)
    best_cost = all_routes[0].cost
    optimal_routes = tuple(route for route in all_routes if route.cost == best_cost)
    if len(optimal_routes) != 1:
        raise RuntimeError(
            f"expected one unique optimum, found {len(optimal_routes)} at cost {best_cost}"
        )
    method_b = optimal_routes[0]
    if method_a.node_path != method_b.node_path or method_a.cost != method_b.cost:
        raise RuntimeError("NetworkX and independent DFS exact references disagree")

    more_expensive_costs = [route.cost for route in all_routes if route.cost > best_cost]
    if not more_expensive_costs:
        raise RuntimeError("a second-best path is required for the Day-1 reference")
    second_best_cost = min(more_expensive_costs)
    second_best_routes = tuple(
        route for route in all_routes if route.cost == second_best_cost
    )
    graph_file = Path(graph_path).resolve()
    graph_sha256 = hashlib.sha256(graph_file.read_bytes()).hexdigest()

    edge_order = []
    for u, v in get_edge_order(graph):
        edge = graph.edges[u, v]
        edge_order.append(
            {
                "qubit_index": edge["qubit_index"],
                "edge_id": edge["edge_id"],
                "u": u,
                "v": v,
                "weight": edge["weight"],
            }
        )

    route_rows = []
    for rank, route in enumerate(all_routes, start=1):
        row = route.as_dict()
        row.update(
            rank=rank,
            is_optimal=route.cost == best_cost,
            is_second_best=route.cost == second_best_cost,
        )
        route_rows.append(row)

    payload: dict[str, object] = {
        "schema": REFERENCE_SCHEMA,
        "version": REFERENCE_VERSION,
        "graph": {
            "path": "data/graph.json",
            "sha256": graph_sha256,
            "source": graph.graph["source"],
            "target": graph.graph["target"],
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "edge_order": edge_order,
        },
        "method_a": {
            "name": "networkx_weighted_shortest_path",
            **method_a.as_dict(),
        },
        "method_b": {
            "name": "independent_deterministic_dfs_simple_path_enumeration",
            **method_b.as_dict(),
        },
        "agreement": {
            "same_node_path": method_a.node_path == method_b.node_path,
            "same_edge_path": method_a.edge_path == method_b.edge_path,
            "same_cost": method_a.cost == method_b.cost,
            "valid": True,
        },
        "exact_reference": method_b.as_dict(),
        "simple_path_count": len(all_routes),
        "unique_optimum": True,
        "second_best_cost": second_best_cost,
        "second_best_route_count": len(second_best_routes),
        "optimality_gap": second_best_cost - best_cost,
        "all_simple_paths": route_rows,
        "reference_role": "correctness and evaluation oracle; not a competing algorithm",
    }
    return payload, all_routes


def write_reference_files(
    payload: dict[str, object],
    all_routes: tuple[RouteRecord, ...],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Write deterministic JSON and compact route-table CSV artifacts."""

    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    best_cost = int(payload["exact_reference"]["cost"])
    second_best_cost = int(payload["second_best_cost"])
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "rank",
            "cost",
            "is_optimal",
            "is_second_best",
            "node_path",
            "edge_path",
            "edge_bitstring",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rank, route in enumerate(all_routes, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "cost": route.cost,
                    "is_optimal": route.cost == best_cost,
                    "is_second_best": route.cost == second_best_cost,
                    "node_path": "->".join(str(node) for node in route.node_path),
                    "edge_path": " | ".join(f"{u}->{v}" for u, v in route.edge_path),
                    "edge_bitstring": route.bitstring_text,
                }
            )


def edge_order_as_pairs(graph: nx.DiGraph) -> tuple[Edge, ...]:
    """Small public alias useful in notebooks and future stages."""

    return get_edge_order(graph)
