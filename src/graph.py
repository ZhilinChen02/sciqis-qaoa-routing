"""Load the routing graph and convert paths to edge bit strings."""

import json
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "graph.json"

EXPECTED_NODES = tuple(range(7))
EXPECTED_EDGE_COUNT = 14

Edge = tuple[int, int]
EdgeVector = tuple[int, ...]


def validate_edge_vector(edge_vector) -> EdgeVector:
    """Return a checked vector ``(x0, ..., x13)`` containing only 0 and 1."""

    bits = tuple(edge_vector)
    if len(bits) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"edge vector must have {EXPECTED_EDGE_COUNT} bits, got {len(bits)}"
        )
    if any(isinstance(bit, bool) or bit not in (0, 1) for bit in bits):
        raise ValueError("edge vector entries must be integer 0 or 1")
    return tuple(int(bit) for bit in bits)


def _clean_bitstring(text, name="bitstring") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be text")
    text = "".join(text.split())
    if len(text) != EXPECTED_EDGE_COUNT or any(bit not in "01" for bit in text):
        raise ValueError(f"{name} must contain exactly {EXPECTED_EDGE_COUNT} bits")
    return text


def edge_vector_to_canonical_bitstring(edge_vector) -> str:
    """Write an edge vector in project order, from q0 to q13."""

    return "".join(map(str, validate_edge_vector(edge_vector)))


def canonical_bitstring_to_edge_vector(canonical_bitstring: str) -> EdgeVector:
    """Read a q0-to-q13 bit string as an edge vector."""

    return tuple(map(int, _clean_bitstring(canonical_bitstring, "canonical bitstring")))


def edge_vector_to_qiskit_display_bitstring(edge_vector) -> str:
    """Write an edge vector in Qiskit's reversed display order."""

    return edge_vector_to_canonical_bitstring(edge_vector)[::-1]


def load_graph(path=DEFAULT_GRAPH_PATH) -> nx.DiGraph:
    """Load ``graph.json`` as a directed NetworkX graph."""

    graph_path = Path(path).resolve()
    data = json.loads(graph_path.read_text(encoding="utf-8"))

    graph = nx.DiGraph(source=data["source"], target=data["target"])
    graph.add_nodes_from(data["nodes"])

    for edge in data["edges"]:
        graph.add_edge(
            edge["u"],
            edge["v"],
            weight=edge["weight"],
            qubit_index=edge["qubit_index"],
            edge_id=edge["edge_id"],
        )

    validate_graph(graph)
    return graph


def validate_graph(graph: nx.DiGraph) -> None:
    """Check the few assumptions used by the QUBO code."""

    if not isinstance(graph, nx.DiGraph):
        raise ValueError("graph must be a NetworkX DiGraph")
    if tuple(sorted(graph.nodes)) != EXPECTED_NODES:
        raise ValueError("graph must contain nodes 0 through 6")
    if (graph.graph.get("source"), graph.graph.get("target")) != (0, 6):
        raise ValueError("source and target must be 0 and 6")
    if graph.number_of_edges() != EXPECTED_EDGE_COUNT:
        raise ValueError(f"graph must contain {EXPECTED_EDGE_COUNT} edges")

    edge_data = [data for _, _, data in graph.edges(data=True)]
    if {data.get("qubit_index") for data in edge_data} != set(range(14)):
        raise ValueError("qubit indices must be 0 through 13")
    weights = [data.get("weight") for data in edge_data]
    if any(
        not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0
        for weight in weights
    ):
        raise ValueError("edge weights must be positive integers")


def get_edge_order(graph: nx.DiGraph) -> tuple[Edge, ...]:
    """Return graph edges in q0-to-q13 order."""

    ordered = sorted(graph.edges(data=True), key=lambda edge: edge[2]["qubit_index"])
    return tuple((u, v) for u, v, _ in ordered)


def path_edges(path) -> tuple[Edge, ...]:
    """Convert a node path such as ``(0, 1, 4)`` to directed edges."""

    nodes = tuple(path)
    if len(nodes) < 2:
        raise ValueError("a path must contain at least two nodes")
    return tuple(zip(nodes, nodes[1:]))


def path_cost(graph: nx.DiGraph, path) -> int:
    """Add the weights along a directed path."""

    total = 0
    for edge in path_edges(path):
        if edge not in graph.edges:
            raise ValueError(f"path uses absent directed edge {edge}")
        total += graph.edges[edge]["weight"]
    return total


def path_to_edge_bitstring(graph: nx.DiGraph, path) -> EdgeVector:
    """Encode a path with one bit per graph edge."""

    selected = set(path_edges(path))
    missing = selected.difference(graph.edges)
    if missing:
        raise ValueError(f"path uses absent directed edges: {sorted(missing)}")
    return tuple(int(edge in selected) for edge in get_edge_order(graph))
