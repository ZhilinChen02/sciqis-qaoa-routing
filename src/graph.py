"""Read the routing graph and convert routes to edge bit strings.

The graph itself is stored in ``data/graph.json``.  It has seven nodes and
fourteen directed edges.  Edge number ``i`` is also QUBO variable ``x_i`` and
qubit ``q_i``, so the edge order must not change during an experiment.
"""

import json
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "graph.json"

EXPECTED_NODES = tuple(range(7))
EXPECTED_EDGE_COUNT = 14
EXPECTED_QUBIT_INDICES = tuple(range(EXPECTED_EDGE_COUNT))

Edge = tuple[int, int]
EdgeVector = tuple[int, ...]


class GraphValidationError(ValueError):
    """The graph file or the NetworkX graph is not the expected course graph."""


# ---------------------------------------------------------------------------
# Bit-string conversion
# ---------------------------------------------------------------------------

def validate_edge_vector(edge_vector) -> EdgeVector:
    """Check and return a vector such as ``(1, 0, ..., 1)``."""

    values = tuple(edge_vector)
    if len(values) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"edge_vector length must be {EXPECTED_EDGE_COUNT}, got {len(values)}"
        )

    for value in values:
        # bool is a subclass of int, but True/False is not accepted here.
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError("edge_vector entries must be integer 0 or 1")

    return tuple(int(value) for value in values)


def _clean_bitstring(text, name: str) -> str:
    """Remove whitespace and check that a bit string contains fourteen bits."""

    if not isinstance(text, str):
        raise TypeError(f"{name} must be text")

    compact = "".join(text.split())
    if len(compact) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"{name} length must be {EXPECTED_EDGE_COUNT}, got {len(compact)}"
        )
    if any(character not in "01" for character in compact):
        raise ValueError(f"{name} must contain only 0 and 1")
    return compact


def edge_vector_to_canonical_bitstring(edge_vector) -> str:
    """Convert ``(x0, ..., x13)`` to text in ``q0 -> q13`` order."""

    return "".join(str(bit) for bit in validate_edge_vector(edge_vector))


def canonical_bitstring_to_edge_vector(canonical_bitstring: str) -> EdgeVector:
    """Convert text in ``q0 -> q13`` order to ``(x0, ..., x13)``."""

    compact = _clean_bitstring(canonical_bitstring, "canonical_bitstring")
    return tuple(int(character) for character in compact)


def canonical_bitstring_to_qiskit_display_bitstring(
    canonical_bitstring: str,
) -> str:
    """Reverse ``q0 -> q13`` text to Qiskit's ``q13 -> q0`` display order."""

    vector = canonical_bitstring_to_edge_vector(canonical_bitstring)
    return "".join(str(bit) for bit in reversed(vector))


def qiskit_display_bitstring_to_canonical_bitstring(
    qiskit_display_bitstring: str,
) -> str:
    """Reverse Qiskit's ``q13 -> q0`` text to the project's normal order."""

    compact = _clean_bitstring(
        qiskit_display_bitstring,
        "qiskit_display_bitstring",
    )
    return compact[::-1]


def edge_vector_to_qiskit_display_bitstring(edge_vector) -> str:
    """Convert ``(x0, ..., x13)`` directly to Qiskit display text."""

    normal_text = edge_vector_to_canonical_bitstring(edge_vector)
    return canonical_bitstring_to_qiskit_display_bitstring(normal_text)


def qiskit_display_bitstring_to_edge_vector(
    qiskit_display_bitstring: str,
) -> EdgeVector:
    """Convert Qiskit display text back to ``(x0, ..., x13)``."""

    normal_text = qiskit_display_bitstring_to_canonical_bitstring(
        qiskit_display_bitstring
    )
    return canonical_bitstring_to_edge_vector(normal_text)


# ---------------------------------------------------------------------------
# Loading and checking the graph
# ---------------------------------------------------------------------------

def _validate_payload(payload: dict, path: Path) -> None:
    """Check the values read directly from ``graph.json``."""

    if payload.get("schema") != "dtu-sciqis-routing-graph":
        raise GraphValidationError(f"{path}: unsupported or missing graph schema")
    if payload.get("version") != "1.0":
        raise GraphValidationError(f"{path}: unsupported or missing graph version")
    if payload.get("directed") is not True:
        raise GraphValidationError(f"{path}: directed must be true")

    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 7:
        actual = len(nodes) if isinstance(nodes, list) else "missing"
        raise GraphValidationError(f"{path}: node count must be 7, got {actual}")
    if tuple(nodes) != EXPECTED_NODES:
        raise GraphValidationError(
            f"{path}: nodes must be exactly {list(EXPECTED_NODES)}"
        )

    source = payload.get("source")
    target = payload.get("target")
    if source not in nodes:
        raise GraphValidationError(f"{path}: source is missing from the node list")
    if target not in nodes:
        raise GraphValidationError(f"{path}: target is missing from the node list")
    if source == target:
        raise GraphValidationError(f"{path}: source and target must be different")

    edges = payload.get("edges")
    if not isinstance(edges, list) or len(edges) != EXPECTED_EDGE_COUNT:
        actual = len(edges) if isinstance(edges, list) else "missing"
        raise GraphValidationError(
            f"{path}: edge count must be {EXPECTED_EDGE_COUNT}, got {actual}"
        )

    edge_pairs = []
    edge_ids = []
    qubit_indices = []

    for position, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise GraphValidationError(
                f"{path}: edge record {position} is missing a required field"
            )

        required = ("u", "v", "weight", "edge_id", "qubit_index")
        if any(field not in edge for field in required):
            raise GraphValidationError(
                f"{path}: edge record {position} is missing a required field"
            )

        u = edge["u"]
        v = edge["v"]
        weight = edge["weight"]
        edge_id = edge["edge_id"]
        qubit_index = edge["qubit_index"]

        if not isinstance(u, int) or not isinstance(v, int):
            raise GraphValidationError(
                f"{path}: edge record {position} has invalid endpoints"
            )
        if u not in nodes or v not in nodes:
            raise GraphValidationError(
                f"{path}: edge record {position} has invalid endpoints"
            )
        if u == v:
            raise GraphValidationError(f"{path}: self-loop {(u, v)} is not allowed")
        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise GraphValidationError(
                f"{path}: edge {(u, v)} weight must be a positive integer"
            )
        if not isinstance(qubit_index, int) or isinstance(qubit_index, bool):
            raise GraphValidationError(f"{path}: qubit_index must be an integer")
        if edge_id != f"e{position:02d}":
            raise GraphValidationError(
                f"{path}: edge_id at position {position} must be e{position:02d}"
            )

        edge_pairs.append((u, v))
        edge_ids.append(edge_id)
        qubit_indices.append(qubit_index)

    if len(set(edge_pairs)) != len(edge_pairs):
        raise GraphValidationError(f"{path}: duplicate directed edges are not allowed")
    if len(set(edge_ids)) != len(edge_ids):
        raise GraphValidationError(f"{path}: edge_id values must be unique")
    if tuple(qubit_indices) != EXPECTED_QUBIT_INDICES:
        raise GraphValidationError(
            f"{path}: qubit indices must be exactly {list(EXPECTED_QUBIT_INDICES)}"
        )
    if edge_pairs != sorted(edge_pairs):
        raise GraphValidationError(
            f"{path}: edge records must be lexicographically ordered by (u, v)"
        )
    if payload.get("edge_order") != "lexicographic_by_(u,v)":
        raise GraphValidationError(f"{path}: edge_order contract is inconsistent")


def load_graph(path=DEFAULT_GRAPH_PATH) -> nx.DiGraph:
    """Read ``graph.json`` and return a checked NetworkX directed graph."""

    graph_path = Path(path).resolve()
    payload = json.loads(graph_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise GraphValidationError(
            f"{graph_path}: top-level JSON value must be an object"
        )
    _validate_payload(payload, graph_path)

    # Start with an empty directed graph.
    graph = nx.DiGraph()

    # Information about the whole routing problem is stored on graph.graph.
    graph.graph["schema"] = payload["schema"]
    graph.graph["version"] = payload["version"]
    graph.graph["name"] = payload["name"]
    graph.graph["directed"] = True
    graph.graph["source"] = payload["source"]
    graph.graph["target"] = payload["target"]
    graph.graph["edge_order"] = payload["edge_order"]
    graph.graph["source_path"] = str(graph_path)

    graph.add_nodes_from(payload["nodes"])

    # Each JSON record becomes one directed and weighted edge.
    for edge in payload["edges"]:
        graph.add_edge(
            edge["u"],
            edge["v"],
            weight=edge["weight"],
            edge_id=edge["edge_id"],
            qubit_index=edge["qubit_index"],
        )

    validate_graph(graph)
    return graph


def validate_graph(graph: nx.DiGraph) -> None:
    """Check a NetworkX graph against the fixed seven-node course problem."""

    if not isinstance(graph, nx.DiGraph) or not graph.is_directed():
        raise GraphValidationError("graph must be a directed NetworkX DiGraph")
    if graph.number_of_nodes() != 7:
        raise GraphValidationError(
            f"node count must be 7, got {graph.number_of_nodes()}"
        )
    if tuple(sorted(graph.nodes)) != EXPECTED_NODES:
        raise GraphValidationError(f"nodes must be exactly {list(EXPECTED_NODES)}")

    source = graph.graph.get("source")
    target = graph.graph.get("target")
    if source not in graph:
        raise GraphValidationError("source is missing from the graph")
    if target not in graph:
        raise GraphValidationError("target is missing from the graph")
    if source == target:
        raise GraphValidationError("source and target must be different")
    if source != 0 or target != 6:
        raise GraphValidationError("the frozen source/target contract must be 0 -> 6")

    if graph.number_of_edges() != EXPECTED_EDGE_COUNT:
        raise GraphValidationError(
            f"edge count must be {EXPECTED_EDGE_COUNT}, got {graph.number_of_edges()}"
        )

    records = []
    for u, v, attributes in graph.edges(data=True):
        weight = attributes.get("weight")
        edge_id = attributes.get("edge_id")
        qubit_index = attributes.get("qubit_index")

        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise GraphValidationError(
                f"edge {(u, v)} weight must be a positive integer"
            )
        if not isinstance(qubit_index, int) or isinstance(qubit_index, bool):
            raise GraphValidationError(f"edge {(u, v)} has an invalid qubit_index")
        if not isinstance(edge_id, str):
            raise GraphValidationError(f"edge {(u, v)} has an invalid edge_id")

        records.append((qubit_index, edge_id, (u, v), weight))

    records.sort(key=lambda item: item[0])
    indices = tuple(item[0] for item in records)
    edge_ids = tuple(item[1] for item in records)
    edge_order = tuple(item[2] for item in records)

    if indices != EXPECTED_QUBIT_INDICES:
        raise GraphValidationError(
            f"qubit indices must be exactly {list(EXPECTED_QUBIT_INDICES)}"
        )

    expected_ids = tuple(f"e{index:02d}" for index in EXPECTED_QUBIT_INDICES)
    if edge_ids != expected_ids:
        raise GraphValidationError("edge_id values are inconsistent with qubit indices")
    if edge_order != tuple(sorted(graph.edges())):
        raise GraphValidationError("edge ordering is not lexicographic by (u, v)")
    if graph.graph.get("edge_order") != "lexicographic_by_(u,v)":
        raise GraphValidationError("edge_order graph metadata is inconsistent")
    if not nx.is_directed_acyclic_graph(graph):
        raise GraphValidationError("the frozen routing graph must be a DAG")


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def get_edge_order(graph: nx.DiGraph) -> tuple[Edge, ...]:
    """Return edges in qubit order: q0 edge first and q13 edge last."""

    validate_graph(graph)
    numbered_edges = []
    for u, v, attributes in graph.edges(data=True):
        numbered_edges.append((attributes["qubit_index"], (u, v)))
    numbered_edges.sort()
    return tuple(edge for _index, edge in numbered_edges)


def path_edges(path) -> tuple[Edge, ...]:
    """Convert ``(0, 1, 4, 6)`` to ``((0, 1), (1, 4), (4, 6))``."""

    if len(path) < 2:
        raise ValueError("a path must contain at least two nodes")

    edges = []
    for position in range(len(path) - 1):
        edges.append((path[position], path[position + 1]))
    return tuple(edges)


def path_cost(graph: nx.DiGraph, path) -> int:
    """Add the weights of all directed edges in a path."""

    validate_graph(graph)
    total = 0
    for edge in path_edges(path):
        if not graph.has_edge(*edge):
            raise ValueError(f"path uses absent directed edge {edge}")
        total += graph.edges[edge]["weight"]
    return total


def path_to_edge_bitstring(graph: nx.DiGraph, path) -> EdgeVector:
    """Mark the edges used by a path with 1 and all other edges with 0."""

    selected_edges = set(path_edges(path))
    missing_edges = selected_edges.difference(graph.edges())
    if missing_edges:
        raise ValueError(f"path uses absent directed edges: {sorted(missing_edges)}")

    bits = []
    for edge in get_edge_order(graph):
        bits.append(1 if edge in selected_edges else 0)
    return tuple(bits)


def edge_bitstring_to_edges(graph: nx.DiGraph, bitstring) -> tuple[Edge, ...]:
    """Return the graph edges whose corresponding bits are 1."""

    if isinstance(bitstring, str):
        compact = "".join(bitstring.split())
        if any(character not in "01" for character in compact):
            raise ValueError("bitstring must contain only 0 and 1")
        bits = tuple(int(character) for character in compact)
    else:
        raw_bits = tuple(bitstring)
        if any(value not in (0, 1) for value in raw_bits):
            raise ValueError("bitstring entries must be 0 or 1")
        bits = tuple(int(value) for value in raw_bits)

    if len(bits) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"bitstring length must be {EXPECTED_EDGE_COUNT}, got {len(bits)}"
        )

    selected_edges = []
    for edge, bit in zip(get_edge_order(graph), bits):
        if bit == 1:
            selected_edges.append(edge)
    return tuple(selected_edges)


def edge_order_records(graph: nx.DiGraph) -> list[dict]:
    """Return the edge order in a format that is easy to print or save as JSON."""

    records = []
    for u, v in get_edge_order(graph):
        attributes = graph.edges[u, v]
        records.append(
            {
                "qubit_index": attributes["qubit_index"],
                "edge_id": attributes["edge_id"],
                "u": u,
                "v": v,
                "weight": attributes["weight"],
            }
        )
    return records
