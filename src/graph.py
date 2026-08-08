"""Frozen Day-1 weighted-routing graph and deterministic edge-bit utilities.

``data/graph.json`` is the canonical scientific contract. In particular,
``qubit_index`` defines the edge-variable order that future project stages must
reuse; NetworkX edge iteration order is never used for that purpose.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "graph.json"
EXPECTED_NODES = tuple(range(7))
EXPECTED_EDGE_COUNT = 14
EXPECTED_QUBIT_INDICES = tuple(range(EXPECTED_EDGE_COUNT))

Edge = tuple[int, int]
EdgeVector = tuple[int, ...]


class GraphValidationError(ValueError):
    """Raised when the frozen graph contract is malformed or inconsistent."""


def validate_edge_vector(edge_vector: Iterable[int]) -> EdgeVector:
    """Return a validated canonical ``[x0, ..., x13]`` edge vector.

    This function deliberately does not accept strings: textual representations
    have named conversion helpers below so that their direction is never
    implicit.
    """

    values = tuple(edge_vector)
    if len(values) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"edge_vector length must be {EXPECTED_EDGE_COUNT}, got {len(values)}"
        )
    if any(isinstance(value, bool) or value not in (0, 1) for value in values):
        raise ValueError("edge_vector entries must be integer 0 or 1")
    return tuple(int(value) for value in values)


def edge_vector_to_canonical_bitstring(edge_vector: Iterable[int]) -> str:
    """Encode ``[x0, ..., x13]`` as text in explicit ``q0 -> q13`` order."""

    return "".join(str(bit) for bit in validate_edge_vector(edge_vector))


def canonical_bitstring_to_edge_vector(canonical_bitstring: str) -> EdgeVector:
    """Decode text in explicit ``q0 -> q13`` order to ``[x0, ..., x13]``."""

    if not isinstance(canonical_bitstring, str):
        raise TypeError("canonical_bitstring must be text")
    compact = "".join(canonical_bitstring.split())
    if len(compact) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            "canonical_bitstring length must be "
            f"{EXPECTED_EDGE_COUNT}, got {len(compact)}"
        )
    if any(character not in "01" for character in compact):
        raise ValueError("canonical_bitstring must contain only 0 and 1")
    return tuple(int(character) for character in compact)


def canonical_bitstring_to_qiskit_display_bitstring(
    canonical_bitstring: str,
) -> str:
    """Convert named ``q0 -> q13`` text to Qiskit-style ``q13 -> q0`` text."""

    vector = canonical_bitstring_to_edge_vector(canonical_bitstring)
    return "".join(str(bit) for bit in reversed(vector))


def qiskit_display_bitstring_to_canonical_bitstring(
    qiskit_display_bitstring: str,
) -> str:
    """Convert named Qiskit-style ``q13 -> q0`` text to ``q0 -> q13`` text."""

    if not isinstance(qiskit_display_bitstring, str):
        raise TypeError("qiskit_display_bitstring must be text")
    compact = "".join(qiskit_display_bitstring.split())
    if len(compact) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            "qiskit_display_bitstring length must be "
            f"{EXPECTED_EDGE_COUNT}, got {len(compact)}"
        )
    if any(character not in "01" for character in compact):
        raise ValueError("qiskit_display_bitstring must contain only 0 and 1")
    return edge_vector_to_canonical_bitstring(reversed(tuple(map(int, compact))))


def edge_vector_to_qiskit_display_bitstring(edge_vector: Iterable[int]) -> str:
    """Encode ``[x0, ..., x13]`` as named Qiskit-style ``q13 -> q0`` text."""

    canonical = edge_vector_to_canonical_bitstring(edge_vector)
    return canonical_bitstring_to_qiskit_display_bitstring(canonical)


def qiskit_display_bitstring_to_edge_vector(
    qiskit_display_bitstring: str,
) -> EdgeVector:
    """Decode named Qiskit-style ``q13 -> q0`` text to ``[x0, ..., x13]``."""

    canonical = qiskit_display_bitstring_to_canonical_bitstring(
        qiskit_display_bitstring
    )
    return canonical_bitstring_to_edge_vector(canonical)


def _validate_payload(payload: dict[str, Any], path: Path) -> None:
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
        raise GraphValidationError(f"{path}: nodes must be exactly {list(EXPECTED_NODES)}")

    source = payload.get("source")
    target = payload.get("target")
    if source not in nodes:
        raise GraphValidationError(f"{path}: source is missing from the node list")
    if target not in nodes:
        raise GraphValidationError(f"{path}: target is missing from the node list")
    if source == target:
        raise GraphValidationError(f"{path}: source and target must be different")

    records = payload.get("edges")
    if not isinstance(records, list) or len(records) != EXPECTED_EDGE_COUNT:
        actual = len(records) if isinstance(records, list) else "missing"
        raise GraphValidationError(
            f"{path}: edge count must be {EXPECTED_EDGE_COUNT}, got {actual}"
        )

    pairs: list[Edge] = []
    edge_ids: list[str] = []
    indices: list[int] = []
    for position, record in enumerate(records):
        try:
            u = record["u"]
            v = record["v"]
            weight = record["weight"]
            edge_id = record["edge_id"]
            qubit_index = record["qubit_index"]
        except (KeyError, TypeError) as exc:
            raise GraphValidationError(
                f"{path}: edge record {position} is missing a required field"
            ) from exc
        if not isinstance(u, int) or not isinstance(v, int) or u not in nodes or v not in nodes:
            raise GraphValidationError(f"{path}: edge record {position} has invalid endpoints")
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
        pairs.append((u, v))
        edge_ids.append(edge_id)
        indices.append(qubit_index)

    if len(set(pairs)) != len(pairs):
        raise GraphValidationError(f"{path}: duplicate directed edges are not allowed")
    if len(set(edge_ids)) != len(edge_ids):
        raise GraphValidationError(f"{path}: edge_id values must be unique")
    if tuple(indices) != EXPECTED_QUBIT_INDICES:
        raise GraphValidationError(
            f"{path}: qubit indices must be exactly {list(EXPECTED_QUBIT_INDICES)}"
        )
    if pairs != sorted(pairs):
        raise GraphValidationError(
            f"{path}: edge records must be lexicographically ordered by (u, v)"
        )
    if payload.get("edge_order") != "lexicographic_by_(u,v)":
        raise GraphValidationError(f"{path}: edge_order contract is inconsistent")


def load_graph(path: str | Path = DEFAULT_GRAPH_PATH) -> nx.DiGraph:
    """Load and validate the canonical JSON graph as a NetworkX ``DiGraph``."""

    graph_path = Path(path).resolve()
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GraphValidationError(f"{graph_path}: top-level JSON value must be an object")
    _validate_payload(payload, graph_path)

    graph = nx.DiGraph()
    graph.graph.update(
        schema=payload["schema"],
        version=payload["version"],
        name=payload["name"],
        directed=True,
        source=payload["source"],
        target=payload["target"],
        edge_order=payload["edge_order"],
        source_path=str(graph_path),
    )
    graph.add_nodes_from(payload["nodes"])
    for record in payload["edges"]:
        graph.add_edge(
            record["u"],
            record["v"],
            weight=record["weight"],
            edge_id=record["edge_id"],
            qubit_index=record["qubit_index"],
        )
    validate_graph(graph)
    return graph


def validate_graph(graph: nx.DiGraph) -> None:
    """Fail clearly unless ``graph`` satisfies the complete Day-1 contract."""

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

    records: list[tuple[int, str, Edge, int]] = []
    for u, v, attributes in graph.edges(data=True):
        weight = attributes.get("weight")
        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise GraphValidationError(f"edge {(u, v)} weight must be a positive integer")
        qubit_index = attributes.get("qubit_index")
        edge_id = attributes.get("edge_id")
        if not isinstance(qubit_index, int) or isinstance(qubit_index, bool):
            raise GraphValidationError(f"edge {(u, v)} has an invalid qubit_index")
        if not isinstance(edge_id, str):
            raise GraphValidationError(f"edge {(u, v)} has an invalid edge_id")
        records.append((qubit_index, edge_id, (u, v), weight))

    ordered = sorted(records, key=lambda record: record[0])
    indices = tuple(record[0] for record in ordered)
    if indices != EXPECTED_QUBIT_INDICES:
        raise GraphValidationError(
            f"qubit indices must be exactly {list(EXPECTED_QUBIT_INDICES)}"
        )
    if tuple(record[1] for record in ordered) != tuple(
        f"e{index:02d}" for index in EXPECTED_QUBIT_INDICES
    ):
        raise GraphValidationError("edge_id values are inconsistent with qubit indices")
    edge_order = tuple(record[2] for record in ordered)
    if edge_order != tuple(sorted(graph.edges())):
        raise GraphValidationError("edge ordering is not lexicographic by (u, v)")
    if graph.graph.get("edge_order") != "lexicographic_by_(u,v)":
        raise GraphValidationError("edge_order graph metadata is inconsistent")
    if not nx.is_directed_acyclic_graph(graph):
        raise GraphValidationError("the frozen routing graph must be a DAG")


def get_edge_order(graph: nx.DiGraph) -> tuple[Edge, ...]:
    """Return edges in the frozen qubit order, never NetworkX iteration order."""

    validate_graph(graph)
    return tuple(
        edge
        for _index, edge in sorted(
            (attributes["qubit_index"], (u, v))
            for u, v, attributes in graph.edges(data=True)
        )
    )


def path_edges(path: Sequence[int]) -> tuple[Edge, ...]:
    """Convert a node path to its consecutive directed edges."""

    if len(path) < 2:
        raise ValueError("a path must contain at least two nodes")
    return tuple(zip(path[:-1], path[1:]))


def path_cost(graph: nx.DiGraph, path: Sequence[int]) -> int:
    """Return the exact integer weight of a directed node path."""

    validate_graph(graph)
    total = 0
    for edge in path_edges(path):
        if not graph.has_edge(*edge):
            raise ValueError(f"path uses absent directed edge {edge}")
        total += graph.edges[edge]["weight"]
    return total


def path_to_edge_bitstring(graph: nx.DiGraph, path: Sequence[int]) -> tuple[int, ...]:
    """Encode a directed node path in the frozen edge/qubit ordering."""

    selected = set(path_edges(path))
    absent = selected.difference(graph.edges())
    if absent:
        raise ValueError(f"path uses absent directed edges: {sorted(absent)}")
    return tuple(int(edge in selected) for edge in get_edge_order(graph))


def edge_bitstring_to_edges(
    graph: nx.DiGraph,
    bitstring: str | Iterable[int],
) -> tuple[Edge, ...]:
    """Decode selected edges from a length-14 bitstring in qubit order."""

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
    return tuple(edge for edge, bit in zip(get_edge_order(graph), bits) if bit)


def edge_order_records(graph: nx.DiGraph) -> list[dict[str, int | str]]:
    """Return a presentation/JSON-friendly view of the frozen edge order."""

    records = []
    for edge in get_edge_order(graph):
        attributes = graph.edges[edge]
        records.append(
            {
                "qubit_index": attributes["qubit_index"],
                "edge_id": attributes["edge_id"],
                "u": edge[0],
                "v": edge[1],
                "weight": attributes["weight"],
            }
        )
    return records
