"""Build the routing QUBO and enumerate its 14-bit state space.

For edge variables ``x``, the energy is

    selected edge weights + A * sum(flow_error[node] ** 2)

The source sends one unit of flow, the target receives one unit, and all other
nodes conserve flow.
"""

from dataclasses import dataclass
from fractions import Fraction

import networkx as nx

from graph import (
    Edge,
    EdgeVector,
    EXPECTED_EDGE_COUNT,
    edge_vector_to_canonical_bitstring,
    get_edge_order,
    validate_edge_vector,
    validate_graph,
)


Number = int | float | Fraction
Pair = tuple[int, int]


def _fraction(value: Number) -> Fraction:
    """Use exact decimal coefficients in the small teaching model."""

    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


@dataclass(frozen=True)
class QuboPolynomial:
    constant: Fraction
    linear: tuple[Fraction, ...]
    pair: dict[Pair, Fraction]

    def evaluate(self, edge_vector) -> Fraction:
        """Evaluate the polynomial for one edge selection."""

        x = validate_edge_vector(edge_vector)
        linear_energy = sum(
            coefficient * x[i] for i, coefficient in enumerate(self.linear)
        )
        pair_energy = sum(
            coefficient * x[i] * x[j]
            for (i, j), coefficient in self.pair.items()
        )
        return self.constant + linear_energy + pair_energy


@dataclass(frozen=True)
class StateRecord:
    """Classical information about one of the 16,384 edge selections."""

    state_index: int
    edge_vector: EdgeVector
    routing_cost: int
    flow_residuals: tuple[int, ...]
    flow_penalty: int
    decoded_route: tuple[int, ...] | None

    @property
    def canonical_bitstring(self) -> str:
        return edge_vector_to_canonical_bitstring(self.edge_vector)

    @property
    def is_decoder_valid(self) -> bool:
        return self.decoded_route is not None


def node_supplies(graph: nx.DiGraph) -> dict[int, int]:
    """Return +1 for the source, -1 for the target, and 0 elsewhere."""

    source = graph.graph["source"]
    target = graph.graph["target"]
    return {
        node: 1 if node == source else -1 if node == target else 0
        for node in sorted(graph.nodes)
    }


def incidence_matrix(graph: nx.DiGraph) -> tuple[tuple[int, ...], ...]:
    """Return the node-edge matrix, using +1 outgoing and -1 incoming."""

    edges = get_edge_order(graph)
    return tuple(
        tuple(int(u == node) - int(v == node) for u, v in edges)
        for node in sorted(graph.nodes)
    )


def routing_cost(graph: nx.DiGraph, edge_vector) -> int:
    """Add the weights of all selected edges."""

    x = validate_edge_vector(edge_vector)
    return sum(
        graph.edges[edge]["weight"] * bit
        for edge, bit in zip(get_edge_order(graph), x)
    )


def node_flow_residuals(graph: nx.DiGraph, edge_vector) -> dict[int, int]:
    """Calculate outgoing flow - incoming flow - required supply."""

    x = validate_edge_vector(edge_vector)
    supplies = node_supplies(graph)
    return {
        node: sum(coefficient * bit for coefficient, bit in zip(row, x))
        - supplies[node]
        for node, row in zip(sorted(graph.nodes), incidence_matrix(graph))
    }


def flow_penalty(graph: nx.DiGraph, edge_vector) -> int:
    """Return the sum of squared flow errors."""

    return sum(value**2 for value in node_flow_residuals(graph, edge_vector).values())


def build_qubo(graph: nx.DiGraph, penalty: Number) -> QuboPolynomial:
    """Expand route cost plus squared flow errors into QUBO coefficients."""

    validate_graph(graph)
    penalty = _fraction(penalty)
    if penalty <= 0:
        raise ValueError("penalty must be positive")

    nodes = tuple(sorted(graph.nodes))
    edges = get_edge_order(graph)
    supplies = node_supplies(graph)
    matrix = incidence_matrix(graph)

    constant = penalty * sum(supply**2 for supply in supplies.values())

    linear = []
    for i, edge in enumerate(edges):
        flow_coefficient = sum(
            row[i] ** 2 - 2 * supplies[node] * row[i]
            for node, row in zip(nodes, matrix)
        )
        edge_weight = Fraction(graph.edges[edge]["weight"])
        linear.append(edge_weight + penalty * flow_coefficient)

    pair = {}
    for i in range(EXPECTED_EDGE_COUNT):
        for j in range(i + 1, EXPECTED_EDGE_COUNT):
            coefficient = penalty * sum(2 * row[i] * row[j] for row in matrix)
            if coefficient:
                pair[i, j] = coefficient

    return QuboPolynomial(constant, tuple(linear), pair)


def state_index_to_edge_vector(state_index: int) -> EdgeVector:
    """Convert basis index 0..16383 to bits, with q0 as the lowest bit."""

    if not isinstance(state_index, int) or isinstance(state_index, bool):
        raise TypeError("state index must be an integer")
    if not 0 <= state_index < 2**EXPECTED_EDGE_COUNT:
        raise ValueError("state index must be in range 0..16383")
    return tuple((state_index >> i) & 1 for i in range(EXPECTED_EDGE_COUNT))


def edge_vector_to_state_index(edge_vector) -> int:
    """Convert fourteen edge bits to an integer basis index."""

    return sum(bit << i for i, bit in enumerate(validate_edge_vector(edge_vector)))


def decode_valid_route(graph: nx.DiGraph, edge_vector) -> tuple[int, ...] | None:
    """Return the route if the selected edges form exactly one source-target path."""

    x = validate_edge_vector(edge_vector)
    return _decode_selected_route(graph, get_edge_order(graph), x)


def _decode_selected_route(
    graph: nx.DiGraph,
    edges: tuple[Edge, ...],
    x: EdgeVector,
) -> tuple[int, ...] | None:
    selected = {edge for edge, bit in zip(edges, x) if bit}
    if not selected:
        return None

    target = graph.graph["target"]
    path = [graph.graph["source"]]

    while path[-1] != target:
        outgoing = [edge for edge in selected if edge[0] == path[-1]]
        if len(outgoing) != 1:
            return None

        edge = outgoing[0]
        selected.remove(edge)
        if edge[1] in path:
            return None
        path.append(edge[1])

    return tuple(path) if not selected else None


def enumerate_state_space(graph: nx.DiGraph) -> tuple[StateRecord, ...]:
    """Calculate cost, flow errors, and decoded route for all 2^14 states."""

    validate_graph(graph)
    nodes = tuple(sorted(graph.nodes))
    edges = get_edge_order(graph)
    supplies = node_supplies(graph)
    matrix = incidence_matrix(graph)
    weights = tuple(graph.edges[edge]["weight"] for edge in edges)

    states = []
    for state_index in range(2**EXPECTED_EDGE_COUNT):
        x = state_index_to_edge_vector(state_index)
        residuals = tuple(
            sum(coefficient * bit for coefficient, bit in zip(row, x))
            - supplies[node]
            for node, row in zip(nodes, matrix)
        )
        states.append(
            StateRecord(
                state_index=state_index,
                edge_vector=x,
                routing_cost=sum(weight * bit for weight, bit in zip(weights, x)),
                flow_residuals=residuals,
                flow_penalty=sum(value**2 for value in residuals),
                decoded_route=_decode_selected_route(graph, edges, x),
            )
        )
    return tuple(states)


def minimum_energy_states(states, penalty: Number):
    """Return the minimum penalized energy and every state that reaches it."""

    penalty = _fraction(penalty)
    energies = [
        Fraction(state.routing_cost) + penalty * state.flow_penalty for state in states
    ]
    minimum = min(energies)
    return minimum, tuple(
        state for state, energy in zip(states, energies) if energy == minimum
    )


@dataclass(frozen=True)
class IsingHamiltonian:
    """Constant, one-qubit, and two-qubit Ising coefficients."""

    constant: Fraction
    h: tuple[Fraction, ...]
    coupling: dict[Pair, Fraction]

    def basis_energy(self, edge_vector) -> Fraction:
        z = tuple(1 - 2 * bit for bit in validate_edge_vector(edge_vector))
        single_energy = sum(coefficient * z[i] for i, coefficient in enumerate(self.h))
        pair_energy = sum(
            coefficient * z[i] * z[j]
            for (i, j), coefficient in self.coupling.items()
        )
        return self.constant + single_energy + pair_energy


def qubo_to_ising(qubo: QuboPolynomial) -> IsingHamiltonian:
    """Substitute ``x_i = (1 - z_i) / 2`` into the QUBO."""

    constant = qubo.constant + sum(coefficient / 2 for coefficient in qubo.linear)
    h = [-coefficient / 2 for coefficient in qubo.linear]
    coupling = {}

    for (i, j), coefficient in qubo.pair.items():
        quarter = coefficient / 4
        constant += quarter
        h[i] -= quarter
        h[j] -= quarter
        coupling[i, j] = quarter

    return IsingHamiltonian(constant, tuple(h), coupling)


def max_qubo_ising_error(states, qubos) -> Fraction:
    """Return the largest basis-energy difference between QUBO and Ising."""

    maximum = Fraction(0)
    for qubo in qubos:
        ising = qubo_to_ising(qubo)
        for state in states:
            difference = qubo.evaluate(state.edge_vector) - ising.basis_energy(
                state.edge_vector
            )
            maximum = max(maximum, abs(difference))
    return maximum
