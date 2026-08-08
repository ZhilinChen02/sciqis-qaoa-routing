"""Explicit directed-flow penalty and canonical QUBO construction.

The polynomial convention is always

    c + sum_i q_i x_i + sum_{i<j} q_ij x_i x_j,

with ``x_i`` in the frozen Day-1 edge order.  No black-box converter and no
ambiguous matrix convention are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

import networkx as nx

from graph import (
    Edge,
    EdgeVector,
    EXPECTED_EDGE_COUNT,
    edge_vector_to_canonical_bitstring,
    edge_vector_to_qiskit_display_bitstring,
    get_edge_order,
    validate_edge_vector,
    validate_graph,
)


Number = int | float | Fraction
Pair = tuple[int, int]
QUBO_CONVENTION = "c + sum_i q_i*x_i + sum_{i<j} q_ij*x_i*x_j"
FLOW_DEFINITION = (
    "f_v(x) = sum_{e outgoing from v} x_e - "
    "sum_{e incoming to v} x_e - b_v"
)


def as_fraction(value: Number) -> Fraction:
    """Convert public numeric input to a stable exact rational."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric coefficient")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    raise TypeError(f"unsupported numeric coefficient {value!r}")


@dataclass(frozen=True)
class QuboPolynomial:
    """Canonical upper-triangular polynomial coefficients."""

    constant: Fraction
    linear: tuple[Fraction, ...]
    pair: Mapping[Pair, Fraction]

    def __post_init__(self) -> None:
        if len(self.linear) != EXPECTED_EDGE_COUNT:
            raise ValueError("QUBO must have exactly 14 linear coefficients")
        for (i, j), coefficient in self.pair.items():
            if not 0 <= i < j < EXPECTED_EDGE_COUNT:
                raise ValueError(f"QUBO pair key must satisfy 0 <= i < j < 14: {(i, j)}")
            if coefficient == 0:
                raise ValueError(f"zero QUBO pair coefficient should be omitted: {(i, j)}")

    def evaluate(self, edge_vector: Iterable[int]) -> Fraction:
        vector = validate_edge_vector(edge_vector)
        value = self.constant
        value += sum(coefficient * vector[i] for i, coefficient in enumerate(self.linear))
        value += sum(
            coefficient * vector[i] * vector[j]
            for (i, j), coefficient in self.pair.items()
        )
        return value


@dataclass(frozen=True)
class StateRecord:
    """Exact classical data for one computational-basis edge selection."""

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
    def qiskit_display_bitstring(self) -> str:
        return edge_vector_to_qiskit_display_bitstring(self.edge_vector)

    @property
    def is_decoder_valid(self) -> bool:
        return self.decoded_route is not None


@dataclass(frozen=True)
class PenaltyChoice:
    label: str
    value: Fraction


def node_supplies(graph: nx.DiGraph) -> dict[int, int]:
    """Return ``b_source=+1``, ``b_target=-1``, and zero otherwise."""

    validate_graph(graph)
    source = graph.graph["source"]
    target = graph.graph["target"]
    return {
        node: 1 if node == source else -1 if node == target else 0
        for node in sorted(graph.nodes)
    }


def incidence_matrix(graph: nx.DiGraph) -> tuple[tuple[int, ...], ...]:
    """Return rows ``a[v,e]`` using +1 outgoing and -1 incoming."""

    validate_graph(graph)
    edges = get_edge_order(graph)
    return tuple(
        tuple(1 if u == node else -1 if v == node else 0 for u, v in edges)
        for node in sorted(graph.nodes)
    )


def flow_equation_strings(graph: nx.DiGraph) -> tuple[str, ...]:
    """Return compact display equations for the seven frozen node residuals."""

    matrix = incidence_matrix(graph)
    supplies = node_supplies(graph)
    equations = []
    for node, row in zip(sorted(graph.nodes), matrix):
        terms = []
        for index, coefficient in enumerate(row):
            if coefficient:
                terms.append(("+" if coefficient > 0 else "-") + f" x{index}")
        left = " ".join(terms).lstrip("+ ") or "0"
        if supplies[node] > 0:
            left += f" - {supplies[node]}"
        elif supplies[node] < 0:
            left += f" + {-supplies[node]}"
        equations.append(f"f_{node}(x) = {left}")
    return tuple(equations)


def routing_cost(graph: nx.DiGraph, edge_vector: Iterable[int]) -> int:
    """Evaluate ``C(x) = sum_e w_e x_e`` exactly."""

    vector = validate_edge_vector(edge_vector)
    return sum(
        graph.edges[edge]["weight"] * bit
        for edge, bit in zip(get_edge_order(graph), vector)
    )


def node_flow_residuals(
    graph: nx.DiGraph,
    edge_vector: Iterable[int],
) -> dict[int, int]:
    """Evaluate every directed flow residual ``f_v(x)`` exactly."""

    vector = validate_edge_vector(edge_vector)
    supplies = node_supplies(graph)
    matrix = incidence_matrix(graph)
    return {
        node: sum(coefficient * bit for coefficient, bit in zip(row, vector))
        - supplies[node]
        for node, row in zip(sorted(graph.nodes), matrix)
    }


def flow_penalty(graph: nx.DiGraph, edge_vector: Iterable[int]) -> int:
    """Evaluate ``P_flow(x) = sum_v f_v(x)^2`` exactly."""

    return sum(
        residual * residual
        for residual in node_flow_residuals(graph, edge_vector).values()
    )


def penalty_objective(
    graph: nx.DiGraph,
    edge_vector: Iterable[int],
    penalty: Number,
) -> Fraction:
    """Evaluate ``Q_A(x) = C(x) + A P_flow(x)`` directly."""

    vector = validate_edge_vector(edge_vector)
    return Fraction(routing_cost(graph, vector)) + as_fraction(penalty) * flow_penalty(
        graph, vector
    )


def flow_penalty_coefficients(graph: nx.DiGraph) -> QuboPolynomial:
    """Expand the squared flow residuals analytically using ``x_i^2=x_i``."""

    matrix = incidence_matrix(graph)
    supplies = node_supplies(graph)
    nodes = tuple(sorted(graph.nodes))
    constant = Fraction(sum(supplies[node] ** 2 for node in nodes))

    linear = []
    for i in range(EXPECTED_EDGE_COUNT):
        coefficient = sum(
            matrix[row_index][i] ** 2
            - 2 * supplies[node] * matrix[row_index][i]
            for row_index, node in enumerate(nodes)
        )
        linear.append(Fraction(coefficient))

    pair: dict[Pair, Fraction] = {}
    for i in range(EXPECTED_EDGE_COUNT):
        for j in range(i + 1, EXPECTED_EDGE_COUNT):
            coefficient = 2 * sum(row[i] * row[j] for row in matrix)
            if coefficient:
                pair[i, j] = Fraction(coefficient)
    return QuboPolynomial(constant, tuple(linear), pair)


def build_qubo(graph: nx.DiGraph, penalty: Number) -> QuboPolynomial:
    """Add routing weights to ``A * P_flow`` in canonical polynomial form."""

    penalty_value = as_fraction(penalty)
    if penalty_value <= 0:
        raise ValueError("penalty A must be positive")
    flow = flow_penalty_coefficients(graph)
    weights = tuple(
        Fraction(graph.edges[edge]["weight"]) for edge in get_edge_order(graph)
    )
    return QuboPolynomial(
        constant=penalty_value * flow.constant,
        linear=tuple(
            weight + penalty_value * coefficient
            for weight, coefficient in zip(weights, flow.linear)
        ),
        pair={
            pair: penalty_value * coefficient
            for pair, coefficient in flow.pair.items()
            if penalty_value * coefficient
        },
    )


def state_index_to_edge_vector(state_index: int) -> EdgeVector:
    """Map integer basis index to ``[x0,...,x13]`` with q0 least-significant."""

    if not isinstance(state_index, int) or isinstance(state_index, bool):
        raise TypeError("state_index must be an integer")
    if not 0 <= state_index < 2**EXPECTED_EDGE_COUNT:
        raise ValueError("state_index must be in range 0..16383")
    return tuple((state_index >> index) & 1 for index in range(EXPECTED_EDGE_COUNT))


def edge_vector_to_state_index(edge_vector: Iterable[int]) -> int:
    """Map ``[x0,...,x13]`` to the integer basis index with q0 as LSB."""

    vector = validate_edge_vector(edge_vector)
    return sum(bit << index for index, bit in enumerate(vector))


def decode_valid_route(
    graph: nx.DiGraph,
    edge_vector: Iterable[int],
) -> tuple[int, ...] | None:
    """Independently accept exactly one selected source-to-target path.

    This decoder intentionally does not call ``flow_penalty``.  It traverses
    selected edges from the source and rejects branches, dead ends, cycles, and
    any extra disconnected or unused selected edge.
    """

    vector = validate_edge_vector(edge_vector)
    return _decode_selected_route(graph, get_edge_order(graph), vector)


def _decode_selected_route(
    graph: nx.DiGraph,
    ordered_edges: tuple[Edge, ...],
    vector: EdgeVector,
) -> tuple[int, ...] | None:
    """Internal decoder using a prevalidated graph/order for exhaustive loops."""

    selected = {
        edge for edge, bit in zip(ordered_edges, vector) if bit
    }
    if not selected:
        return None

    source = graph.graph["source"]
    target = graph.graph["target"]
    path = [source]
    used: set[Edge] = set()
    current = source
    while current != target:
        outgoing = sorted(edge for edge in selected if edge[0] == current)
        if len(outgoing) != 1:
            return None
        edge = outgoing[0]
        next_node = edge[1]
        if edge in used or next_node in path:
            return None
        used.add(edge)
        path.append(next_node)
        current = next_node

    if used != selected:
        return None
    return tuple(path)


def enumerate_state_space(graph: nx.DiGraph) -> tuple[StateRecord, ...]:
    """Evaluate all ``2^14`` classical basis states deterministically."""

    validate_graph(graph)
    nodes = tuple(sorted(graph.nodes))
    ordered_edges = get_edge_order(graph)
    matrix = incidence_matrix(graph)
    supplies = node_supplies(graph)
    weights = tuple(graph.edges[edge]["weight"] for edge in ordered_edges)
    records = []
    for state_index in range(2**EXPECTED_EDGE_COUNT):
        vector = state_index_to_edge_vector(state_index)
        residual_values = tuple(
            sum(coefficient * bit for coefficient, bit in zip(row, vector))
            - supplies[node]
            for node, row in zip(nodes, matrix)
        )
        records.append(
            StateRecord(
                state_index=state_index,
                edge_vector=vector,
                routing_cost=sum(weight * bit for weight, bit in zip(weights, vector)),
                flow_residuals=residual_values,
                flow_penalty=sum(value * value for value in residual_values),
                decoded_route=_decode_selected_route(graph, ordered_edges, vector),
            )
        )
    return tuple(records)


def flow_feasibility_verdict(states: Sequence[StateRecord]) -> dict[str, object]:
    """Compare zero-flow-penalty states with independently decoder-valid states."""

    zero_penalty = {state.state_index for state in states if state.flow_penalty == 0}
    decoder_valid = {state.state_index for state in states if state.is_decoder_valid}
    return {
        "flow_penalty_zero_count": len(zero_penalty),
        "decoder_valid_route_count": len(decoder_valid),
        "sets_identical": zero_penalty == decoder_valid,
        "zero_only_state_indices": sorted(zero_penalty - decoder_valid),
        "decoder_only_state_indices": sorted(decoder_valid - zero_penalty),
    }


def derive_critical_penalty(
    states: Sequence[StateRecord],
    optimal_feasible_cost: int,
) -> tuple[Fraction, tuple[StateRecord, ...]]:
    """Compute the exact infeasible-state crossing threshold ``A_crit``."""

    crossings = [
        (
            Fraction(optimal_feasible_cost - state.routing_cost, state.flow_penalty),
            state,
        )
        for state in states
        if state.flow_penalty > 0 and state.routing_cost < optimal_feasible_cost
    ]
    if not crossings:
        return Fraction(0), tuple()
    critical_value = max(crossing for crossing, _state in crossings)
    critical_states = tuple(
        state for crossing, state in crossings if crossing == critical_value
    )
    return critical_value, critical_states


def frozen_penalty_grid(critical_penalty: Fraction) -> tuple[PenaltyChoice, ...]:
    """Derive the four-point teaching grid only from the exact threshold."""

    critical = as_fraction(critical_penalty)
    if critical <= 0:
        raise ValueError("A_crit must be positive for the frozen teaching grid")
    weak_integer = critical.numerator // (2 * critical.denominator)
    weak = Fraction(weak_integer) if weak_integer > 0 else critical / 2
    floor_critical = critical.numerator // critical.denominator
    just_supercritical = Fraction(floor_critical + 1)
    if just_supercritical <= critical:
        just_supercritical += 1
    strong = 2 * just_supercritical
    choices = (
        PenaltyChoice("weak", weak),
        PenaltyChoice("critical", critical),
        PenaltyChoice("just-supercritical", just_supercritical),
        PenaltyChoice("strong", strong),
    )
    if not (0 < weak < critical < just_supercritical < strong):
        raise RuntimeError("derived penalty grid does not strictly straddle A_crit")
    return choices


def minimum_energy_states(
    states: Sequence[StateRecord],
    penalty: Number,
) -> tuple[Fraction, tuple[StateRecord, ...]]:
    """Return exact minimum ``C + A P`` and every state attaining it."""

    value = as_fraction(penalty)
    energies = tuple(
        Fraction(state.routing_cost) + value * state.flow_penalty for state in states
    )
    minimum = min(energies)
    minimizers = tuple(
        state for state, energy in zip(states, energies) if energy == minimum
    )
    return minimum, minimizers


def max_qubo_expansion_error(
    graph: nx.DiGraph,
    states: Sequence[StateRecord],
    penalties: Sequence[Number],
) -> Fraction:
    """Exhaustively compare direct and coefficient QUBO values."""

    maximum = Fraction(0)
    for penalty in penalties:
        qubo = build_qubo(graph, penalty)
        penalty_value = as_fraction(penalty)
        for state in states:
            direct = Fraction(state.routing_cost) + penalty_value * state.flow_penalty
            maximum = max(maximum, abs(direct - qubo.evaluate(state.edge_vector)))
    return maximum
