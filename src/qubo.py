"""Build and evaluate the QUBO for the routing problem.

For every edge there is one binary variable.  A value of 1 means that the edge
is selected.  The QUBO energy is

    route cost + A * flow penalty

where ``A`` is a positive penalty chosen by the experiment.
"""

from dataclasses import dataclass
from fractions import Fraction

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
    """Convert an integer or decimal input to an exact Fraction."""

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
    """The constant, linear and quadratic coefficients of a QUBO."""

    constant: Fraction
    linear: tuple[Fraction, ...]
    pair: dict[Pair, Fraction]

    def __post_init__(self):
        if len(self.linear) != EXPECTED_EDGE_COUNT:
            raise ValueError("QUBO must have exactly 14 linear coefficients")

        for (first, second), coefficient in self.pair.items():
            if not 0 <= first < second < EXPECTED_EDGE_COUNT:
                raise ValueError(
                    "QUBO pair key must satisfy 0 <= i < j < 14: "
                    f"{(first, second)}"
                )
            if coefficient == 0:
                raise ValueError(
                    "zero QUBO pair coefficient should be omitted: "
                    f"{(first, second)}"
                )

    def evaluate(self, edge_vector) -> Fraction:
        """Calculate the QUBO energy of one fourteen-bit state."""

        vector = validate_edge_vector(edge_vector)
        value = self.constant

        for index, coefficient in enumerate(self.linear):
            value += coefficient * vector[index]

        for (first, second), coefficient in self.pair.items():
            value += coefficient * vector[first] * vector[second]

        return value


@dataclass(frozen=True)
class StateRecord:
    """Classical information about one of the 2^14 edge selections."""

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


# ---------------------------------------------------------------------------
# Flow equations
# ---------------------------------------------------------------------------

def node_supplies(graph: nx.DiGraph) -> dict[int, int]:
    """Use +1 at the source, -1 at the target and 0 elsewhere."""

    validate_graph(graph)
    source = graph.graph["source"]
    target = graph.graph["target"]

    supplies = {}
    for node in sorted(graph.nodes):
        if node == source:
            supplies[node] = 1
        elif node == target:
            supplies[node] = -1
        else:
            supplies[node] = 0
    return supplies


def incidence_matrix(graph: nx.DiGraph) -> tuple[tuple[int, ...], ...]:
    """Build the node-edge matrix: outgoing=+1, incoming=-1."""

    validate_graph(graph)
    ordered_edges = get_edge_order(graph)
    matrix = []

    for node in sorted(graph.nodes):
        row = []
        for start, end in ordered_edges:
            if start == node:
                row.append(1)
            elif end == node:
                row.append(-1)
            else:
                row.append(0)
        matrix.append(tuple(row))

    return tuple(matrix)


def flow_equation_strings(graph: nx.DiGraph) -> tuple[str, ...]:
    """Make readable versions of the seven flow equations."""

    matrix = incidence_matrix(graph)
    supplies = node_supplies(graph)
    equations = []

    for node, row in zip(sorted(graph.nodes), matrix):
        terms = []
        for variable_index, coefficient in enumerate(row):
            if coefficient == 1:
                terms.append(f"+ x{variable_index}")
            elif coefficient == -1:
                terms.append(f"- x{variable_index}")

        left_side = " ".join(terms).lstrip("+ ") or "0"
        if supplies[node] == 1:
            left_side += " - 1"
        elif supplies[node] == -1:
            left_side += " + 1"
        equations.append(f"f_{node}(x) = {left_side}")

    return tuple(equations)


def routing_cost(graph: nx.DiGraph, edge_vector) -> int:
    """Add the weights of the selected edges."""

    vector = validate_edge_vector(edge_vector)
    total = 0
    for edge, bit in zip(get_edge_order(graph), vector):
        total += graph.edges[edge]["weight"] * bit
    return total


def node_flow_residuals(graph: nx.DiGraph, edge_vector) -> dict[int, int]:
    """Calculate the left side of every flow constraint."""

    vector = validate_edge_vector(edge_vector)
    supplies = node_supplies(graph)
    matrix = incidence_matrix(graph)
    residuals = {}

    for node, row in zip(sorted(graph.nodes), matrix):
        value = 0
        for coefficient, bit in zip(row, vector):
            value += coefficient * bit
        residuals[node] = value - supplies[node]

    return residuals


def flow_penalty(graph: nx.DiGraph, edge_vector) -> int:
    """Return the sum of squared flow-constraint errors."""

    residuals = node_flow_residuals(graph, edge_vector)
    total = 0
    for value in residuals.values():
        total += value * value
    return total


def penalty_objective(graph: nx.DiGraph, edge_vector, penalty: Number) -> Fraction:
    """Calculate route cost + A times flow penalty directly."""

    vector = validate_edge_vector(edge_vector)
    cost = routing_cost(graph, vector)
    invalid_route_penalty = flow_penalty(graph, vector)
    return Fraction(cost) + as_fraction(penalty) * invalid_route_penalty


# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------

def flow_penalty_coefficients(graph: nx.DiGraph) -> QuboPolynomial:
    """Expand the squared flow equations into QUBO coefficients."""

    matrix = incidence_matrix(graph)
    supplies = node_supplies(graph)
    nodes = tuple(sorted(graph.nodes))

    constant = Fraction(0)
    for node in nodes:
        constant += supplies[node] ** 2

    linear = []
    for variable in range(EXPECTED_EDGE_COUNT):
        coefficient = 0
        for row_number, node in enumerate(nodes):
            matrix_value = matrix[row_number][variable]
            coefficient += matrix_value**2
            coefficient -= 2 * supplies[node] * matrix_value
        linear.append(Fraction(coefficient))

    pair = {}
    for first in range(EXPECTED_EDGE_COUNT):
        for second in range(first + 1, EXPECTED_EDGE_COUNT):
            coefficient = 0
            for row in matrix:
                coefficient += 2 * row[first] * row[second]
            if coefficient != 0:
                pair[first, second] = Fraction(coefficient)

    return QuboPolynomial(constant, tuple(linear), pair)


def build_qubo(graph: nx.DiGraph, penalty: Number) -> QuboPolynomial:
    """Build the complete routing QUBO for a chosen penalty A."""

    penalty_value = as_fraction(penalty)
    if penalty_value <= 0:
        raise ValueError("penalty A must be positive")

    flow_qubo = flow_penalty_coefficients(graph)

    linear = []
    for edge, flow_coefficient in zip(get_edge_order(graph), flow_qubo.linear):
        edge_weight = Fraction(graph.edges[edge]["weight"])
        linear.append(edge_weight + penalty_value * flow_coefficient)

    pair = {}
    for indices, flow_coefficient in flow_qubo.pair.items():
        coefficient = penalty_value * flow_coefficient
        if coefficient != 0:
            pair[indices] = coefficient

    return QuboPolynomial(
        constant=penalty_value * flow_qubo.constant,
        linear=tuple(linear),
        pair=pair,
    )


# ---------------------------------------------------------------------------
# State and route conversion
# ---------------------------------------------------------------------------

def state_index_to_edge_vector(state_index: int) -> EdgeVector:
    """Convert basis index 0..16383 to bits, with q0 as the lowest bit."""

    if not isinstance(state_index, int) or isinstance(state_index, bool):
        raise TypeError("state_index must be an integer")
    if not 0 <= state_index < 2**EXPECTED_EDGE_COUNT:
        raise ValueError("state_index must be in range 0..16383")

    bits = []
    for bit_position in range(EXPECTED_EDGE_COUNT):
        bits.append((state_index >> bit_position) & 1)
    return tuple(bits)


def edge_vector_to_state_index(edge_vector) -> int:
    """Convert fourteen edge bits to an integer basis index."""

    vector = validate_edge_vector(edge_vector)
    state_index = 0
    for bit_position, bit in enumerate(vector):
        state_index += bit << bit_position
    return state_index


def decode_valid_route(graph: nx.DiGraph, edge_vector) -> tuple[int, ...] | None:
    """Decode selected edges if they form exactly one source-to-target route."""

    vector = validate_edge_vector(edge_vector)
    ordered_edges = get_edge_order(graph)
    return _decode_selected_route(graph, ordered_edges, vector)


def _decode_selected_route(
    graph: nx.DiGraph,
    ordered_edges: tuple[Edge, ...],
    vector: EdgeVector,
) -> tuple[int, ...] | None:
    """Internal route decoder used repeatedly during state enumeration."""

    selected_edges = set()
    for edge, bit in zip(ordered_edges, vector):
        if bit == 1:
            selected_edges.add(edge)

    if not selected_edges:
        return None

    source = graph.graph["source"]
    target = graph.graph["target"]
    path = [source]
    used_edges = set()
    current_node = source

    while current_node != target:
        outgoing = []
        for edge in selected_edges:
            if edge[0] == current_node:
                outgoing.append(edge)

        if len(outgoing) != 1:
            return None

        edge = outgoing[0]
        next_node = edge[1]
        if edge in used_edges or next_node in path:
            return None

        used_edges.add(edge)
        path.append(next_node)
        current_node = next_node

    if used_edges != selected_edges:
        return None
    return tuple(path)


def enumerate_state_space(graph: nx.DiGraph) -> tuple[StateRecord, ...]:
    """Calculate classical information for all 16,384 edge selections."""

    validate_graph(graph)
    nodes = tuple(sorted(graph.nodes))
    ordered_edges = get_edge_order(graph)
    matrix = incidence_matrix(graph)
    supplies = node_supplies(graph)
    weights = tuple(graph.edges[edge]["weight"] for edge in ordered_edges)

    records = []
    for state_index in range(2**EXPECTED_EDGE_COUNT):
        vector = state_index_to_edge_vector(state_index)

        residual_values = []
        for node, row in zip(nodes, matrix):
            residual = 0
            for coefficient, bit in zip(row, vector):
                residual += coefficient * bit
            residual_values.append(residual - supplies[node])

        route_cost = 0
        for weight, bit in zip(weights, vector):
            route_cost += weight * bit

        penalty_value = 0
        for residual in residual_values:
            penalty_value += residual * residual

        records.append(
            StateRecord(
                state_index=state_index,
                edge_vector=vector,
                routing_cost=route_cost,
                flow_residuals=tuple(residual_values),
                flow_penalty=penalty_value,
                decoded_route=_decode_selected_route(graph, ordered_edges, vector),
            )
        )

    return tuple(records)


# ---------------------------------------------------------------------------
# Checks and penalty selection used by the experiments
# ---------------------------------------------------------------------------

def flow_feasibility_verdict(states) -> dict[str, object]:
    """Check that zero flow penalty means a route was decoded successfully."""

    zero_penalty = set()
    decoder_valid = set()
    for state in states:
        if state.flow_penalty == 0:
            zero_penalty.add(state.state_index)
        if state.is_decoder_valid:
            decoder_valid.add(state.state_index)

    return {
        "flow_penalty_zero_count": len(zero_penalty),
        "decoder_valid_route_count": len(decoder_valid),
        "sets_identical": zero_penalty == decoder_valid,
        "zero_only_state_indices": sorted(zero_penalty - decoder_valid),
        "decoder_only_state_indices": sorted(decoder_valid - zero_penalty),
    }


def derive_critical_penalty(states, optimal_feasible_cost: int):
    """Find the penalty where an invalid state can no longer beat the optimum."""

    crossings = []
    for state in states:
        if state.flow_penalty > 0 and state.routing_cost < optimal_feasible_cost:
            crossing = Fraction(
                optimal_feasible_cost - state.routing_cost,
                state.flow_penalty,
            )
            crossings.append((crossing, state))

    if not crossings:
        return Fraction(0), tuple()

    critical_value = max(crossing for crossing, _state in crossings)
    critical_states = []
    for crossing, state in crossings:
        if crossing == critical_value:
            critical_states.append(state)
    return critical_value, tuple(critical_states)


def frozen_penalty_grid(critical_penalty: Fraction) -> tuple[PenaltyChoice, ...]:
    """Create weak, critical, just-supercritical and strong penalty choices."""

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


def minimum_energy_states(states, penalty: Number):
    """Return the minimum QUBO energy and all states with this energy."""

    penalty_value = as_fraction(penalty)
    energies = []
    for state in states:
        energy = Fraction(state.routing_cost)
        energy += penalty_value * state.flow_penalty
        energies.append(energy)

    minimum = min(energies)
    minimizers = []
    for state, energy in zip(states, energies):
        if energy == minimum:
            minimizers.append(state)
    return minimum, tuple(minimizers)


def max_qubo_expansion_error(graph, states, penalties) -> Fraction:
    """Compare direct energies with the expanded QUBO for every given state."""

    maximum_error = Fraction(0)
    for penalty in penalties:
        qubo = build_qubo(graph, penalty)
        penalty_value = as_fraction(penalty)

        for state in states:
            direct = Fraction(state.routing_cost)
            direct += penalty_value * state.flow_penalty
            expanded = qubo.evaluate(state.edge_vector)
            error = abs(direct - expanded)
            if error > maximum_error:
                maximum_error = error

    return maximum_error


# ---------------------------------------------------------------------------
# Ising conversion
# ---------------------------------------------------------------------------

ISING_CONVENTION = "H = c0*I + sum_i h_i*Z_i + sum_{i<j} J_ij*Z_i*Z_j"
BIT_TO_Z_CONVENTION = "x_i = (1 - z_i)/2; equivalently z_i = 1 - 2*x_i"


@dataclass(frozen=True)
class IsingHamiltonian:
    """Constant, single-qubit and two-qubit Ising coefficients."""

    constant: Fraction
    h: tuple[Fraction, ...]
    coupling: dict[Pair, Fraction]

    def __post_init__(self):
        if len(self.h) != EXPECTED_EDGE_COUNT:
            raise ValueError("Ising Hamiltonian must have exactly 14 h coefficients")

        for (first, second), coefficient in self.coupling.items():
            if not 0 <= first < second < EXPECTED_EDGE_COUNT:
                raise ValueError(
                    "Ising pair key must satisfy 0 <= i < j < 14: "
                    f"{(first, second)}"
                )
            if coefficient == 0:
                raise ValueError(
                    f"zero Ising coupling should be omitted: {(first, second)}"
                )

    def basis_energy(self, edge_vector) -> Fraction:
        z_values = z_eigenvalues(edge_vector)
        energy = self.constant

        for index, coefficient in enumerate(self.h):
            energy += coefficient * z_values[index]
        for (first, second), coefficient in self.coupling.items():
            energy += coefficient * z_values[first] * z_values[second]
        return energy


def qubo_to_ising(qubo: QuboPolynomial) -> IsingHamiltonian:
    """Apply x=(1-z)/2 to every term of a QUBO."""

    constant = qubo.constant
    h = []

    for coefficient in qubo.linear:
        constant += coefficient / 2
        h.append(-coefficient / 2)

    coupling = {}
    for (first, second), coefficient in qubo.pair.items():
        quarter = coefficient / 4
        constant += quarter
        h[first] -= quarter
        h[second] -= quarter
        coupling[first, second] = quarter

    return IsingHamiltonian(constant, tuple(h), coupling)


def max_qubo_ising_error(states, qubos) -> Fraction:
    """Return the largest energy difference between QUBO and Ising forms."""

    maximum_error = Fraction(0)
    for qubo in qubos:
        ising = qubo_to_ising(qubo)
        for state in states:
            qubo_energy = qubo.evaluate(state.edge_vector)
            ising_energy = ising.basis_energy(state.edge_vector)
            error = abs(qubo_energy - ising_energy)
            if error > maximum_error:
                maximum_error = error
    return maximum_error


def z_eigenvalues(edge_vector) -> tuple[int, ...]:
    """Map edge bit 0 to Z value +1 and edge bit 1 to Z value -1."""

    vector = validate_edge_vector(edge_vector)
    return tuple(1 - 2 * bit for bit in vector)
