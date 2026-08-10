"""Logical feasibility-preserving Warm-Start QAOA for the frozen course graph.

Q2-F is an exact, enumeration-based teaching reference.  Its computational
basis consists only of valid simple source-to-target routes.  It is not a
hardware-efficient encoding and makes no scalability or quantum-advantage
claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi
from time import perf_counter
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, minimize

from exact_reference import compute_exact_reference, enumerate_simple_paths_independent
from graph import (
    DEFAULT_GRAPH_PATH,
    Edge,
    edge_vector_to_canonical_bitstring,
    path_cost,
    path_edges,
    path_to_edge_bitstring,
    validate_edge_vector,
    validate_graph,
)
from objectives import probability_weighted_cvar, probability_weighted_expectation
from optimization import EvaluationBudgetExhausted, SOURCE_UNIFORM, initial_parameters
from qubo import decode_valid_route
from warm_start import greedy_incumbent_route


UNIFORM_FEASIBLE = "uniform_feasible"
INCUMBENT_BIASED_FEASIBLE = "incumbent_biased_feasible"
INITIALIZATION_MODES = (UNIFORM_FEASIBLE, INCUMBENT_BIASED_FEASIBLE)

EXPECTATION_OBJECTIVE = "expectation"
FIXED_CVAR_OBJECTIVE = "cvar"
ASCENDING_CVAR_OBJECTIVE = "ascending_cvar"
Q2F_OBJECTIVES = (
    EXPECTATION_OBJECTIVE,
    FIXED_CVAR_OBJECTIVE,
    ASCENDING_CVAR_OBJECTIVE,
)

PATH_EXCHANGE_MIXER = "logical_path_exchange_mixer"
PARAMETER_ORDER = "all_gammas_then_all_betas"
LAYER_ORDER = "cost_then_mixer"
PROBABILITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class FeasibleRoute:
    route_id: int
    node_sequence: tuple[int, ...]
    edge_sequence: tuple[Edge, ...]
    edge_bitstring: tuple[int, ...]
    routing_cost: int
    exact_optimal: bool

    @property
    def bitstring_text(self) -> str:
        return edge_vector_to_canonical_bitstring(self.edge_bitstring)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "node_sequence": list(self.node_sequence),
            "edge_sequence": [list(edge) for edge in self.edge_sequence],
            "edge_bitstring": list(self.edge_bitstring),
            "edge_bitstring_text": self.bitstring_text,
            "routing_cost": self.routing_cost,
            "exact_optimal": self.exact_optimal,
        }


@dataclass(frozen=True)
class FeasibleRouteBasis:
    routes: tuple[FeasibleRoute, ...]
    source: int
    target: int
    exact_optimal_route_id: int
    incumbent_route_id: int
    exact_optimal_route: tuple[int, ...]
    incumbent_route: tuple[int, ...]
    ordering: str = "routing_cost_then_lexicographic_node_sequence"

    def __post_init__(self) -> None:
        route_ids = tuple(route.route_id for route in self.routes)
        if not self.routes or route_ids != tuple(range(len(self.routes))):
            raise ValueError("feasible_route_ids_must_be_contiguous")
        bitstrings = tuple(route.edge_bitstring for route in self.routes)
        node_sequences = tuple(route.node_sequence for route in self.routes)
        if len(set(bitstrings)) != len(bitstrings) or len(set(node_sequences)) != len(node_sequences):
            raise ValueError("duplicate_feasible_route")
        optimal_ids = tuple(route.route_id for route in self.routes if route.exact_optimal)
        if optimal_ids != (self.exact_optimal_route_id,):
            raise ValueError("feasible_basis_requires_one_exact_optimum")
        if self.routes[self.incumbent_route_id].node_sequence != self.incumbent_route:
            raise ValueError("incumbent_route_id_mismatch")

    @property
    def size(self) -> int:
        return len(self.routes)

    @property
    def raw_costs(self) -> np.ndarray:
        return np.asarray([route.routing_cost for route in self.routes], dtype=np.float64)

    @property
    def edge_bitstrings(self) -> tuple[tuple[int, ...], ...]:
        return tuple(route.edge_bitstring for route in self.routes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "dtu-sciqis-q2f-feasible-route-basis",
            "version": "1.0",
            "basis_size": self.size,
            "source": self.source,
            "target": self.target,
            "ordering": self.ordering,
            "exact_optimal_route_id": self.exact_optimal_route_id,
            "incumbent_route_id": self.incumbent_route_id,
            "exact_optimal_route": list(self.exact_optimal_route),
            "incumbent_route": list(self.incumbent_route),
            "routes": [route.as_dict() for route in self.routes],
        }


def build_feasible_route_basis(
    graph: nx.DiGraph,
    *,
    graph_path: str = str(DEFAULT_GRAPH_PATH),
) -> FeasibleRouteBasis:
    """Build and independently verify the complete deterministic route basis."""

    validate_graph(graph)
    source = int(graph.graph["source"])
    target = int(graph.graph["target"])
    enumerated = enumerate_simple_paths_independent(graph)
    networkx_paths = {
        tuple(path) for path in nx.all_simple_paths(graph, source=source, target=target)
    }
    enumerated_paths = {record.node_path for record in enumerated}
    if enumerated_paths != networkx_paths:
        raise RuntimeError("independent_feasible_route_enumerations_disagree")

    exact_payload, exact_routes = compute_exact_reference(graph, graph_path=graph_path)
    if tuple(record.node_path for record in exact_routes) != tuple(
        record.node_path for record in enumerated
    ):
        raise RuntimeError("exact_reference_route_order_disagrees")
    exact_route = tuple(exact_payload["exact_reference"]["node_path"])
    exact_cost = int(exact_payload["exact_reference"]["cost"])

    routes: list[FeasibleRoute] = []
    for route_id, record in enumerate(enumerated):
        nodes = tuple(record.node_path)
        edges = tuple(record.edge_path)
        bits = validate_edge_vector(record.edge_bitstring)
        if nodes[0] != source or nodes[-1] != target or not nx.is_simple_path(graph, nodes):
            raise RuntimeError(f"invalid_enumerated_route:{nodes}")
        if edges != path_edges(nodes) or len(set(edges)) != len(edges):
            raise RuntimeError(f"invalid_enumerated_edge_sequence:{nodes}")
        if tuple(path_to_edge_bitstring(graph, nodes)) != bits:
            raise RuntimeError(f"route_bitstring_encoding_mismatch:{nodes}")
        if decode_valid_route(graph, bits) != nodes:
            raise RuntimeError(f"route_bitstring_round_trip_mismatch:{nodes}")
        independently_summed = path_cost(graph, nodes)
        if independently_summed != record.cost:
            raise RuntimeError(f"route_cost_mismatch:{nodes}")
        routes.append(
            FeasibleRoute(
                route_id=route_id,
                node_sequence=nodes,
                edge_sequence=edges,
                edge_bitstring=bits,
                routing_cost=int(record.cost),
                exact_optimal=bool(nodes == exact_route and record.cost == exact_cost),
            )
        )

    if len({route.node_sequence for route in routes}) != len(routes):
        raise RuntimeError("duplicate_node_route_in_feasible_basis")
    if len({route.edge_bitstring for route in routes}) != len(routes):
        raise RuntimeError("duplicate_edge_bitstring_in_feasible_basis")
    optimal_ids = [route.route_id for route in routes if route.exact_optimal]
    if len(optimal_ids) != 1 or routes[optimal_ids[0]].routing_cost != exact_cost:
        raise RuntimeError("logical_optimum_disagrees_with_exact_reference")
    incumbent = tuple(greedy_incumbent_route(graph))
    incumbent_ids = [route.route_id for route in routes if route.node_sequence == incumbent]
    if len(incumbent_ids) != 1:
        raise RuntimeError("historical_incumbent_missing_from_feasible_basis")
    return FeasibleRouteBasis(
        routes=tuple(routes),
        source=source,
        target=target,
        exact_optimal_route_id=optimal_ids[0],
        incumbent_route_id=incumbent_ids[0],
        exact_optimal_route=exact_route,
        incumbent_route=incumbent,
    )


@dataclass(frozen=True)
class LogicalCostHamiltonian:
    raw_energies: tuple[float, ...]
    normalized_energies: tuple[float, ...]
    normalization_shift: float
    normalization_scale: float
    normalization: str

    @property
    def raw_matrix(self) -> np.ndarray:
        return np.diag(np.asarray(self.raw_energies, dtype=np.float64))

    @property
    def normalized_matrix(self) -> np.ndarray:
        return np.diag(np.asarray(self.normalized_energies, dtype=np.float64))

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_energies": list(self.raw_energies),
            "normalized_energies": list(self.normalized_energies),
            "normalization_shift": self.normalization_shift,
            "normalization_scale": self.normalization_scale,
            "normalization": self.normalization,
        }


def build_logical_cost_hamiltonian(basis: FeasibleRouteBasis) -> LogicalCostHamiltonian:
    """Use transparent min-max normalization while retaining every raw cost."""

    raw = basis.raw_costs
    shift = float(np.min(raw))
    scale = float(np.max(raw) - shift)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("logical_cost_range_must_be_positive")
    normalized = (raw - shift) / scale
    return LogicalCostHamiltonian(
        raw_energies=tuple(map(float, raw)),
        normalized_energies=tuple(map(float, normalized)),
        normalization_shift=shift,
        normalization_scale=scale,
        normalization="(raw_route_cost - minimum_route_cost) / (maximum_route_cost - minimum_route_cost)",
    )


@dataclass(frozen=True)
class PathExchange:
    left_route_id: int
    right_route_id: int
    boundary_start: int
    boundary_end: int
    edge_hamming_distance: int
    weight: float = 1.0
    kind: str = "divergence_reconvergence"


@dataclass(frozen=True)
class LogicalPathExchangeMixer:
    hamiltonian: np.ndarray
    exchanges: tuple[PathExchange, ...]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    connected: bool
    mixer_name: str = PATH_EXCHANGE_MIXER
    weight_rule: str = "uniform"

    @property
    def dimension(self) -> int:
        return int(self.hamiltonian.shape[0])

    def unitary(self, beta: float) -> np.ndarray:
        phases = np.exp(-1j * float(beta) * self.eigenvalues)
        return (self.eigenvectors * phases) @ self.eigenvectors.conj().T

    def evolve(self, state: Sequence[complex], beta: float) -> np.ndarray:
        vector = np.asarray(state, dtype=np.complex128)
        if vector.shape != (self.dimension,):
            raise ValueError("logical_mixer_state_dimension_mismatch")
        phases = np.exp(-1j * float(beta) * self.eigenvalues)
        coefficients = self.eigenvectors.conj().T @ vector
        return self.eigenvectors @ (phases * coefficients)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mixer_name": self.mixer_name,
            "construction": "validated divergence-reconvergence path exchanges",
            "dimension": self.dimension,
            "weight_rule": self.weight_rule,
            "connected": self.connected,
            "exchange_count": len(self.exchanges),
            "hermitian_error": float(
                np.max(np.abs(self.hamiltonian - self.hamiltonian.conj().T), initial=0.0)
            ),
            "hamiltonian": self.hamiltonian.real.tolist(),
            "exchanges": [asdict(exchange) for exchange in self.exchanges],
        }


def _path_exchange_boundaries(
    left: FeasibleRoute,
    right: FeasibleRoute,
) -> tuple[int, int] | None:
    """Return one existing divergence-reconvergence exchange, if present.

    Remove the maximal common edge prefix and suffix, then require two nonempty
    alternative subpaths with common endpoints and disjoint internal nodes.
    """

    left_edges, right_edges = left.edge_sequence, right.edge_sequence
    prefix = 0
    while (
        prefix < min(len(left_edges), len(right_edges))
        and left_edges[prefix] == right_edges[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(left_edges) - prefix, len(right_edges) - prefix)
        and left_edges[len(left_edges) - 1 - suffix]
        == right_edges[len(right_edges) - 1 - suffix]
    ):
        suffix += 1
    left_middle = left_edges[prefix : len(left_edges) - suffix if suffix else len(left_edges)]
    right_middle = right_edges[prefix : len(right_edges) - suffix if suffix else len(right_edges)]
    if (
        not left_middle
        or not right_middle
        or left_middle[0][0] != right_middle[0][0]
        or left_middle[-1][1] != right_middle[-1][1]
    ):
        return None
    left_internal = {edge[1] for edge in left_middle[:-1]}
    right_internal = {edge[1] for edge in right_middle[:-1]}
    if left_internal & right_internal:
        return None
    return int(left_middle[0][0]), int(left_middle[-1][1])


def build_logical_path_exchange_mixer(
    basis: FeasibleRouteBasis,
) -> LogicalPathExchangeMixer:
    """Build the connected uniform path-exchange adjacency on the route basis."""

    graph = nx.Graph()
    graph.add_nodes_from(range(basis.size))
    exchanges: list[PathExchange] = []
    for left_id, left in enumerate(basis.routes):
        for right in basis.routes[left_id + 1 :]:
            boundaries = _path_exchange_boundaries(left, right)
            if boundaries is None:
                continue
            distance = sum(
                left_bit != right_bit
                for left_bit, right_bit in zip(left.edge_bitstring, right.edge_bitstring)
            )
            exchange = PathExchange(
                left_route_id=left.route_id,
                right_route_id=right.route_id,
                boundary_start=boundaries[0],
                boundary_end=boundaries[1],
                edge_hamming_distance=distance,
            )
            exchanges.append(exchange)
            graph.add_edge(left.route_id, right.route_id)
    connected = bool(nx.is_connected(graph))
    if not connected:
        raise RuntimeError("logical_path_exchange_graph_is_not_connected")
    hamiltonian = np.zeros((basis.size, basis.size), dtype=np.complex128)
    for exchange in exchanges:
        i, j = exchange.left_route_id, exchange.right_route_id
        hamiltonian[i, j] = exchange.weight
        hamiltonian[j, i] = exchange.weight
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-14, rtol=0.0):
        raise RuntimeError("logical_path_exchange_hamiltonian_not_hermitian")
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return LogicalPathExchangeMixer(
        hamiltonian=hamiltonian,
        exchanges=tuple(exchanges),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        connected=connected,
    )


def incumbent_biased_probabilities(
    route_bitstrings: Sequence[Sequence[int]],
    incumbent_bitstring: Sequence[int],
    *,
    bias_lambda: float = 1.0,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return exp(-lambda * edge-Hamming-distance) route probabilities.

    The construction receives no optimum identity, cost, or optimality flag.
    """

    lam = float(bias_lambda)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("bias_lambda_must_be_finite_and_nonnegative")
    incumbent = validate_edge_vector(incumbent_bitstring)
    bitstrings = tuple(validate_edge_vector(bits) for bits in route_bitstrings)
    if not bitstrings or len(set(bitstrings)) != len(bitstrings):
        raise ValueError("route_bitstrings_must_be_nonempty_and_unique")
    distances = tuple(
        sum(bit != incumbent_bit for bit, incumbent_bit in zip(bits, incumbent))
        for bits in bitstrings
    )
    weights = np.exp(-lam * np.asarray(distances, dtype=np.float64))
    probabilities = weights / float(np.sum(weights))
    return probabilities, distances


@dataclass(frozen=True)
class FeasibleInitialState:
    mode: str
    amplitudes: tuple[complex, ...]
    probabilities: tuple[float, ...]
    route_distances: tuple[int, ...]
    bias_lambda: float | None
    incumbent_route_id: int
    incumbent_route: tuple[int, ...]
    p_feas: float
    p_opt: float
    expected_route_cost: float
    expected_normalized_cost: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "probabilities": list(self.probabilities),
            "route_distances": list(self.route_distances),
            "bias_lambda": self.bias_lambda,
            "incumbent_route_id": self.incumbent_route_id,
            "incumbent_route": list(self.incumbent_route),
            "p_feas": self.p_feas,
            "p_opt": self.p_opt,
            "expected_route_cost": self.expected_route_cost,
            "expected_normalized_cost": self.expected_normalized_cost,
        }


def build_feasible_initial_state(
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    *,
    mode: str,
    bias_lambda: float = 1.0,
) -> FeasibleInitialState:
    if mode not in INITIALIZATION_MODES:
        raise ValueError(f"unsupported_feasible_initialization:{mode}")
    if mode == UNIFORM_FEASIBLE:
        probabilities = np.full(basis.size, 1.0 / basis.size, dtype=np.float64)
        distances = tuple(
            sum(a != b for a, b in zip(route.edge_bitstring, basis.routes[basis.incumbent_route_id].edge_bitstring))
            for route in basis.routes
        )
        applied_lambda: float | None = None
    else:
        probabilities, distances = incumbent_biased_probabilities(
            basis.edge_bitstrings,
            basis.routes[basis.incumbent_route_id].edge_bitstring,
            bias_lambda=bias_lambda,
        )
        applied_lambda = float(bias_lambda)
    amplitudes = np.sqrt(probabilities).astype(np.complex128)
    if abs(float(np.vdot(amplitudes, amplitudes).real) - 1.0) > PROBABILITY_TOLERANCE:
        raise RuntimeError("feasible_initial_state_not_normalized")
    raw = np.asarray(costs.raw_energies, dtype=np.float64)
    normalized = np.asarray(costs.normalized_energies, dtype=np.float64)
    return FeasibleInitialState(
        mode=mode,
        amplitudes=tuple(map(complex, amplitudes)),
        probabilities=tuple(map(float, probabilities)),
        route_distances=distances,
        bias_lambda=applied_lambda,
        incumbent_route_id=basis.incumbent_route_id,
        incumbent_route=basis.incumbent_route,
        p_feas=float(np.sum(probabilities)),
        p_opt=float(probabilities[basis.exact_optimal_route_id]),
        expected_route_cost=float(probabilities @ raw),
        expected_normalized_cost=float(probabilities @ normalized),
    )


@dataclass(frozen=True)
class LayerNormRecord:
    layer: int
    after_cost_norm: float
    after_mixer_norm: float


@dataclass(frozen=True)
class LogicalSimulation:
    state: tuple[complex, ...]
    probabilities: tuple[float, ...]
    initial_norm: float
    final_norm: float
    layer_norms: tuple[LayerNormRecord, ...]


def simulate_logical_qaoa(
    initial_state: Sequence[complex],
    normalized_costs: Sequence[float],
    mixer: LogicalPathExchangeMixer,
    parameters: Sequence[float],
    *,
    depth: int,
) -> LogicalSimulation:
    """Apply grouped-parameter cost-then-mixer QAOA entirely inside F."""

    p = int(depth)
    if p not in (1, 2, 3):
        raise ValueError("q2f_depth_must_be_1_2_or_3")
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (2 * p,) or np.any(~np.isfinite(values)):
        raise ValueError("q2f_parameter_count_or_finiteness_error")
    costs = np.asarray(normalized_costs, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.complex128).copy()
    if state.shape != (mixer.dimension,) or costs.shape != (mixer.dimension,):
        raise ValueError("q2f_logical_dimension_mismatch")
    initial_norm = float(np.linalg.norm(state))
    if abs(initial_norm - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError("q2f_initial_state_not_normalized")
    gammas, betas = values[:p], values[p:]
    norms: list[LayerNormRecord] = []
    for layer in range(p):
        state *= np.exp(-1j * float(gammas[layer]) * costs)
        after_cost = float(np.linalg.norm(state))
        if abs(after_cost - 1.0) > PROBABILITY_TOLERANCE:
            raise RuntimeError("q2f_cost_layer_changed_norm")
        state = mixer.evolve(state, float(betas[layer]))
        after_mixer = float(np.linalg.norm(state))
        if abs(after_mixer - 1.0) > PROBABILITY_TOLERANCE:
            raise RuntimeError("q2f_mixer_layer_changed_norm")
        norms.append(LayerNormRecord(layer + 1, after_cost, after_mixer))
    probabilities = np.abs(state) ** 2
    total = float(np.sum(probabilities))
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise RuntimeError("q2f_final_probability_not_normalized")
    probabilities = probabilities / total
    return LogicalSimulation(
        state=tuple(map(complex, state)),
        probabilities=tuple(map(float, probabilities)),
        initial_norm=initial_norm,
        final_norm=float(np.linalg.norm(state)),
        layer_norms=tuple(norms),
    )


def ascending_cvar_alpha(
    evaluation_index: int,
    total_budget: int,
    *,
    alpha_start: float = 0.25,
    alpha_end: float = 1.0,
) -> float:
    """Return the frozen 0-based alpha schedule for a declared request cap."""

    index, budget = int(evaluation_index), int(total_budget)
    start, end = float(alpha_start), float(alpha_end)
    if budget < 1 or index < 0:
        raise ValueError("ascending_cvar_index_and_budget_must_be_nonnegative")
    if not 0.0 < start <= end <= 1.0:
        raise ValueError("ascending_cvar_alpha_bounds_invalid")
    if budget == 1:
        return start
    value = start + (end - start) * index / (budget - 1)
    return float(np.clip(value, start, end))


def evaluate_q2f_objective(
    probabilities: Sequence[float],
    normalized_energies: Sequence[float],
    objective_mode: str,
    *,
    evaluation_index: int,
    total_budget: int,
    fixed_cvar_alpha: float = 0.25,
    alpha_start: float = 0.25,
    alpha_end: float = 1.0,
) -> tuple[float, float | None]:
    if objective_mode == EXPECTATION_OBJECTIVE:
        return probability_weighted_expectation(probabilities, normalized_energies), None
    if objective_mode == FIXED_CVAR_OBJECTIVE:
        alpha = float(fixed_cvar_alpha)
    elif objective_mode == ASCENDING_CVAR_OBJECTIVE:
        alpha = ascending_cvar_alpha(
            evaluation_index,
            total_budget,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
        )
    else:
        raise ValueError(f"unsupported_q2f_objective:{objective_mode}")
    return probability_weighted_cvar(probabilities, normalized_energies, alpha), alpha


@dataclass(frozen=True)
class FeasibleMetrics:
    p_feas: float
    p_opt: float
    optimal_state_rank: int
    top3_lowest_cost_mass: float
    top5_lowest_cost_mass: float
    expected_route_cost: float
    expected_normalized_cost: float
    most_probable_route_id: int
    most_probable_route: tuple[int, ...]
    most_probable_route_cost: int
    most_probable_route_probability: float
    probability_entropy: float
    initial_p_opt: float
    final_p_opt: float
    p_opt_amplification: float | None


def feasible_route_metrics(
    probabilities: Sequence[float],
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    *,
    initial_p_opt: float,
) -> FeasibleMetrics:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (basis.size,) or np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("invalid_q2f_probability_vector")
    p_feas = float(np.sum(probs))
    if abs(p_feas - 1.0) > PROBABILITY_TOLERANCE:
        raise RuntimeError(f"q2f_feasibility_invariant_failed:{p_feas}")
    optimal_ids = [route.route_id for route in basis.routes if route.exact_optimal]
    p_opt = float(np.sum(probs[optimal_ids]))
    optimal_probability = float(np.max(probs[optimal_ids]))
    optimal_rank = 1 + int(np.sum(probs > optimal_probability + 1e-14))
    order = np.lexsort((np.arange(basis.size), -np.round(probs, 15)))
    most_id = int(order[0])
    most = basis.routes[most_id]
    positive = probs[probs > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    initial = float(initial_p_opt)
    amplification = None if initial <= 0.0 else float(p_opt / initial)
    raw = np.asarray(costs.raw_energies, dtype=np.float64)
    normalized = np.asarray(costs.normalized_energies, dtype=np.float64)
    return FeasibleMetrics(
        p_feas=p_feas,
        p_opt=p_opt,
        optimal_state_rank=optimal_rank,
        top3_lowest_cost_mass=float(np.sum(probs[: min(3, basis.size)])),
        top5_lowest_cost_mass=float(np.sum(probs[: min(5, basis.size)])),
        expected_route_cost=float(probs @ raw),
        expected_normalized_cost=float(probs @ normalized),
        most_probable_route_id=most_id,
        most_probable_route=most.node_sequence,
        most_probable_route_cost=most.routing_cost,
        most_probable_route_probability=float(probs[most_id]),
        probability_entropy=entropy,
        initial_p_opt=initial,
        final_p_opt=p_opt,
        p_opt_amplification=amplification,
    )


@dataclass(frozen=True)
class Q2FEvaluationRecord:
    evaluation_index: int
    schedule_index: int
    parameters: tuple[float, ...]
    objective_value: float
    alpha: float | None
    in_bounds: bool


@dataclass(frozen=True)
class Q2FOptimizationResult:
    initial_parameters: tuple[float, ...]
    final_parameters: tuple[float, ...]
    initial_objective_value: float
    optimizer_reported_objective_value: float
    final_objective_at_terminal_alpha: float
    terminal_alpha: float | None
    evaluations: int
    statevector_evaluations: int
    success: bool
    reason: str
    message: str
    runtime: float
    retention_policy: str
    evaluation_trace: tuple[Q2FEvaluationRecord, ...]


def optimize_logical_qaoa(
    initial_state: Sequence[complex],
    costs: LogicalCostHamiltonian,
    mixer: LogicalPathExchangeMixer,
    *,
    depth: int,
    seed: int,
    objective_mode: str,
    evaluation_budget: int = 100,
    fixed_cvar_alpha: float = 0.25,
    alpha_start: float = 0.25,
    alpha_end: float = 1.0,
    rhobeg: float = 0.5,
    tolerance: float = 1e-8,
) -> tuple[Q2FOptimizationResult, LogicalSimulation]:
    """Run bounded COBYLA with a complete objective/alpha request trace."""

    p, budget = int(depth), int(evaluation_budget)
    if p not in (1, 2, 3) or budget < 1 or objective_mode not in Q2F_OBJECTIVES:
        raise ValueError("invalid_q2f_optimizer_configuration")
    initial = initial_parameters(p, int(seed), strategy=SOURCE_UNIFORM)
    lower = np.asarray([0.0] * p + [0.0] * p, dtype=np.float64)
    upper = np.asarray([2.0 * pi] * p + [pi] * p, dtype=np.float64)
    normalized = np.asarray(costs.normalized_energies, dtype=np.float64)
    trace: list[Q2FEvaluationRecord] = []
    evaluations = 0
    statevector_evaluations = 0
    best_value = float("inf")
    best_parameters = initial.copy()
    last_in_bounds = initial.copy()

    def counted(parameters: np.ndarray) -> float:
        nonlocal evaluations, statevector_evaluations, best_value, best_parameters, last_in_bounds
        if evaluations >= budget:
            raise EvaluationBudgetExhausted
        candidate = np.asarray(parameters, dtype=np.float64)
        schedule_index = evaluations
        evaluations += 1
        outside = float(
            np.sum(np.maximum(lower - candidate, 0.0) + np.maximum(candidate - upper, 0.0))
        )
        if outside:
            _, alpha = evaluate_q2f_objective(
                np.full(mixer.dimension, 1.0 / mixer.dimension),
                normalized,
                objective_mode,
                evaluation_index=schedule_index,
                total_budget=budget,
                fixed_cvar_alpha=fixed_cvar_alpha,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
            )
            value = 1_000_000.0 + outside
            in_bounds = False
        else:
            simulation = simulate_logical_qaoa(
                initial_state, normalized, mixer, candidate, depth=p
            )
            statevector_evaluations += 1
            value, alpha = evaluate_q2f_objective(
                simulation.probabilities,
                normalized,
                objective_mode,
                evaluation_index=schedule_index,
                total_budget=budget,
                fixed_cvar_alpha=fixed_cvar_alpha,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
            )
            in_bounds = True
            last_in_bounds = candidate.copy()
            if value < best_value:
                best_value = float(value)
                best_parameters = candidate.copy()
        if not np.isfinite(value):
            raise ValueError("non_finite_q2f_objective")
        trace.append(
            Q2FEvaluationRecord(
                evaluation_index=evaluations,
                schedule_index=schedule_index,
                parameters=tuple(map(float, candidate)),
                objective_value=float(value),
                alpha=alpha,
                in_bounds=in_bounds,
            )
        )
        return float(value)

    started = perf_counter()
    scipy_result = None
    exhausted = False
    try:
        scipy_result = minimize(
            counted,
            initial,
            method="COBYLA",
            bounds=Bounds(lower, upper),
            options={
                "maxiter": budget,
                "rhobeg": float(rhobeg),
                "tol": float(tolerance),
                "catol": float(tolerance),
            },
        )
    except EvaluationBudgetExhausted:
        exhausted = True
    runtime = perf_counter() - started
    if not trace or not np.isfinite(best_value):
        raise RuntimeError("q2f_optimizer_produced_no_finite_in_bounds_evaluation")

    scipy_candidate_in_bounds = False
    if scipy_result is not None:
        candidate = np.asarray(scipy_result.x, dtype=np.float64)
        scipy_candidate_in_bounds = bool(
            candidate.shape == initial.shape
            and np.all(candidate >= lower)
            and np.all(candidate <= upper)
        )
    if objective_mode == ASCENDING_CVAR_OBJECTIVE:
        if scipy_result is not None and scipy_candidate_in_bounds:
            final_parameters = np.asarray(scipy_result.x, dtype=np.float64)
            optimizer_reported = float(scipy_result.fun)
            retention = "scipy_terminal_parameters_for_nonstationary_schedule"
        else:
            final_parameters = last_in_bounds
            optimizer_reported = float(trace[-1].objective_value)
            retention = "last_in_bounds_parameters_after_hard_cap"
    else:
        if (
            scipy_result is not None
            and scipy_candidate_in_bounds
            and float(scipy_result.fun) < best_value
        ):
            best_parameters = np.asarray(scipy_result.x, dtype=np.float64)
            best_value = float(scipy_result.fun)
        final_parameters = best_parameters
        optimizer_reported = best_value
        retention = "best_finite_in_bounds_objective"

    terminal_alpha = trace[-1].alpha
    final_simulation = simulate_logical_qaoa(
        initial_state, normalized, mixer, final_parameters, depth=p
    )
    final_objective, _ = evaluate_q2f_objective(
        final_simulation.probabilities,
        normalized,
        objective_mode,
        evaluation_index=trace[-1].schedule_index,
        total_budget=budget,
        fixed_cvar_alpha=fixed_cvar_alpha,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
    )
    statevector_evaluations += 1
    success = bool(scipy_result is not None and scipy_result.success and not exhausted)
    if exhausted or evaluations >= budget:
        reason = "evaluation_budget_exhausted"
    elif success:
        reason = "converged"
    else:
        reason = f"optimizer_failed:{getattr(scipy_result, 'message', 'unknown_failure')}"
    message = str(getattr(scipy_result, "message", reason))
    return (
        Q2FOptimizationResult(
            initial_parameters=tuple(map(float, initial)),
            final_parameters=tuple(map(float, final_parameters)),
            initial_objective_value=float(trace[0].objective_value),
            optimizer_reported_objective_value=float(optimizer_reported),
            final_objective_at_terminal_alpha=float(final_objective),
            terminal_alpha=terminal_alpha,
            evaluations=evaluations,
            statevector_evaluations=statevector_evaluations,
            success=success,
            reason=reason,
            message=message,
            runtime=float(runtime),
            retention_policy=retention,
            evaluation_trace=tuple(trace),
        ),
        final_simulation,
    )


@dataclass(frozen=True)
class Q2FRunResult:
    run_id: str
    objective_mode: str
    depth: int
    seed: int
    evaluation_budget: int
    initialization_mode: str
    bias_lambda: float
    fixed_cvar_alpha: float
    alpha_start: float
    alpha_end: float
    optimizer: Q2FOptimizationResult
    final_probabilities: tuple[float, ...]
    final_layer_norms: tuple[LayerNormRecord, ...]
    metrics: FeasibleMetrics

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_q2f_cell(
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    mixer: LogicalPathExchangeMixer,
    initial_state: FeasibleInitialState,
    *,
    objective_mode: str,
    depth: int,
    seed: int,
    evaluation_budget: int = 100,
    fixed_cvar_alpha: float = 0.25,
    alpha_start: float = 0.25,
    alpha_end: float = 1.0,
    rhobeg: float = 0.5,
    tolerance: float = 1e-8,
) -> Q2FRunResult:
    if initial_state.mode != INCUMBENT_BIASED_FEASIBLE:
        raise ValueError("primary_q2f_cells_require_incumbent_biased_initialization")
    optimized, final_simulation = optimize_logical_qaoa(
        initial_state.amplitudes,
        costs,
        mixer,
        depth=depth,
        seed=seed,
        objective_mode=objective_mode,
        evaluation_budget=evaluation_budget,
        fixed_cvar_alpha=fixed_cvar_alpha,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        rhobeg=rhobeg,
        tolerance=tolerance,
    )
    metrics = feasible_route_metrics(
        final_simulation.probabilities,
        basis,
        costs,
        initial_p_opt=initial_state.p_opt,
    )
    if abs(metrics.p_feas - 1.0) > PROBABILITY_TOLERANCE:
        raise RuntimeError("q2f_primary_feasibility_invariant_failed")
    return Q2FRunResult(
        run_id=f"{objective_mode}_p{int(depth)}_seed{int(seed)}",
        objective_mode=objective_mode,
        depth=int(depth),
        seed=int(seed),
        evaluation_budget=int(evaluation_budget),
        initialization_mode=initial_state.mode,
        bias_lambda=float(initial_state.bias_lambda or 0.0),
        fixed_cvar_alpha=float(fixed_cvar_alpha),
        alpha_start=float(alpha_start),
        alpha_end=float(alpha_end),
        optimizer=optimized,
        final_probabilities=final_simulation.probabilities,
        final_layer_norms=final_simulation.layer_norms,
        metrics=metrics,
    )
