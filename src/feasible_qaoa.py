"""QAOA in a small basis containing only valid source-to-target routes.

The graph used in this course project has only 20 simple routes, so we can use
one basis state per route.  The code below follows the calculation directly:

1. enumerate and sort the routes;
2. use their normalized lengths as the cost Hamiltonian;
3. connect routes that differ by one path exchange;
4. alternate cost and mixer layers;
5. optimize the QAOA angles with COBYLA.

This is an ideal teaching simulation.  Enumerating every route first is not a
scalable solution to a large routing problem.
"""

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np

from metrics import indexed_probability_mass, probability_mass, shannon_entropy
from graph import (
    Edge,
    edge_vector_to_canonical_bitstring,
    greedy_route,
    path_cost,
    path_edges,
    path_to_edge_bitstring,
    simple_routes,
    validate_edge_vector,
)


UNIFORM_FEASIBLE = "uniform_feasible"
INCUMBENT_BIASED_FEASIBLE = "incumbent_biased_feasible"
INITIALIZATION_MODES = (UNIFORM_FEASIBLE, INCUMBENT_BIASED_FEASIBLE)

PROBABILITY_TOLERANCE = 1e-12


# These small data classes only give names to values used by the experiments.


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

@dataclass(frozen=True)
class FeasibleRouteBasis:
    routes: tuple[FeasibleRoute, ...]
    source: int
    target: int
    exact_optimal_route_id: int
    incumbent_route_id: int
    exact_optimal_route: tuple[int, ...]
    incumbent_route: tuple[int, ...]
    @property
    def size(self) -> int:
        return len(self.routes)

    @property
    def raw_costs(self) -> np.ndarray:
        return np.asarray([route.routing_cost for route in self.routes], dtype=float)

    @property
    def edge_bitstrings(self) -> tuple[tuple[int, ...], ...]:
        return tuple(route.edge_bitstring for route in self.routes)

@dataclass(frozen=True)
class LogicalCostHamiltonian:
    raw_energies: tuple[float, ...]
    normalized_energies: tuple[float, ...]
    normalization_shift: float
    normalization_scale: float


@dataclass(frozen=True)
class PathExchange:
    left_route_id: int
    right_route_id: int
    boundary_start: int
    boundary_end: int
    edge_hamming_distance: int


@dataclass(frozen=True)
class LogicalPathExchangeMixer:
    hamiltonian: np.ndarray
    exchanges: tuple[PathExchange, ...]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def dimension(self) -> int:
        return len(self.hamiltonian)

    def unitary(self, beta: float) -> np.ndarray:
        phases = np.exp(-1j * beta * self.eigenvalues)
        return (self.eigenvectors * phases) @ self.eigenvectors.conj().T

    def evolve(self, state: Sequence[complex], beta: float) -> np.ndarray:
        state = np.asarray(state, dtype=complex)
        coefficients = self.eigenvectors.conj().T @ state
        phases = np.exp(-1j * beta * self.eigenvalues)
        return self.eigenvectors @ (phases * coefficients)

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


def build_feasible_route_basis(
    graph: nx.DiGraph,
) -> FeasibleRouteBasis:
    """Enumerate the graph's simple routes and assign one basis index to each."""

    source = int(graph.graph["source"])
    target = int(graph.graph["target"])
    paths = simple_routes(graph)
    if not paths:
        raise ValueError("the graph has no source-to-target route")

    best_cost = path_cost(graph, paths[0])
    best_paths = [path for path in paths if path_cost(graph, path) == best_cost]
    if len(best_paths) != 1:
        raise ValueError("this example expects one shortest route")
    best_path = best_paths[0]

    routes = tuple(
        FeasibleRoute(
            route_id=route_id,
            node_sequence=path,
            edge_sequence=path_edges(path),
            edge_bitstring=path_to_edge_bitstring(graph, path),
            routing_cost=path_cost(graph, path),
            exact_optimal=path == best_path,
        )
        for route_id, path in enumerate(paths)
    )

    incumbent = greedy_route(graph)
    incumbent_id = next(
        route.route_id for route in routes if route.node_sequence == incumbent
    )
    return FeasibleRouteBasis(
        routes=routes,
        source=source,
        target=target,
        exact_optimal_route_id=0,  # routes are sorted by cost
        incumbent_route_id=incumbent_id,
        exact_optimal_route=best_path,
        incumbent_route=incumbent,
    )


def build_logical_cost_hamiltonian(
    basis: FeasibleRouteBasis,
) -> LogicalCostHamiltonian:
    """Normalize the route costs to [0, 1] for the QAOA phase layer."""

    raw = basis.raw_costs
    minimum = float(raw.min())
    scale = float(raw.max() - minimum)
    if scale == 0:
        raise ValueError("all routes have the same cost")
    normalized = (raw - minimum) / scale
    return LogicalCostHamiltonian(
        raw_energies=tuple(map(float, raw)),
        normalized_energies=tuple(map(float, normalized)),
        normalization_shift=minimum,
        normalization_scale=scale,
    )


def _path_exchange_boundaries(
    left: FeasibleRoute,
    right: FeasibleRoute,
) -> tuple[int, int] | None:
    """Find where two routes split and join again."""

    left_edges = left.edge_sequence
    right_edges = right.edge_sequence

    prefix = 0
    while prefix < min(len(left_edges), len(right_edges)):
        if left_edges[prefix] != right_edges[prefix]:
            break
        prefix += 1

    suffix = 0
    remaining = min(len(left_edges), len(right_edges)) - prefix
    while suffix < remaining:
        if left_edges[-1 - suffix] != right_edges[-1 - suffix]:
            break
        suffix += 1

    left_end = len(left_edges) - suffix if suffix else len(left_edges)
    right_end = len(right_edges) - suffix if suffix else len(right_edges)
    left_middle = left_edges[prefix:left_end]
    right_middle = right_edges[prefix:right_end]
    if not left_middle or not right_middle:
        return None
    if left_middle[0][0] != right_middle[0][0]:
        return None
    if left_middle[-1][1] != right_middle[-1][1]:
        return None

    left_internal = {v for _, v in left_middle[:-1]}
    right_internal = {v for _, v in right_middle[:-1]}
    if left_internal & right_internal:
        return None
    return left_middle[0][0], left_middle[-1][1]


def build_logical_path_exchange_mixer(
    basis: FeasibleRouteBasis,
) -> LogicalPathExchangeMixer:
    """Connect route states that differ by one alternative subpath."""

    exchanges = []
    route_graph = nx.Graph()
    route_graph.add_nodes_from(range(basis.size))

    for i, left in enumerate(basis.routes):
        for right in basis.routes[i + 1 :]:
            boundaries = _path_exchange_boundaries(left, right)
            if boundaries is None:
                continue
            distance = sum(
                left_bit != right_bit
                for left_bit, right_bit in zip(
                    left.edge_bitstring, right.edge_bitstring
                )
            )
            exchanges.append(
                PathExchange(
                    left.route_id,
                    right.route_id,
                    boundaries[0],
                    boundaries[1],
                    distance,
                )
            )
            route_graph.add_edge(left.route_id, right.route_id)

    if not nx.is_connected(route_graph):
        raise ValueError("the path-exchange mixer is not connected")

    hamiltonian = np.zeros((basis.size, basis.size), dtype=complex)
    for exchange in exchanges:
        i = exchange.left_route_id
        j = exchange.right_route_id
        hamiltonian[i, j] = hamiltonian[j, i] = 1.0

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return LogicalPathExchangeMixer(
        hamiltonian,
        tuple(exchanges),
        eigenvalues,
        eigenvectors,
    )


def incumbent_biased_probabilities(
    route_bitstrings: Sequence[Sequence[int]],
    incumbent_bitstring: Sequence[int],
    *,
    bias_lambda: float = 1.0,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Give routes near the greedy route more starting probability."""

    bias_lambda = float(bias_lambda)
    if not np.isfinite(bias_lambda) or bias_lambda < 0:
        raise ValueError("bias_lambda must be non-negative")

    incumbent = validate_edge_vector(incumbent_bitstring)
    bitstrings = [validate_edge_vector(bits) for bits in route_bitstrings]
    distances = tuple(
        sum(bit != reference for bit, reference in zip(bits, incumbent))
        for bits in bitstrings
    )
    weights = np.exp(-bias_lambda * np.asarray(distances, dtype=float))
    return weights / weights.sum(), distances


def build_feasible_initial_state(
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    *,
    mode: str,
    bias_lambda: float = 1.0,
) -> FeasibleInitialState:
    """Prepare either a uniform or greedy-route-biased feasible state."""

    if mode not in INITIALIZATION_MODES:
        raise ValueError(f"unknown initialization mode: {mode}")

    incumbent_bits = basis.routes[basis.incumbent_route_id].edge_bitstring
    _, distances = incumbent_biased_probabilities(
        basis.edge_bitstrings, incumbent_bits, bias_lambda=0.0
    )
    if mode == UNIFORM_FEASIBLE:
        probabilities = np.full(basis.size, 1 / basis.size)
        applied_lambda = None
    else:
        probabilities, distances = incumbent_biased_probabilities(
            basis.edge_bitstrings,
            incumbent_bits,
            bias_lambda=bias_lambda,
        )
        applied_lambda = float(bias_lambda)

    raw = np.asarray(costs.raw_energies)
    normalized = np.asarray(costs.normalized_energies)
    amplitudes = np.sqrt(probabilities).astype(complex)
    return FeasibleInitialState(
        mode=mode,
        amplitudes=tuple(map(complex, amplitudes)),
        probabilities=tuple(map(float, probabilities)),
        route_distances=distances,
        bias_lambda=applied_lambda,
        incumbent_route_id=basis.incumbent_route_id,
        incumbent_route=basis.incumbent_route,
        p_feas=probability_mass(probabilities),
        p_opt=indexed_probability_mass(
            probabilities, [basis.exact_optimal_route_id]
        ),
        expected_route_cost=float(probabilities @ raw),
        expected_normalized_cost=float(probabilities @ normalized),
    )


def simulate_logical_qaoa(
    initial_state: Sequence[complex],
    normalized_costs: Sequence[float],
    mixer,
    parameters: Sequence[float],
    *,
    depth: int,
) -> LogicalSimulation:
    """Apply p cost layers and p path-exchange mixer layers."""

    depth = int(depth)
    parameters = np.asarray(parameters, dtype=float)
    costs = np.asarray(normalized_costs, dtype=float)
    state = np.asarray(initial_state, dtype=complex).copy()
    if depth < 1 or len(parameters) != 2 * depth:
        raise ValueError("depth and parameter count do not match")
    if state.shape != (mixer.dimension,) or costs.shape != (mixer.dimension,):
        raise ValueError("state, costs and mixer must have the same dimension")

    initial_norm = float(np.linalg.norm(state))
    if not np.isclose(initial_norm, 1.0, atol=PROBABILITY_TOLERANCE, rtol=0):
        raise ValueError("initial state is not normalized")

    gammas = parameters[:depth]
    betas = parameters[depth:]
    layer_norms = []
    for layer, (gamma, beta) in enumerate(zip(gammas, betas), start=1):
        state *= np.exp(-1j * gamma * costs)
        after_cost = float(np.linalg.norm(state))
        state = mixer.evolve(state, beta)
        after_mixer = float(np.linalg.norm(state))
        layer_norms.append(LayerNormRecord(layer, after_cost, after_mixer))

    probabilities = np.abs(state) ** 2
    probabilities /= probabilities.sum()
    return LogicalSimulation(
        state=tuple(map(complex, state)),
        probabilities=tuple(map(float, probabilities)),
        initial_norm=initial_norm,
        final_norm=float(np.linalg.norm(state)),
        layer_norms=tuple(layer_norms),
    )


def feasible_route_metrics(
    probabilities: Sequence[float],
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    *,
    initial_p_opt: float,
) -> FeasibleMetrics:
    """Summarize the final probability distribution over feasible routes."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (basis.size,) or np.any(probabilities < 0):
        raise ValueError("invalid probability vector")

    p_feas = probability_mass(probabilities)
    if abs(p_feas - 1) > PROBABILITY_TOLERANCE:
        raise RuntimeError(f"feasibility_invariant_failed: {p_feas}")

    optimal_id = basis.exact_optimal_route_id
    p_opt = indexed_probability_mass(probabilities, [optimal_id])
    optimal_rank = 1 + int(np.sum(probabilities > p_opt + 1e-14))
    order = np.lexsort((np.arange(basis.size), -np.round(probabilities, 15)))
    most_id = int(order[0])
    most_route = basis.routes[most_id]
    initial_p_opt = float(initial_p_opt)

    return FeasibleMetrics(
        p_feas=p_feas,
        p_opt=p_opt,
        optimal_state_rank=optimal_rank,
        top3_lowest_cost_mass=indexed_probability_mass(probabilities, range(3)),
        top5_lowest_cost_mass=indexed_probability_mass(probabilities, range(5)),
        expected_route_cost=float(probabilities @ np.asarray(costs.raw_energies)),
        expected_normalized_cost=float(
            probabilities @ np.asarray(costs.normalized_energies)
        ),
        most_probable_route_id=most_id,
        most_probable_route=most_route.node_sequence,
        most_probable_route_cost=most_route.routing_cost,
        most_probable_route_probability=float(probabilities[most_id]),
        probability_entropy=shannon_entropy(probabilities),
        initial_p_opt=initial_p_opt,
        final_p_opt=p_opt,
        p_opt_amplification=None if initial_p_opt <= 0 else p_opt / initial_p_opt,
    )
