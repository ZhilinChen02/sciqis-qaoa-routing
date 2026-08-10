"""Small readable Penalty-X QAOA demonstration for the course core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exact_reference import compute_exact_reference
from graph import DEFAULT_GRAPH_PATH, load_graph
from ising import max_qubo_ising_error, qubo_to_ising
from metrics import DistributionMetrics, distribution_metrics
from optimization import OptimizationResult, optimize_cobyla
from qaoa import Q1_PENALTY_X, normalized_diagonal, simulate_qaoa_state, state_probabilities
from qubo import (
    build_qubo,
    enumerate_state_space,
    minimum_energy_states,
)


@dataclass(frozen=True)
class PenaltyDemoResult:
    node_count: int
    edge_count: int
    state_count: int
    penalty: float
    depth: int
    seed: int
    exact_route: tuple[int, ...]
    exact_cost: int
    qubo_ising_max_error: float
    ground_state_count: int
    optimizer: OptimizationResult
    p_feas: float
    p_opt: float
    expected_feasible_cost: float | None
    most_probable_feasible_route: tuple[int, ...] | None
    most_probable_feasible_cost: int | None
    probability_sum: float


def run_penalty_demo(
    *,
    penalty: float = 6.0,
    depth: int = 1,
    seed: int = 2601,
    evaluation_budget: int = 100,
) -> PenaltyDemoResult:
    """Build graph→QUBO→Ising→Penalty-X and evaluate the full distribution."""

    graph = load_graph(DEFAULT_GRAPH_PATH)
    exact_payload, _ = compute_exact_reference(graph, graph_path=DEFAULT_GRAPH_PATH)
    exact_route = tuple(exact_payload["exact_reference"]["node_path"])
    exact_cost = int(exact_payload["exact_reference"]["cost"])
    states = enumerate_state_space(graph)
    qubo = build_qubo(graph, float(penalty))
    ising = qubo_to_ising(qubo)
    qubo_ising_error = float(max_qubo_ising_error(states, [qubo]))
    if qubo_ising_error != 0.0:
        raise RuntimeError(f"penalty_demo_qubo_ising_mismatch:{qubo_ising_error}")
    _, ground_states = minimum_energy_states(states, float(penalty))
    if not ground_states or any(
        not state.is_decoder_valid or state.routing_cost != exact_cost
        for state in ground_states
    ):
        raise RuntimeError("penalty_demo_ground_state_not_exact_route")
    raw_diagonal = np.asarray(
        [float(qubo.evaluate(state.edge_vector)) for state in states], dtype=np.float64
    )
    normalized, _, _ = normalized_diagonal(raw_diagonal)

    def expectation(parameters: np.ndarray) -> float:
        state = simulate_qaoa_state(
            normalized,
            parameters,
            depth=int(depth),
            solver=Q1_PENALTY_X,
        )
        probabilities = state_probabilities(state)
        return float(probabilities @ normalized)

    optimizer = optimize_cobyla(
        expectation,
        depth=int(depth),
        seed=int(seed),
        evaluation_budget=int(evaluation_budget),
    )
    final_state = simulate_qaoa_state(
        normalized,
        optimizer.parameters,
        depth=int(depth),
        solver=Q1_PENALTY_X,
    )
    probabilities = state_probabilities(final_state)
    metrics: DistributionMetrics = distribution_metrics(
        probabilities, states, optimal_cost=exact_cost
    )
    return PenaltyDemoResult(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        state_count=len(states),
        penalty=float(penalty),
        depth=int(depth),
        seed=int(seed),
        exact_route=exact_route,
        exact_cost=exact_cost,
        qubo_ising_max_error=qubo_ising_error,
        ground_state_count=len(ground_states),
        optimizer=optimizer,
        p_feas=metrics.p_feas,
        p_opt=metrics.p_opt,
        expected_feasible_cost=metrics.feasible_conditional_cost,
        most_probable_feasible_route=metrics.best_decoded_route,
        most_probable_feasible_cost=metrics.best_decoded_cost,
        probability_sum=float(np.sum(probabilities)),
    )
