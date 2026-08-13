"""Archived tuning helpers for incumbent-product Warm-Start QAOA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from support.exact_reference import compute_exact_reference
from graph import DEFAULT_GRAPH_PATH, load_graph, path_cost, path_to_edge_bitstring
from qubo import IsingHamiltonian, qubo_to_ising
from utils import DistributionMetrics, distribution_metrics, top_state_rows
from utils import SOURCE_UNIFORM, initial_parameters, optimize_cobyla
from qaoa import (
    Q2_WARM_START,
    CircuitStatistics,
    build_qaoa_circuit,
    circuit_statistics,
    normalized_diagonal,
    simulate_qaoa_state,
    state_probabilities,
)
from qubo import (
    QuboPolynomial,
    StateRecord,
    build_qubo,
    edge_vector_to_state_index,
    enumerate_state_space,
)
from utils import greedy_incumbent_route, incumbent_relaxation, product_state


@dataclass(frozen=True)
class TuningContext:
    graph: nx.DiGraph
    states: tuple[StateRecord, ...]
    qubo: QuboPolynomial
    ising: IsingHamiltonian
    raw_diagonal: np.ndarray
    normalized_diagonal: np.ndarray
    normalization_scale: float
    optimal_route: tuple[int, ...]
    optimal_cost: int
    optimal_state_index: int
    incumbent_route: tuple[int, ...]
    incumbent_cost: int
    incumbent_state_index: int


@dataclass(frozen=True)
class TuningRun:
    epsilon: float
    p: int
    seed: int
    optimizer_budget: int
    initialization_strategy: str
    initial_parameters: str
    final_parameters: str
    initial_objective: float
    optimized_normalized_energy: float
    optimized_energy: float
    p_feas: float
    p_opt: float
    feasible_conditional_cost: float | None
    initial_incumbent_probability: float
    incumbent_probability: float
    incumbent_amplification: float | None
    initial_optimal_probability: float
    optimal_state_probability: float
    optimum_amplification: float | None
    optimal_state_rank: int
    highest_probability_route: str
    highest_probability_route_cost: int | None
    best_state_probability: float
    best_state_valid: bool
    best_state_optimal: bool
    optimizer_evaluations: int
    optimizer_time_s: float
    total_runtime_s: float
    optimizer_success: bool
    optimizer_reason: str
    degenerate_parameter_flag: bool
    degenerate_parameters: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def build_tuning_context(penalty: float = 6.0) -> TuningContext:
    graph = load_graph()
    exact, _ = compute_exact_reference(graph, graph_path=DEFAULT_GRAPH_PATH)
    optimal_route = tuple(exact["exact_reference"]["node_path"])
    optimal_cost = int(exact["exact_reference"]["cost"])
    states = enumerate_state_space(graph)
    qubo = build_qubo(graph, penalty)
    ising = qubo_to_ising(qubo)
    raw = np.asarray([float(qubo.evaluate(state.edge_vector)) for state in states])
    normalized, _, scale = normalized_diagonal(raw)
    incumbent_route = greedy_incumbent_route(graph)
    incumbent_vector = path_to_edge_bitstring(graph, incumbent_route)
    optimal_vector = path_to_edge_bitstring(graph, optimal_route)
    return TuningContext(
        graph=graph,
        states=states,
        qubo=qubo,
        ising=ising,
        raw_diagonal=raw,
        normalized_diagonal=normalized,
        normalization_scale=scale,
        optimal_route=optimal_route,
        optimal_cost=optimal_cost,
        optimal_state_index=edge_vector_to_state_index(optimal_vector),
        incumbent_route=incumbent_route,
        incumbent_cost=path_cost(graph, incumbent_route),
        incumbent_state_index=edge_vector_to_state_index(incumbent_vector),
    )


def warm_start_probabilities(
    context: TuningContext,
    epsilon: float,
) -> tuple[tuple[float, ...], np.ndarray, float, float]:
    """Return clipped values, full initial distribution, and tracked overlaps."""

    rows = incumbent_relaxation(context.graph, epsilon)
    values = tuple(row.clipped_value for row in rows)
    probabilities = state_probabilities(product_state(values))
    return (
        values,
        probabilities,
        float(probabilities[context.incumbent_state_index]),
        float(probabilities[context.optimal_state_index]),
    )


def degenerate_parameters(
    parameters: Sequence[float],
    depth: int,
    *,
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """Flag zero or optimizer-boundary gamma/beta coordinates."""

    values = np.asarray(parameters, dtype=float)
    if values.shape != (2 * int(depth),):
        raise ValueError("qaoa_parameter_count_mismatch")
    labels: list[str] = []
    for index, value in enumerate(values[:depth]):
        if abs(value) < tolerance:
            labels.append(f"gamma_{index + 1}:lower_zero")
        elif abs(value - 2.0 * np.pi) < tolerance:
            labels.append(f"gamma_{index + 1}:upper_2pi")
    for index, value in enumerate(values[depth:]):
        if abs(value) < tolerance:
            labels.append(f"beta_{index + 1}:lower_zero")
        elif abs(value - np.pi) < tolerance:
            labels.append(f"beta_{index + 1}:upper_pi")
    return bool(labels), "|".join(labels)


def _safe_amplification(final: float, initial: float) -> float | None:
    return None if initial <= 0.0 else float(final / initial)


def run_tuning_start(
    context: TuningContext,
    *,
    epsilon: float,
    depth: int,
    seed: int,
    optimizer_budget: int,
    initialization_strategy: str = SOURCE_UNIFORM,
    optimizer_tolerance: float = 1e-8,
    optimizer_rhobeg: float = 0.5,
) -> TuningRun:
    """Optimize expected energy only, then evaluate route probabilities."""

    started = perf_counter()
    values, initial_probabilities, initial_incumbent, initial_optimal = (
        warm_start_probabilities(context, epsilon)
    )

    def simulate(parameters: np.ndarray) -> np.ndarray:
        return simulate_qaoa_state(
            context.normalized_diagonal,
            parameters,
            depth=depth,
            solver=Q2_WARM_START,
            warm_start_values=values,
        )

    def objective(parameters: np.ndarray) -> float:
        probabilities = state_probabilities(simulate(parameters))
        return float(probabilities @ context.normalized_diagonal)

    initial = initial_parameters(
        depth, seed, strategy=initialization_strategy
    )
    optimized = optimize_cobyla(
        objective,
        depth=depth,
        seed=seed,
        evaluation_budget=optimizer_budget,
        tolerance=optimizer_tolerance,
        rhobeg=optimizer_rhobeg,
        initialization_strategy=initialization_strategy,
    )
    probabilities = state_probabilities(
        simulate(np.asarray(optimized.parameters, dtype=float))
    )
    evaluated = distribution_metrics(
        probabilities, context.states, optimal_cost=context.optimal_cost
    )
    incumbent_probability = float(probabilities[context.incumbent_state_index])
    optimal_probability = float(probabilities[context.optimal_state_index])
    is_degenerate, labels = degenerate_parameters(optimized.parameters, depth)
    return TuningRun(
        epsilon=float(epsilon),
        p=int(depth),
        seed=int(seed),
        optimizer_budget=int(optimizer_budget),
        initialization_strategy=initialization_strategy,
        initial_parameters=json.dumps(list(map(float, initial))),
        final_parameters=json.dumps(list(optimized.parameters)),
        initial_objective=float(optimized.initial_objective),
        optimized_normalized_energy=float(probabilities @ context.normalized_diagonal),
        optimized_energy=float(probabilities @ context.raw_diagonal),
        p_feas=evaluated.p_feas,
        p_opt=evaluated.p_opt,
        feasible_conditional_cost=evaluated.feasible_conditional_cost,
        initial_incumbent_probability=initial_incumbent,
        incumbent_probability=incumbent_probability,
        incumbent_amplification=_safe_amplification(
            incumbent_probability, initial_incumbent
        ),
        initial_optimal_probability=initial_optimal,
        optimal_state_probability=optimal_probability,
        optimum_amplification=_safe_amplification(
            optimal_probability, initial_optimal
        ),
        optimal_state_rank=evaluated.optimal_state_rank,
        highest_probability_route=(
            "" if evaluated.best_decoded_route is None
            else "->".join(map(str, evaluated.best_decoded_route))
        ),
        highest_probability_route_cost=evaluated.best_decoded_cost,
        best_state_probability=evaluated.best_state_probability,
        best_state_valid=evaluated.best_state_valid,
        best_state_optimal=evaluated.best_state_optimal,
        optimizer_evaluations=optimized.evaluations,
        optimizer_time_s=optimized.wall_time,
        total_runtime_s=perf_counter() - started,
        optimizer_success=optimized.success,
        optimizer_reason=optimized.reason,
        degenerate_parameter_flag=is_degenerate,
        degenerate_parameters=labels,
    )


def energy_selected_winner(runs: Sequence[TuningRun]) -> TuningRun:
    """Select only by expected raw energy, with deterministic neutral ties."""

    if not runs:
        raise ValueError("cannot select a winner from no runs")
    keys = {
        (run.epsilon, run.p, run.optimizer_budget, run.initialization_strategy)
        for run in runs
    }
    if len(keys) != 1:
        raise ValueError("winner candidates must share epsilon, p, budget, and strategy")
    return min(runs, key=lambda run: (run.optimized_energy, run.seed))


def summarize_runs(runs: Sequence[TuningRun]) -> dict[str, object]:
    winner = energy_selected_winner(runs)

    def values(name: str) -> list[float]:
        return [float(getattr(run, name)) for run in runs]

    row = winner.as_row()
    row["winner_seed"] = winner.seed
    for name in (
        "optimized_energy",
        "p_feas",
        "p_opt",
        "incumbent_probability",
        "optimum_amplification",
        "optimizer_time_s",
    ):
        observations = values(name)
        row[f"{name}_median"] = median(observations)
        row[f"{name}_best"] = (
            min(observations) if name in {"optimized_energy", "incumbent_probability", "optimizer_time_s"}
            else max(observations)
        )
        row[f"{name}_worst"] = (
            max(observations) if name in {"optimized_energy", "incumbent_probability", "optimizer_time_s"}
            else min(observations)
        )
    row["optimizer_success_rate"] = sum(run.optimizer_success for run in runs) / len(runs)
    row["degenerate_run_count"] = sum(run.degenerate_parameter_flag for run in runs)
    row["num_starts"] = len(runs)
    return row


def circuit_resources(
    context: TuningContext,
    run: TuningRun,
) -> CircuitStatistics:
    values = tuple(
        row.clipped_value for row in incumbent_relaxation(context.graph, run.epsilon)
    )
    circuit = build_qaoa_circuit(
        context.ising,
        json.loads(run.final_parameters),
        depth=run.p,
        solver=Q2_WARM_START,
        normalization_scale=context.normalization_scale,
        warm_start_values=values,
    )
    return circuit_statistics(circuit)


def final_distribution(
    context: TuningContext,
    run: TuningRun,
) -> tuple[np.ndarray, tuple[float, ...]]:
    values = tuple(
        row.clipped_value for row in incumbent_relaxation(context.graph, run.epsilon)
    )
    state = simulate_qaoa_state(
        context.normalized_diagonal,
        json.loads(run.final_parameters),
        depth=run.p,
        solver=Q2_WARM_START,
        warm_start_values=values,
    )
    return state_probabilities(state), values


def top_rows_for_run(
    context: TuningContext,
    run: TuningRun,
    top_k: int = 12,
) -> list[dict[str, object]]:
    probabilities, _ = final_distribution(context, run)
    return top_state_rows(
        probabilities,
        context.states,
        optimal_cost=context.optimal_cost,
        top_k=top_k,
    )


def rank_promising_epsilons(summary_rows: Sequence[dict[str, object]]) -> list[tuple[float, float]]:
    """Joint robust rank: energy, feasibility, optimal mass, incumbent release."""

    primary = [
        row for row in summary_rows
        if int(row["optimizer_budget"]) == 100
        and row["initialization_strategy"] == SOURCE_UNIFORM
        and int(row["p"]) in (1, 2)
    ]
    epsilons = sorted({float(row["epsilon"]) for row in primary})
    aggregated: dict[float, dict[str, float]] = {}
    for epsilon in epsilons:
        rows = [row for row in primary if float(row["epsilon"]) == epsilon]
        if len(rows) != 2:
            raise ValueError("promising-epsilon ranking requires both p=1 and p=2")
        aggregated[epsilon] = {
            "energy": float(np.mean([float(row["optimized_energy_median"]) for row in rows])),
            "p_feas": float(np.mean([float(row["p_feas_median"]) for row in rows])),
            "p_opt": float(np.mean([float(row["p_opt_median"]) for row in rows])),
            "incumbent": float(np.mean([float(row["incumbent_probability_median"]) for row in rows])),
        }

    def ranks(metric: str, reverse: bool) -> dict[float, int]:
        ordered = sorted(epsilons, key=lambda eps: aggregated[eps][metric], reverse=reverse)
        return {epsilon: rank for rank, epsilon in enumerate(ordered, start=1)}

    energy_rank = ranks("energy", False)
    feasibility_rank = ranks("p_feas", True)
    optimal_rank = ranks("p_opt", True)
    incumbent_rank = ranks("incumbent", False)
    scored = []
    for epsilon in epsilons:
        score = float(
            energy_rank[epsilon]
            + feasibility_rank[epsilon]
            + optimal_rank[epsilon]
            + incumbent_rank[epsilon]
        )
        scored.append((epsilon, score))
    return sorted(scored, key=lambda pair: (pair[1], pair[0]))
