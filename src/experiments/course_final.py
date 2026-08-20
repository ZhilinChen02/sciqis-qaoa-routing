"""Load the final configuration and run threshold Grover QAOA for each seed."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from feasible_qaoa import (
    UNIFORM_FEASIBLE,
    FeasibleInitialState,
    FeasibleRouteBasis,
    LogicalCostHamiltonian,
    build_feasible_initial_state,
    build_feasible_route_basis,
    build_logical_cost_hamiltonian,
)
from graph import load_graph
from feasible_experiments import (
    BSP_LOSS,
    FinalImprovementMetrics,
    GroverFeasibleMixer,
    IncumbentThreshold,
    build_grover_feasible_mixer,
    build_incumbent_threshold,
    final_improvement_metrics,
    optimize_final_variant,
    threshold_phase_values,
)
from qaoa import OptimizationResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "course_final.json"


def load_course_final_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the explicit final course configuration."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CourseFinalContext:
    basis: FeasibleRouteBasis
    costs: LogicalCostHamiltonian
    initial_state: FeasibleInitialState
    grover_mixer: GroverFeasibleMixer
    threshold: IncumbentThreshold


def build_course_final_context(config: dict[str, Any]) -> CourseFinalContext:
    """Construct the feasible basis and incumbent-derived marked set."""

    graph_path = PROJECT_ROOT / config["graph_path"]
    graph = load_graph(graph_path)
    basis = build_feasible_route_basis(graph)
    costs = build_logical_cost_hamiltonian(basis)
    incumbent_cost = float(costs.raw_energies[basis.incumbent_route_id])
    if incumbent_cost != float(config["incumbent_cost"]):
        raise RuntimeError(
            f"course_final_incumbent_cost_changed:{incumbent_cost}"
        )
    threshold = build_incumbent_threshold(costs.raw_energies, incumbent_cost)
    if basis.size != int(config["expected_feasible_route_count"]):
        raise RuntimeError(f"course_final_basis_size_changed:{basis.size}")
    if threshold.better_route_count != int(config["expected_marked_route_count"]):
        raise RuntimeError(
            f"course_final_marked_count_changed:{threshold.better_route_count}"
        )
    initial_state = build_feasible_initial_state(
        basis, costs, mode=UNIFORM_FEASIBLE
    )
    return CourseFinalContext(
        basis=basis,
        costs=costs,
        initial_state=initial_state,
        grover_mixer=build_grover_feasible_mixer(basis.size),
        threshold=threshold,
    )


@dataclass(frozen=True)
class CourseFinalRun:
    seed: int
    depth: int
    evaluation_budget: int
    p_feas: float
    p_opt: float
    threshold_probability: float
    expected_cost: float
    evaluations: int
    termination: str
    final_probabilities: tuple[float, ...]
    optimizer: OptimizationResult
    metrics: FinalImprovementMetrics

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_course_final_seed(
    context: CourseFinalContext,
    *,
    seed: int,
    depth: int,
    evaluation_budget: int,
    rhobeg: float = 0.5,
    tolerance: float = 1e-8,
) -> CourseFinalRun:
    """Run one bounded GM-Th-QAOA seed without supplying an optimum to optimization."""

    threshold_phase = threshold_phase_values(
        context.costs.raw_energies,
        context.threshold.incumbent_raw_cost,
    )
    optimized, simulation = optimize_final_variant(
        context.initial_state.amplitudes,
        threshold_phase,
        context.grover_mixer,
        context.costs.normalized_energies,
        context.threshold.better_mask,
        loss_kind=BSP_LOSS,
        depth=int(depth),
        seed=int(seed),
        evaluation_budget=int(evaluation_budget),
        rhobeg=float(rhobeg),
        tolerance=float(tolerance),
    )
    metrics = final_improvement_metrics(
        simulation.probabilities,
        context.basis,
        context.costs,
        context.threshold.better_mask,
        initial_p_opt=context.initial_state.p_opt,
    )
    if abs(metrics.p_feas - 1.0) > 1e-12:
        raise RuntimeError(f"course_final_feasibility_invariant_failed:{metrics.p_feas}")
    return CourseFinalRun(
        seed=int(seed),
        depth=int(depth),
        evaluation_budget=int(evaluation_budget),
        p_feas=metrics.p_feas,
        p_opt=metrics.p_opt,
        threshold_probability=metrics.bsp,
        expected_cost=metrics.expected_route_cost,
        evaluations=optimized.evaluations,
        termination=optimized.reason,
        final_probabilities=simulation.probabilities,
        optimizer=optimized,
        metrics=metrics,
    )


def run_course_final(
    config: dict[str, Any],
    *,
    seeds: Sequence[int] | None = None,
    depth: int | None = None,
) -> tuple[CourseFinalContext, tuple[CourseFinalRun, ...]]:
    """Execute the configured seed set and return every retained run."""

    context = build_course_final_context(config)
    selected_seeds = tuple(map(int, config["seeds"] if seeds is None else seeds))
    selected_depth = int(config["depth"] if depth is None else depth)
    if not selected_seeds or selected_depth not in (1, 2, 3, 4):
        raise ValueError("invalid_course_final_seed_or_depth_request")
    runs = []
    for seed in selected_seeds:
        runs.append(
            run_course_final_seed(
                context,
                seed=seed,
                depth=selected_depth,
                evaluation_budget=int(config["objective_evaluation_cap"]),
                rhobeg=float(config["optimizer_rhobeg"]),
                tolerance=float(config["optimizer_tolerance"]),
            )
        )
    return context, tuple(runs)


def course_final_payload(
    config: dict[str, Any],
    context: CourseFinalContext,
    runs: Sequence[CourseFinalRun],
) -> dict[str, Any]:
    """Build an optional development-output object for an explicit path."""

    return {
        "method": config["method"],
        "depth": runs[0].depth,
        "seeds": [run.seed for run in runs],
        "basis_size": context.basis.size,
        "incumbent_route": list(context.basis.incumbent_route),
        "incumbent_cost": context.threshold.incumbent_raw_cost,
        "marked_route_count": context.threshold.better_route_count,
        "marked_route_ids": list(context.threshold.better_route_ids),
        "initial_uniform_p_opt": context.initial_state.p_opt,
        "median_p_opt": float(median(run.p_opt for run in runs)),
        "runs": [run.as_dict() for run in runs],
    }
