"""QAOA experiments in the 20-dimensional feasible-route space.

There are three variants in the comparison:

* path-exchange mixer with the route-cost phase;
* Grover mixer with the route-cost phase;
* Grover mixer with a phase that marks routes better than a greedy route.

All three use the same simulation loop and COBYLA helper below.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from metrics import probability_mass
from qaoa import OptimizationResult, optimize_cobyla
from feasible_qaoa import (
    FeasibleMetrics,
    FeasibleRouteBasis,
    LogicalCostHamiltonian,
    LogicalSimulation,
    feasible_route_metrics,
    simulate_logical_qaoa,
)


BSP_LOSS = "negative_better_solution_probability"
EXPECTATION_LOSS = "normalized_route_cost_expectation"


def better_than_incumbent_mask(
    raw_route_costs: Sequence[float], incumbent_raw_cost: float
) -> np.ndarray:
    """Mark every route whose cost is strictly below the greedy route cost."""

    costs = np.asarray(raw_route_costs, dtype=float)
    if costs.ndim != 1 or len(costs) == 0:
        raise ValueError("route costs must be a non-empty vector")
    return costs < float(incumbent_raw_cost)


def better_solution_probability(
    probabilities: Sequence[float], better_mask: Sequence[bool]
) -> float:
    """Add the probabilities of the routes marked as better."""

    probabilities = np.asarray(probabilities, dtype=float)
    mask = np.asarray(better_mask, dtype=bool)
    if probabilities.shape != mask.shape:
        raise ValueError("probabilities and mask must have the same size")
    return probability_mass(probabilities, mask)


@dataclass(frozen=True)
class IncumbentThreshold:
    incumbent_raw_cost: float
    better_mask: tuple[bool, ...]
    better_route_ids: tuple[int, ...]

    @property
    def better_route_count(self) -> int:
        return len(self.better_route_ids)


def build_incumbent_threshold(
    raw_route_costs: Sequence[float], incumbent_raw_cost: float
) -> IncumbentThreshold:
    mask = better_than_incumbent_mask(raw_route_costs, incumbent_raw_cost)
    route_ids = tuple(map(int, np.flatnonzero(mask)))
    return IncumbentThreshold(
        float(incumbent_raw_cost), tuple(map(bool, mask)), route_ids
    )


@dataclass(frozen=True)
class GroverFeasibleMixer:
    """Grover mixer H = |F><F| for the uniform feasible state |F>."""

    uniform_state: tuple[complex, ...]
    hamiltonian: np.ndarray

    @property
    def dimension(self) -> int:
        return len(self.uniform_state)

    def unitary(self, beta: float) -> np.ndarray:
        phase = np.exp(-1j * beta)
        identity = np.eye(self.dimension, dtype=complex)
        return identity + (phase - 1) * self.hamiltonian

    def evolve(self, state: Sequence[complex], beta: float) -> np.ndarray:
        state = np.asarray(state, dtype=complex)
        uniform = np.asarray(self.uniform_state)
        overlap = np.vdot(uniform, state)
        return state + (np.exp(-1j * beta) - 1) * uniform * overlap


def build_grover_feasible_mixer(route_count: int) -> GroverFeasibleMixer:
    """Build the rank-one mixer from a uniform vector over all routes."""

    route_count = int(route_count)
    if route_count < 1:
        raise ValueError("the feasible space cannot be empty")
    uniform = np.full(route_count, 1 / np.sqrt(route_count), dtype=complex)
    hamiltonian = np.outer(uniform, uniform.conj())
    return GroverFeasibleMixer(tuple(map(complex, uniform)), hamiltonian)


def threshold_phase_values(
    raw_route_costs: Sequence[float], incumbent_raw_cost: float
) -> np.ndarray:
    """Return 1 for routes better than the incumbent and 0 otherwise."""

    return better_than_incumbent_mask(
        raw_route_costs, incumbent_raw_cost
    ).astype(float)


# The final simulator has exactly the same result fields as the basic logical
# simulator, so one small data class is enough for both.
FinalImprovementSimulation = LogicalSimulation


def simulate_final_improvement(
    initial_state: Sequence[complex],
    phase_values: Sequence[float],
    mixer,
    parameters: Sequence[float],
    *,
    depth: int,
) -> FinalImprovementSimulation:
    """Alternate phase and mixer operations for p QAOA layers."""

    return simulate_logical_qaoa(
        initial_state,
        phase_values,
        mixer,
        parameters,
        depth=depth,
    )


def evaluate_final_loss(
    probabilities: Sequence[float],
    *,
    loss_kind: str,
    normalized_costs: Sequence[float],
    better_mask: Sequence[bool],
) -> float:
    """Return either negative BSP or expected normalized route cost."""

    probabilities = np.asarray(probabilities, dtype=float)
    if loss_kind == BSP_LOSS:
        return -better_solution_probability(probabilities, better_mask)
    if loss_kind == EXPECTATION_LOSS:
        return float(probabilities @ np.asarray(normalized_costs))
    raise ValueError(f"unknown loss: {loss_kind}")


@dataclass(frozen=True)
class FinalImprovementMetrics(FeasibleMetrics):
    bsp: float


def final_improvement_metrics(
    probabilities: Sequence[float],
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    better_mask: Sequence[bool],
    *,
    initial_p_opt: float,
) -> FinalImprovementMetrics:
    """Add better-solution probability to the common route metrics."""

    base: FeasibleMetrics = feasible_route_metrics(
        probabilities, basis, costs, initial_p_opt=initial_p_opt
    )
    return FinalImprovementMetrics(
        **vars(base),
        bsp=better_solution_probability(probabilities, better_mask),
    )


def optimize_final_variant(
    initial_state,
    phase_values,
    mixer,
    normalized_costs,
    better_mask,
    *,
    loss_kind,
    depth,
    seed,
    evaluation_budget=150,
    rhobeg=0.5,
    tolerance=1e-8,
) -> tuple[OptimizationResult, LogicalSimulation]:
    """Optimize one feasible-space variant and return its final simulation."""

    def loss(parameters):
        simulation = simulate_final_improvement(
            initial_state, phase_values, mixer, parameters, depth=depth
        )
        return evaluate_final_loss(
            simulation.probabilities,
            loss_kind=loss_kind,
            normalized_costs=normalized_costs,
            better_mask=better_mask,
        )

    result = optimize_cobyla(
        loss,
        depth=depth,
        seed=seed,
        evaluation_budget=evaluation_budget,
        rhobeg=rhobeg,
        tolerance=tolerance,
    )
    simulation = simulate_final_improvement(
        initial_state, phase_values, mixer, result.parameters, depth=depth
    )
    return result, simulation
