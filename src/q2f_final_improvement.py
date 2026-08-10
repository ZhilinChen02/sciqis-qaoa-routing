"""Bounded final Q2-F mechanism study in the enumerated feasible-route basis.

This module intentionally does not change :mod:`feasible_qaoa`.  It reuses the
sealed Q2-F basis, cost data, warm start, and path-exchange mixer, and adds only
the three prospectively bounded variants requested for the final course study.
The simulation is a 20-dimensional logical reference, not a scalable circuit
implementation or a quantum-advantage claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

from feasible_qaoa import (
    FeasibleInitialState,
    FeasibleMetrics,
    FeasibleRouteBasis,
    LayerNormRecord,
    LogicalCostHamiltonian,
    LogicalPathExchangeMixer,
    PROBABILITY_TOLERANCE,
    feasible_route_metrics,
)
from optimization import EvaluationBudgetExhausted, SOURCE_UNIFORM, initial_parameters


BSP_PATH_EXCHANGE = "bsp_path_exchange"
GM_QAOA_EXPECTATION = "gm_qaoa_expectation"
GM_TH_QAOA = "gm_th_qaoa"
FINAL_METHODS = (BSP_PATH_EXCHANGE, GM_QAOA_EXPECTATION, GM_TH_QAOA)

PATH_EXCHANGE_KIND = "logical_path_exchange_mixer"
GROVER_MIXER_KIND = "logical_grover_feasible_mixer"
COST_PHASE = "normalized_route_cost_phase"
THRESHOLD_PHASE = "strict_incumbent_threshold_phase"
BSP_LOSS = "negative_better_solution_probability"
EXPECTATION_LOSS = "normalized_route_cost_expectation"


def better_than_incumbent_mask(
    raw_route_costs: Sequence[float], incumbent_raw_cost: float
) -> np.ndarray:
    """Return the strict minimization BSP set using only the incumbent cost.

    No optimum index, label, or cost is accepted by this constructor.
    """

    costs = np.asarray(raw_route_costs, dtype=np.float64)
    threshold = float(incumbent_raw_cost)
    if costs.ndim != 1 or costs.size == 0 or np.any(~np.isfinite(costs)):
        raise ValueError("raw_route_costs_must_be_a_finite_nonempty_vector")
    if not np.isfinite(threshold):
        raise ValueError("incumbent_raw_cost_must_be_finite")
    return costs < threshold


def better_solution_probability(
    probabilities: Sequence[float], better_mask: Sequence[bool]
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(better_mask, dtype=bool)
    if probs.ndim != 1 or probs.shape != mask.shape:
        raise ValueError("bsp_probability_and_mask_dimensions_differ")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("invalid_bsp_probability_vector")
    return float(np.sum(probs[mask]))


@dataclass(frozen=True)
class IncumbentThreshold:
    incumbent_raw_cost: float
    better_mask: tuple[bool, ...]
    better_route_ids: tuple[int, ...]

    @property
    def better_route_count(self) -> int:
        return len(self.better_route_ids)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_incumbent_threshold(
    raw_route_costs: Sequence[float], incumbent_raw_cost: float
) -> IncumbentThreshold:
    mask = better_than_incumbent_mask(raw_route_costs, incumbent_raw_cost)
    route_ids = tuple(int(index) for index in np.flatnonzero(mask))
    return IncumbentThreshold(
        incumbent_raw_cost=float(incumbent_raw_cost),
        better_mask=tuple(map(bool, mask)),
        better_route_ids=route_ids,
    )


@dataclass(frozen=True)
class GroverFeasibleMixer:
    """Rank-one GM-QAOA mixer H_G=|F><F| in the exact logical basis."""

    uniform_state: tuple[complex, ...]
    hamiltonian: np.ndarray
    mixer_name: str = GROVER_MIXER_KIND

    @property
    def dimension(self) -> int:
        return len(self.uniform_state)

    def unitary(self, beta: float) -> np.ndarray:
        phase = np.exp(-1j * float(beta))
        return np.eye(self.dimension, dtype=np.complex128) + (phase - 1.0) * self.hamiltonian

    def evolve(self, state: Sequence[complex], beta: float) -> np.ndarray:
        vector = np.asarray(state, dtype=np.complex128)
        if vector.shape != (self.dimension,):
            raise ValueError("grover_mixer_state_dimension_mismatch")
        feasible = np.asarray(self.uniform_state, dtype=np.complex128)
        overlap = np.vdot(feasible, vector)
        return vector + (np.exp(-1j * float(beta)) - 1.0) * feasible * overlap

    def as_dict(self) -> dict[str, Any]:
        return {
            "mixer_name": self.mixer_name,
            "definition": "H_G=|F><F|; U_G(beta)=I+(exp(-i beta)-1)|F><F|",
            "dimension": self.dimension,
            "uniform_state_real": [float(value.real) for value in self.uniform_state],
            "hamiltonian_real": self.hamiltonian.real.tolist(),
            "hermitian_error": float(np.max(np.abs(self.hamiltonian - self.hamiltonian.conj().T))),
        }


def build_grover_feasible_mixer(route_count: int) -> GroverFeasibleMixer:
    size = int(route_count)
    if size < 1:
        raise ValueError("grover_feasible_space_must_be_nonempty")
    uniform = np.full(size, 1.0 / np.sqrt(size), dtype=np.complex128)
    hamiltonian = np.outer(uniform, uniform.conj())
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-14, rtol=0.0):
        raise RuntimeError("grover_mixer_hamiltonian_not_hermitian")
    return GroverFeasibleMixer(tuple(map(complex, uniform)), hamiltonian)


def threshold_phase_values(
    raw_route_costs: Sequence[float], incumbent_raw_cost: float
) -> np.ndarray:
    """Boolean h_T for minimization: one iff C(P) < C(incumbent)."""

    return better_than_incumbent_mask(raw_route_costs, incumbent_raw_cost).astype(np.float64)


class LogicalMixer(Protocol):
    @property
    def dimension(self) -> int: ...

    def evolve(self, state: Sequence[complex], beta: float) -> np.ndarray: ...


@dataclass(frozen=True)
class FinalImprovementSimulation:
    state: tuple[complex, ...]
    probabilities: tuple[float, ...]
    initial_norm: float
    final_norm: float
    layer_norms: tuple[LayerNormRecord, ...]


def simulate_final_improvement(
    initial_state: Sequence[complex],
    phase_values: Sequence[float],
    mixer: LogicalMixer,
    parameters: Sequence[float],
    *,
    depth: int,
) -> FinalImprovementSimulation:
    """Apply cost/threshold phase then a feasibility-preserving logical mixer."""

    p = int(depth)
    if p not in (1, 2, 3, 4):
        raise ValueError("q2f_final_depth_must_be_1_to_4")
    values = np.asarray(parameters, dtype=np.float64)
    phase = np.asarray(phase_values, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.complex128).copy()
    if values.shape != (2 * p,) or np.any(~np.isfinite(values)):
        raise ValueError("q2f_final_parameter_count_or_finiteness_error")
    if state.shape != (mixer.dimension,) or phase.shape != (mixer.dimension,):
        raise ValueError("q2f_final_logical_dimension_mismatch")
    if np.any(~np.isfinite(phase)):
        raise ValueError("q2f_final_phase_values_nonfinite")
    initial_norm = float(np.linalg.norm(state))
    if abs(initial_norm - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError("q2f_final_initial_state_not_normalized")
    gammas, betas = values[:p], values[p:]
    norms: list[LayerNormRecord] = []
    for layer in range(p):
        state *= np.exp(-1j * float(gammas[layer]) * phase)
        after_phase = float(np.linalg.norm(state))
        if abs(after_phase - 1.0) > PROBABILITY_TOLERANCE:
            raise RuntimeError("q2f_final_phase_changed_norm")
        state = mixer.evolve(state, float(betas[layer]))
        after_mixer = float(np.linalg.norm(state))
        if abs(after_mixer - 1.0) > PROBABILITY_TOLERANCE:
            raise RuntimeError("q2f_final_mixer_changed_norm")
        norms.append(LayerNormRecord(layer + 1, after_phase, after_mixer))
    probabilities = np.abs(state) ** 2
    total = float(np.sum(probabilities))
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise RuntimeError(f"q2f_final_feasibility_invariant_failed:{total}")
    probabilities /= total
    return FinalImprovementSimulation(
        state=tuple(map(complex, state)),
        probabilities=tuple(map(float, probabilities)),
        initial_norm=initial_norm,
        final_norm=float(np.linalg.norm(state)),
        layer_norms=tuple(norms),
    )


def evaluate_final_loss(
    probabilities: Sequence[float],
    *,
    loss_kind: str,
    normalized_costs: Sequence[float],
    better_mask: Sequence[bool],
) -> float:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    costs = np.asarray(normalized_costs, dtype=np.float64)
    if probabilities_array.shape != costs.shape:
        raise ValueError("final_loss_probability_cost_dimension_mismatch")
    if loss_kind == BSP_LOSS:
        return -better_solution_probability(probabilities_array, better_mask)
    if loss_kind == EXPECTATION_LOSS:
        return float(probabilities_array @ costs)
    raise ValueError(f"unsupported_q2f_final_loss:{loss_kind}")


@dataclass(frozen=True)
class FinalEvaluationRecord:
    evaluation_index: int
    parameters: tuple[float, ...]
    loss: float
    bsp: float | None
    expected_normalized_cost: float | None
    p_feas: float | None
    in_bounds: bool


@dataclass(frozen=True)
class FinalOptimizationResult:
    initial_parameters: tuple[float, ...]
    final_parameters: tuple[float, ...]
    initial_loss: float
    final_loss: float
    optimizer_reported_loss: float
    evaluations: int
    statevector_evaluations: int
    success: bool
    reason: str
    message: str
    runtime_seconds: float
    retention_policy: str
    evaluation_trace: tuple[FinalEvaluationRecord, ...]


def optimize_final_variant(
    initial_state: Sequence[complex],
    phase_values: Sequence[float],
    mixer: LogicalMixer,
    normalized_costs: Sequence[float],
    better_mask: Sequence[bool],
    *,
    loss_kind: str,
    depth: int,
    seed: int,
    evaluation_budget: int = 150,
    rhobeg: float = 0.5,
    tolerance: float = 1e-8,
) -> tuple[FinalOptimizationResult, FinalImprovementSimulation]:
    """Bounded stationary COBYLA; this function receives no optimum identity."""

    p, budget = int(depth), int(evaluation_budget)
    if p not in (1, 2, 3, 4) or budget < 1 or loss_kind not in (BSP_LOSS, EXPECTATION_LOSS):
        raise ValueError("invalid_q2f_final_optimizer_configuration")
    normalized = np.asarray(normalized_costs, dtype=np.float64)
    marked = np.asarray(better_mask, dtype=bool)
    if normalized.shape != (mixer.dimension,) or marked.shape != (mixer.dimension,):
        raise ValueError("q2f_final_optimizer_dimension_mismatch")
    initial = initial_parameters(p, int(seed), strategy=SOURCE_UNIFORM)
    lower = np.asarray([0.0] * p + [0.0] * p, dtype=np.float64)
    upper = np.asarray([2.0 * pi] * p + [pi] * p, dtype=np.float64)
    trace: list[FinalEvaluationRecord] = []
    evaluations = 0
    statevector_evaluations = 0
    best_loss = float("inf")
    best_parameters = initial.copy()

    def counted(parameters: np.ndarray) -> float:
        nonlocal evaluations, statevector_evaluations, best_loss, best_parameters
        if evaluations >= budget:
            raise EvaluationBudgetExhausted
        candidate = np.asarray(parameters, dtype=np.float64)
        evaluations += 1
        outside = float(np.sum(np.maximum(lower - candidate, 0.0) + np.maximum(candidate - upper, 0.0)))
        if outside:
            loss = 1_000_000.0 + outside
            bsp = expected = p_feas = None
            in_bounds = False
        else:
            simulation = simulate_final_improvement(
                initial_state, phase_values, mixer, candidate, depth=p
            )
            statevector_evaluations += 1
            probs = np.asarray(simulation.probabilities, dtype=np.float64)
            p_feas = float(np.sum(probs))
            if abs(p_feas - 1.0) > PROBABILITY_TOLERANCE:
                raise RuntimeError("q2f_final_optimizer_feasibility_invariant_failed")
            bsp = better_solution_probability(probs, marked)
            expected = float(probs @ normalized)
            loss = evaluate_final_loss(
                probs,
                loss_kind=loss_kind,
                normalized_costs=normalized,
                better_mask=marked,
            )
            in_bounds = True
            if loss < best_loss:
                best_loss = float(loss)
                best_parameters = candidate.copy()
        if not np.isfinite(loss):
            raise ValueError("non_finite_q2f_final_objective")
        trace.append(
            FinalEvaluationRecord(
                evaluation_index=evaluations,
                parameters=tuple(map(float, candidate)),
                loss=float(loss),
                bsp=bsp,
                expected_normalized_cost=expected,
                p_feas=p_feas,
                in_bounds=in_bounds,
            )
        )
        return float(loss)

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
    if not trace or not np.isfinite(best_loss):
        raise RuntimeError("q2f_final_optimizer_produced_no_finite_in_bounds_evaluation")
    if scipy_result is not None:
        candidate = np.asarray(scipy_result.x, dtype=np.float64)
        candidate_in_bounds = bool(
            candidate.shape == initial.shape
            and np.all(candidate >= lower)
            and np.all(candidate <= upper)
        )
        if candidate_in_bounds and float(scipy_result.fun) < best_loss:
            best_loss = float(scipy_result.fun)
            best_parameters = candidate.copy()
    final_simulation = simulate_final_improvement(
        initial_state, phase_values, mixer, best_parameters, depth=p
    )
    statevector_evaluations += 1
    final_loss = evaluate_final_loss(
        final_simulation.probabilities,
        loss_kind=loss_kind,
        normalized_costs=normalized,
        better_mask=marked,
    )
    success = bool(scipy_result is not None and scipy_result.success and not exhausted)
    if exhausted or evaluations >= budget:
        reason = "evaluation_budget_exhausted"
    elif success:
        reason = "converged"
    else:
        reason = f"optimizer_failed:{getattr(scipy_result, 'message', 'unknown_failure')}"
    message = str(getattr(scipy_result, "message", reason))
    return (
        FinalOptimizationResult(
            initial_parameters=tuple(map(float, initial)),
            final_parameters=tuple(map(float, best_parameters)),
            initial_loss=float(trace[0].loss),
            final_loss=float(final_loss),
            optimizer_reported_loss=float(best_loss),
            evaluations=evaluations,
            statevector_evaluations=statevector_evaluations,
            success=success,
            reason=reason,
            message=message,
            runtime_seconds=float(runtime),
            retention_policy="best_finite_in_bounds_stationary_loss",
            evaluation_trace=tuple(trace),
        ),
        final_simulation,
    )


@dataclass(frozen=True)
class FinalImprovementMetrics:
    p_feas: float
    p_opt: float
    bsp: float
    expected_route_cost: float
    expected_normalized_cost: float
    top3_lowest_cost_mass: float
    top5_lowest_cost_mass: float
    probability_entropy: float
    optimal_state_rank: int
    most_probable_route_id: int
    most_probable_route: tuple[int, ...]
    most_probable_route_cost: int
    most_probable_route_probability: float
    initial_p_opt: float
    p_opt_amplification: float | None


def final_improvement_metrics(
    probabilities: Sequence[float],
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    better_mask: Sequence[bool],
    *,
    initial_p_opt: float,
) -> FinalImprovementMetrics:
    base: FeasibleMetrics = feasible_route_metrics(
        probabilities, basis, costs, initial_p_opt=float(initial_p_opt)
    )
    return FinalImprovementMetrics(
        p_feas=base.p_feas,
        p_opt=base.p_opt,
        bsp=better_solution_probability(probabilities, better_mask),
        expected_route_cost=base.expected_route_cost,
        expected_normalized_cost=base.expected_normalized_cost,
        top3_lowest_cost_mass=base.top3_lowest_cost_mass,
        top5_lowest_cost_mass=base.top5_lowest_cost_mass,
        probability_entropy=base.probability_entropy,
        optimal_state_rank=base.optimal_state_rank,
        most_probable_route_id=base.most_probable_route_id,
        most_probable_route=base.most_probable_route,
        most_probable_route_cost=base.most_probable_route_cost,
        most_probable_route_probability=base.most_probable_route_probability,
        initial_p_opt=base.initial_p_opt,
        p_opt_amplification=base.p_opt_amplification,
    )


@dataclass(frozen=True)
class FinalImprovementRun:
    run_id: str
    method: str
    depth: int
    seed: int
    evaluation_budget: int
    initialization_mode: str
    mixer_kind: str
    phase_kind: str
    loss_kind: str
    incumbent_raw_cost_threshold: float
    better_route_ids: tuple[int, ...]
    optimizer: FinalOptimizationResult
    final_probabilities: tuple[float, ...]
    final_layer_norms: tuple[LayerNormRecord, ...]
    metrics: FinalImprovementMetrics

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_final_improvement_cell(
    basis: FeasibleRouteBasis,
    costs: LogicalCostHamiltonian,
    path_exchange_mixer: LogicalPathExchangeMixer,
    grover_mixer: GroverFeasibleMixer,
    incumbent_biased_state: FeasibleInitialState,
    uniform_state: FeasibleInitialState,
    threshold: IncumbentThreshold,
    *,
    method: str,
    depth: int,
    seed: int,
    evaluation_budget: int = 150,
    rhobeg: float = 0.5,
    tolerance: float = 1e-8,
) -> FinalImprovementRun:
    normalized = np.asarray(costs.normalized_energies, dtype=np.float64)
    marked = np.asarray(threshold.better_mask, dtype=bool)
    if method == BSP_PATH_EXCHANGE:
        initial, mixer, phase, loss = (
            incumbent_biased_state,
            path_exchange_mixer,
            normalized,
            BSP_LOSS,
        )
        mixer_kind, phase_kind = PATH_EXCHANGE_KIND, COST_PHASE
    elif method == GM_QAOA_EXPECTATION:
        initial, mixer, phase, loss = uniform_state, grover_mixer, normalized, EXPECTATION_LOSS
        mixer_kind, phase_kind = GROVER_MIXER_KIND, COST_PHASE
    elif method == GM_TH_QAOA:
        initial, mixer, phase, loss = (
            uniform_state,
            grover_mixer,
            threshold_phase_values(costs.raw_energies, threshold.incumbent_raw_cost),
            BSP_LOSS,
        )
        mixer_kind, phase_kind = GROVER_MIXER_KIND, THRESHOLD_PHASE
    else:
        raise ValueError(f"unsupported_q2f_final_method:{method}")
    optimized, simulation = optimize_final_variant(
        initial.amplitudes,
        phase,
        mixer,
        normalized,
        marked,
        loss_kind=loss,
        depth=depth,
        seed=seed,
        evaluation_budget=evaluation_budget,
        rhobeg=rhobeg,
        tolerance=tolerance,
    )
    metrics = final_improvement_metrics(
        simulation.probabilities,
        basis,
        costs,
        marked,
        initial_p_opt=initial.p_opt,
    )
    if abs(metrics.p_feas - 1.0) > PROBABILITY_TOLERANCE:
        raise RuntimeError("q2f_final_cell_feasibility_invariant_failed")
    return FinalImprovementRun(
        run_id=f"{method}_p{int(depth)}_seed{int(seed)}",
        method=method,
        depth=int(depth),
        seed=int(seed),
        evaluation_budget=int(evaluation_budget),
        initialization_mode=initial.mode,
        mixer_kind=mixer_kind,
        phase_kind=phase_kind,
        loss_kind=loss,
        incumbent_raw_cost_threshold=threshold.incumbent_raw_cost,
        better_route_ids=threshold.better_route_ids,
        optimizer=optimized,
        final_probabilities=simulation.probabilities,
        final_layer_norms=simulation.layer_norms,
        metrics=metrics,
    )
