"""Archived literature-guided revision of the aligned-mixer Q2 path.

Q2-R keeps the checked-in incumbent-product state, matched incumbent-dependent
mixer, cost Hamiltonian, decoder, and final metrics.  It adds independently
switchable training objectives and an observable incremental-depth controller.
The optimization controller is deliberately separated from exact-optimum
evaluation so no oracle quantity can enter a continuation decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import pi
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

from graph import path_cost
from metrics import distribution_metrics
from qaoa import (
    CVAR,
    EXPECTATION,
    OBJECTIVE_MODES,
    evaluate_optimization_objective,
    probability_weighted_expectation,
)
from qaoa import (
    EvaluationBudgetExhausted,
    INITIALIZATION_STRATEGIES,
    OptimizationResult,
    SOURCE_UNIFORM,
    initial_parameters,
)
from qaoa import (
    Q2_WARM_START,
    build_qaoa_circuit,
    circuit_statistics,
    simulate_qaoa_state,
    state_probabilities,
)
from support.warm_start import greedy_incumbent_route, incumbent_relaxation, product_state


METHOD_REVISION = "Q2-R literature-guided v1"
WARM_START_MODE = "incumbent_product_aligned_existing"
MIXER_MODE = "incumbent_dependent_aligned_existing"
MIXER_SCALE_MODE = "inverse_num_qubits_existing"
FIXED_DEPTH = "fixed"
INCREMENTAL_DEPTH = "incremental"
DEPTH_MODES = (FIXED_DEPTH, INCREMENTAL_DEPTH)
OBJECTIVE_IMPROVEMENT_TRIGGER = "objective_improvement"
NEUTRAL_TRANSFER = "neutral_zero_append_grouped"
NEUTRAL_TRANSFER_EQUIVALENCE_TOLERANCE = 1e-12
BASELINE_OPTIMIZER_PROGRESS = "baseline_optimizer_progress"
MARGINAL_DEPTH_GAIN = "marginal_depth_gain"


@dataclass(frozen=True)
class DepthTriggerConfig:
    """Frozen trigger for named baseline progress or marginal depth gain."""

    mode: str = OBJECTIVE_IMPROVEMENT_TRIGGER
    absolute_improvement_tolerance: float = 1e-6
    relative_improvement_tolerance: float = 0.0
    relative_scale_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.mode != OBJECTIVE_IMPROVEMENT_TRIGGER:
            raise ValueError(f"unsupported depth trigger: {self.mode}")
        if not np.isfinite(self.absolute_improvement_tolerance) or self.absolute_improvement_tolerance < 0:
            raise ValueError("absolute improvement tolerance must be finite and non-negative")
        if not np.isfinite(self.relative_improvement_tolerance) or self.relative_improvement_tolerance < 0:
            raise ValueError("relative improvement tolerance must be finite and non-negative")
        if not np.isfinite(self.relative_scale_floor) or self.relative_scale_floor <= 0:
            raise ValueError("relative scale floor must be finite and positive")


@dataclass(frozen=True)
class Q2RevisionConfig:
    """Explicit A0--A3 configuration; defaults reproduce fixed-depth Q2 policy."""

    objective_mode: str = EXPECTATION
    cvar_alpha: float | None = None
    depth_mode: str = FIXED_DEPTH
    initial_depth: int = 1
    max_depth: int = 1
    depth_trigger: DepthTriggerConfig = field(default_factory=DepthTriggerConfig)
    transfer_mode: str = NEUTRAL_TRANSFER
    warm_start_mode: str = WARM_START_MODE
    mixer_mode: str = MIXER_MODE
    mixer_scale_mode: str = MIXER_SCALE_MODE
    per_depth_budget: int = 100
    total_cumulative_budget: int | None = 100
    seed: int = 2601
    optimizer_tolerance: float = 1e-8
    optimizer_rhobeg: float = 0.5
    initialization_strategy: str = SOURCE_UNIFORM

    def __post_init__(self) -> None:
        if self.objective_mode not in OBJECTIVE_MODES:
            raise ValueError(f"unsupported objective mode: {self.objective_mode}")
        if self.objective_mode == EXPECTATION and self.cvar_alpha is not None:
            raise ValueError("expectation mode requires cvar_alpha=None")
        if self.objective_mode == CVAR:
            if self.cvar_alpha is None or not np.isfinite(self.cvar_alpha):
                raise ValueError("cvar mode requires a finite cvar_alpha")
            if not 0.0 < float(self.cvar_alpha) <= 1.0:
                raise ValueError("cvar_alpha must satisfy 0 < alpha <= 1")
        if self.depth_mode not in DEPTH_MODES:
            raise ValueError(f"unsupported depth mode: {self.depth_mode}")
        if int(self.initial_depth) != self.initial_depth or int(self.max_depth) != self.max_depth:
            raise ValueError("depths must be integers")
        if self.initial_depth < 1 or self.max_depth < self.initial_depth:
            raise ValueError("depths must satisfy 1 <= initial_depth <= max_depth")
        if self.depth_mode == FIXED_DEPTH and self.max_depth != self.initial_depth:
            raise ValueError("fixed depth requires max_depth == initial_depth")
        if self.transfer_mode != NEUTRAL_TRANSFER:
            raise ValueError(f"unsupported transfer mode: {self.transfer_mode}")
        if self.warm_start_mode != WARM_START_MODE:
            raise ValueError(f"unsupported warm-start mode: {self.warm_start_mode}")
        if self.mixer_mode != MIXER_MODE:
            raise ValueError(f"unsupported mixer mode: {self.mixer_mode}")
        if self.mixer_scale_mode != MIXER_SCALE_MODE:
            raise ValueError(f"unsupported mixer scale mode: {self.mixer_scale_mode}")
        if int(self.per_depth_budget) != self.per_depth_budget or self.per_depth_budget < 1:
            raise ValueError("per_depth_budget must be a positive integer")
        if self.total_cumulative_budget is not None:
            if (
                int(self.total_cumulative_budget) != self.total_cumulative_budget
                or self.total_cumulative_budget < 1
            ):
                raise ValueError("total_cumulative_budget must be a positive integer or None")
        if self.initialization_strategy not in INITIALIZATION_STRATEGIES:
            raise ValueError(
                f"unsupported initialization strategy: {self.initialization_strategy}"
            )
        if not np.isfinite(self.optimizer_tolerance) or self.optimizer_tolerance <= 0:
            raise ValueError("optimizer_tolerance must be finite and positive")
        if not np.isfinite(self.optimizer_rhobeg) or self.optimizer_rhobeg <= 0:
            raise ValueError("optimizer_rhobeg must be finite and positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Q2RevisionConfig":
        values = dict(payload)
        trigger = values.get("depth_trigger")
        if isinstance(trigger, Mapping):
            values["depth_trigger"] = DepthTriggerConfig(**dict(trigger))
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_total_budget(self) -> int:
        if self.total_cumulative_budget is not None:
            return int(self.total_cumulative_budget)
        attempted_depths = 1 if self.depth_mode == FIXED_DEPTH else self.max_depth - self.initial_depth + 1
        return int(self.per_depth_budget * attempted_depths)


@dataclass(frozen=True)
class DepthTriggerDecision:
    gain_semantics: str
    trigger_gain_absolute: float
    trigger_gain_relative: float
    continuation_decision: bool
    continuation_reason: str


@dataclass(frozen=True)
class ObjectiveEvaluationRecord:
    """One budget-counted optimizer objective request."""

    depth_evaluation_index: int
    cumulative_evaluation_index: int
    evaluation_role: str
    parameters: tuple[float, ...]
    objective_value: float
    in_bounds: bool


@dataclass(frozen=True)
class DepthHistoryEntry:
    depth_started: int
    depth_completed: int
    input_parameters: tuple[float, ...]
    output_parameters: tuple[float, ...]
    objective_before: float
    objective_after: float
    optimizer_progress_absolute: float
    optimizer_progress_relative: float
    inherited_parent_depth: int | None
    inherited_parent_objective: float | None
    neutral_transfer_objective: float | None
    neutral_transfer_equivalence_error: float | None
    marginal_depth_gain_absolute: float | None
    marginal_depth_gain_relative: float | None
    expectation_value: float
    raw_expectation_value: float
    objective_evaluations: int
    objective_trace: tuple[ObjectiveEvaluationRecord, ...]
    cumulative_objective_evaluations: int
    statevector_evaluations: int
    cumulative_statevector_evaluations: int
    optimizer_runtime: float
    cumulative_optimizer_runtime: float
    optimizer_success: bool
    optimizer_reason: str
    continuation_decision: bool
    continuation_reason: str


@dataclass(frozen=True)
class Q2OptimizationTrace:
    final_parameters: tuple[float, ...]
    final_probabilities: tuple[float, ...]
    final_objective_value: float
    final_expectation_value: float
    final_raw_expectation_value: float
    realized_depth: int
    cumulative_objective_evaluations: int
    cumulative_statevector_evaluations: int
    cumulative_runtime: float
    overall_runtime: float
    stop_reason: str
    depth_history: tuple[DepthHistoryEntry, ...]


@dataclass(frozen=True)
class Q2RevisionResult:
    method_revision: str
    task_identity: str
    warm_start_mode: str
    mixer_mode: str
    mixer_scale_mode: str
    warm_start_epsilon: float
    optimization_objective_mode: str
    cvar_alpha: float | None
    optimized_objective_value: float
    final_expectation_value: float
    final_expected_energy: float
    depth_mode: str
    initial_depth: int
    maximum_allowed_depth: int
    realized_depth: int
    depth_trigger: DepthTriggerConfig
    transfer_mode: str
    cumulative_objective_evaluations: int
    cumulative_statevector_evaluations: int
    cumulative_runtime: float
    overall_runtime: float
    stop_reason: str
    initial_parameters: tuple[float, ...]
    final_parameters: tuple[float, ...]
    depth_history: tuple[DepthHistoryEntry, ...]
    p_feas: float
    p_opt: float
    invalid_probability_mass: float
    feasible_conditional_cost: float | None
    incumbent_route: tuple[int, ...]
    incumbent_cost: int
    initial_incumbent_probability: float
    incumbent_probability: float
    incumbent_amplification: float | None
    initial_optimal_probability: float
    optimal_state_probability: float
    optimum_amplification: float | None
    best_decoded_route: tuple[int, ...] | None
    best_decoded_cost: int | None
    best_state_probability: float
    best_state_valid: bool
    best_state_optimal: bool
    optimal_state_rank: int
    probability_sum: float
    circuit_depth: int
    total_gate_count: int
    two_qubit_gate_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Q2ParameterEvaluation:
    """Optimizer-independent evaluation of one existing-Q2 parameter vector."""

    initial_state: tuple[complex, ...]
    statevector: tuple[complex, ...]
    probabilities: tuple[float, ...]
    optimization_objective_value: float
    expectation_value: float
    raw_expectation_value: float


def ablation_configurations(
    *,
    cvar_alpha: float,
    fixed_depth: int = 1,
    incremental_max_depth: int = 3,
    per_depth_budget: int = 100,
    total_cumulative_budget: int | None = 300,
    seed: int = 2601,
    depth_trigger: DepthTriggerConfig | None = None,
) -> dict[str, Q2RevisionConfig]:
    """Return A0--A3 while holding every non-objective/depth setting fixed."""

    trigger = depth_trigger or DepthTriggerConfig()
    common: dict[str, Any] = {
        "initial_depth": int(fixed_depth),
        "depth_trigger": trigger,
        "transfer_mode": NEUTRAL_TRANSFER,
        "warm_start_mode": WARM_START_MODE,
        "mixer_mode": MIXER_MODE,
        "mixer_scale_mode": MIXER_SCALE_MODE,
        "per_depth_budget": int(per_depth_budget),
        "total_cumulative_budget": total_cumulative_budget,
        "seed": int(seed),
    }
    fixed = {**common, "depth_mode": FIXED_DEPTH, "max_depth": int(fixed_depth)}
    incremental = {
        **common,
        "depth_mode": INCREMENTAL_DEPTH,
        "max_depth": int(incremental_max_depth),
    }
    return {
        "A0": Q2RevisionConfig(objective_mode=EXPECTATION, cvar_alpha=None, **fixed),
        "A1": Q2RevisionConfig(objective_mode=CVAR, cvar_alpha=cvar_alpha, **fixed),
        "A2": Q2RevisionConfig(objective_mode=EXPECTATION, cvar_alpha=None, **incremental),
        "A3": Q2RevisionConfig(objective_mode=CVAR, cvar_alpha=cvar_alpha, **incremental),
    }


def budget_matched_fixed_depth_config(
    incremental_config: Q2RevisionConfig,
    target_depth: int,
) -> Q2RevisionConfig:
    """Build a fixed-depth control with the incremental cumulative budget.

    With an incremental schedule starting at ``p0`` and per-depth cap ``B``, a
    fixed control at depth ``p`` receives ``(p-p0+1)B`` evaluations, capped by
    the incremental run's total cumulative budget.  The returned configuration
    performs one fixed-depth optimization with that full matched cap.
    """

    depth = int(target_depth)
    if incremental_config.depth_mode != INCREMENTAL_DEPTH:
        raise ValueError("budget matching requires an incremental reference config")
    if depth < incremental_config.initial_depth or depth > incremental_config.max_depth:
        raise ValueError("target depth must lie inside the incremental depth range")
    attempted_depths = depth - incremental_config.initial_depth + 1
    matched_budget = min(
        incremental_config.effective_total_budget,
        attempted_depths * int(incremental_config.per_depth_budget),
    )
    return replace(
        incremental_config,
        depth_mode=FIXED_DEPTH,
        initial_depth=depth,
        max_depth=depth,
        per_depth_budget=matched_budget,
        total_cumulative_budget=matched_budget,
    )


def transfer_neutral_parameters(parameters: Sequence[float], depth: int) -> np.ndarray:
    """Append zero gamma/beta under the existing grouped parameter convention.

    Existing Q2 stores ``[gamma_1,...,gamma_p,beta_1,...,beta_p]``.  Therefore
    the neutral transfer is ``[old gammas,0,old betas,0]``.  Both zero angles
    make the appended cost-then-mixer layer exactly the identity.
    """

    old_depth = int(depth)
    values = np.asarray(parameters, dtype=np.float64)
    if old_depth < 1 or values.shape != (2 * old_depth,) or np.any(~np.isfinite(values)):
        raise ValueError("qaoa_parameter_count_mismatch")
    return np.concatenate(
        (values[:old_depth], np.zeros(1), values[old_depth:], np.zeros(1))
    ).astype(np.float64)


def evaluate_q2_parameters(
    normalized_cost_diagonal: Sequence[float],
    raw_cost_diagonal: Sequence[float],
    warm_start_values: Sequence[float],
    parameters: Sequence[float],
    *,
    depth: int,
    config: Q2RevisionConfig,
) -> Q2ParameterEvaluation:
    """Evaluate Q2-R directly through the unchanged historical Q2 primitives."""

    normalized = np.asarray(normalized_cost_diagonal, dtype=np.float64)
    raw = np.asarray(raw_cost_diagonal, dtype=np.float64)
    warm = tuple(map(float, warm_start_values))
    values = np.asarray(parameters, dtype=np.float64)
    if normalized.ndim != 1 or raw.shape != normalized.shape or len(normalized) == 0:
        raise ValueError("normalized and raw diagonals must be equal non-empty vectors")
    if np.any(~np.isfinite(normalized)) or np.any(~np.isfinite(raw)):
        raise ValueError("cost diagonals must be finite")
    if len(normalized) & (len(normalized) - 1):
        raise ValueError("cost diagonal dimension must be a power of two")
    if len(warm) != len(normalized).bit_length() - 1:
        raise ValueError("Q2 requires one warm-start value per qubit")
    if values.shape != (2 * int(depth),):
        raise ValueError("qaoa_parameter_count_mismatch")
    initial_state = product_state(warm)
    state = simulate_qaoa_state(
        normalized,
        values,
        depth=int(depth),
        solver=Q2_WARM_START,
        warm_start_values=warm,
    )
    probabilities = state_probabilities(state)
    return Q2ParameterEvaluation(
        initial_state=tuple(map(complex, initial_state)),
        statevector=tuple(map(complex, state)),
        probabilities=tuple(map(float, probabilities)),
        optimization_objective_value=evaluate_optimization_objective(
            probabilities,
            normalized,
            config.objective_mode,
            config.cvar_alpha,
        ),
        expectation_value=probability_weighted_expectation(probabilities, normalized),
        raw_expectation_value=probability_weighted_expectation(probabilities, raw),
    )


def decide_depth_continuation(
    *,
    depth_mode: str,
    current_depth: int,
    maximum_depth: int,
    objective_before: float,
    objective_after: float,
    gain_semantics: str,
    cumulative_objective_evaluations: int,
    total_cumulative_budget: int,
    trigger: DepthTriggerConfig,
) -> DepthTriggerDecision:
    """Make an oracle-free decision from a named gain and resource limits."""

    if depth_mode not in DEPTH_MODES:
        raise ValueError(f"unsupported depth mode: {depth_mode}")
    if gain_semantics not in (BASELINE_OPTIMIZER_PROGRESS, MARGINAL_DEPTH_GAIN):
        raise ValueError(f"unsupported continuation gain semantics: {gain_semantics}")
    before, after = float(objective_before), float(objective_after)
    if not np.isfinite(before) or not np.isfinite(after):
        raise ValueError("depth trigger objectives must be finite")
    improvement = before - after
    relative = improvement / max(abs(before), trigger.relative_scale_floor)
    if depth_mode == FIXED_DEPTH:
        return DepthTriggerDecision(
            gain_semantics, improvement, relative, False, "fixed_depth_complete"
        )
    if int(current_depth) >= int(maximum_depth):
        return DepthTriggerDecision(
            gain_semantics, improvement, relative, False, "maximum_depth_reached"
        )
    if int(cumulative_objective_evaluations) >= int(total_cumulative_budget):
        return DepthTriggerDecision(
            gain_semantics,
            improvement,
            relative,
            False,
            "cumulative_evaluation_budget_exhausted",
        )
    if gain_semantics == BASELINE_OPTIMIZER_PROGRESS:
        return DepthTriggerDecision(
            gain_semantics,
            improvement,
            relative,
            True,
            "baseline_complete_required_first_transfer",
        )
    should_continue = bool(
        improvement > trigger.absolute_improvement_tolerance
        and relative > trigger.relative_improvement_tolerance
    )
    return DepthTriggerDecision(
        gain_semantics,
        improvement,
        relative,
        should_continue,
        (
            f"{gain_semantics}_exceeds_threshold"
            if should_continue
            else f"{gain_semantics}_plateau"
        ),
    )


def _optimize_cobyla_from_parameters(
    objective: Callable[[np.ndarray], float],
    initial: Sequence[float],
    *,
    depth: int,
    evaluation_budget: int,
    tolerance: float,
    rhobeg: float,
    gamma_bounds: Sequence[float] = (0.0, 2.0 * pi),
    beta_bounds: Sequence[float] = (0.0, pi),
    evaluation_observer: Callable[
        [int, tuple[float, ...], float, bool], None
    ]
    | None = None,
) -> OptimizationResult:
    """Revision-only COBYLA entry point accepting transferred parameters."""

    depth, budget = int(depth), int(evaluation_budget)
    values = np.asarray(initial, dtype=np.float64)
    if depth < 1 or budget < 1 or values.shape != (2 * depth,):
        raise ValueError("invalid_optimizer_depth_budget_or_initial_parameters")
    lower = np.asarray([gamma_bounds[0]] * depth + [beta_bounds[0]] * depth, dtype=float)
    upper = np.asarray([gamma_bounds[1]] * depth + [beta_bounds[1]] * depth, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < lower) or np.any(values > upper):
        raise ValueError("initial parameters must be finite and inside optimizer bounds")

    evaluations = 0
    best_value = float("inf")
    best_parameters = values.copy()
    initial_value: float | None = None

    def counted(parameters: np.ndarray) -> float:
        nonlocal evaluations, best_value, best_parameters, initial_value
        if evaluations >= budget:
            raise EvaluationBudgetExhausted
        evaluations += 1
        candidate = np.asarray(parameters, dtype=np.float64)
        outside = float(
            np.sum(np.maximum(lower - candidate, 0.0) + np.maximum(candidate - upper, 0.0))
        )
        value = 1_000_000.0 + outside if outside else float(objective(candidate))
        if not np.isfinite(value):
            raise ValueError("non_finite_optimizer_objective")
        if evaluation_observer is not None:
            evaluation_observer(
                evaluations,
                tuple(map(float, candidate)),
                float(value),
                not bool(outside),
            )
        if initial_value is None:
            initial_value = value
        if not outside and value < best_value:
            best_value = value
            best_parameters = candidate.copy()
        return value

    started = perf_counter()
    scipy_result = None
    exhausted = False
    try:
        scipy_result = minimize(
            counted,
            values,
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
    wall_time = perf_counter() - started
    if evaluations == 0 or initial_value is None:
        raise RuntimeError("optimizer_performed_no_evaluations")
    if scipy_result is not None:
        candidate = np.asarray(scipy_result.x, dtype=np.float64)
        in_bounds = bool(np.all(candidate >= lower) and np.all(candidate <= upper))
        if in_bounds and float(scipy_result.fun) < best_value:
            best_value = float(scipy_result.fun)
            best_parameters = candidate
    success = bool(scipy_result is not None and scipy_result.success and not exhausted)
    if exhausted or evaluations >= budget:
        reason = "evaluation_budget_exhausted"
    elif success:
        reason = "converged"
    else:
        reason = f"optimizer_failed:{getattr(scipy_result, 'message', 'unknown_failure')}"
    return OptimizationResult(
        parameters=tuple(map(float, best_parameters)),
        objective_value=float(best_value),
        initial_objective=float(initial_value),
        evaluations=evaluations,
        success=success,
        reason=reason,
        wall_time=float(wall_time),
    )


def run_q2_optimization(
    normalized_cost_diagonal: Sequence[float],
    raw_cost_diagonal: Sequence[float],
    warm_start_values: Sequence[float],
    config: Q2RevisionConfig,
) -> Q2OptimizationTrace:
    """Optimize Q2-R without accepting an exact optimum or final route label."""

    normalized = np.asarray(normalized_cost_diagonal, dtype=np.float64)
    raw = np.asarray(raw_cost_diagonal, dtype=np.float64)
    warm = tuple(map(float, warm_start_values))
    if normalized.ndim != 1 or raw.shape != normalized.shape or len(normalized) == 0:
        raise ValueError("normalized and raw diagonals must be equal non-empty vectors")
    if np.any(~np.isfinite(normalized)) or np.any(~np.isfinite(raw)):
        raise ValueError("cost diagonals must be finite")
    if len(normalized) & (len(normalized) - 1):
        raise ValueError("cost diagonal dimension must be a power of two")
    if len(warm) != len(normalized).bit_length() - 1:
        raise ValueError("Q2 requires one warm-start value per qubit")
    # Validate the unchanged aligned-state input before optimization begins.
    product_state(warm)

    started = perf_counter()
    total_budget = config.effective_total_budget
    current_depth = int(config.initial_depth)
    current_parameters = initial_parameters(
        current_depth,
        config.seed,
        strategy=config.initialization_strategy,
    )
    cumulative_evaluations = 0
    cumulative_statevectors = 0
    cumulative_runtime = 0.0
    history: list[DepthHistoryEntry] = []
    final_probabilities: np.ndarray | None = None

    while True:
        remaining_budget = total_budget - cumulative_evaluations
        if remaining_budget < 1:
            raise RuntimeError("no cumulative evaluation budget remained for an attempted depth")
        depth_budget = min(int(config.per_depth_budget), remaining_budget)
        statevector_calls = 0

        def simulate(parameters: np.ndarray) -> np.ndarray:
            nonlocal statevector_calls
            statevector_calls += 1
            return simulate_qaoa_state(
                normalized,
                parameters,
                depth=current_depth,
                solver=Q2_WARM_START,
                warm_start_values=warm,
            )

        input_parameters = tuple(map(float, current_parameters))
        input_probabilities = state_probabilities(simulate(current_parameters))
        input_objective = evaluate_optimization_objective(
            input_probabilities,
            normalized,
            config.objective_mode,
            config.cvar_alpha,
        )
        cached_initial_available = True
        objective_trace: list[ObjectiveEvaluationRecord] = []

        def objective(parameters: np.ndarray) -> float:
            nonlocal cached_initial_available
            candidate = np.asarray(parameters, dtype=np.float64)
            if cached_initial_available:
                if not np.array_equal(candidate, current_parameters):
                    raise RuntimeError(
                        "COBYLA did not request the declared initial parameters first"
                    )
                cached_initial_available = False
                return input_objective
            probabilities = state_probabilities(simulate(candidate))
            return evaluate_optimization_objective(
                probabilities,
                normalized,
                config.objective_mode,
                config.cvar_alpha,
            )

        def observe_objective(
            depth_index: int,
            parameters: tuple[float, ...],
            value: float,
            in_bounds: bool,
        ) -> None:
            objective_trace.append(
                ObjectiveEvaluationRecord(
                    depth_evaluation_index=depth_index,
                    cumulative_evaluation_index=cumulative_evaluations + depth_index,
                    evaluation_role=(
                        "initial_diagnostic_and_optimizer_initial"
                        if depth_index == 1 and not history
                        else "neutral_transfer_diagnostic_and_optimizer_initial"
                        if depth_index == 1
                        else "optimizer"
                    ),
                    parameters=parameters,
                    objective_value=value,
                    in_bounds=in_bounds,
                )
            )

        optimized = _optimize_cobyla_from_parameters(
            objective,
            current_parameters,
            depth=current_depth,
            evaluation_budget=depth_budget,
            tolerance=config.optimizer_tolerance,
            rhobeg=config.optimizer_rhobeg,
            evaluation_observer=observe_objective,
        )
        final_state = simulate_qaoa_state(
            normalized,
            optimized.parameters,
            depth=current_depth,
            solver=Q2_WARM_START,
            warm_start_values=warm,
        )
        statevector_calls += 1
        final_probabilities = state_probabilities(final_state)
        final_objective = optimized.objective_value
        final_expectation = probability_weighted_expectation(
            final_probabilities, normalized
        )
        final_raw_expectation = probability_weighted_expectation(
            final_probabilities, raw
        )
        if abs(optimized.initial_objective - input_objective) > 1e-12:
            raise RuntimeError("optimizer initial objective changed before optimization")
        if len(objective_trace) != optimized.evaluations:
            raise RuntimeError("objective trace does not match counted evaluations")

        cumulative_evaluations += optimized.evaluations
        cumulative_statevectors += statevector_calls
        cumulative_runtime += optimized.wall_time
        optimizer_progress_absolute = input_objective - final_objective
        optimizer_progress_relative = optimizer_progress_absolute / max(
            abs(input_objective), config.depth_trigger.relative_scale_floor
        )
        parent = history[-1] if history else None
        if parent is None:
            inherited_parent_depth = None
            inherited_parent_objective = None
            neutral_transfer_objective = None
            neutral_transfer_equivalence_error = None
            marginal_depth_gain_absolute = None
            marginal_depth_gain_relative = None
            gain_semantics = BASELINE_OPTIMIZER_PROGRESS
            comparison_objective = input_objective
        else:
            inherited_parent_depth = parent.depth_completed
            inherited_parent_objective = parent.objective_after
            neutral_transfer_objective = input_objective
            neutral_transfer_equivalence_error = abs(
                neutral_transfer_objective - inherited_parent_objective
            )
            if (
                neutral_transfer_equivalence_error
                > NEUTRAL_TRANSFER_EQUIVALENCE_TOLERANCE
            ):
                raise RuntimeError(
                    "neutral transfer changed the inherited parent objective: "
                    f"error={neutral_transfer_equivalence_error}"
                )
            marginal_depth_gain_absolute = (
                inherited_parent_objective - final_objective
            )
            marginal_depth_gain_relative = marginal_depth_gain_absolute / max(
                abs(inherited_parent_objective),
                config.depth_trigger.relative_scale_floor,
            )
            gain_semantics = MARGINAL_DEPTH_GAIN
            comparison_objective = inherited_parent_objective
        decision = decide_depth_continuation(
            depth_mode=config.depth_mode,
            current_depth=current_depth,
            maximum_depth=config.max_depth,
            objective_before=comparison_objective,
            objective_after=final_objective,
            gain_semantics=gain_semantics,
            cumulative_objective_evaluations=cumulative_evaluations,
            total_cumulative_budget=total_budget,
            trigger=config.depth_trigger,
        )
        history.append(
            DepthHistoryEntry(
                depth_started=current_depth,
                depth_completed=current_depth,
                input_parameters=input_parameters,
                output_parameters=optimized.parameters,
                objective_before=input_objective,
                objective_after=final_objective,
                optimizer_progress_absolute=optimizer_progress_absolute,
                optimizer_progress_relative=optimizer_progress_relative,
                inherited_parent_depth=inherited_parent_depth,
                inherited_parent_objective=inherited_parent_objective,
                neutral_transfer_objective=neutral_transfer_objective,
                neutral_transfer_equivalence_error=neutral_transfer_equivalence_error,
                marginal_depth_gain_absolute=marginal_depth_gain_absolute,
                marginal_depth_gain_relative=marginal_depth_gain_relative,
                expectation_value=final_expectation,
                raw_expectation_value=final_raw_expectation,
                objective_evaluations=optimized.evaluations,
                objective_trace=tuple(objective_trace),
                cumulative_objective_evaluations=cumulative_evaluations,
                statevector_evaluations=statevector_calls,
                cumulative_statevector_evaluations=cumulative_statevectors,
                optimizer_runtime=optimized.wall_time,
                cumulative_optimizer_runtime=cumulative_runtime,
                optimizer_success=optimized.success,
                optimizer_reason=optimized.reason,
                continuation_decision=decision.continuation_decision,
                continuation_reason=decision.continuation_reason,
            )
        )
        if not decision.continuation_decision:
            break
        current_parameters = transfer_neutral_parameters(
            optimized.parameters, current_depth
        )
        current_depth += 1

    if final_probabilities is None:
        raise RuntimeError("Q2 revision completed without a final distribution")
    return Q2OptimizationTrace(
        final_parameters=history[-1].output_parameters,
        final_probabilities=tuple(map(float, final_probabilities)),
        final_objective_value=history[-1].objective_after,
        final_expectation_value=history[-1].expectation_value,
        final_raw_expectation_value=history[-1].raw_expectation_value,
        realized_depth=history[-1].depth_completed,
        cumulative_objective_evaluations=cumulative_evaluations,
        cumulative_statevector_evaluations=cumulative_statevectors,
        cumulative_runtime=cumulative_runtime,
        overall_runtime=perf_counter() - started,
        stop_reason=history[-1].continuation_reason,
        depth_history=tuple(history),
    )


def _safe_amplification(final: float, initial: float) -> float | None:
    return None if initial <= 0.0 else float(final / initial)


def run_q2_revision(
    context: Any,
    *,
    epsilon: float,
    config: Q2RevisionConfig,
) -> Q2RevisionResult:
    """Run Q2-R, then attach exact-oracle metrics only after optimization stops."""

    warm_rows = incumbent_relaxation(context.graph, epsilon)
    warm_values = tuple(row.clipped_value for row in warm_rows)
    initial_probabilities = state_probabilities(product_state(warm_values))

    # This call has no access to context, route labels, p_opt, or exact cost.
    optimized = run_q2_optimization(
        context.normalized_diagonal,
        context.raw_diagonal,
        warm_values,
        config,
    )

    final_probabilities = np.asarray(optimized.final_probabilities, dtype=np.float64)
    evaluated = distribution_metrics(
        final_probabilities,
        context.states,
        optimal_cost=context.optimal_cost,
    )
    incumbent_route = greedy_incumbent_route(context.graph)
    incumbent_cost = path_cost(context.graph, incumbent_route)
    initial_incumbent = float(initial_probabilities[context.incumbent_state_index])
    final_incumbent = float(final_probabilities[context.incumbent_state_index])
    initial_optimal = float(initial_probabilities[context.optimal_state_index])
    final_optimal = float(final_probabilities[context.optimal_state_index])

    circuit = build_qaoa_circuit(
        context.ising,
        optimized.final_parameters,
        depth=optimized.realized_depth,
        solver=Q2_WARM_START,
        normalization_scale=context.normalization_scale,
        warm_start_values=warm_values,
    )
    resources = circuit_statistics(circuit)
    graph = context.graph
    task_identity = (
        f"{graph.graph.get('schema', 'routing-graph')}:"
        f"{graph.graph.get('name', 'unnamed')}:"
        f"{graph.number_of_nodes()}n-{graph.number_of_edges()}e"
    )
    return Q2RevisionResult(
        method_revision=METHOD_REVISION,
        task_identity=task_identity,
        warm_start_mode=config.warm_start_mode,
        mixer_mode=config.mixer_mode,
        mixer_scale_mode=config.mixer_scale_mode,
        warm_start_epsilon=float(epsilon),
        optimization_objective_mode=config.objective_mode,
        cvar_alpha=config.cvar_alpha,
        optimized_objective_value=optimized.final_objective_value,
        final_expectation_value=optimized.final_expectation_value,
        final_expected_energy=optimized.final_raw_expectation_value,
        depth_mode=config.depth_mode,
        initial_depth=config.initial_depth,
        maximum_allowed_depth=config.max_depth,
        realized_depth=optimized.realized_depth,
        depth_trigger=config.depth_trigger,
        transfer_mode=config.transfer_mode,
        cumulative_objective_evaluations=optimized.cumulative_objective_evaluations,
        cumulative_statevector_evaluations=optimized.cumulative_statevector_evaluations,
        cumulative_runtime=optimized.cumulative_runtime,
        overall_runtime=optimized.overall_runtime,
        stop_reason=optimized.stop_reason,
        initial_parameters=optimized.depth_history[0].input_parameters,
        final_parameters=optimized.final_parameters,
        depth_history=optimized.depth_history,
        p_feas=evaluated.p_feas,
        p_opt=evaluated.p_opt,
        invalid_probability_mass=max(0.0, 1.0 - evaluated.p_feas),
        feasible_conditional_cost=evaluated.feasible_conditional_cost,
        incumbent_route=incumbent_route,
        incumbent_cost=incumbent_cost,
        initial_incumbent_probability=initial_incumbent,
        incumbent_probability=final_incumbent,
        incumbent_amplification=_safe_amplification(final_incumbent, initial_incumbent),
        initial_optimal_probability=initial_optimal,
        optimal_state_probability=final_optimal,
        optimum_amplification=_safe_amplification(final_optimal, initial_optimal),
        best_decoded_route=evaluated.best_decoded_route,
        best_decoded_cost=evaluated.best_decoded_cost,
        best_state_probability=evaluated.best_state_probability,
        best_state_valid=evaluated.best_state_valid,
        best_state_optimal=evaluated.best_state_optimal,
        optimal_state_rank=evaluated.optimal_state_rank,
        probability_sum=float(np.sum(final_probabilities)),
        circuit_depth=resources.circuit_depth,
        total_gate_count=resources.total_gate_count,
        two_qubit_gate_count=resources.two_qubit_gate_count,
    )
