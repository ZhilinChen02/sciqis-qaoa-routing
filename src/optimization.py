"""Small reproducible COBYLA wrapper preserving the validated Q2 policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize


SOURCE_UNIFORM = "source_uniform"
SMALL_RANDOM = "small_random"
INITIALIZATION_STRATEGIES = (SOURCE_UNIFORM, SMALL_RANDOM)


class EvaluationBudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizationResult:
    parameters: tuple[float, ...]
    objective_value: float
    initial_objective: float
    evaluations: int
    success: bool
    reason: str
    wall_time: float


def seeded_initial_parameters(depth: int, seed: int) -> np.ndarray:
    """Draw gammas first, then betas, exactly as in the source Q2 code."""

    if int(depth) < 1:
        raise ValueError("depth_must_be_positive")
    rng = np.random.default_rng(int(seed))
    return np.concatenate(
        (
            rng.uniform(0.0, 2.0 * pi, size=int(depth)),
            rng.uniform(0.0, pi, size=int(depth)),
        )
    ).astype(np.float64)


def initial_parameters(
    depth: int,
    seed: int,
    *,
    strategy: str = SOURCE_UNIFORM,
    small_random_scale: float = 0.3,
) -> np.ndarray:
    """Return reproducible source-uniform or small-angle parameters."""

    if strategy == SOURCE_UNIFORM:
        return seeded_initial_parameters(depth, seed)
    if strategy != SMALL_RANDOM:
        raise ValueError(f"unsupported initialization strategy: {strategy}")
    depth = int(depth)
    scale = float(small_random_scale)
    if depth < 1 or not 0.0 < scale <= np.pi:
        raise ValueError("invalid small-random depth or scale")
    rng = np.random.default_rng(int(seed))
    return np.concatenate(
        (
            rng.uniform(0.0, scale, size=depth),
            rng.uniform(0.0, scale, size=depth),
        )
    ).astype(np.float64)


def optimize_cobyla(
    objective: Callable[[np.ndarray], float],
    *,
    depth: int,
    seed: int,
    evaluation_budget: int,
    gamma_bounds: Sequence[float] = (0.0, 2.0 * pi),
    beta_bounds: Sequence[float] = (0.0, pi),
    tolerance: float = 1e-8,
    rhobeg: float = 0.5,
    initialization_strategy: str = SOURCE_UNIFORM,
    small_random_scale: float = 0.3,
) -> OptimizationResult:
    """Run bounded COBYLA with a hard, counted objective-evaluation cap."""

    depth, budget = int(depth), int(evaluation_budget)
    if depth < 1 or budget < 1:
        raise ValueError("invalid_optimizer_depth_or_budget")
    lower = np.asarray([gamma_bounds[0]] * depth + [beta_bounds[0]] * depth, dtype=float)
    upper = np.asarray([gamma_bounds[1]] * depth + [beta_bounds[1]] * depth, dtype=float)
    initial = initial_parameters(
        depth,
        seed,
        strategy=initialization_strategy,
        small_random_scale=small_random_scale,
    )
    evaluations = 0
    best_value = float("inf")
    best_parameters = initial.copy()
    initial_value: float | None = None

    def counted(parameters: np.ndarray) -> float:
        nonlocal evaluations, best_value, best_parameters, initial_value
        if evaluations >= budget:
            raise EvaluationBudgetExhausted
        evaluations += 1
        candidate = np.asarray(parameters, dtype=np.float64)
        outside = float(np.sum(np.maximum(lower - candidate, 0.0) + np.maximum(candidate - upper, 0.0)))
        value = 1_000_000.0 + outside if outside else float(objective(candidate))
        if not np.isfinite(value):
            raise ValueError("non_finite_optimizer_objective")
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
