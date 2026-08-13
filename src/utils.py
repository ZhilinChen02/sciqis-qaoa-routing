"""Small calculations shared by the QAOA scripts."""

from dataclasses import dataclass
from math import pi
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, minimize

from support.metrics import (  # noqa: F401 - public helpers collected here
    DistributionMetrics,
    distribution_metrics,
    top_state_rows,
)
from support.warm_start import (  # noqa: F401 - public helpers collected here
    WarmStartValue,
    apply_single_qubit_hamiltonian_rotation,
    apply_warm_start_mixer,
    clip_relaxed_values,
    greedy_incumbent_route,
    incumbent_bitstring,
    incumbent_product_warm_start_state,
    incumbent_relaxation,
    mixer_hamiltonian,
    product_state,
    warm_start_single_qubit_mixer,
)

EXPECTATION = "expectation"
CVAR = "cvar"
OBJECTIVE_MODES = (EXPECTATION, CVAR)
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


def seeded_initial_parameters(depth, seed):
    """Pick reproducible starting angles for QAOA."""

    depth = int(depth)
    if depth < 1:
        raise ValueError("depth_must_be_positive")
    random = np.random.default_rng(int(seed))
    gammas = random.uniform(0, 2 * pi, size=depth)
    betas = random.uniform(0, pi, size=depth)
    return np.concatenate((gammas, betas)).astype(float)


def initial_parameters(depth, seed, *, strategy=SOURCE_UNIFORM, small_random_scale=0.3):
    """Pick normal or small random starting angles."""

    if strategy == SOURCE_UNIFORM:
        return seeded_initial_parameters(depth, seed)
    if strategy != SMALL_RANDOM:
        raise ValueError(f"unsupported initialization strategy: {strategy}")
    depth = int(depth)
    scale = float(small_random_scale)
    if depth < 1 or not 0 < scale <= pi:
        raise ValueError("invalid small-random depth or scale")
    random = np.random.default_rng(int(seed))
    return np.concatenate(
        (random.uniform(0, scale, depth), random.uniform(0, scale, depth))
    ).astype(float)


def optimize_cobyla(
    objective,
    *,
    depth,
    seed,
    evaluation_budget,
    gamma_bounds=(0.0, 2.0 * pi),
    beta_bounds=(0.0, pi),
    tolerance=1e-8,
    rhobeg=0.5,
    initialization_strategy=SOURCE_UNIFORM,
    small_random_scale=0.3,
):
    """Run bounded COBYLA and remember the best point it tries."""

    depth = int(depth)
    budget = int(evaluation_budget)
    if depth < 1 or budget < 1:
        raise ValueError("invalid_optimizer_depth_or_budget")

    lower = np.array([gamma_bounds[0]] * depth + [beta_bounds[0]] * depth)
    upper = np.array([gamma_bounds[1]] * depth + [beta_bounds[1]] * depth)
    start = initial_parameters(
        depth,
        seed,
        strategy=initialization_strategy,
        small_random_scale=small_random_scale,
    )
    evaluations = 0
    initial_value = None
    best_value = float("inf")
    best_parameters = start.copy()

    def counted_objective(parameters):
        nonlocal evaluations, initial_value, best_value, best_parameters
        if evaluations >= budget:
            raise EvaluationBudgetExhausted
        evaluations += 1
        parameters = np.asarray(parameters, dtype=float)
        outside = float(
            np.maximum(lower - parameters, 0).sum()
            + np.maximum(parameters - upper, 0).sum()
        )
        value = 1_000_000 + outside if outside else float(objective(parameters))
        if not np.isfinite(value):
            raise ValueError("non_finite_optimizer_objective")
        if initial_value is None:
            initial_value = value
        if not outside and value < best_value:
            best_value = value
            best_parameters = parameters.copy()
        return value

    started = perf_counter()
    result = None
    budget_exhausted = False
    try:
        result = minimize(
            counted_objective,
            start,
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
        budget_exhausted = True

    if evaluations == 0 or initial_value is None:
        raise RuntimeError("optimizer_performed_no_evaluations")
    if result is not None:
        final_parameters = np.asarray(result.x, dtype=float)
        in_bounds = np.all(final_parameters >= lower) and np.all(final_parameters <= upper)
        if in_bounds and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_parameters = final_parameters

    success = bool(result is not None and result.success and not budget_exhausted)
    if budget_exhausted or evaluations >= budget:
        reason = "evaluation_budget_exhausted"
    elif success:
        reason = "converged"
    else:
        reason = f"optimizer_failed:{getattr(result, 'message', 'unknown_failure')}"
    return OptimizationResult(
        tuple(float(value) for value in best_parameters),
        float(best_value),
        float(initial_value),
        evaluations,
        success,
        reason,
        float(perf_counter() - started),
    )


def _check_probabilities(probabilities, energies, negative_tolerance, sum_tolerance):
    """Convert the inputs to arrays and check the probability distribution."""

    probabilities = np.asarray(probabilities, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if (
        probabilities.ndim != 1
        or probabilities.shape != energies.shape
        or len(probabilities) == 0
    ):
        raise ValueError(
            "probabilities and energies must be non-empty vectors of equal length"
        )
    if np.any(~np.isfinite(probabilities)) or np.any(~np.isfinite(energies)):
        raise ValueError("probabilities and energies must be finite")
    if negative_tolerance < 0 or sum_tolerance < 0:
        raise ValueError("probability tolerances must be non-negative")
    if np.any(probabilities < -negative_tolerance):
        raise ValueError("probabilities contain a negative value beyond tolerance")

    had_small_negative = bool(np.any(probabilities < 0))
    probabilities = np.maximum(probabilities, 0)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probability mass must be positive and finite")
    if not np.isclose(total, 1, rtol=0, atol=sum_tolerance):
        raise ValueError(f"probabilities must sum to one within tolerance; got {total}")
    if had_small_negative:
        probabilities = probabilities / total
    return probabilities, energies


def probability_weighted_expectation(
    probabilities,
    energies,
    *,
    negative_probability_tolerance=1e-12,
    normalization_tolerance=1e-10,
):
    """Calculate the ordinary expected energy."""

    probabilities, energies = _check_probabilities(
        probabilities,
        energies,
        negative_probability_tolerance,
        normalization_tolerance,
    )
    return float(probabilities @ energies)


def probability_weighted_cvar(
    probabilities,
    energies,
    alpha,
    *,
    negative_probability_tolerance=1e-12,
    normalization_tolerance=1e-10,
):
    """Calculate the mean energy in the best ``alpha`` probability mass."""

    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("cvar alpha must satisfy 0 < alpha <= 1")
    probabilities, energies = _check_probabilities(
        probabilities,
        energies,
        negative_probability_tolerance,
        normalization_tolerance,
    )
    if alpha == 1:
        return float(probabilities @ energies)

    remaining = alpha
    weighted_energy = 0.0
    for index in np.argsort(energies, kind="stable"):
        used = min(float(probabilities[index]), remaining)
        weighted_energy += used * float(energies[index])
        remaining -= used
        if remaining <= 0:
            break
    if remaining > normalization_tolerance:
        raise RuntimeError("validated distribution did not contain the requested CVaR mass")
    return weighted_energy / alpha


def evaluate_optimization_objective(
    probabilities,
    energies,
    mode,
    cvar_alpha=None,
    *,
    negative_probability_tolerance=1e-12,
    normalization_tolerance=1e-10,
):
    """Choose expectation or CVaR from the experiment setting."""

    options = {
        "negative_probability_tolerance": negative_probability_tolerance,
        "normalization_tolerance": normalization_tolerance,
    }
    if mode == EXPECTATION:
        if cvar_alpha is not None:
            raise ValueError("cvar_alpha must be None in expectation mode")
        return probability_weighted_expectation(probabilities, energies, **options)
    if mode == CVAR:
        if cvar_alpha is None:
            raise ValueError("cvar_alpha is required in cvar mode")
        return probability_weighted_cvar(
            probabilities, energies, cvar_alpha, **options
        )
    raise ValueError(f"unsupported optimization objective mode: {mode}")
