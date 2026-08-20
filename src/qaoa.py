"""Exact QAOA evolution, Expectation/CVaR objectives, and COBYLA."""

from dataclasses import dataclass
from math import pi
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, minimize

X_MIXER = "x"
GROVER_MIXER = "grover"
EXPECTATION = "expectation"
CVAR = "cvar"
SOURCE_UNIFORM = "source_uniform"
SMALL_RANDOM = "small_random"

GROVER_GLOBAL = "grover_global"


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


def _cost_diagonal(values):
    values = np.asarray(values, dtype=np.float64)
    size = values.size
    if (
        values.ndim != 1
        or size < 2
        or size & (size - 1)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("cost diagonal must have a finite power-of-two length")
    return values, size.bit_length() - 1


def _angles(values, depth):
    values, depth = np.asarray(values, dtype=np.float64), int(depth)
    if depth < 1 or values.shape != (2 * depth,) or not np.all(np.isfinite(values)):
        raise ValueError("QAOA needs p gamma values followed by p beta values")
    return values[:depth], values[depth:]


def initial_parameters(depth, seed, *, strategy=SOURCE_UNIFORM, small_random_scale=0.3):
    """Draw the frozen seed-based gamma and beta initialization."""

    depth = int(depth)
    if depth < 1:
        raise ValueError("depth_must_be_positive")
    random = np.random.default_rng(int(seed))
    if strategy == SOURCE_UNIFORM:
        gamma_high, beta_high = 2 * pi, pi
    elif strategy == SMALL_RANDOM and 0 < float(small_random_scale) <= pi:
        gamma_high = beta_high = float(small_random_scale)
    else:
        raise ValueError(f"unsupported initialization strategy: {strategy}")
    return np.concatenate(
        (random.uniform(0, gamma_high, depth), random.uniform(0, beta_high, depth))
    ).astype(float)


def normalized_diagonal(values):
    """Shift and scale energies to 0..1."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("invalid cost diagonal")
    shift = float(values.min())
    scale = float(values.max() - shift) or 1.0
    return (values - shift) / scale, shift, scale


def initial_state(num_qubits):
    """Uniform |+> state over all edge bit strings."""

    dimension = 1 << int(num_qubits)
    if dimension < 2:
        raise ValueError("num_qubits must be positive")
    return np.full(dimension, 1 / np.sqrt(dimension), dtype=np.complex128)


def apply_cost(state, energies, gamma):
    """Apply exp(-i gamma H_C); probabilities stay unchanged."""

    state = np.asarray(state, dtype=np.complex128).copy()
    energies = np.asarray(energies, dtype=np.float64)
    if state.shape != energies.shape:
        raise ValueError("state and cost diagonal must have the same size")
    state *= np.exp(-1j * float(gamma) * energies)
    return state


def apply_x_mixer(state, beta, num_qubits):
    """Apply exp(-i beta X) independently to every qubit."""

    state = np.asarray(state, dtype=np.complex128).copy()
    if state.shape != (1 << int(num_qubits),):
        raise ValueError("statevector dimension does not match num_qubits")
    cosine, sine = np.cos(float(beta)), np.sin(float(beta))
    for qubit in range(int(num_qubits)):
        blocks = state.reshape(-1, 2, 1 << qubit)
        zero, one = blocks[:, 0, :].copy(), blocks[:, 1, :].copy()
        blocks[:, 0, :] = cosine * zero - 1j * sine * one
        blocks[:, 1, :] = cosine * one - 1j * sine * zero
    return state


def apply_grover_mixer(state, beta):
    """Apply exp(-i beta |s><s|) without constructing a dense matrix."""

    state = np.asarray(state, dtype=np.complex128)
    dimension = len(state)
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("state length must be a power of two")
    uniform = 1 / np.sqrt(dimension)
    overlap = np.sum(state) * uniform
    return state + (np.exp(-1j * float(beta)) - 1) * overlap * uniform


def qaoa_state(energies, parameters, *, depth, mixer=X_MIXER, scale_x=True):
    """Alternate cost and mixer operations for p QAOA layers."""

    energies, num_qubits = _cost_diagonal(energies)
    gammas, betas = _angles(parameters, depth)
    state = initial_state(num_qubits)
    for gamma, beta in zip(gammas, betas):
        state = apply_cost(state, energies, gamma)
        if mixer == X_MIXER:
            angle = beta / num_qubits if scale_x else beta
            state = apply_x_mixer(state, angle, num_qubits)
        elif mixer == GROVER_MIXER:
            state = apply_grover_mixer(state, beta)
        else:
            raise ValueError(f"unknown mixer: {mixer}")
    return state


def probabilities(state):
    """Convert normalized amplitudes to probabilities."""

    values = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    total = float(values.sum())
    if not np.isfinite(total) or not np.isclose(total, 1, atol=1e-10):
        raise RuntimeError(f"statevector is not normalized: probability sum={total}")
    return values / total


def _distribution(probability_values, energies, negative_tolerance, sum_tolerance):
    probability_values = np.asarray(probability_values, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if probability_values.ndim != 1 or probability_values.shape != energies.shape:
        raise ValueError("probabilities and energies must be equal non-empty vectors")
    if len(probability_values) == 0 or np.any(~np.isfinite(probability_values + energies)):
        raise ValueError("probabilities and energies must be finite")
    if negative_tolerance < 0 or sum_tolerance < 0:
        raise ValueError("probability tolerances must be non-negative")
    if np.any(probability_values < -negative_tolerance):
        raise ValueError("probabilities contain a negative value beyond tolerance")
    had_negative = bool(np.any(probability_values < 0))
    probability_values = np.maximum(probability_values, 0)
    total = float(probability_values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probability mass must be positive and finite")
    if not np.isclose(total, 1, rtol=0, atol=sum_tolerance):
        raise ValueError(f"probabilities must sum to one within tolerance; got {total}")
    if had_negative:
        probability_values /= total
    return probability_values, energies


def expectation(
    probability_values,
    energies,
    *,
    negative_probability_tolerance=1e-12,
    normalization_tolerance=1e-10,
):
    """Probability-weighted mean energy."""

    probability_values, energies = _distribution(
        probability_values,
        energies,
        negative_probability_tolerance,
        normalization_tolerance,
    )
    return float(probability_values @ energies)


def cvar(
    probability_values,
    energies,
    alpha,
    *,
    negative_probability_tolerance=1e-12,
    normalization_tolerance=1e-10,
):
    """Mean energy of the lowest-energy alpha probability mass."""

    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("cvar alpha must satisfy 0 < alpha <= 1")
    probability_values, energies = _distribution(
        probability_values,
        energies,
        negative_probability_tolerance,
        normalization_tolerance,
    )
    if alpha == 1:
        return float(probability_values @ energies)
    remaining, weighted_energy = alpha, 0.0
    for index in np.argsort(energies, kind="stable"):
        used = min(float(probability_values[index]), remaining)
        weighted_energy += used * float(energies[index])
        remaining -= used
        if remaining <= 0:
            break
    if remaining > normalization_tolerance:
        raise RuntimeError("distribution does not contain the requested CVaR mass")
    return weighted_energy / alpha


def objective_value(probability_values, energies, objective, alpha=None):
    """Choose the ordinary expectation or CVaR objective."""

    if objective == EXPECTATION and alpha is None:
        return expectation(probability_values, energies)
    if objective == CVAR and alpha is not None:
        return cvar(probability_values, energies, alpha)
    if objective == EXPECTATION:
        raise ValueError("cvar_alpha must be None in expectation mode")
    if objective == CVAR:
        raise ValueError("cvar_alpha is required in cvar mode")
    raise ValueError(f"unsupported optimization objective mode: {objective}")


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
    """Bounded COBYLA with the frozen evaluation-counting policy."""

    depth, budget = int(depth), int(evaluation_budget)
    if depth < 1 or budget < 1:
        raise ValueError("invalid_optimizer_depth_or_budget")
    lower = np.array([gamma_bounds[0]] * depth + [beta_bounds[0]] * depth)
    upper = np.array([gamma_bounds[1]] * depth + [beta_bounds[1]] * depth)
    start = initial_parameters(
        depth, seed, strategy=initialization_strategy, small_random_scale=small_random_scale
    )
    evaluations, initial_value = 0, None
    best_value, best_parameters = float("inf"), start.copy()

    def counted(parameters):
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
            best_value, best_parameters = value, parameters.copy()
        return value

    started, scipy_result, exhausted = perf_counter(), None, False
    try:
        scipy_result = minimize(
            counted,
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
        exhausted = True
    if evaluations == 0 or initial_value is None:
        raise RuntimeError("optimizer_performed_no_evaluations")
    if scipy_result is not None:
        candidate = np.asarray(scipy_result.x, dtype=float)
        if np.all(candidate >= lower) and np.all(candidate <= upper) and float(
            scipy_result.fun
        ) < best_value:
            best_value, best_parameters = float(scipy_result.fun), candidate
    success = bool(scipy_result is not None and scipy_result.success and not exhausted)
    if exhausted or evaluations >= budget:
        reason = "evaluation_budget_exhausted"
    elif success:
        reason = "converged"
    else:
        reason = f"optimizer_failed:{getattr(scipy_result, 'message', 'unknown_failure')}"
    return OptimizationResult(
        tuple(map(float, best_parameters)),
        float(best_value),
        float(initial_value),
        evaluations,
        success,
        reason,
        float(perf_counter() - started),
    )


def optimize(
    energies,
    *,
    depth,
    mixer,
    objective=EXPECTATION,
    alpha=None,
    seed=2601,
    evaluation_budget=100,
):
    """Optimize one mixer/objective and return its result and probabilities."""

    energies = np.asarray(energies, dtype=float)

    def loss(parameters):
        values = probabilities(qaoa_state(energies, parameters, depth=depth, mixer=mixer))
        return objective_value(values, energies, objective, alpha)

    result = optimize_cobyla(
        loss, depth=depth, seed=seed, evaluation_budget=evaluation_budget
    )
    final = probabilities(qaoa_state(energies, result.parameters, depth=depth, mixer=mixer))
    return result, final


class GlobalGroverMixer:
    """Small object adapter used by the saved experiment code."""

    def __init__(self, num_qubits):
        self.num_qubits = int(num_qubits)
        if isinstance(num_qubits, bool) or self.num_qubits < 1:
            raise ValueError("num_qubits must be positive")

    @property
    def dimension(self):
        return 1 << self.num_qubits

    def initial_state(self):
        return initial_state(self.num_qubits)

    def evolve(self, state, beta):
        return apply_grover_mixer(state, beta)


def build_global_grover_mixer(num_qubits):
    return GlobalGroverMixer(num_qubits)


def simulate_global_grover_state(cost_diagonal, parameters, *, depth):
    return qaoa_state(cost_diagonal, parameters, depth=depth, mixer=GROVER_MIXER)
