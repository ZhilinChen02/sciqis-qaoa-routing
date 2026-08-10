"""Optimization objectives for exact diagonal-QAOA probability distributions.

The CVaR convention is the lower-energy tail used for minimization.  Unlike a
fixed state-count truncation, :func:`probability_weighted_cvar` consumes exactly
``alpha`` probability mass and takes only the required fraction of an atom at
the cutoff energy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


EXPECTATION = "expectation"
CVAR = "cvar"
OBJECTIVE_MODES = (EXPECTATION, CVAR)

DEFAULT_NEGATIVE_PROBABILITY_TOLERANCE = 1e-12
DEFAULT_NORMALIZATION_TOLERANCE = 1e-10


def _validated_distribution(
    probabilities: Sequence[float],
    energies: Sequence[float],
    *,
    negative_probability_tolerance: float,
    normalization_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    costs = np.asarray(energies, dtype=np.float64)
    if probs.ndim != 1 or costs.ndim != 1 or probs.shape != costs.shape or len(probs) == 0:
        raise ValueError("probabilities and energies must be non-empty vectors of equal length")
    if np.any(~np.isfinite(probs)) or np.any(~np.isfinite(costs)):
        raise ValueError("probabilities and energies must be finite")

    negative_tolerance = float(negative_probability_tolerance)
    normalization_tolerance = float(normalization_tolerance)
    if negative_tolerance < 0.0 or normalization_tolerance < 0.0:
        raise ValueError("probability tolerances must be non-negative")
    if np.any(probs < -negative_tolerance):
        raise ValueError("probabilities contain a negative value beyond tolerance")

    # Exact statevector arithmetic can produce tiny negative artifacts after
    # other numerical transformations.  Values within the documented tolerance
    # are projected to zero before the normalization check.
    had_negative_artifact = bool(np.any(probs < 0.0))
    cleaned = np.maximum(probs, 0.0)
    total = float(np.sum(cleaned))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("probability mass must be positive and finite")
    if not np.isclose(total, 1.0, rtol=0.0, atol=normalization_tolerance):
        raise ValueError(f"probabilities must sum to one within tolerance; got {total}")
    # Preserve an already non-negative distribution verbatim after validation.
    # This avoids perturbing the historical expectation objective by a second
    # normalization after ``state_probabilities`` has normalized it.  Clipping a
    # negative artifact does change mass, so only that case is renormalized.
    if had_negative_artifact:
        cleaned = cleaned / total
    return cleaned, costs


def probability_weighted_expectation(
    probabilities: Sequence[float],
    energies: Sequence[float],
    *,
    negative_probability_tolerance: float = DEFAULT_NEGATIVE_PROBABILITY_TOLERANCE,
    normalization_tolerance: float = DEFAULT_NORMALIZATION_TOLERANCE,
) -> float:
    """Return the ordinary expectation after robust distribution validation."""

    probs, costs = _validated_distribution(
        probabilities,
        energies,
        negative_probability_tolerance=negative_probability_tolerance,
        normalization_tolerance=normalization_tolerance,
    )
    return float(probs @ costs)


def probability_weighted_cvar(
    probabilities: Sequence[float],
    energies: Sequence[float],
    alpha: float,
    *,
    negative_probability_tolerance: float = DEFAULT_NEGATIVE_PROBABILITY_TOLERANCE,
    normalization_tolerance: float = DEFAULT_NORMALIZATION_TOLERANCE,
) -> float:
    """Return minimization CVaR over exactly the lowest ``alpha`` mass.

    If the cumulative distribution crosses ``alpha`` inside one state (or one
    tied-energy atom), only the probability required to reach ``alpha`` enters
    the numerator.  Sorting is by energy, never by state count.
    """

    tail_mass = float(alpha)
    if not np.isfinite(tail_mass) or not 0.0 < tail_mass <= 1.0:
        raise ValueError("cvar alpha must satisfy 0 < alpha <= 1")
    probs, costs = _validated_distribution(
        probabilities,
        energies,
        negative_probability_tolerance=negative_probability_tolerance,
        normalization_tolerance=normalization_tolerance,
    )
    if tail_mass == 1.0:
        return float(probs @ costs)
    # For a strict partial tail, consume alpha against a unit-mass vector even
    # when the accepted input differs from one only within normalization
    # tolerance.  The alpha=1 control above intentionally remains identical to
    # the ordinary expectation function.
    probs = probs / float(np.sum(probs))
    order = np.argsort(costs, kind="stable")
    remaining = tail_mass
    weighted_tail = 0.0
    for index in order:
        available = float(probs[int(index)])
        if available <= 0.0:
            continue
        consumed = min(available, remaining)
        weighted_tail += consumed * float(costs[int(index)])
        remaining -= consumed
        if remaining <= 0.0:
            break
    if remaining > normalization_tolerance:
        raise RuntimeError("validated distribution did not contain the requested CVaR mass")
    return float(weighted_tail / tail_mass)


def evaluate_optimization_objective(
    probabilities: Sequence[float],
    energies: Sequence[float],
    mode: str,
    cvar_alpha: float | None = None,
    *,
    negative_probability_tolerance: float = DEFAULT_NEGATIVE_PROBABILITY_TOLERANCE,
    normalization_tolerance: float = DEFAULT_NORMALIZATION_TOLERANCE,
) -> float:
    """Evaluate one declared training objective without changing final metrics."""

    if mode == EXPECTATION:
        if cvar_alpha is not None:
            raise ValueError("cvar_alpha must be None in expectation mode")
        return probability_weighted_expectation(
            probabilities,
            energies,
            negative_probability_tolerance=negative_probability_tolerance,
            normalization_tolerance=normalization_tolerance,
        )
    if mode == CVAR:
        if cvar_alpha is None:
            raise ValueError("cvar_alpha is required in cvar mode")
        return probability_weighted_cvar(
            probabilities,
            energies,
            cvar_alpha,
            negative_probability_tolerance=negative_probability_tolerance,
            normalization_tolerance=normalization_tolerance,
        )
    raise ValueError(f"unsupported optimization objective mode: {mode}")
