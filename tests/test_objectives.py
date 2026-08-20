from __future__ import annotations

import numpy as np
import pytest

from qaoa import (
    CVAR,
    EXPECTATION,
    cvar as probability_weighted_cvar,
    expectation as probability_weighted_expectation,
    objective_value as evaluate_optimization_objective,
)


def test_cvar_alpha_one_equals_expectation():
    probabilities = [0.2, 0.5, 0.3]
    energies = [-2.0, 1.0, 4.0]
    assert probability_weighted_cvar(probabilities, energies, 1.0) == pytest.approx(
        probability_weighted_expectation(probabilities, energies)
    )


def test_cvar_deterministic_distribution():
    assert probability_weighted_cvar(
        [0.0, 1.0, 0.0], [8.0, -3.0, 2.0], 0.1
    ) == pytest.approx(-3.0)


def test_cvar_exact_boundary_cutoff():
    assert probability_weighted_cvar([0.2, 0.3, 0.5], [0.0, 2.0, 9.0], 0.5) == pytest.approx(1.2)


def test_cvar_partial_cutoff():
    value = probability_weighted_cvar([0.2, 0.5, 0.3], [0.0, 1.0, 4.0], 0.4)
    assert value == pytest.approx((0.2 * 0.0 + 0.2 * 1.0) / 0.4)


def test_cvar_required_hand_example_returns_point_six():
    value = probability_weighted_cvar([0.2, 0.5, 0.3], [0.0, 1.0, 4.0], 0.5)
    assert value == pytest.approx(0.6)


def test_cvar_sorts_unsorted_energies():
    sorted_value = probability_weighted_cvar([0.2, 0.5, 0.3], [0.0, 1.0, 4.0], 0.5)
    unsorted_value = probability_weighted_cvar([0.3, 0.2, 0.5], [4.0, 0.0, 1.0], 0.5)
    assert unsorted_value == pytest.approx(sorted_value)


def test_cvar_tied_energies_are_mass_safe():
    first = probability_weighted_cvar([0.15, 0.45, 0.40], [0.0, 0.0, 2.0], 0.3)
    second = probability_weighted_cvar([0.45, 0.15, 0.40], [0.0, 0.0, 2.0], 0.3)
    assert first == second == 0.0


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0001, np.inf, np.nan])
def test_cvar_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        probability_weighted_cvar([1.0], [0.0], alpha)


def test_probability_normalization_and_negative_artifact_policy():
    with pytest.raises(ValueError, match="sum to one"):
        probability_weighted_cvar([0.2, 0.7], [0.0, 1.0], 0.5)
    with pytest.raises(ValueError, match="negative value"):
        probability_weighted_cvar([-1e-4, 1.0001], [0.0, 1.0], 0.5)
    assert probability_weighted_cvar([-1e-14, 1.0 + 1e-14], [0.0, 1.0], 0.5) == 1.0


def test_probability_normalization_tolerance_boundary():
    assert probability_weighted_cvar(
        [0.4, 0.60000000005], [0.0, 1.0], 0.5
    ) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="sum to one"):
        probability_weighted_cvar([0.4, 0.6000000002], [0.0, 1.0], 0.5)


def test_cvar_is_invariant_to_pair_permutations():
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
    energies = np.asarray([5.0, -1.0, 2.0, 2.0])
    expected = probability_weighted_cvar(probabilities, energies, 0.55)
    for permutation in (
        [3, 2, 1, 0],
        [1, 3, 0, 2],
        [2, 0, 3, 1],
    ):
        assert probability_weighted_cvar(
            probabilities[permutation], energies[permutation], 0.55
        ) == pytest.approx(expected)


def test_objective_abstraction_keeps_modes_explicit():
    probabilities = [0.2, 0.5, 0.3]
    energies = [0.0, 1.0, 4.0]
    assert evaluate_optimization_objective(
        probabilities, energies, EXPECTATION
    ) == pytest.approx(1.7)
    assert evaluate_optimization_objective(
        probabilities, energies, CVAR, 0.5
    ) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="cvar_alpha"):
        evaluate_optimization_objective(probabilities, energies, EXPECTATION, 0.5)
    with pytest.raises(ValueError, match="required"):
        evaluate_optimization_objective(probabilities, energies, CVAR)
