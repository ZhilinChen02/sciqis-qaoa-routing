"""Archived tests for the superseded warm-start tuning study."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from utils import SMALL_RANDOM, SOURCE_UNIFORM, initial_parameters
from tuning import (
    build_tuning_context,
    degenerate_parameters,
    energy_selected_winner,
    run_tuning_start,
    warm_start_probabilities,
)
from utils import clip_relaxed_values, incumbent_relaxation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tuning_context():
    return build_tuning_context()


def test_configurable_epsilon_and_bounds(graph):
    assert {row.clipped_value for row in incumbent_relaxation(graph, 0.02)} == {0.02, 0.98}
    assert {row.clipped_value for row in incumbent_relaxation(graph, 0.40)} == {0.4, 0.6}
    with pytest.raises(ValueError):
        clip_relaxed_values([0, 1], 0.0)
    with pytest.raises(ValueError):
        clip_relaxed_values([0, 1], 0.5)


def test_initial_incumbent_and_optimum_probabilities(tuning_context):
    _, _, incumbent, optimum = warm_start_probabilities(tuning_context, 0.1)
    assert np.isclose(incumbent, 0.9**14)
    assert np.isclose(optimum, 0.9**11 * 0.1**3)
    _, _, weaker_incumbent, _ = warm_start_probabilities(tuning_context, 0.3)
    assert weaker_incumbent < incumbent


def test_seed_and_multistart_reproducibility(tuning_context):
    assert np.array_equal(
        initial_parameters(2, 4, strategy=SOURCE_UNIFORM),
        initial_parameters(2, 4, strategy=SOURCE_UNIFORM),
    )
    assert np.array_equal(
        initial_parameters(2, 4, strategy=SMALL_RANDOM),
        initial_parameters(2, 4, strategy=SMALL_RANDOM),
    )
    first = run_tuning_start(
        tuning_context, epsilon=0.2, depth=1, seed=2, optimizer_budget=6
    )
    second = run_tuning_start(
        tuning_context, epsilon=0.2, depth=1, seed=2, optimizer_budget=6
    )
    assert first.final_parameters == second.final_parameters
    assert first.optimized_energy == second.optimized_energy
    assert first.p_opt == second.p_opt


def test_energy_winner_selection_never_uses_p_opt(tuning_context):
    template = run_tuning_start(
        tuning_context, epsilon=0.2, depth=1, seed=0, optimizer_budget=6
    )
    lower_energy_lower_popt = replace(
        template, seed=0, optimized_energy=20.0, p_opt=0.00001
    )
    higher_energy_higher_popt = replace(
        template, seed=1, optimized_energy=21.0, p_opt=0.9
    )
    assert energy_selected_winner(
        [higher_energy_higher_popt, lower_energy_lower_popt]
    ) is lower_energy_lower_popt


def test_distribution_shift_and_budget_metrics(tuning_context):
    run = run_tuning_start(
        tuning_context, epsilon=0.25, depth=2, seed=0, optimizer_budget=6
    )
    assert run.optimizer_budget == 6
    assert run.optimizer_evaluations <= 6
    assert np.isclose(
        run.incumbent_amplification,
        run.incumbent_probability / run.initial_incumbent_probability,
    )
    assert np.isclose(
        run.optimum_amplification,
        run.optimal_state_probability / run.initial_optimal_probability,
    )


def test_degenerate_parameter_flag():
    flagged, labels = degenerate_parameters([0.0, 2 * np.pi, 0.2, np.pi], 2)
    assert flagged is True
    assert "gamma_1:lower_zero" in labels
    assert "gamma_2:upper_2pi" in labels
    assert "beta_2:upper_pi" in labels
    assert degenerate_parameters([0.2, 0.3], 1)[0] is False


def test_tuning_csv_schemas():
    required = {
        "epsilon_sweep.csv": {
            "epsilon", "p", "optimized_energy", "p_feas", "p_opt",
            "initial_incumbent_probability", "incumbent_probability",
            "initial_optimal_probability", "optimal_state_probability",
            "circuit_depth", "two_qubit_gate_count",
        },
        "multistart_runs.csv": {
            "epsilon", "p", "seed", "initial_parameters", "final_parameters",
            "optimized_energy", "p_feas", "p_opt", "optimizer_evaluations",
            "optimizer_time_s", "degenerate_parameter_flag",
        },
        "budget_study.csv": {"epsilon", "p", "optimizer_budget", "winner_seed"},
        "distribution_shift.csv": {
            "initial_incumbent_probability", "incumbent_probability",
            "incumbent_amplification", "initial_optimal_probability",
            "optimal_state_probability", "optimum_amplification",
        },
        "tuning_summary.csv": {
            "optimized_energy_median", "optimized_energy_best", "optimized_energy_worst",
            "p_feas_median", "p_opt_median", "num_starts", "selected_final",
        },
    }
    for filename, columns in required.items():
        path = ROOT / "results" / "tuning" / filename
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = set(next(reader))
        assert columns <= header
