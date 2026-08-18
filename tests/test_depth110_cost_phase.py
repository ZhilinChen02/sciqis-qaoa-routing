"""Small checks for the route-cost-phase Grover-Mixer main track."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth110_cost_phase import (  # noqa: E402
    cost_phase_setup,
    evaluate_cost_phase_grover,
    simulate_cost_phase_grover_state,
)
from depth_sweep import continuation_initial_parameters  # noqa: E402
from dynamics_study import prepare_study_context  # noqa: E402


@pytest.fixture(scope="module")
def context():
    return prepare_study_context(penalty=6.0)


def test_main_track_phase_is_the_raw_route_cost(context):
    phase, marked, _ = cost_phase_setup(context)
    expected = np.asarray(context.feasible_metadata.route_costs, dtype=np.float64)
    assert np.array_equal(phase, expected)
    assert np.unique(phase).size > 2
    assert not np.array_equal(phase, marked.astype(np.float64))


def test_one_layer_matches_direct_cost_then_mixer(context):
    phase, _, _ = cost_phase_setup(context)
    parameters = np.asarray([0.17, 0.29], dtype=np.float64)
    actual = simulate_cost_phase_grover_state(
        context.feasible_initial_state,
        phase,
        context.feasible_mixer,
        parameters,
        depth=1,
    )
    expected = np.asarray(context.feasible_initial_state, dtype=np.complex128)
    expected = expected * np.exp(-1j * parameters[0] * phase)
    expected = context.feasible_mixer.evolve(expected, parameters[1])
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.linalg.norm(actual) == pytest.approx(1.0, abs=1e-10)


def test_metrics_are_finite_and_structurally_feasible(context):
    phase, marked, _ = cost_phase_setup(context)
    parameters = continuation_initial_parameters(None, depth=1, seed=2601)
    diagnostics = evaluate_cost_phase_grover(
        context,
        parameters,
        1,
        route_cost_phase=phase,
        marked=marked,
    )
    assert diagnostics["p_feas"] == pytest.approx(1.0, abs=1e-12)
    assert diagnostics["invalid_mass"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= diagnostics["p_opt"] <= 1.0
    assert np.isfinite(diagnostics["expected_cost_full"])
    assert np.sum(diagnostics["probabilities"]) == pytest.approx(1.0, abs=1e-10)
