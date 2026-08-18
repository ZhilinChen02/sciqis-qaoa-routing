"""Checks for the Depth-110 summary definitions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_depth_sweep_v2_figures import (  # noqa: E402
    first_grover_above_penalty,
    first_penalty_above_grover,
    saturation_depth,
)


def rows(values: list[float]) -> list[dict[str, float | int]]:
    return [
        {"depth": depth, "p_opt": value}
        for depth, value in enumerate(values, start=1)
    ]


def test_saturation_starts_at_second_depth_for_constant_curve():
    assert saturation_depth(rows([0.1] * 20), window=10) == 2


def test_saturation_returns_first_depth_in_change_window():
    values = [float(depth) for depth in range(1, 12)] + [11.0] * 12
    assert saturation_depth(rows(values), window=10) == 12


def test_saturation_is_not_reported_without_full_window():
    assert saturation_depth(rows([0.1] * 10), window=10) is None


def test_plan_crossover_uses_grover_above_penalty_direction():
    penalty = rows([0.001, 0.01, 0.02])
    grover = rows([0.05, 0.20, 0.40])
    assert first_grover_above_penalty(penalty, grover) == 1


def test_penalty_overtake_is_reported_separately():
    penalty = rows([0.001, 0.30, 0.20])
    grover = rows([0.05, 0.20, 0.40])
    assert first_grover_above_penalty(penalty, grover) == 1
    assert first_penalty_above_grover(penalty, grover) == 2


def test_penalty_overtake_can_be_absent():
    penalty = rows([0.001, 0.01, 0.02])
    grover = rows([0.05, 0.20, 0.40])
    assert first_penalty_above_grover(penalty, grover) is None
