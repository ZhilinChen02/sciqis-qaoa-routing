from __future__ import annotations

import json

import pytest

from experiments.course_final import (
    DEFAULT_CONFIG_PATH,
    build_course_final_context,
    load_course_final_config,
    run_course_final_seed,
)
from main import run


def test_final_config_is_explicit_and_has_no_optimum_label():
    config = load_course_final_config()
    assert config["method"] == "gm_threshold_qaoa"
    assert config["representation"] == "logical_feasible_routes"
    assert config["depth"] == 3
    assert config["seeds"] == [2601, 2602, 2603]
    assert config["objective_evaluation_cap"] == 150
    assert config["threshold_source"] == "incumbent_cost"
    assert config["incumbent_cost"] == 11
    serialized = json.dumps(config).lower()
    assert "optimum_route_id" not in serialized
    assert "optimal_route_id" not in serialized
    assert DEFAULT_CONFIG_PATH.is_file()


def test_final_context_uses_the_incumbent_threshold_and_unique_current_mark():
    config = load_course_final_config()
    context = build_course_final_context(config)
    assert context.basis.size == 20
    assert context.threshold.incumbent_raw_cost == 11.0
    assert context.threshold.better_route_count == 1
    marked_costs = [
        context.costs.raw_energies[index]
        for index in context.threshold.better_route_ids
    ]
    assert marked_costs == [10.0]
    assert all(cost < 11 for cost in marked_costs)
    assert all(
        (index in context.threshold.better_route_ids) == (cost < 11)
        for index, cost in enumerate(context.costs.raw_energies)
    )
    assert context.initial_state.p_opt == pytest.approx(0.05)


def test_final_course_seed_is_deterministic_and_feasible_with_small_budget():
    context = build_course_final_context(load_course_final_config())
    left = run_course_final_seed(
        context, seed=2601, depth=1, evaluation_budget=8
    )
    right = run_course_final_seed(
        context, seed=2601, depth=1, evaluation_budget=8
    )
    assert left.optimizer.parameters == right.optimizer.parameters
    assert left.p_opt == pytest.approx(right.p_opt, abs=1e-14)
    assert abs(left.p_feas - 1.0) <= 1e-12
    assert left.evaluations <= 8


def test_penalty_demo_covers_qubo_ising_statevector_and_metrics():
    result = run(evaluation_budget=4)
    assert result["edges"] == 14
    assert result["states"] == 2**14
    assert result["exact_route"] == (0, 1, 2, 4, 5, 6)
    for values in result["results"].values():
        assert 0.0 <= values["p_opt"] <= values["p_feas"] <= 1.0
