from __future__ import annotations

import json

import pytest

from experiments.course_final import (
    DEFAULT_CONFIG_PATH,
    build_course_final_context,
    load_course_final_config,
    run_course_final_seed,
)
from experiments.penalty_demo import run_penalty_demo
from support.sealed_results import verify_all_sealed_results


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
    assert left.optimizer.final_parameters == right.optimizer.final_parameters
    assert left.p_opt == pytest.approx(right.p_opt, abs=1e-14)
    assert abs(left.p_feas - 1.0) <= 1e-12
    assert left.evaluations <= 8


def test_penalty_demo_covers_qubo_ising_statevector_and_metrics():
    result = run_penalty_demo(evaluation_budget=4)
    assert result.node_count == 7
    assert result.edge_count == 14
    assert result.state_count == 2**14
    assert result.qubo_ising_max_error == 0.0
    assert result.ground_state_count == 1
    assert abs(result.probability_sum - 1.0) <= 1e-12
    assert 0.0 <= result.p_opt <= result.p_feas <= 1.0


def test_all_sealed_results_remain_verifiable():
    checks = verify_all_sealed_results()
    assert [item["status"] for item in checks] == ["PASS", "PASS", "PASS"]
    assert [item["artifact_count"] for item in checks] == [15, 51, 50]
