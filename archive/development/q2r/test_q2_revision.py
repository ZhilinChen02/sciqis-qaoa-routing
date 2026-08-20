"""Archived tests for the sealed Q2-R development implementation."""

from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import q2_revision
import qaoa as historical_optimization
from metrics import distribution_metrics
from qaoa import CVAR, EXPECTATION
from qaoa import OptimizationResult, optimize_cobyla, seeded_initial_parameters
from q2_revision import (
    BASELINE_OPTIMIZER_PROGRESS,
    FIXED_DEPTH,
    INCREMENTAL_DEPTH,
    MARGINAL_DEPTH_GAIN,
    MIXER_MODE,
    MIXER_SCALE_MODE,
    NEUTRAL_TRANSFER,
    WARM_START_MODE,
    DepthTriggerDecision,
    DepthTriggerConfig,
    Q2RevisionConfig,
    ablation_configurations,
    budget_matched_fixed_depth_config,
    decide_depth_continuation,
    evaluate_q2_parameters,
    run_q2_optimization,
    run_q2_revision,
    transfer_neutral_parameters,
)
from qaoa import Q2_WARM_START, simulate_qaoa_state, state_probabilities
from tuning import build_tuning_context
from support.warm_start import (
    greedy_incumbent_route,
    incumbent_relaxation,
    mixer_hamiltonian,
    product_state,
)


@pytest.fixture(scope="module")
def revision_context():
    return build_tuning_context()


@pytest.mark.parametrize("depth", range(1, 7))
def test_neutral_parameter_transfer_preserves_every_old_layer(depth):
    parameters = np.arange(1, 2 * depth + 1, dtype=float) / (2 * depth + 2)
    transferred = transfer_neutral_parameters(parameters, depth)
    assert transferred.shape == (2 * (depth + 1),)
    assert np.array_equal(transferred[:depth], parameters[:depth])
    assert np.array_equal(transferred[depth + 1 : 2 * depth + 1], parameters[depth:])
    assert transferred[depth] == 0.0
    assert transferred[-1] == 0.0


def test_neutral_transfer_is_seed_independent_and_rejects_bad_length():
    parameters = np.asarray([0.2, 0.4, 0.6, 0.8])
    expected = transfer_neutral_parameters(parameters, 2)
    np.random.default_rng(99).random(100)
    assert np.array_equal(transfer_neutral_parameters(parameters, 2), expected)
    with pytest.raises(ValueError, match="parameter_count"):
        transfer_neutral_parameters([0.1, 0.2, 0.3], 2)


def test_zero_appended_layer_preserves_the_q2_state():
    diagonal = np.asarray([0.0, 0.2, 0.7, 1.0])
    warm = [0.1, 0.9]
    depth_one = np.asarray([0.7, 0.4])
    depth_two = transfer_neutral_parameters(depth_one, 1)
    first = simulate_qaoa_state(
        diagonal,
        depth_one,
        depth=1,
        solver=Q2_WARM_START,
        warm_start_values=warm,
    )
    transferred = simulate_qaoa_state(
        diagonal,
        depth_two,
        depth=2,
        solver=Q2_WARM_START,
        warm_start_values=warm,
    )
    assert np.allclose(transferred, first, atol=1e-14)


def _decision(**overrides):
    values = {
        "depth_mode": INCREMENTAL_DEPTH,
        "current_depth": 1,
        "maximum_depth": 3,
        "objective_before": 1.0,
        "objective_after": 0.8,
        "gain_semantics": MARGINAL_DEPTH_GAIN,
        "cumulative_objective_evaluations": 10,
        "total_cumulative_budget": 30,
        "trigger": DepthTriggerConfig(absolute_improvement_tolerance=0.1),
    }
    values.update(overrides)
    return decide_depth_continuation(**values)


def test_depth_trigger_stops_below_threshold_and_continues_above_it():
    stopped = _decision(objective_after=0.95)
    assert stopped.continuation_decision is False
    assert stopped.continuation_reason == "marginal_depth_gain_plateau"
    continued = _decision(objective_after=0.8)
    assert continued.continuation_decision is True
    assert continued.continuation_reason == "marginal_depth_gain_exceeds_threshold"


def test_p1_baseline_progress_is_recorded_but_cannot_block_p2():
    decision = _decision(
        gain_semantics=BASELINE_OPTIMIZER_PROGRESS,
        objective_before=1.0,
        objective_after=1.0,
    )
    assert decision.trigger_gain_absolute == 0.0
    assert decision.continuation_decision is True
    assert decision.continuation_reason == "baseline_complete_required_first_transfer"


def test_depth_trigger_stops_at_maximum_depth_and_cumulative_budget():
    maximum = _decision(current_depth=3)
    assert maximum.continuation_decision is False
    assert maximum.continuation_reason == "maximum_depth_reached"
    budget = _decision(cumulative_objective_evaluations=30)
    assert budget.continuation_decision is False
    assert budget.continuation_reason == "cumulative_evaluation_budget_exhausted"


def test_fixed_depth_trigger_never_continues():
    decision = _decision(depth_mode=FIXED_DEPTH)
    assert decision.continuation_decision is False
    assert decision.continuation_reason == "fixed_depth_complete"


def test_incremental_history_records_neutral_inheritance_at_every_depth(monkeypatch):
    def fake_optimizer(
        objective,
        initial,
        *,
        depth,
        evaluation_budget,
        tolerance,
        rhobeg,
        evaluation_observer,
    ):
        del evaluation_budget, tolerance, rhobeg
        initial_value = float(objective(np.asarray(initial, dtype=float)))
        for index in range(1, 4):
            evaluation_observer(
                index,
                tuple(map(float, initial)),
                initial_value,
                True,
            )
        return OptimizationResult(
            parameters=tuple(map(float, initial)),
            objective_value=initial_value,
            initial_objective=initial_value,
            evaluations=3,
            success=True,
            reason="fake_converged",
            wall_time=0.01,
        )

    monkeypatch.setattr(q2_revision, "_optimize_cobyla_from_parameters", fake_optimizer)

    def force_all_depths(**kwargs):
        current_depth = int(kwargs["current_depth"])
        maximum_depth = int(kwargs["maximum_depth"])
        semantics = str(kwargs["gain_semantics"])
        should_continue = current_depth < maximum_depth
        return DepthTriggerDecision(
            gain_semantics=semantics,
            trigger_gain_absolute=0.0,
            trigger_gain_relative=0.0,
            continuation_decision=should_continue,
            continuation_reason=(
                "forced_test_continuation" if should_continue else "maximum_depth_reached"
            ),
        )

    monkeypatch.setattr(q2_revision, "decide_depth_continuation", force_all_depths)
    config = Q2RevisionConfig(
        depth_mode=INCREMENTAL_DEPTH,
        initial_depth=1,
        max_depth=4,
        per_depth_budget=3,
        total_cumulative_budget=12,
        depth_trigger=DepthTriggerConfig(absolute_improvement_tolerance=0.0),
    )
    result = run_q2_optimization([0.0, 1.0], [10.0, 20.0], [0.2], config)
    assert [entry.depth_started for entry in result.depth_history] == [1, 2, 3, 4]
    assert [entry.objective_evaluations for entry in result.depth_history] == [3, 3, 3, 3]
    assert [len(entry.objective_trace) for entry in result.depth_history] == [3, 3, 3, 3]
    assert [
        entry.cumulative_objective_evaluations for entry in result.depth_history
    ] == [3, 6, 9, 12]
    assert result.cumulative_objective_evaluations == 12
    assert result.realized_depth == 4
    assert result.stop_reason == "maximum_depth_reached"
    assert result.depth_history[0].continuation_decision is True
    baseline = result.depth_history[0]
    assert baseline.inherited_parent_depth is None
    assert baseline.inherited_parent_objective is None
    assert baseline.neutral_transfer_objective is None
    assert baseline.neutral_transfer_equivalence_error is None
    assert baseline.marginal_depth_gain_absolute is None
    assert baseline.marginal_depth_gain_relative is None
    for parent, transferred in zip(result.depth_history, result.depth_history[1:]):
        assert transferred.inherited_parent_depth == parent.depth_completed
        assert transferred.inherited_parent_objective == pytest.approx(
            parent.objective_after, abs=1e-12
        )
        assert transferred.neutral_transfer_objective == pytest.approx(
            parent.objective_after, abs=1e-12
        )
        assert transferred.neutral_transfer_equivalence_error <= 1e-12
        assert transferred.marginal_depth_gain_absolute == pytest.approx(
            parent.objective_after - transferred.objective_after, abs=1e-12
        )
        expected_relative = transferred.marginal_depth_gain_absolute / max(
            abs(parent.objective_after), config.depth_trigger.relative_scale_floor
        )
        assert transferred.marginal_depth_gain_relative == pytest.approx(
            expected_relative, abs=1e-12
        )
    assert result.depth_history[-1].continuation_decision is False


def test_optimization_controller_has_no_exact_optimum_or_p_opt_input(monkeypatch):
    parameters = set(inspect.signature(run_q2_optimization).parameters)
    assert "optimal_cost" not in parameters
    assert "p_opt" not in parameters
    assert "optimal_state" not in parameters
    monkeypatch.setattr(
        q2_revision,
        "distribution_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("oracle read")),
    )
    config = Q2RevisionConfig(per_depth_budget=4, total_cumulative_budget=4)
    result = run_q2_optimization([0.0, 1.0], [10.0, 20.0], [0.2], config)
    assert result.realized_depth == 1


def test_fixed_expectation_mode_reproduces_existing_q2_optimizer_path(revision_context):
    warm = tuple(
        row.clipped_value for row in incumbent_relaxation(revision_context.graph, 0.1)
    )

    def objective(parameters):
        probabilities = state_probabilities(
            simulate_qaoa_state(
                revision_context.normalized_diagonal,
                parameters,
                depth=1,
                solver=Q2_WARM_START,
                warm_start_values=warm,
            )
        )
        return float(probabilities @ revision_context.normalized_diagonal)

    legacy = optimize_cobyla(
        objective,
        depth=1,
        seed=2601,
        evaluation_budget=6,
        tolerance=1e-8,
        rhobeg=0.5,
    )
    revision = run_q2_optimization(
        revision_context.normalized_diagonal,
        revision_context.raw_diagonal,
        warm,
        Q2RevisionConfig(per_depth_budget=6, total_cumulative_budget=6),
    )
    assert revision.final_parameters == legacy.parameters
    assert revision.final_objective_value == pytest.approx(legacy.objective_value)
    assert revision.cumulative_objective_evaluations == legacy.evaluations
    assert revision.stop_reason == "fixed_depth_complete"


def test_historical_q2_and_a0_optimizer_trajectories_are_equivalent(
    revision_context, monkeypatch
):
    frozen_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "baseline"
        / "experiment_config.json"
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    epsilon = float(frozen["warm_start_epsilon"])
    budget = int(frozen["optimizer_budget"])
    seed = int(frozen["seed"])
    tolerance = float(frozen["optimizer_tolerance"])
    rhobeg = float(frozen["optimizer_rhobeg"])
    # Historical Q2 includes p=2.  Using p=2 makes this gate sensitive to the
    # grouped [all gammas, all betas] ordering, which is ambiguous at p=1.
    depth = 2
    warm = tuple(
        row.clipped_value
        for row in incumbent_relaxation(revision_context.graph, epsilon)
    )

    def historical_objective(parameters):
        probabilities = state_probabilities(
            simulate_qaoa_state(
                revision_context.normalized_diagonal,
                parameters,
                depth=depth,
                solver=Q2_WARM_START,
                warm_start_values=warm,
            )
        )
        return float(probabilities @ revision_context.normalized_diagonal)

    historical_trace: list[tuple[tuple[float, ...], float]] = []
    original_minimize = historical_optimization.minimize

    def tracing_minimize(function, *args, **kwargs):
        def traced(parameters):
            value = float(function(parameters))
            historical_trace.append((tuple(map(float, parameters)), value))
            return value

        return original_minimize(traced, *args, **kwargs)

    monkeypatch.setattr(historical_optimization, "minimize", tracing_minimize)
    historical = historical_optimization.optimize_cobyla(
        historical_objective,
        depth=depth,
        seed=seed,
        evaluation_budget=budget,
        tolerance=tolerance,
        rhobeg=rhobeg,
    )
    revised = run_q2_optimization(
        revision_context.normalized_diagonal,
        revision_context.raw_diagonal,
        warm,
        Q2RevisionConfig(
            objective_mode=EXPECTATION,
            depth_mode=FIXED_DEPTH,
            initial_depth=depth,
            max_depth=depth,
            per_depth_budget=budget,
            total_cumulative_budget=budget,
            seed=seed,
            optimizer_tolerance=tolerance,
            optimizer_rhobeg=rhobeg,
        ),
    )
    revised_trace = revised.depth_history[0].objective_trace

    assert historical.evaluations == revised.cumulative_objective_evaluations
    assert historical.evaluations == len(historical_trace) == len(revised_trace)
    for (historical_parameters, historical_value), revised_record in zip(
        historical_trace, revised_trace
    ):
        assert np.max(
            np.abs(np.asarray(historical_parameters) - revised_record.parameters)
        ) <= 1e-12
        assert revised_record.objective_value == pytest.approx(
            historical_value, abs=1e-12
        )
    assert np.max(
        np.abs(np.asarray(historical.parameters) - revised.final_parameters)
    ) <= 1e-12
    assert revised.final_objective_value == pytest.approx(
        historical.objective_value, abs=1e-12
    )

    historical_probabilities = state_probabilities(
        simulate_qaoa_state(
            revision_context.normalized_diagonal,
            historical.parameters,
            depth=depth,
            solver=Q2_WARM_START,
            warm_start_values=warm,
        )
    )
    revised_probabilities = np.asarray(revised.final_probabilities)
    assert np.max(np.abs(historical_probabilities - revised_probabilities)) <= 1e-12
    historical_metrics = distribution_metrics(
        historical_probabilities,
        revision_context.states,
        optimal_cost=revision_context.optimal_cost,
    )
    revised_metrics = distribution_metrics(
        revised_probabilities,
        revision_context.states,
        optimal_cost=revision_context.optimal_cost,
    )
    assert revised_metrics.p_feas == pytest.approx(historical_metrics.p_feas, abs=1e-12)
    assert revised_metrics.p_opt == pytest.approx(historical_metrics.p_opt, abs=1e-12)


def test_historical_q2_and_a0_are_equivalent_without_optimizer(revision_context):
    epsilon = 0.1
    depth = 2
    tolerance = 1e-12
    parameters = seeded_initial_parameters(depth, 2601)
    warm = tuple(
        row.clipped_value
        for row in incumbent_relaxation(revision_context.graph, epsilon)
    )

    historical_initial_state = product_state(warm)
    historical_state = simulate_qaoa_state(
        revision_context.normalized_diagonal,
        parameters,
        depth=depth,
        solver=Q2_WARM_START,
        warm_start_values=warm,
    )
    historical_probabilities = state_probabilities(historical_state)
    historical_objective = float(
        historical_probabilities @ revision_context.normalized_diagonal
    )
    historical_expected_energy = float(
        historical_probabilities @ revision_context.raw_diagonal
    )
    historical_metrics = distribution_metrics(
        historical_probabilities,
        revision_context.states,
        optimal_cost=revision_context.optimal_cost,
    )

    a0 = Q2RevisionConfig(
        objective_mode=EXPECTATION,
        depth_mode=FIXED_DEPTH,
        initial_depth=depth,
        max_depth=depth,
        per_depth_budget=10,
        total_cumulative_budget=10,
    )
    revised = evaluate_q2_parameters(
        revision_context.normalized_diagonal,
        revision_context.raw_diagonal,
        warm,
        parameters,
        depth=depth,
        config=a0,
    )
    revised_initial_state = np.asarray(revised.initial_state)
    revised_state = np.asarray(revised.statevector)
    revised_probabilities = np.asarray(revised.probabilities)
    revised_metrics = distribution_metrics(
        revised_probabilities,
        revision_context.states,
        optimal_cost=revision_context.optimal_cost,
    )

    assert greedy_incumbent_route(revision_context.graph) == revision_context.incumbent_route
    assert np.max(np.abs(revised_initial_state - historical_initial_state)) <= tolerance
    assert np.max(np.abs(revised_state - historical_state)) <= tolerance
    assert np.max(np.abs(revised_probabilities - historical_probabilities)) <= tolerance
    assert revised.optimization_objective_value == pytest.approx(
        historical_objective, abs=tolerance
    )
    assert revised.expectation_value == pytest.approx(historical_objective, abs=tolerance)
    assert revised.raw_expectation_value == pytest.approx(
        historical_expected_energy, abs=tolerance
    )
    assert revised_metrics.p_feas == pytest.approx(historical_metrics.p_feas, abs=tolerance)
    assert revised_metrics.p_opt == pytest.approx(historical_metrics.p_opt, abs=tolerance)
    assert 1.0 - revised_metrics.p_feas == pytest.approx(
        1.0 - historical_metrics.p_feas, abs=tolerance
    )
    assert revised_metrics.feasible_conditional_cost == pytest.approx(
        historical_metrics.feasible_conditional_cost, abs=tolerance
    )
    assert revised_metrics.best_decoded_route == historical_metrics.best_decoded_route
    assert revised_metrics.best_decoded_cost == historical_metrics.best_decoded_cost
    assert revised_metrics.best_state_probability == pytest.approx(
        historical_metrics.best_state_probability, abs=tolerance
    )
    assert revised_metrics.best_state_valid == historical_metrics.best_state_valid
    assert revised_metrics.best_state_optimal == historical_metrics.best_state_optimal
    assert revised_metrics.optimal_state_rank == historical_metrics.optimal_state_rank


def test_cvar_alpha_one_matches_expectation_through_optimizer_wrapper():
    normalized = [0.0, 0.2, 0.7, 1.0]
    raw = [10.0, 12.0, 17.0, 20.0]
    warm = [0.1, 0.9]
    common = {
        "per_depth_budget": 6,
        "total_cumulative_budget": 6,
        "seed": 2601,
    }
    expectation = run_q2_optimization(
        normalized,
        raw,
        warm,
        Q2RevisionConfig(objective_mode=EXPECTATION, **common),
    )
    cvar_one = run_q2_optimization(
        normalized,
        raw,
        warm,
        Q2RevisionConfig(objective_mode=CVAR, cvar_alpha=1.0, **common),
    )
    assert cvar_one.final_parameters == expectation.final_parameters
    assert cvar_one.final_objective_value == expectation.final_objective_value
    assert cvar_one.final_expectation_value == expectation.final_expectation_value
    assert np.array_equal(cvar_one.final_probabilities, expectation.final_probabilities)
    assert cvar_one.cumulative_objective_evaluations == (
        expectation.cumulative_objective_evaluations
    )


def test_budget_matched_fixed_depth_controls_receive_cumulative_caps():
    incremental = Q2RevisionConfig(
        depth_mode=INCREMENTAL_DEPTH,
        initial_depth=1,
        max_depth=3,
        per_depth_budget=7,
        total_cumulative_budget=21,
    )
    controls = {
        depth: budget_matched_fixed_depth_config(incremental, depth)
        for depth in (1, 2, 3)
    }
    assert [controls[depth].per_depth_budget for depth in (1, 2, 3)] == [7, 14, 21]
    assert [
        controls[depth].total_cumulative_budget for depth in (1, 2, 3)
    ] == [7, 14, 21]
    for depth, control in controls.items():
        assert control.depth_mode == FIXED_DEPTH
        assert control.initial_depth == control.max_depth == depth
        assert control.objective_mode == incremental.objective_mode
        assert control.warm_start_mode == incremental.warm_start_mode
        assert control.mixer_mode == incremental.mixer_mode
        assert control.mixer_scale_mode == incremental.mixer_scale_mode
    with pytest.raises(ValueError, match="incremental reference"):
        budget_matched_fixed_depth_config(controls[1], 1)
    with pytest.raises(ValueError, match="depth range"):
        budget_matched_fixed_depth_config(incremental, 4)


def test_fixed_and_incremental_objective_call_accounting_is_identical():
    normalized = [0.0, 0.2, 0.7, 1.0]
    raw = [10.0, 12.0, 17.0, 20.0]
    warm = [0.1, 0.9]
    per_depth = 6
    fixed = run_q2_optimization(
        normalized,
        raw,
        warm,
        Q2RevisionConfig(
            per_depth_budget=per_depth,
            total_cumulative_budget=per_depth,
        ),
    )
    incremental = run_q2_optimization(
        normalized,
        raw,
        warm,
        Q2RevisionConfig(
            depth_mode=INCREMENTAL_DEPTH,
            initial_depth=1,
            max_depth=3,
            per_depth_budget=per_depth,
            total_cumulative_budget=3 * per_depth,
            depth_trigger=DepthTriggerConfig(
                absolute_improvement_tolerance=1e6,
                relative_improvement_tolerance=1e6,
            ),
        ),
    )
    assert len(fixed.depth_history) == 1
    assert len(incremental.depth_history) == 2
    assert incremental.stop_reason == "marginal_depth_gain_plateau"
    assert incremental.cumulative_objective_evaluations == sum(
        entry.objective_evaluations for entry in incremental.depth_history
    )
    assert incremental.cumulative_objective_evaluations < 3 * per_depth
    for entry in (*fixed.depth_history, *incremental.depth_history):
        assert entry.objective_evaluations == len(entry.objective_trace)
        assert entry.objective_evaluations <= per_depth
        assert entry.objective_trace[0].depth_evaluation_index == 1
        assert entry.objective_trace[0].evaluation_role in {
            "initial_diagnostic_and_optimizer_initial",
            "neutral_transfer_diagnostic_and_optimizer_initial",
        }
    assert fixed.depth_history[0].objective_trace[0].evaluation_role == (
        "initial_diagnostic_and_optimizer_initial"
    )
    assert incremental.depth_history[0].objective_trace[0].evaluation_role == (
        "initial_diagnostic_and_optimizer_initial"
    )
    assert incremental.depth_history[1].objective_trace[0].evaluation_role == (
        "neutral_transfer_diagnostic_and_optimizer_initial"
    )


def test_ablation_configuration_differences_are_isolated():
    configs = ablation_configurations(
        cvar_alpha=0.25,
        fixed_depth=1,
        incremental_max_depth=3,
        per_depth_budget=10,
        total_cumulative_budget=30,
    )
    assert set(configs) == {"A0", "A1", "A2", "A3"}
    assert (configs["A0"].objective_mode, configs["A0"].depth_mode) == (
        EXPECTATION,
        FIXED_DEPTH,
    )
    assert (configs["A1"].objective_mode, configs["A1"].depth_mode) == (
        CVAR,
        FIXED_DEPTH,
    )
    assert (configs["A2"].objective_mode, configs["A2"].depth_mode) == (
        EXPECTATION,
        INCREMENTAL_DEPTH,
    )
    assert (configs["A3"].objective_mode, configs["A3"].depth_mode) == (
        CVAR,
        INCREMENTAL_DEPTH,
    )

    ignored_objective = {"objective_mode", "cvar_alpha"}
    ignored_depth = {"depth_mode", "max_depth"}
    a0, a1, a2, a3 = (asdict(configs[name]) for name in ("A0", "A1", "A2", "A3"))
    assert {key: value for key, value in a0.items() if key not in ignored_objective} == {
        key: value for key, value in a1.items() if key not in ignored_objective
    }
    assert {key: value for key, value in a0.items() if key not in ignored_depth} == {
        key: value for key, value in a2.items() if key not in ignored_depth
    }
    assert {
        key: value
        for key, value in a0.items()
        if key not in ignored_objective | ignored_depth
    } == {
        key: value
        for key, value in a3.items()
        if key not in ignored_objective | ignored_depth
    }
    assert all(config.warm_start_mode == WARM_START_MODE for config in configs.values())
    assert all(config.mixer_mode == MIXER_MODE for config in configs.values())
    assert all(
        config.mixer_scale_mode == MIXER_SCALE_MODE for config in configs.values()
    )
    assert all(config.transfer_mode == NEUTRAL_TRANSFER for config in configs.values())


def test_ablation_variants_share_task_state_mixer_decoder_and_metrics(revision_context):
    configs = ablation_configurations(cvar_alpha=0.25)
    epsilon = 0.1
    incumbent = greedy_incumbent_route(revision_context.graph)
    warm = tuple(
        row.clipped_value
        for row in incumbent_relaxation(revision_context.graph, epsilon)
    )
    initial_state = product_state(warm)
    mixers = tuple(mixer_hamiltonian(value) for value in warm)
    parameters = seeded_initial_parameters(1, 2601)
    graph_identity = (
        revision_context.graph.graph["schema"],
        revision_context.graph.graph["name"],
        tuple(revision_context.graph.nodes),
        tuple(
            (left, right, revision_context.graph.edges[left, right]["weight"])
            for left, right in revision_context.graph.edges
        ),
    )
    reference_evaluation = None
    reference_metrics = None
    for config in configs.values():
        assert config.warm_start_mode == WARM_START_MODE
        assert config.mixer_mode == MIXER_MODE
        assert config.mixer_scale_mode == MIXER_SCALE_MODE
        assert epsilon == 0.1
        assert graph_identity[0] == "dtu-sciqis-routing-graph"
        assert greedy_incumbent_route(revision_context.graph) == incumbent
        assert np.array_equal(product_state(warm), initial_state)
        assert all(
            np.array_equal(mixer_hamiltonian(value), expected)
            for value, expected in zip(warm, mixers)
        )
        assert revision_context.states is revision_context.states
        assert distribution_metrics is q2_revision.distribution_metrics
        evaluation = evaluate_q2_parameters(
            revision_context.normalized_diagonal,
            revision_context.raw_diagonal,
            warm,
            parameters,
            depth=1,
            config=config,
        )
        metrics = distribution_metrics(
            evaluation.probabilities,
            revision_context.states,
            optimal_cost=revision_context.optimal_cost,
        )
        if reference_evaluation is None:
            reference_evaluation = evaluation
            reference_metrics = metrics
        else:
            assert np.array_equal(evaluation.initial_state, reference_evaluation.initial_state)
            assert np.array_equal(evaluation.statevector, reference_evaluation.statevector)
            assert np.array_equal(evaluation.probabilities, reference_evaluation.probabilities)
            assert evaluation.expectation_value == reference_evaluation.expectation_value
            assert evaluation.raw_expectation_value == reference_evaluation.raw_expectation_value
            assert metrics.p_feas == reference_metrics.p_feas
            assert metrics.p_opt == reference_metrics.p_opt
            assert metrics.feasible_conditional_cost == reference_metrics.feasible_conditional_cost


def test_revision_config_mapping_round_trip_and_validation():
    config = Q2RevisionConfig.from_mapping(
        {
            "objective_mode": "cvar",
            "cvar_alpha": 1.0,
            "depth_mode": "incremental",
            "initial_depth": 1,
            "max_depth": 2,
            "per_depth_budget": 5,
            "total_cumulative_budget": 9,
            "depth_trigger": {"absolute_improvement_tolerance": 1e-5},
        }
    )
    assert config.cvar_alpha == 1.0
    assert config.depth_trigger.absolute_improvement_tolerance == 1e-5
    with pytest.raises(ValueError, match="cvar_alpha"):
        Q2RevisionConfig(objective_mode="cvar", cvar_alpha=None)
    with pytest.raises(ValueError, match="fixed depth"):
        Q2RevisionConfig(depth_mode="fixed", initial_depth=1, max_depth=2)


def test_machine_readable_ablation_example_loads_all_four_configs():
    path = Path(__file__).resolve().parents[1] / "data" / "q2_revision_config.example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "dtu-sciqis-q2-literature-guided-revision"
    loaded = {
        name: Q2RevisionConfig.from_mapping(values)
        for name, values in payload["configurations"].items()
    }
    assert set(loaded) == {"A0", "A1", "A2", "A3"}
    assert loaded["A0"].objective_mode == EXPECTATION
    assert loaded["A1"].objective_mode == CVAR
    assert loaded["A2"].depth_mode == INCREMENTAL_DEPTH
    assert loaded["A3"].depth_mode == INCREMENTAL_DEPTH


def test_preregistration_contract_matches_frozen_historical_configuration():
    root = Path(__file__).resolve().parents[1]
    historical = json.loads(
        (root / "results" / "baseline" / "experiment_config.json").read_text(
            encoding="utf-8"
        )
    )
    contract = json.loads(
        (root / "data" / "q2_revision_preregistration.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["status"] == "FROZEN_NOT_EXECUTED"
    assert contract["formal_execution_authorized"] is False
    assert contract["smoke_data_status"] == "DEVELOPMENT_ONLY_NOT_CONFIRMATORY"
    assert contract["frozen_historical_q2"]["warm_start_epsilon"] == historical[
        "warm_start_epsilon"
    ]
    assert contract["frozen_historical_q2"]["fixed_penalty"] == historical[
        "fixed_penalty"
    ]
    assert contract["budget"][
        "historical_per_depth_objective_evaluation_budget_B"
    ] == historical["optimizer_budget"]
    assert contract["optimizer"]["name"] == historical["optimizer"]
    assert contract["optimizer"]["rhobeg"] == historical["optimizer_rhobeg"]
    assert contract["optimizer"]["tolerance"] == historical[
        "optimizer_tolerance"
    ]
    assert contract["optimizer"]["constraint_tolerance"] == historical[
        "optimizer_tolerance"
    ]
    assert contract["seed_policy"]["seeds"] == [historical["seed"]]
    assert contract["seed_policy"]["multistart_count_per_arm"] == 1


def test_preregistered_arm_matrix_and_budget_accounting_are_machine_validated():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "data" / "q2_revision_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    arms = {
        name: Q2RevisionConfig.from_mapping(payload["q2_config"])
        for name, payload in contract["primary_arms"].items()
    }

    assert set(arms) == {"A0", "A1", "A2", "A3"}
    assert (arms["A0"].objective_mode, arms["A0"].cvar_alpha) == (
        EXPECTATION,
        None,
    )
    assert (arms["A1"].objective_mode, arms["A1"].cvar_alpha) == (CVAR, 0.25)
    assert (arms["A2"].objective_mode, arms["A2"].cvar_alpha) == (
        EXPECTATION,
        None,
    )
    assert (arms["A3"].objective_mode, arms["A3"].cvar_alpha) == (CVAR, 0.25)
    for name in ("A0", "A1"):
        assert arms[name].depth_mode == FIXED_DEPTH
        assert arms[name].initial_depth == arms[name].max_depth == 3
        assert arms[name].per_depth_budget == 300
        assert arms[name].effective_total_budget == 300
    for name in ("A2", "A3"):
        assert arms[name].depth_mode == INCREMENTAL_DEPTH
        assert arms[name].initial_depth == 1
        assert arms[name].max_depth == 3
        assert arms[name].per_depth_budget == 100
        assert arms[name].effective_total_budget == 300
    for config in arms.values():
        assert config.seed == 2601
        assert config.optimizer_tolerance == 1e-8
        assert config.optimizer_rhobeg == 0.5
        assert config.depth_trigger.absolute_improvement_tolerance == 1e-12
        assert config.depth_trigger.relative_improvement_tolerance == 0.01
        assert config.warm_start_mode == WARM_START_MODE
        assert config.mixer_mode == MIXER_MODE
        assert config.mixer_scale_mode == MIXER_SCALE_MODE

    budget = contract["budget"]
    assert budget["fixed_p1_diagnostic"] == 100
    assert budget["fixed_p2_diagnostic"] == 200
    assert budget["fixed_p3_primary"] == 300
    assert budget["incremental_per_attempted_depth"] == 100
    assert budget["incremental_total_cap"] == 300
    assert budget["unused_incremental_budget_reallocated"] is False
    accounting = budget["objective_call_accounting"]
    assert accounting["fixed_and_incremental_semantics_identical"] is True
    assert accounting["post_optimizer_objective_recomputation"] is False


def test_revision_result_keeps_cvar_and_final_metrics_separate(revision_context):
    config = Q2RevisionConfig(
        objective_mode=CVAR,
        cvar_alpha=0.25,
        per_depth_budget=4,
        total_cumulative_budget=4,
    )
    result = run_q2_revision(revision_context, epsilon=0.1, config=config)
    warm = tuple(
        row.clipped_value for row in incumbent_relaxation(revision_context.graph, 0.1)
    )
    probabilities = state_probabilities(
        simulate_qaoa_state(
            revision_context.normalized_diagonal,
            result.final_parameters,
            depth=result.realized_depth,
            solver=Q2_WARM_START,
            warm_start_values=warm,
        )
    )
    expected_metrics = distribution_metrics(
        probabilities,
        revision_context.states,
        optimal_cost=revision_context.optimal_cost,
    )
    assert result.optimization_objective_mode == CVAR
    assert result.cvar_alpha == 0.25
    assert result.final_expectation_value == pytest.approx(
        probabilities @ revision_context.normalized_diagonal
    )
    assert result.final_expected_energy == pytest.approx(
        probabilities @ revision_context.raw_diagonal
    )
    assert result.p_feas == pytest.approx(expected_metrics.p_feas)
    assert result.p_opt == pytest.approx(expected_metrics.p_opt)
    assert result.invalid_probability_mass == pytest.approx(1.0 - result.p_feas)
    required = {
        "method_revision",
        "warm_start_mode",
        "mixer_mode",
        "mixer_scale_mode",
        "optimization_objective_mode",
        "cvar_alpha",
        "optimized_objective_value",
        "final_expectation_value",
        "depth_mode",
        "initial_depth",
        "maximum_allowed_depth",
        "realized_depth",
        "depth_trigger",
        "transfer_mode",
        "cumulative_objective_evaluations",
        "cumulative_runtime",
        "depth_history",
        "p_feas",
        "p_opt",
        "invalid_probability_mass",
    }
    assert required <= result.as_dict().keys()
    history_required = {
        "objective_trace",
        "inherited_parent_depth",
        "inherited_parent_objective",
        "neutral_transfer_objective",
        "neutral_transfer_equivalence_error",
        "marginal_depth_gain_absolute",
        "marginal_depth_gain_relative",
    }
    assert history_required <= result.as_dict()["depth_history"][0].keys()
