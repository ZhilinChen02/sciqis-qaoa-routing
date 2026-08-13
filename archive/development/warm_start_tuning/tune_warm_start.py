#!/usr/bin/env python3
"""Archived one-off warm-start tuning controller."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

from utils import SMALL_RANDOM, SOURCE_UNIFORM
from tuning import (
    TuningContext,
    TuningRun,
    build_tuning_context,
    circuit_resources,
    energy_selected_winner,
    final_distribution,
    rank_promising_epsilons,
    run_tuning_start,
    summarize_runs,
    top_rows_for_run,
)
from utils import incumbent_relaxation


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline"
TUNING = ROOT / "results" / "tuning"
FINAL = ROOT / "results" / "final"
EPSILONS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
SEEDS = tuple(range(10))
BASELINE_PFEAS = {1: 0.3761915333269981, 2: 0.638288143351339}
MINIMUM_FEASIBILITY_RETENTION = 0.60


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def run_grid(
    context: TuningContext,
    *,
    epsilons: Iterable[float],
    depths: Iterable[int],
    budgets: Iterable[int],
    strategies: Iterable[str],
) -> list[TuningRun]:
    runs: list[TuningRun] = []
    for epsilon in epsilons:
        for depth in depths:
            for budget in budgets:
                for strategy in strategies:
                    group = []
                    for seed in SEEDS:
                        group.append(
                            run_tuning_start(
                                context,
                                epsilon=epsilon,
                                depth=depth,
                                seed=seed,
                                optimizer_budget=budget,
                                initialization_strategy=strategy,
                            )
                        )
                    runs.extend(group)
                    winner = energy_selected_winner(group)
                    print(
                        f"epsilon={epsilon:.2f} p={depth} budget={budget} "
                        f"init={strategy}: winner seed={winner.seed}, "
                        f"E={winner.optimized_energy:.6f}, "
                        f"p_feas={winner.p_feas:.6f}, p_opt={winner.p_opt:.6f}"
                    )
    return runs


def grouped(runs: Sequence[TuningRun]) -> list[list[TuningRun]]:
    keys = sorted(
        {
            (run.epsilon, run.p, run.optimizer_budget, run.initialization_strategy)
            for run in runs
        }
    )
    return [
        [
            run for run in runs
            if (run.epsilon, run.p, run.optimizer_budget, run.initialization_strategy) == key
        ]
        for key in keys
    ]


def summaries_with_resources(
    context: TuningContext,
    runs: Sequence[TuningRun],
) -> list[dict[str, object]]:
    rows = []
    for group in grouped(runs):
        row = summarize_runs(group)
        winner = energy_selected_winner(group)
        resources = circuit_resources(context, winner)
        warm = incumbent_relaxation(context.graph, winner.epsilon)
        row.update(
            clipped_warm_start_values=json.dumps(
                [value.clipped_value for value in warm]
            ),
            circuit_depth=resources.circuit_depth,
            total_gate_count=resources.total_gate_count,
            two_qubit_gate_count=resources.two_qubit_gate_count,
        )
        rows.append(row)
    return rows


def _rank(values: Sequence[float], *, reverse: bool = False) -> list[int]:
    """Dense rank with numerical ties preserved."""

    rounded = [round(float(value), 12) for value in values]
    unique = sorted(set(rounded), reverse=reverse)
    lookup = {value: rank for rank, value in enumerate(unique, start=1)}
    return [lookup[value] for value in rounded]


def recommendation_scores(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Rank robust medians plus circuit/runtime cost; never a lucky maximum."""

    all_candidates = [dict(row) for row in rows if int(row["p"]) in (1, 2)]
    candidates = []
    for row in all_candidates:
        if row["initialization_strategy"] != SMALL_RANDOM:
            candidates.append(row)
            continue
        source = next(
            (
                other for other in all_candidates
                if float(other["epsilon"]) == float(row["epsilon"])
                and int(other["p"]) == int(row["p"])
                and int(other["optimizer_budget"]) == int(row["optimizer_budget"])
                and other["initialization_strategy"] == SOURCE_UNIFORM
            ),
            None,
        )
        # Keep the simpler validated source initialization unless the added
        # strategy materially lowers robust median energy.
        if source is None or (
            float(row["optimized_energy_median"])
            < float(source["optimized_energy_median"]) - 1e-8
        ):
            candidates.append(row)
    metrics = {
        "energy": _rank([float(row["optimized_energy_median"]) for row in candidates]),
        "p_feas": _rank([float(row["p_feas_median"]) for row in candidates], reverse=True),
        "p_opt": _rank([float(row["p_opt_median"]) for row in candidates], reverse=True),
        "opt_amp": _rank([float(row["optimum_amplification_median"]) for row in candidates], reverse=True),
        "incumbent": _rank([float(row["incumbent_probability_median"]) for row in candidates]),
        "two_qubit": _rank([float(row["two_qubit_gate_count"]) for row in candidates]),
        # Runtime is measured but coarsened to 10 ms before ranking so normal
        # host jitter cannot change the scientific recommendation.
        "runtime": _rank([round(float(row["optimizer_time_s_median"]), 2) for row in candidates]),
        "budget": _rank([float(row["optimizer_budget"]) for row in candidates]),
    }
    for index, row in enumerate(candidates):
        row["recommendation_score"] = (
            2.0 * metrics["energy"][index]
            + metrics["p_feas"][index]
            + metrics["p_opt"][index]
            + metrics["opt_amp"][index]
            + metrics["incumbent"][index]
            + 0.5 * metrics["two_qubit"][index]
            + 0.5 * metrics["runtime"][index]
            + 0.5 * metrics["budget"][index]
        )
        feasibility_floor = (
            MINIMUM_FEASIBILITY_RETENTION * BASELINE_PFEAS[int(row["p"])]
        )
        row["minimum_retained_p_feas"] = feasibility_floor
        row["passes_feasibility_retention"] = (
            float(row["p_feas_median"]) >= feasibility_floor
        )
        if not row["passes_feasibility_retention"]:
            row["recommendation_score"] += 100.0
    return sorted(
        candidates,
        key=lambda row: (
            float(row["recommendation_score"]),
            row["initialization_strategy"] != SOURCE_UNIFORM,
            int(row["optimizer_budget"]),
            float(row["optimized_energy_median"]),
        ),
    )


def p2_consistently_beats_p1(
    runs: Sequence[TuningRun],
    p2_summary: dict[str, object],
) -> bool:
    """Require paired improvement in both metrics for at least 7/10 seeds."""

    epsilon = float(p2_summary["epsilon"])
    budget = int(p2_summary["optimizer_budget"])
    strategy = str(p2_summary["initialization_strategy"])
    p1 = {
        run.seed: run for run in runs
        if run.epsilon == epsilon and run.p == 1
        and run.optimizer_budget == budget and run.initialization_strategy == strategy
    }
    p2 = {
        run.seed: run for run in runs
        if run.epsilon == epsilon and run.p == 2
        and run.optimizer_budget == budget and run.initialization_strategy == strategy
    }
    if set(p1) != set(SEEDS) or set(p2) != set(SEEDS):
        return False
    improvements = sum(
        p2[seed].p_feas > p1[seed].p_feas and p2[seed].p_opt > p1[seed].p_opt
        for seed in SEEDS
    )
    return improvements >= 7


def distribution_rows(summaries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "epsilon", "p", "optimizer_budget", "initialization_strategy", "winner_seed",
        "initial_incumbent_probability", "incumbent_probability", "incumbent_amplification",
        "initial_optimal_probability", "optimal_state_probability", "optimum_amplification",
        "p_feas", "p_opt", "optimal_state_rank", "optimized_energy",
    )
    return [{name: row[name] for name in fields} for row in summaries]


def final_metric_row(
    context: TuningContext,
    run: TuningRun,
    experiment_id: str,
) -> dict[str, object]:
    resources = circuit_resources(context, run)
    return {
        "experiment_id": experiment_id,
        "solver": "Q2 Warm-Start tuned",
        "p": run.p,
        "num_qubits": 14,
        "num_basis_states": len(context.states),
        "epsilon": run.epsilon,
        "penalty": 6.0,
        "optimizer": "COBYLA multistart (energy-selected)",
        "optimizer_budget": run.optimizer_budget,
        "optimizer_evaluations": run.optimizer_evaluations,
        "optimizer_success": run.optimizer_success,
        "optimizer_reason": run.optimizer_reason,
        "seed": run.seed,
        "initial_parameters": run.initial_parameters,
        "optimized_parameters": run.final_parameters,
        "optimized_energy": run.optimized_energy,
        "optimized_normalized_energy": run.optimized_normalized_energy,
        "p_feas": run.p_feas,
        "p_opt": run.p_opt,
        "feasible_conditional_cost": run.feasible_conditional_cost,
        "best_decoded_route": run.highest_probability_route,
        "best_decoded_cost": run.highest_probability_route_cost,
        "best_state_probability": run.best_state_probability,
        "best_state_valid": run.best_state_valid,
        "best_state_optimal": run.best_state_optimal,
        "optimal_state_rank": run.optimal_state_rank,
        "circuit_depth": resources.circuit_depth,
        "total_gate_count": resources.total_gate_count,
        "two_qubit_gate_count": resources.two_qubit_gate_count,
        "optimization_time_s": run.optimizer_time_s,
        "optimization_cpu_time_s": "",
        "statevector_evaluation_time_s": "",
        "statevector_evaluation_cpu_s": "",
        "total_runtime_s": run.total_runtime_s,
        "total_cpu_time_s": "",
        "probability_sum": 1.0,
        "circuit_probability_max_error": "",
        "initial_incumbent_probability": run.initial_incumbent_probability,
        "final_incumbent_probability": run.incumbent_probability,
        "initial_optimal_probability": run.initial_optimal_probability,
        "optimum_amplification": run.optimum_amplification,
        "degenerate_parameter_flag": run.degenerate_parameter_flag,
    }


def write_final_results(
    context: TuningContext,
    runs: Sequence[TuningRun],
    scored: Sequence[dict[str, object]],
    all_summaries: Sequence[dict[str, object]],
) -> list[TuningRun]:
    FINAL.mkdir(parents=True, exist_ok=True)
    selected_summaries = []
    for depth in (1, 2):
        selected_summaries.append(next(row for row in scored if int(row["p"]) == depth))
    overall = scored[0]
    p3_summaries = [row for row in all_summaries if int(row["p"]) == 3]
    if p3_summaries:
        selected_summaries.append(
            min(p3_summaries, key=lambda row: float(row["optimized_energy_median"]))
        )
    selected_runs = []
    seen = set()
    for row in selected_summaries:
        key = (
            float(row["epsilon"]), int(row["p"]), int(row["optimizer_budget"]),
            str(row["initialization_strategy"]),
        )
        group = [
            run for run in runs
            if (run.epsilon, run.p, run.optimizer_budget, run.initialization_strategy) == key
        ]
        winner = energy_selected_winner(group)
        if key not in seen:
            selected_runs.append(winner)
            seen.add(key)

    with (BASELINE / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        baseline_metrics = list(csv.DictReader(handle))
    metric_rows: list[dict[str, object]] = list(baseline_metrics)
    for run in selected_runs:
        metric_rows.append(
            final_metric_row(
                context,
                run,
                f"q2_tuned_p{run.p}_eps{str(run.epsilon).replace('.', 'p')}_b{run.optimizer_budget}",
            )
        )
    write_csv(FINAL / "metrics.csv", metric_rows)

    with (BASELINE / "top_bitstrings.csv").open(encoding="utf-8", newline="") as handle:
        top_rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for run in selected_runs:
        experiment_id = f"q2_tuned_p{run.p}_eps{str(run.epsilon).replace('.', 'p')}_b{run.optimizer_budget}"
        for row in top_rows_for_run(context, run):
            top_rows.append(
                {"experiment_id": experiment_id, "solver": "Q2 Warm-Start tuned", "p": run.p, **row}
            )
    write_csv(FINAL / "top_bitstrings.csv", top_rows)

    recommended_key = (
        float(overall["epsilon"]), int(overall["p"]), int(overall["optimizer_budget"]),
        str(overall["initialization_strategy"]),
    )
    recommended = next(
        run for run in selected_runs
        if (run.epsilon, run.p, run.optimizer_budget, run.initialization_strategy) == recommended_key
    )
    warm_rows = []
    for row in incumbent_relaxation(context.graph, recommended.epsilon):
        warm_rows.append(
            {
                "experiment_id": "recommended_q2",
                "epsilon": recommended.epsilon,
                **row.__dict__,
            }
        )
    write_csv(FINAL / "warm_start_values.csv", warm_rows)

    with (BASELINE / "runtime_profile.csv").open(encoding="utf-8", newline="") as handle:
        runtime_rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for run in selected_runs:
        experiment_id = f"q2_tuned_p{run.p}_eps{str(run.epsilon).replace('.', 'p')}_b{run.optimizer_budget}"
        runtime_rows.extend(
            [
                {
                    "experiment_id": experiment_id, "solver": "Q2 Warm-Start tuned", "p": run.p,
                    "stage": "energy-selected winner optimization", "wall_time_s": run.optimizer_time_s,
                    "cpu_time_s": "",
                },
                {
                    "experiment_id": experiment_id, "solver": "Q2 Warm-Start tuned", "p": run.p,
                    "stage": "total winner run", "wall_time_s": run.total_runtime_s,
                    "cpu_time_s": "",
                },
            ]
        )
    write_csv(FINAL / "runtime_profile.csv", runtime_rows)
    return selected_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-small-random", action="store_true")
    parser.add_argument("--skip-p3", action="store_true")
    arguments = parser.parse_args()
    missing = [
        name for name in (
            "metrics.csv", "top_bitstrings.csv", "warm_start_values.csv",
            "runtime_profile.csv", "experiment_summary.json", "experiment_config.json",
        )
        if not (BASELINE / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"baseline archive incomplete: {missing}")

    context = build_tuning_context()
    TUNING.mkdir(parents=True, exist_ok=True)
    print("Phase 1: full epsilon sweep, budget=100, source initialization")
    runs = run_grid(
        context,
        epsilons=EPSILONS,
        depths=(1, 2),
        budgets=(100,),
        strategies=(SOURCE_UNIFORM,),
    )
    primary_summaries = summaries_with_resources(context, runs)
    promising_ranking = rank_promising_epsilons(primary_summaries)
    promising = tuple(epsilon for epsilon, _score in promising_ranking[:3])
    print(f"Joint energy/feasibility/optimal/incumbent ranking selected: {promising}")

    print("Phase 2: optimizer budgets 200 and 400 for three promising epsilons")
    runs.extend(
        run_grid(
            context,
            epsilons=promising,
            depths=(1, 2),
            budgets=(200, 400),
            strategies=(SOURCE_UNIFORM,),
        )
    )
    if not arguments.skip_small_random:
        print("Phase 3: small-random initialization diagnostic at budget=100")
        runs.extend(
            run_grid(
                context,
                epsilons=promising,
                depths=(1, 2),
                budgets=(100,),
                strategies=(SMALL_RANDOM,),
            )
        )

    summaries = summaries_with_resources(context, runs)
    scored = recommendation_scores(summaries)
    best_p2 = next(row for row in scored if int(row["p"]) == 2)
    p3_ran = False
    if not arguments.skip_p3 and p2_consistently_beats_p1(runs, best_p2):
        p3_ran = True
        print("Phase 4: paired-seed criterion justifies one secondary p=3 run")
        runs.extend(
            run_grid(
                context,
                epsilons=(float(best_p2["epsilon"]),),
                depths=(3,),
                budgets=(min(400, int(best_p2["optimizer_budget"])),),
                strategies=(str(best_p2["initialization_strategy"]),),
            )
        )
        summaries = summaries_with_resources(context, runs)
        scored = recommendation_scores(summaries)
    else:
        print("Secondary p=3 not justified by the paired-seed consistency criterion")

    primary_keys = {(epsilon, depth, 100, SOURCE_UNIFORM) for epsilon in EPSILONS for depth in (1, 2)}
    epsilon_rows = [
        row for row in summaries
        if (float(row["epsilon"]), int(row["p"]), int(row["optimizer_budget"]), str(row["initialization_strategy"])) in primary_keys
    ]
    ranking_lookup = {epsilon: rank for rank, (epsilon, _score) in enumerate(promising_ranking, start=1)}
    for row in epsilon_rows:
        row["promising_epsilon_rank"] = ranking_lookup[float(row["epsilon"])]
        row["selected_for_budget_study"] = float(row["epsilon"]) in promising
    budget_rows = [
        row for row in summaries
        if float(row["epsilon"]) in promising
        and str(row["initialization_strategy"]) == SOURCE_UNIFORM
        and int(row["optimizer_budget"]) in (100, 200, 400)
        and int(row["p"]) in (1, 2)
    ]
    selected_signature = (
        float(scored[0]["epsilon"]), int(scored[0]["p"]),
        int(scored[0]["optimizer_budget"]), str(scored[0]["initialization_strategy"]),
    )
    scored_lookup = {
        (float(row["epsilon"]), int(row["p"]), int(row["optimizer_budget"]), str(row["initialization_strategy"])): row
        for row in scored
    }
    for row in summaries:
        signature = (
            float(row["epsilon"]), int(row["p"]),
            int(row["optimizer_budget"]), str(row["initialization_strategy"]),
        )
        row["recommendation_score"] = scored_lookup.get(signature, {}).get("recommendation_score", "")
        row["selected_final"] = signature == selected_signature
        row["secondary_p3"] = int(row["p"]) == 3

    write_csv(TUNING / "multistart_runs.csv", [run.as_row() for run in runs])
    write_csv(TUNING / "epsilon_sweep.csv", epsilon_rows)
    write_csv(TUNING / "budget_study.csv", budget_rows)
    write_csv(TUNING / "distribution_shift.csv", distribution_rows(summaries))
    write_csv(TUNING / "tuning_summary.csv", summaries)
    selected_runs = write_final_results(context, runs, scored, summaries)

    metadata = {
        "promising_epsilons": list(promising),
        "promising_ranking": promising_ranking,
        "p3_ran": p3_ran,
        "selection_policy": (
            "rank of cross-seed medians with >=60% baseline feasibility retention; "
            "every start winner selected only by minimum expected QUBO energy"
        ),
        "minimum_feasibility_retention_fraction": MINIMUM_FEASIBILITY_RETENTION,
        "recommended_configuration": scored[0],
        "final_selected_runs": [run.as_row() for run in selected_runs],
    }
    (TUNING / "selection.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Recommended:",
        f"epsilon={scored[0]['epsilon']} p={scored[0]['p']} ",
        f"budget={scored[0]['optimizer_budget']} init={scored[0]['initialization_strategy']}",
    )


if __name__ == "__main__":
    main()
