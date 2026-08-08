"""Transparent implementation profiling for the frozen Day-5 extension."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np
import qiskit
from qiskit.quantum_info import Statevector
import scipy

from circuit import bind_qaoa_parameters, build_qaoa_circuit
from day5_analysis import (
    EXPECTED_CORE_RESULTS_SHA256,
    load_day5_contract,
    reconcile_frozen_day4,
    representative_distribution,
    selected_cell,
)
from graph import PROJECT_ROOT, load_graph
from ising import qubo_to_ising
from optimization import compute_core_metrics, split_interleaved_parameters
from qubo import build_qubo, enumerate_state_space
from statevector_reference import (
    penalty_energies,
    probabilities,
    reference_qaoa_from_energies,
)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.quantile(values, percentile))


def _timing_summary(durations_ns: list[int]) -> dict[str, float | int]:
    values = np.asarray(durations_ns, dtype=np.int64)
    return {
        "repeat_count": len(durations_ns),
        "median_ns": int(np.median(values)),
        "p10_ns": int(np.quantile(values, 0.10)),
        "p90_ns": int(np.quantile(values, 0.90)),
        "minimum_ns": int(np.min(values)),
        "maximum_ns": int(np.max(values)),
        "median_ms": float(np.median(values) / 1e6),
        "p10_ms": float(np.quantile(values, 0.10) / 1e6),
        "p90_ms": float(np.quantile(values, 0.90) / 1e6),
    }


def _profile_operation(
    operation: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> list[int]:
    for _ in range(warmups):
        operation()
    durations = []
    last_result = None
    for _ in range(repeats):
        start = perf_counter_ns()
        last_result = operation()
        durations.append(perf_counter_ns() - start)
    if last_result is None:
        raise RuntimeError("profiling requires at least one timed repeat")
    return durations


def _environment_payload() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor or "not reported",
        "logical_cpu_count": os.cpu_count(),
        "executable": sys.executable,
    }


def _runtime_distribution(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    return {
        "run_count": len(values),
        "minimum_seconds": float(np.min(data)),
        "p10_seconds": _percentile(data, 0.10),
        "median_seconds": float(np.median(data)),
        "p90_seconds": _percentile(data, 0.90),
        "maximum_seconds": float(np.max(data)),
    }


def run_runtime_profile(
    *,
    root: str | Path = PROJECT_ROOT,
    csv_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Profile the frozen representative implementation without reoptimizing."""

    project = Path(root).resolve()
    contract, contract_hash = load_day5_contract(
        project / "data" / "day5_analysis_contract.json"
    )
    core, _reconciliation = reconcile_frozen_day4(project)
    profile_contract = contract["profiling"]
    warmups = profile_contract["warmup_repeats_per_stage"]
    repeats = profile_contract["timed_repeats_per_stage"]

    graph = load_graph(project / "data" / "graph.json")
    states = enumerate_state_space(graph)
    representative = contract["representative_state"]
    p1_cell = selected_cell(core, penalty=2, depth=1)
    p2_cell = selected_cell(core, penalty=2, depth=2)
    p1_parameters = p1_cell["optimization"]["final_parameters"]
    p2_parameters = p2_cell["optimization"]["final_parameters"]
    p1_gammas, p1_betas = split_interleaved_parameters(p1_parameters, 1)
    p2_gammas, p2_betas = split_interleaved_parameters(p2_parameters, 2)
    energies = penalty_energies(states, representative["A"])
    qubo = build_qubo(graph, representative["A"])
    hamiltonian = qubo_to_ising(qubo)
    _representative_states, _representative_cell, distribution = (
        representative_distribution(project)
    )

    p1_bound = bind_qaoa_parameters(
        build_qaoa_circuit(graph, hamiltonian, p=1),
        gammas=p1_gammas,
        betas=p1_betas,
    )
    p2_bound = bind_qaoa_parameters(
        build_qaoa_circuit(graph, hamiltonian, p=2),
        gammas=p2_gammas,
        betas=p2_betas,
    )
    sampling_rng = np.random.default_rng(profile_contract["finite_shot_stage_seed"])

    operations: list[tuple[str, str, str, Callable[[], Any]]] = [
        (
            "Model",
            "contract_validation",
            "Load and validate all frozen contracts/results",
            lambda: reconcile_frozen_day4(project),
        ),
        (
            "Model",
            "qubo_construction",
            "Construct directed-flow QUBO coefficients",
            lambda: build_qubo(graph, representative["A"]),
        ),
        (
            "Model",
            "ising_conversion",
            "Convert canonical QUBO to Ising coefficients",
            lambda: qubo_to_ising(qubo),
        ),
        (
            "Circuit",
            "qiskit_circuit_build_p1",
            "Build explicit parameterized p=1 Qiskit circuit",
            lambda: build_qaoa_circuit(graph, hamiltonian, p=1),
        ),
        (
            "Circuit",
            "qiskit_circuit_build_p2",
            "Build explicit parameterized p=2 Qiskit circuit",
            lambda: build_qaoa_circuit(graph, hamiltonian, p=2),
        ),
        (
            "Simulation",
            "numpy_statevector_p1",
            "Independent NumPy p=1 state evolution",
            lambda: reference_qaoa_from_energies(
                energies, gammas=p1_gammas, betas=p1_betas
            ),
        ),
        (
            "Simulation",
            "numpy_statevector_p2",
            "Independent NumPy p=2 state evolution",
            lambda: reference_qaoa_from_energies(
                energies, gammas=p2_gammas, betas=p2_betas
            ),
        ),
        (
            "Simulation",
            "qiskit_statevector_p1",
            "Qiskit Statevector execution of bound p=1 circuit",
            lambda: np.asarray(Statevector.from_instruction(p1_bound).data),
        ),
        (
            "Simulation",
            "qiskit_statevector_p2",
            "Qiskit Statevector execution of bound p=2 circuit",
            lambda: np.asarray(Statevector.from_instruction(p2_bound).data),
        ),
        (
            "Analysis",
            "core_metric_calculation",
            "Calculate final metrics over all 16384 states",
            lambda: compute_core_metrics(
                states,
                distribution,
                optimal_index=contract["scientific_identity"]["exact_state_index"],
            ),
        ),
        (
            "Analysis",
            "finite_shot_sampling_4096",
            "One multinomial finite-shot sample with 4096 shots",
            lambda: sampling_rng.multinomial(
                profile_contract["finite_shot_stage_shots"], distribution
            ),
        ),
    ]
    if [stage for _group, stage, _description, _operation in operations] != profile_contract["stages"]:
        raise RuntimeError("implemented profiling stages differ from frozen contract")

    raw_rows = []
    stage_summaries = []
    for group, stage, description, operation in operations:
        durations = _profile_operation(operation, warmups=warmups, repeats=repeats)
        raw_rows.extend(
            {
                "group": group,
                "stage": stage,
                "description": description,
                "repeat_id": repeat_id,
                "duration_ns": duration,
                "duration_ms": duration / 1e6,
            }
            for repeat_id, duration in enumerate(durations)
        )
        stage_summaries.append(
            {
                "group": group,
                "stage": stage,
                "description": description,
                **_timing_summary(durations),
            }
        )

    csv_output = Path(csv_path) if csv_path else project / "results" / "runtime_profile.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(raw_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    day4_runtimes = [float(run["elapsed_seconds"]) for run in core["optimization_runs"]]
    by_depth = {
        str(depth): _runtime_distribution(
            [
                float(run["elapsed_seconds"])
                for run in core["optimization_runs"]
                if run["p"] == depth
            ]
        )
        for depth in (1, 2)
    }
    by_depth["1"]["objective_evaluation_count"] = sum(
        run["objective_evaluation_count"]
        for run in core["optimization_runs"]
        if run["p"] == 1
    )
    by_depth["2"]["objective_evaluation_count"] = sum(
        run["objective_evaluation_count"]
        for run in core["optimization_runs"]
        if run["p"] == 2
    )
    summary = {
        "schema": "dtu-sciqis-routing-runtime-profile",
        "version": "1.0",
        "classification": "POST_CORE_DESCRIPTIVE_EXTENSION",
        "day5_analysis_contract_sha256": contract_hash,
        "core_experiment_results_sha256": EXPECTED_CORE_RESULTS_SHA256,
        "timer": profile_contract["timer"],
        "warmup_repeats_per_stage": warmups,
        "timed_repeats_per_stage": repeats,
        "raw_measurement_count": len(raw_rows),
        "environment": _environment_payload(),
        "stage_summaries": stage_summaries,
        "day4_observed_optimization_runtimes": {
            "source": "committed 24-run result; not rerun for profiling",
            "overall": _runtime_distribution(day4_runtimes),
            "by_depth": by_depth,
            "total_recorded_runtime_seconds": float(sum(day4_runtimes)),
            "total_objective_evaluations": sum(
                run["objective_evaluation_count"]
                for run in core["optimization_runs"]
            ),
        },
        "interpretation_boundary": profile_contract["interpretation_boundary"],
        "workflow_observation": "Repeated state evolution/objective evaluation dominates complete optimization runs; primitive timings are implementation diagnostics, not algorithmic or hardware speed comparisons.",
        "raw_profile_file": "results/runtime_profile.csv",
    }
    summary_output = Path(summary_path) if summary_path else project / "results" / "runtime_profile_summary.json"
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
