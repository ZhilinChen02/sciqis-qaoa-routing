"""Frozen Day-5 descriptive sampling built on immutable Day-4 results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from graph import PROJECT_ROOT, load_graph
from optimization import statevector_for_parameters
from qubo import StateRecord, enumerate_state_space
from statevector_reference import penalty_energies, probabilities


EXPECTED_DAY5_CONTRACT_SHA256 = (
    "c3515a4a54b4492f26d0469ab683ca7b333b3a157633411d6072bcc09432aad9"
)
EXPECTED_GRAPH_SHA256 = (
    "451a6e4cffd01552c1cb6e4290b5ee8d50fc97514aab8ed47a6b33a111f5e54e"
)
EXPECTED_PENALTY_SHA256 = (
    "3e31dc31434dcd06c430e90e75884536cbef8cdfefec195afeb1acc1b77acd57"
)
EXPECTED_CIRCUIT_SHA256 = (
    "368fde5ef32609723758fc12da47439207a9328e152f2599560c5ab9f5287c67"
)
EXPECTED_OPTIMIZATION_SHA256 = (
    "12d44320e83ce06ae9172c6d68bd8b0f617bc4fa2b9273418fc0e8bec2ecba3d"
)
EXPECTED_CORE_RESULTS_SHA256 = (
    "2d382a313403a21ecce8b611cec5c534e585f95c3ce576c1ec8f86a4033df0cb"
)
EXPECTED_CORE_SUMMARY_SHA256 = (
    "26192f1873a525ab66d77cc1d8e66bfbab3478f95b2e93f172eee3053b542167"
)
EXPECTED_OPTIMIZATION_RUNS_SHA256 = (
    "0c753916fa46733ad65ade4d7ec0d0fd444f46bacbc4c45b806e1de53adddac7"
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability_vector_sha256(distribution: np.ndarray) -> str:
    canonical = np.asarray(distribution, dtype="<f8")
    if canonical.shape != (16384,):
        raise ValueError("the frozen probability vector must have length 16384")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_day5_contract(
    path: str | Path = PROJECT_ROOT / "data" / "day5_analysis_contract.json",
) -> tuple[dict[str, Any], str]:
    contract_path = Path(path).resolve()
    digest = file_sha256(contract_path)
    if digest != EXPECTED_DAY5_CONTRACT_SHA256:
        raise RuntimeError(
            f"Day-5 analysis contract changed: expected {EXPECTED_DAY5_CONTRACT_SHA256}, got {digest}"
        )
    contract = _read_json(contract_path)
    if not contract["statement"].startswith(
        "The Day-5 descriptive-extension protocol was frozen"
    ):
        raise RuntimeError("Day-5 pre-extension freeze statement is missing")
    if any(contract["immutability"].values()):
        raise RuntimeError("Day-5 contract may not authorize Day-4 scientific changes")
    return contract, digest


def reconcile_frozen_day4(
    root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project = Path(root).resolve()
    required_hashes = {
        "data/graph.json": EXPECTED_GRAPH_SHA256,
        "data/penalty_contract.json": EXPECTED_PENALTY_SHA256,
        "data/circuit_contract.json": EXPECTED_CIRCUIT_SHA256,
        "data/optimization_contract.json": EXPECTED_OPTIMIZATION_SHA256,
        "results/core_experiment_results.json": EXPECTED_CORE_RESULTS_SHA256,
        "results/core_experiment_summary.csv": EXPECTED_CORE_SUMMARY_SHA256,
        "results/optimization_runs.csv": EXPECTED_OPTIMIZATION_RUNS_SHA256,
    }
    observed = {
        relative: file_sha256(project / relative)
        for relative in required_hashes
    }
    if observed != required_hashes:
        raise RuntimeError(f"frozen Day-4 artifact mismatch: {observed!r}")

    contract, _digest = load_day5_contract(project / "data" / "day5_analysis_contract.json")
    core = _read_json(project / "results" / "core_experiment_results.json")
    identity = core["scientific_identity"]
    expected_identity = contract["scientific_identity"]
    checks = {
        "graph_sha256": identity["graph_sha256"],
        "penalty_contract_sha256": identity["penalty_contract_sha256"],
        "circuit_contract_sha256": identity["circuit_contract_sha256"],
        "exact_route": identity["exact_route"],
        "exact_cost": identity["exact_cost"],
        "exact_state_index": identity["exact_state_index"],
        "A_crit": identity["A_crit"],
        "optimization_run_count": len(core["optimization_runs"]),
        "selected_cell_count": len(core["selected_cells"]),
    }
    expected_checks = {
        key: expected_identity[key]
        for key in checks
    }
    if checks != expected_checks:
        raise RuntimeError(f"frozen Day-4 scientific identity changed: {checks!r}")
    if sorted((cell["identity"]["A"], cell["identity"]["p"]) for cell in core["selected_cells"]) != [
        (2, 1), (2, 2), (5, 1), (5, 2), (6, 1), (6, 2), (12, 1), (12, 2)
    ]:
        raise RuntimeError("the frozen eight-cell result matrix changed")
    return core, {"required_sha256": observed, "scientific_identity": checks}


def selected_cell(
    core: dict[str, Any], *, penalty: int, depth: int
) -> dict[str, Any]:
    matches = [
        cell
        for cell in core["selected_cells"]
        if cell["identity"]["A"] == penalty and cell["identity"]["p"] == depth
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one selected cell for A={penalty}, p={depth}")
    return matches[0]


def representative_distribution(
    root: str | Path = PROJECT_ROOT,
) -> tuple[tuple[StateRecord, ...], dict[str, Any], np.ndarray]:
    project = Path(root).resolve()
    contract, _digest = load_day5_contract(project / "data" / "day5_analysis_contract.json")
    core, _reconciliation = reconcile_frozen_day4(project)
    representative = contract["representative_state"]
    cell = selected_cell(
        core, penalty=representative["A"], depth=representative["p"]
    )
    optimization = cell["optimization"]
    if (
        optimization["selected_run_id"] != representative["selected_run_id"]
        or optimization["selected_start_id"] != representative["selected_start_id"]
        or optimization["final_parameters"] != representative["optimized_parameters"]
    ):
        raise RuntimeError("representative state does not match frozen Day-4 selection")

    graph = load_graph(project / "data" / "graph.json")
    states = enumerate_state_space(graph)
    energies = penalty_energies(states, representative["A"])
    vector = statevector_for_parameters(
        energies,
        p=representative["p"],
        parameters=representative["optimized_parameters"],
    )
    distribution = probabilities(vector)
    feasible = np.asarray([state.flow_penalty == 0 for state in states], dtype=bool)
    p_feas = float(distribution[feasible].sum())
    p_opt = float(distribution[contract["scientific_identity"]["exact_state_index"]])
    if not np.isclose(p_feas, cell["quality"]["p_feas"], atol=1e-15, rtol=0):
        raise RuntimeError("representative p_feas differs from committed Day-4 result")
    if not np.isclose(p_opt, cell["quality"]["p_opt"], atol=1e-15, rtol=0):
        raise RuntimeError("representative p_opt differs from committed Day-4 result")
    return states, cell, distribution


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sampling_summary(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "empirical_standard_deviation": float(np.std(values, ddof=1)),
        "quantile_05": float(np.quantile(values, 0.05)),
        "quantile_50": float(np.quantile(values, 0.50)),
        "quantile_95": float(np.quantile(values, 0.95)),
    }


def run_finite_shot_sampling(
    *,
    root: str | Path = PROJECT_ROOT,
    csv_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate all 800 frozen Monte Carlo sampling replicates."""

    project = Path(root).resolve()
    contract, contract_hash = load_day5_contract(
        project / "data" / "day5_analysis_contract.json"
    )
    states, cell, distribution = representative_distribution(project)
    sampling = contract["finite_shot_sampling"]
    optimum_index = contract["scientific_identity"]["exact_state_index"]
    feasible_indices = np.asarray(
        [state.state_index for state in states if state.flow_penalty == 0], dtype=int
    )
    rng = np.random.default_rng(sampling["seed"])
    rows: list[dict[str, Any]] = []
    for shots in sampling["shot_counts"]:
        for replicate_id in range(sampling["independent_replicates_per_shot_count"]):
            counts = rng.multinomial(shots, distribution)
            feasible_count = int(counts[feasible_indices].sum())
            optimum_count = int(counts[optimum_index])
            most_frequent_index = int(np.argmax(counts))
            most_frequent_state = states[most_frequent_index]
            p_feas_hat = feasible_count / shots
            p_opt_hat = optimum_count / shots
            rows.append(
                {
                    "A": cell["identity"]["A"],
                    "p": cell["identity"]["p"],
                    "selected_start_id": cell["optimization"]["selected_start_id"],
                    "shots": shots,
                    "replicate_id": replicate_id,
                    "p_feas_hat": p_feas_hat,
                    "p_opt_hat": p_opt_hat,
                    "invalid_mass_hat": 1.0 - p_feas_hat,
                    "feasible_observation_count": feasible_count,
                    "exact_optimum_observation_count": optimum_count,
                    "most_frequent_state_index": most_frequent_index,
                    "most_frequent_canonical_bitstring": most_frequent_state.canonical_bitstring,
                    "most_frequent_qiskit_display_bitstring": most_frequent_state.qiskit_display_bitstring,
                    "most_frequent_observation_count": int(counts[most_frequent_index]),
                }
            )

    if len(rows) != sampling["total_replicates"]:
        raise RuntimeError("finite-shot replicate matrix is incomplete")

    csv_output = Path(csv_path) if csv_path else project / "results" / "finite_shot_replicates.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    exact_p_feas = cell["quality"]["p_feas"]
    exact_p_opt = cell["quality"]["p_opt"]
    groups = []
    for shots in sampling["shot_counts"]:
        group = [row for row in rows if row["shots"] == shots]
        p_feas_values = np.asarray([row["p_feas_hat"] for row in group])
        p_opt_values = np.asarray([row["p_opt_hat"] for row in group])
        optimum_counts = np.asarray(
            [row["exact_optimum_observation_count"] for row in group]
        )
        groups.append(
            {
                "shots": shots,
                "replicate_count": len(group),
                "exact_p_feas": exact_p_feas,
                "p_feas_sampling": _sampling_summary(p_feas_values),
                "exact_p_opt": exact_p_opt,
                "p_opt_sampling": _sampling_summary(p_opt_values),
                "zero_exact_optimum_observation_replicates": int(
                    np.count_nonzero(optimum_counts == 0)
                ),
                "zero_exact_optimum_observation_fraction": float(
                    np.mean(optimum_counts == 0)
                ),
                "mean_exact_optimum_observation_count": float(
                    np.mean(optimum_counts)
                ),
            }
        )

    summary = {
        "schema": "dtu-sciqis-routing-finite-shot-demonstration",
        "version": "1.0",
        "classification": "POST_CORE_DESCRIPTIVE_EXTENSION",
        "label": "Monte Carlo sampling variability",
        "not_formal_confidence_intervals": True,
        "not_hardware_experiment": True,
        "day5_analysis_contract_sha256": contract_hash,
        "core_experiment_results_sha256": EXPECTED_CORE_RESULTS_SHA256,
        "source_state": {
            "A": cell["identity"]["A"],
            "p": cell["identity"]["p"],
            "selected_run_id": cell["optimization"]["selected_run_id"],
            "selected_start_id": cell["optimization"]["selected_start_id"],
            "optimized_parameters": cell["optimization"]["final_parameters"],
            "probability_vector_float64_sha256": probability_vector_sha256(distribution),
            "probability_sum": float(distribution.sum()),
            "exact_p_feas": exact_p_feas,
            "exact_p_opt": exact_p_opt,
        },
        "sampling_protocol": sampling,
        "raw_replicate_file": "results/finite_shot_replicates.csv",
        "raw_replicate_count": len(rows),
        "shot_summaries": groups,
        "caption_boundary": "Finite-shot sampling demonstration from a fixed optimized statevector; not a hardware experiment.",
    }
    summary_output = Path(summary_path) if summary_path else project / "results" / "finite_shot_summary.json"
    _write_json(summary_output, summary)
    return summary
