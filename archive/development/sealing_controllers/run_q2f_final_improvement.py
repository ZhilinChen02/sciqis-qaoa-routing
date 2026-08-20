#!/usr/bin/env python3
"""Archived controller that produced the immutable final-improvement root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from feasible_qaoa import (
    INCUMBENT_BIASED_FEASIBLE,
    UNIFORM_FEASIBLE,
    build_feasible_initial_state,
    build_feasible_route_basis,
    build_logical_cost_hamiltonian,
    build_logical_path_exchange_mixer,
)
from graph import load_graph
from feasible_experiments import (
    BSP_PATH_EXCHANGE,
    FINAL_METHODS,
    GM_QAOA_EXPECTATION,
    GM_TH_QAOA,
    build_grover_feasible_mixer,
    build_incumbent_threshold,
    run_final_improvement_cell,
)


CONFIG_PATH = PROJECT / "data" / "q2f_final_improvement_config.json"
Q2F_IDENTITY = "q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7"
Q2F_ROOT = PROJECT / "results" / "q2f_course_extension" / Q2F_IDENTITY
Q2F_MANIFEST_SHA256 = "dfa0313b198824f5925f00494a307311445bd7ac08786f1e16570993d919f99b"
Q2R_IDENTITY = "q2r-9c98b90049545d0508fb20bb488019608a4356d55fe91ff809bce8dc5c5e0c69"
Q2R_ROOT = PROJECT / "results" / "q2_revision_formal" / Q2R_IDENTITY
Q2R_MANIFEST_SHA256 = "896fdbf45bbf19d04f74a802e78739714d286beabf322db6ef8f3594679411d1"
RESULT_PARENT = PROJECT / "results" / "q2f_final_improvement"
SOURCE_PATHS = (
    PROJECT / "src" / "feasible_experiments.py",
    PROJECT / "scripts" / "run_q2f_final_improvement.py",
)
EVIDENCE_SOURCE_PATHS = SOURCE_PATHS + (
    PROJECT / "tests" / "test_q2f_final_improvement.py",
    PROJECT / "docs" / "methods" / "Q2F_FINAL_IMPROVEMENT.md",
    CONFIG_PATH,
)

METHOD_LABELS = {
    BSP_PATH_EXCHANGE: "BSP path-exchange",
    GM_QAOA_EXPECTATION: "GM-QAOA expectation",
    GM_TH_QAOA: "GM-Th-QAOA",
}
METHOD_COLORS = {
    BSP_PATH_EXCHANGE: "#e45756",
    GM_QAOA_EXPECTATION: "#4c78a8",
    GM_TH_QAOA: "#54a24b",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(f"output_appeared_before_publication:{path}")
    os.rename(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).rstrip("\n")


def verify_sealed_root(root: Path, identity: str, expected_manifest_sha: str) -> dict[str, Any]:
    manifest_path = root / "hashes" / "result_manifest.json"
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(f"sealed_manifest_hash_mismatch:{identity}:{actual_manifest_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = []
    for entry in manifest["files"]:
        if entry.get("scope") == "external_report":
            relative = entry["project_relative_path"]
            path = PROJECT / relative
        else:
            relative = entry["relative_path"]
            path = root / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_hash != entry["sha256"] or actual_size != entry["size_bytes"]:
            raise RuntimeError(f"sealed_artifact_mismatch:{identity}:{relative}")
        checked.append(relative)
    return {
        "identity": identity,
        "result_manifest_sha256": actual_manifest_sha,
        "verified_artifact_count": len(checked),
        "artifact_integrity": "PASS",
    }


def verify_old_seals() -> dict[str, Any]:
    return {
        "q2f": verify_sealed_root(Q2F_ROOT, Q2F_IDENTITY, Q2F_MANIFEST_SHA256),
        "q2r": verify_sealed_root(Q2R_ROOT, Q2R_IDENTITY, Q2R_MANIFEST_SHA256),
    }


def load_and_validate_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema") != "dtu-sciqis-q2f-final-improvement-config":
        raise ValueError("invalid_final_improvement_config_schema")
    if config.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise ValueError("final_improvement_config_not_frozen")
    if [item["id"] for item in config["methods"]] != list(FINAL_METHODS):
        raise ValueError("final_improvement_method_order_changed")
    if config["depths"] != [1, 2, 3, 4] or config["seeds"] != [2601, 2602, 2603]:
        raise ValueError("final_improvement_depth_or_seed_matrix_changed")
    optimizer = config["optimizer"]
    if (
        optimizer["name"] != "COBYLA"
        or optimizer["objective_evaluation_cap"] != 150
        or optimizer["rhobeg"] != 0.5
        or optimizer["tolerance"] != 1e-8
        or optimizer["constraint_tolerance"] != 1e-8
    ):
        raise ValueError("final_improvement_optimizer_contract_changed")
    success = config["success_rule"]
    if (
        success["reference_p_opt"] != 0.2649009394
        or success["relative_multiplier"] != 1.1
        or success["required_median_p_opt"] != 0.29139103334
        or config["required_run_count"] != 36
        or config["feasibility_tolerance"] != 1e-12
    ):
        raise ValueError("final_improvement_success_or_size_contract_changed")
    return config


def identity_entries(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(PROJECT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def run_identity(graph_sha: str, source_sha: str, config_sha: str) -> str:
    digest = sha256_bytes(f"{graph_sha}\n{source_sha}\n{config_sha}\n".encode())
    return f"q2fi-{digest}"


def build_study(config: dict[str, Any]):
    graph_path = PROJECT / config["graph_path"]
    graph = load_graph(graph_path)
    basis = build_feasible_route_basis(graph, graph_path=str(graph_path))
    old_basis_payload = json.loads((Q2F_ROOT / "basis.json").read_text(encoding="utf-8"))
    if basis.as_dict() != old_basis_payload["basis"]:
        raise RuntimeError("current_feasible_basis_differs_from_sealed_q2f_basis")
    costs = build_logical_cost_hamiltonian(basis)
    path_mixer = build_logical_path_exchange_mixer(basis)
    uniform = build_feasible_initial_state(basis, costs, mode=UNIFORM_FEASIBLE)
    biased = build_feasible_initial_state(
        basis, costs, mode=INCUMBENT_BIASED_FEASIBLE, bias_lambda=1.0
    )
    grover = build_grover_feasible_mixer(basis.size)
    incumbent_cost = float(costs.raw_energies[basis.incumbent_route_id])
    threshold = build_incumbent_threshold(costs.raw_energies, incumbent_cost)
    if basis.size != config["basis"]["required_route_count"] or len(path_mixer.exchanges) != 106:
        raise RuntimeError("sealed_q2f_basis_or_path_mixer_identity_changed")
    if not threshold.better_route_ids:
        raise RuntimeError("incumbent_threshold_has_no_better_routes")
    return graph_path, basis, costs, path_mixer, grover, uniform, biased, threshold


def summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    optimizer = result["optimizer"]
    metrics = result["metrics"]
    return {
        "run_id": result["run_id"],
        "method": result["method"],
        "depth": result["depth"],
        "seed": result["seed"],
        "initialization_mode": result["initialization_mode"],
        "mixer_kind": result["mixer_kind"],
        "phase_kind": result["phase_kind"],
        "loss_kind": result["loss_kind"],
        "incumbent_raw_cost_threshold": result["incumbent_raw_cost_threshold"],
        "better_route_ids": json.dumps(result["better_route_ids"], separators=(",", ":")),
        "evaluation_budget": result["evaluation_budget"],
        "optimizer_evaluations": optimizer["evaluations"],
        "optimizer_success": optimizer["success"],
        "optimizer_reason": optimizer["reason"],
        "optimizer_message": optimizer["message"],
        "optimizer_runtime_seconds": optimizer["runtime_seconds"],
        "p_feas": metrics["p_feas"],
        "p_opt": metrics["p_opt"],
        "bsp": metrics["bsp"],
        "expected_route_cost": metrics["expected_route_cost"],
        "expected_normalized_cost": metrics["expected_normalized_cost"],
        "top3_lowest_cost_mass": metrics["top3_lowest_cost_mass"],
        "top5_lowest_cost_mass": metrics["top5_lowest_cost_mass"],
        "probability_entropy": metrics["probability_entropy"],
        "optimal_state_rank": metrics["optimal_state_rank"],
        "most_probable_route_id": metrics["most_probable_route_id"],
        "most_probable_route": "->".join(map(str, metrics["most_probable_route"])),
        "most_probable_route_cost": metrics["most_probable_route_cost"],
        "most_probable_route_probability": metrics["most_probable_route_probability"],
        "initial_p_opt": metrics["initial_p_opt"],
        "p_opt_amplification": metrics["p_opt_amplification"],
        "probability_sum": float(sum(result["final_probabilities"])),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = (
        "p_opt",
        "bsp",
        "expected_route_cost",
        "top3_lowest_cost_mass",
        "top5_lowest_cost_mass",
        "probability_entropy",
        "p_opt_amplification",
        "optimizer_evaluations",
        "optimizer_runtime_seconds",
    )
    output: list[dict[str, Any]] = []
    for method in FINAL_METHODS:
        for depth in (1, 2, 3, 4):
            selected = [row for row in rows if row["method"] == method and row["depth"] == depth]
            if len(selected) != 3 or sorted(row["seed"] for row in selected) != [2601, 2602, 2603]:
                raise RuntimeError(f"aggregate_cell_incomplete:{method}:p{depth}")
            aggregate: dict[str, Any] = {
                "method": method,
                "depth": depth,
                "seed_count": 3,
                "seeds": "[2601,2602,2603]",
                "p_feas_min": float(min(row["p_feas"] for row in selected)),
                "p_feas_max": float(max(row["p_feas"] for row in selected)),
                "optimizer_cap_hits": sum(
                    row["optimizer_reason"] == "evaluation_budget_exhausted" for row in selected
                ),
            }
            for field in metric_fields:
                values = np.asarray([float(row[field]) for row in selected], dtype=float)
                aggregate[f"{field}_median"] = float(np.median(values))
                aggregate[f"{field}_min"] = float(np.min(values))
                aggregate[f"{field}_max"] = float(np.max(values))
            output.append(aggregate)
    return output


def load_old_expectation() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_summary = json.loads((Q2F_ROOT / "summary" / "q2f_results.json").read_text(encoding="utf-8"))["results"]
    rows = [row for row in old_summary if row["objective"] == "expectation"]
    raw = [
        json.loads((Q2F_ROOT / "raw" / f"expectation_p{depth}_seed{seed}.json").read_text(encoding="utf-8"))
        for depth in (1, 2, 3) for seed in (2601, 2602, 2603)
    ]
    return rows, raw


def decide(
    aggregates: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    success = config["success_rule"]
    tolerance = float(config["feasibility_tolerance"])
    threshold = float(success["required_median_p_opt"])
    passing = [
        row for row in aggregates
        if row["p_opt_median"] >= threshold
        and abs(row["p_feas_min"] - 1.0) <= tolerance
        and abs(row["p_feas_max"] - 1.0) <= tolerance
    ]
    strongest = max(aggregates, key=lambda row: (row["p_opt_median"], -row["depth"]))
    if not passing:
        return False, strongest, strongest
    highest = max(row["p_opt_median"] for row in passing)
    tied = [row for row in passing if highest - row["p_opt_median"] <= 0.01]
    lowest_depth = min(row["depth"] for row in tied)
    tied = [row for row in tied if row["depth"] == lowest_depth]
    simplicity = {method: index for index, method in enumerate(success["simplicity_order"])}
    selected = min(tied, key=lambda row: simplicity[row["method"]])
    return True, selected, strongest


def save_figure(fig: plt.Figure, path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"refusing_to_overwrite_figure:{path}")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.5),
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def generate_figures(
    root: Path,
    basis,
    rows: list[dict[str, Any]],
    raw_payloads: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    old_rows: list[dict[str, Any]],
    old_raw: list[dict[str, Any]],
    figure_candidate: dict[str, Any],
) -> list[str]:
    configure_plot_style()
    figures = root / "figures"
    outputs: list[str] = []
    depths = (1, 2, 3, 4)

    fig, ax = plt.subplots()
    old_medians = [
        float(np.median([row["final_p_opt"] for row in old_rows if row["depth"] == depth]))
        for depth in (1, 2, 3)
    ]
    ax.scatter(
        [row["depth"] - 0.09 for row in old_rows],
        [row["final_p_opt"] for row in old_rows],
        color="#777777", alpha=0.42, s=25,
    )
    ax.plot((1, 2, 3), old_medians, color="#555555", marker="o", linewidth=2,
            label="Old Q2-F expectation (read-only)")
    offsets = {BSP_PATH_EXCHANGE: -0.03, GM_QAOA_EXPECTATION: 0.03, GM_TH_QAOA: 0.09}
    for method in FINAL_METHODS:
        selected_rows = [row for row in rows if row["method"] == method]
        ax.scatter(
            [row["depth"] + offsets[method] for row in selected_rows],
            [row["p_opt"] for row in selected_rows],
            color=METHOD_COLORS[method], alpha=0.45, s=28,
        )
        medians = [
            next(row["p_opt_median"] for row in aggregates if row["method"] == method and row["depth"] == depth)
            for depth in depths
        ]
        ax.plot(depths, medians, color=METHOD_COLORS[method], marker="o", linewidth=2,
                label=METHOD_LABELS[method])
    ax.axhline(0.29139103334, color="#8c2d2d", linestyle="--", linewidth=1.2,
               label="Predefined 10% threshold")
    ax.set(title="Final bounded Q2-F study: optimal-route probability",
           xlabel="Fixed depth p", ylabel="p_opt", xticks=depths)
    ax.legend(fontsize=9)
    outputs.append(save_figure(fig, figures / "01_median_popt_versus_depth.png"))

    candidate_rows = [
        row for row in rows
        if row["method"] == figure_candidate["method"] and row["depth"] == figure_candidate["depth"]
    ]
    candidate_median = float(np.median([row["p_opt"] for row in candidate_rows]))
    candidate_row = min(candidate_rows, key=lambda row: (abs(row["p_opt"] - candidate_median), row["seed"]))
    candidate_payload = next(item for item in raw_payloads if item["result"]["run_id"] == candidate_row["run_id"])
    old_p3 = [row for row in old_rows if row["depth"] == 3]
    old_median = float(np.median([row["final_p_opt"] for row in old_p3]))
    old_row = min(old_p3, key=lambda row: (abs(row["final_p_opt"] - old_median), row["seed"]))
    old_payload = next(item for item in old_raw if item["result"]["run_id"] == old_row["run_id"])
    route_ids = np.arange(basis.size)
    width = 0.42
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.bar(route_ids - width / 2, old_payload["result"]["final_probabilities"], width,
           color="#777777", alpha=0.78,
           label=f"Old expectation p=3, median seed {old_row['seed']}")
    ax.bar(route_ids + width / 2, candidate_payload["result"]["final_probabilities"], width,
           color=METHOD_COLORS[figure_candidate["method"]], alpha=0.82,
           label=f"{METHOD_LABELS[figure_candidate['method']]} p={figure_candidate['depth']}, median seed {candidate_row['seed']}")
    ax.axvline(basis.exact_optimal_route_id, color="#b22222", linestyle="--", linewidth=1.2,
               label="Exact optimal route (evaluation only)")
    ax.set(title="Feasible-route distribution: strongest robust new cell vs old Q2-F",
           xlabel="Logical route ID (ordered by cost, then route)", ylabel="Probability",
           xticks=route_ids)
    ax.legend(fontsize=9)
    outputs.append(save_figure(fig, figures / "02_strongest_distribution_vs_old_q2f.png"))

    labels = [f"{METHOD_LABELS[row['method']]}\np={row['depth']}" for row in aggregates]
    x = np.arange(len(aggregates))
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.0), sharex=True)
    colors = [METHOD_COLORS[row["method"]] for row in aggregates]
    axes[0].bar(x, [row["p_opt_median"] for row in aggregates], color=colors, alpha=0.82)
    axes[0].axhline(0.2649009394, color="#555555", linestyle=":", label="Old robust reference")
    axes[0].axhline(0.29139103334, color="#8c2d2d", linestyle="--", label="10% success threshold")
    axes[0].set(ylabel="Median p_opt", title="Concentration improved, feasibility remained structural")
    axes[0].legend()
    for index, row in enumerate(rows):
        aggregate_index = next(
            i for i, item in enumerate(aggregates)
            if item["method"] == row["method"] and item["depth"] == row["depth"]
        )
        axes[1].scatter(aggregate_index, row["p_feas"], color=METHOD_COLORS[row["method"]], alpha=0.6, s=24)
    axes[1].axhline(1.0, color="#222222", linewidth=1)
    axes[1].set(ylabel="p_feas (all seed runs)", ylim=(1 - 2e-12, 1 + 2e-12))
    axes[1].set_xticks(x, labels, rotation=52, ha="right")
    fig.text(0.5, -0.02, "p_feas=1 follows from the enumerated feasible basis; it is not quantum advantage.", ha="center")
    outputs.append(save_figure(fig, figures / "03_popt_pfeas_summary.png"))
    return outputs


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def result_report(
    identity: str,
    basis,
    threshold,
    aggregates: list[dict[str, Any]],
    old_rows: list[dict[str, Any]],
    improvement_found: bool,
    selected: dict[str, Any],
    strongest: dict[str, Any],
    rows: list[dict[str, Any]],
    figures: list[str],
) -> str:
    lookup = {(row["method"], row["depth"]): row for row in aggregates}
    old = {
        depth: float(np.median([row["final_p_opt"] for row in old_rows if row["depth"] == depth]))
        for depth in (1, 2, 3)
    }
    popt_rows = [["Old Q2-F expectation", *(f"{old[p]:.12g}" for p in (1, 2, 3)), "N/A"]]
    range_rows = []
    for method in FINAL_METHODS:
        popt_rows.append([METHOD_LABELS[method], *(f"{lookup[method,p]['p_opt_median']:.12g}" for p in (1,2,3,4))])
        range_rows.append([
            METHOD_LABELS[method],
            *(f"{lookup[method,p]['p_opt_min']:.6g}–{lookup[method,p]['p_opt_max']:.6g}" for p in (1,2,3,4)),
        ])
    cap_hits = sum(row["optimizer_reason"] == "evaluation_budget_exhausted" for row in rows)
    pfeas_min = min(row["p_feas"] for row in rows)
    pfeas_max = max(row["p_feas"] for row in rows)
    improvement = strongest["p_opt_median"] / 0.2649009394 - 1.0
    bsp_equals = threshold.better_route_ids == (basis.exact_optimal_route_id,)
    decision = "ADOPT_IMPROVED_VARIANT" if improvement_found else "FREEZE_EXISTING_Q2F"
    selected_text = (
        f"{METHOD_LABELS[selected['method']]} at p={selected['depth']}"
        if improvement_found else "none; retain the existing Q2-F definition"
    )
    return f"""# Q2-F final bounded improvement results

Run identity: `{identity}`

This is a descriptive, teaching-scale mechanism study. Full feasible-route
enumeration is classical preprocessing; structural `p_feas=1` is not quantum
advantage; and three seeds do not support statistical-significance claims.

The incumbent route is `{' -> '.join(map(str, basis.incumbent_route))}` with raw
cost `{threshold.incumbent_raw_cost:g}`. The strict incumbent-derived set
`B={{P_i:C(P_i)<C(P_incumbent)}}` contains `{threshold.better_route_count}` route.
Only after constructing B was it compared with the evaluation-only optimum:
`BSP == p_opt` on this instance is `{str(bsp_equals).lower()}`.

## Median p_opt

{markdown_table(['method', 'p=1', 'p=2', 'p=3', 'p=4'], popt_rows)}

## Seed min–max p_opt

{markdown_table(['method', 'p=1', 'p=2', 'p=3', 'p=4'], range_rows)}

The strongest new median was `{strongest['p_opt_median']:.17g}` for
`{METHOD_LABELS[strongest['method']]}` at `p={strongest['depth']}`, a relative
change of `{improvement:.3%}` from `0.2649009394`. The predefined 10% rule
passed: `{str(improvement_found).upper()}`. Selected final course variant:
`{selected_text}`. Final decision: `{decision}`.

All 36 required rows were retained. The observed p_feas range was
`[{pfeas_min:.17g}, {pfeas_max:.17g}]`; all values satisfy the 1e-12 invariant.
`{cap_hits}` optimizer runs reached the 150-request cap and remain retained.

## Interpretation boundary

BSP and the GM-Th phase use only the incumbent threshold, never an optimum
label. The Grover variants use the exact rank-one mixer
`exp(-i beta |F><F|)` with uniform feasible initialization. GM-Th-QAOA here is
an incumbent-threshold demonstration in a 20-state logical route space, not a
scalable routing claim or quantum-advantage claim.

## Figures

""" + "\n".join(f"- `{Path(path).relative_to(PROJECT)}`" for path in figures) + "\n"


def validate_results(
    rows: list[dict[str, Any]],
    raw_payloads: list[dict[str, Any]],
    basis,
    costs,
    threshold,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
    graph_sha: str,
    old_before: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = {
        f"{method}_p{depth}_seed{seed}"
        for method in FINAL_METHODS for depth in (1,2,3,4) for seed in (2601,2602,2603)
    }
    actual_ids = [row["run_id"] for row in rows]
    tolerance = float(config["feasibility_tolerance"])
    better_expected = np.asarray(costs.raw_energies) < threshold.incumbent_raw_cost
    phase_and_loss_no_optimum = all(
        payload["scientific_identity"]["optimum_identity_used_by_optimizer"] is False
        for payload in raw_payloads
    )
    checks = {
        "basis_is_exact_sealed_q2f_basis": basis.as_dict() == json.loads((Q2F_ROOT / "basis.json").read_text())["basis"],
        "basis_size_is_20": basis.size == 20,
        "all_36_required_runs_exist": len(rows) == 36 and set(actual_ids) == expected_ids,
        "no_duplicate_runs": len(set(actual_ids)) == len(actual_ids),
        "exactly_three_seeds_per_method_depth": all(
            sorted(row["seed"] for row in rows if row["method"] == method and row["depth"] == depth)
            == [2601,2602,2603]
            for method in FINAL_METHODS for depth in (1,2,3,4)
        ),
        "all_budgets_at_most_150": all(row["optimizer_evaluations"] <= 150 for row in rows),
        "all_probability_vectors_normalized": all(abs(row["probability_sum"] - 1.0) <= tolerance for row in rows),
        "all_p_feas_within_1e_12": all(abs(row["p_feas"] - 1.0) <= tolerance for row in rows),
        "strict_threshold_unchanged": all(
            row["incumbent_raw_cost_threshold"] == threshold.incumbent_raw_cost
            and json.loads(row["better_route_ids"]) == list(threshold.better_route_ids)
            for row in rows
        ) and np.array_equal(np.asarray(threshold.better_mask), better_expected),
        "no_optimum_identity_in_optimizer": phase_and_loss_no_optimum,
        "configuration_unchanged": sha256_file(CONFIG_PATH) == config_sha,
        "source_bundle_unchanged": sha256_bytes(canonical_json_bytes(identity_entries(SOURCE_PATHS))) == source_sha,
        "graph_unchanged": sha256_file(PROJECT / config["graph_path"]) == graph_sha,
    }
    old_after = verify_old_seals()
    checks["old_q2f_untouched"] = old_after["q2f"] == old_before["q2f"]
    checks["old_q2r_untouched"] = old_after["q2r"] == old_before["q2r"]
    return {
        "schema": "dtu-sciqis-q2f-final-improvement-validation",
        "version": "1.0",
        "validated_at_utc": now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run_count": len(rows),
        "method_counts": {method: sum(row["method"] == method for row in rows) for method in FINAL_METHODS},
        "p_feas_min": min(row["p_feas"] for row in rows),
        "p_feas_max": max(row["p_feas"] for row in rows),
        "optimizer_evaluations_max": max(row["optimizer_evaluations"] for row in rows),
        "old_seals_before": old_before,
        "old_seals_after": old_after,
    }


def result_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == root / "hashes" / "result_manifest.json":
            continue
        files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "dtu-sciqis-q2f-final-improvement-result-manifest",
        "version": "1.0",
        "created_at_utc": now(),
        "manifest_scope": "every file in result root except this self-referential manifest",
        "file_count": len(files),
        "files": files,
    }


def execute(preflight_only: bool) -> None:
    old_before = verify_old_seals()
    config = load_and_validate_config()
    graph_path, basis, costs, path_mixer, grover, uniform, biased, threshold = build_study(config)
    graph_sha = sha256_file(graph_path)
    config_sha = sha256_file(CONFIG_PATH)
    source_entries = identity_entries(SOURCE_PATHS)
    source_sha = sha256_bytes(canonical_json_bytes(source_entries))
    identity = run_identity(graph_sha, source_sha, config_sha)
    root = RESULT_PARENT / identity
    optimum_coincidence = threshold.better_route_ids == (basis.exact_optimal_route_id,)
    preflight = {
        "old_seals": old_before,
        "run_identity": identity,
        "result_root": str(root),
        "basis_size": basis.size,
        "incumbent_route_id": basis.incumbent_route_id,
        "incumbent_route": list(basis.incumbent_route),
        "incumbent_raw_cost": threshold.incumbent_raw_cost,
        "better_route_ids_constructed_without_optimum": list(threshold.better_route_ids),
        "better_route_count": threshold.better_route_count,
        "post_construction_bsp_equals_p_opt_on_instance": optimum_coincidence,
        "path_exchange_count": len(path_mixer.exchanges),
        "config_sha256": config_sha,
        "source_bundle_sha256": source_sha,
        "graph_sha256": graph_sha,
        "result_root_absent": not root.exists(),
    }
    if preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    if root.exists():
        raise FileExistsError(f"final_improvement_identity_already_exists:{root}")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "summary", "figures", "validation", "hashes"):
        (root / name).mkdir()

    started_at = now()
    payloads: list[dict[str, Any]] = []
    active_run: str | None = None
    try:
        atomic_write_json(
            root / "config.json",
            {
                "schema": "dtu-sciqis-q2f-final-improvement-execution-config",
                "version": "1.0",
                "run_identity": identity,
                "identity_scheme": "q2fi- + sha256(graph_sha256 + newline + source_bundle_sha256 + newline + config_sha256 + newline)",
                "graph_sha256": graph_sha,
                "source_bundle_sha256": source_sha,
                "config_sha256": config_sha,
                "frozen_config": config,
                "preflight": preflight,
            },
        )
        atomic_write_json(
            root / "basis.json",
            {
                "basis": basis.as_dict(),
                "cost_hamiltonian": costs.as_dict(),
                "path_exchange_mixer": path_mixer.as_dict(),
                "grover_mixer": grover.as_dict(),
                "incumbent_threshold": threshold.as_dict(),
                "uniform_initial_state": uniform.as_dict(),
                "incumbent_biased_initial_state": biased.as_dict(),
                "post_construction_analysis": {
                    "better_set_equals_exact_optimum_set_on_this_instance": optimum_coincidence,
                    "note": "This comparison occurred only after constructing B from incumbent cost.",
                },
            },
        )
        atomic_write_json(
            root / "source_manifest.json",
            {
                "source_bundle_sha256": source_sha,
                "identity_sources": source_entries,
                "evidence_sources": identity_entries(EVIDENCE_SOURCE_PATHS),
            },
        )

        total = int(config["required_run_count"])
        completed = 0
        for method in FINAL_METHODS:
            for depth in config["depths"]:
                for seed in config["seeds"]:
                    active_run = f"{method}_p{depth}_seed{seed}"
                    if sha256_file(CONFIG_PATH) != config_sha:
                        raise RuntimeError("configuration_changed_during_execution")
                    result = run_final_improvement_cell(
                        basis,
                        costs,
                        path_mixer,
                        grover,
                        biased,
                        uniform,
                        threshold,
                        method=method,
                        depth=depth,
                        seed=seed,
                        evaluation_budget=config["optimizer"]["objective_evaluation_cap"],
                        rhobeg=config["optimizer"]["rhobeg"],
                        tolerance=config["optimizer"]["tolerance"],
                    )
                    payload = {
                        "schema": "dtu-sciqis-q2f-final-improvement-raw-run",
                        "version": "1.0",
                        "completed_at_utc": now(),
                        "identity": {
                            "run_identity": identity,
                            "graph_sha256": graph_sha,
                            "source_bundle_sha256": source_sha,
                            "config_sha256": config_sha,
                            "old_q2f_result_manifest_sha256": Q2F_MANIFEST_SHA256,
                            "old_q2r_result_manifest_sha256": Q2R_MANIFEST_SHA256,
                        },
                        "scientific_identity": {
                            "basis_size": basis.size,
                            "basis_ordering": basis.ordering,
                            "incumbent_route_id": basis.incumbent_route_id,
                            "incumbent_raw_cost": threshold.incumbent_raw_cost,
                            "better_route_ids": list(threshold.better_route_ids),
                            "better_set_definition": "raw_route_cost < incumbent_raw_cost",
                            "optimum_identity_used_by_optimizer": False,
                            "optimum_identity_use": "post-optimization p_opt evaluation only",
                            "normalization": costs.normalization,
                            "parameter_order": "all_gammas_then_all_betas",
                            "layer_order": "phase_then_mixer",
                        },
                        "result": result.as_dict(),
                        "warnings": [],
                    }
                    atomic_write_json(root / "raw" / f"{active_run}.json", payload)
                    payloads.append(payload)
                    completed += 1
                    print(
                        f"[{completed:02d}/{total}] {active_run} "
                        f"evals={result.optimizer.evaluations} p_opt={result.metrics.p_opt:.12g} "
                        f"p_feas={result.metrics.p_feas:.16g}",
                        flush=True,
                    )
                    active_run = None

        rows = [summary_row(payload) for payload in payloads]
        aggregates = aggregate_rows(rows)
        old_rows, old_raw = load_old_expectation()
        improvement_found, selected, strongest = decide(aggregates, config)
        figure_candidate = selected if improvement_found else strongest
        figures = generate_figures(
            root, basis, rows, payloads, aggregates, old_rows, old_raw, figure_candidate
        )
        atomic_write_text(root / "summary" / "q2f_final_improvement_results.csv", csv_text(list(rows[0]), rows))
        atomic_write_json(
            root / "summary" / "q2f_final_improvement_results.json",
            {"schema": "dtu-sciqis-q2f-final-improvement-results", "version": "1.0", "run_identity": identity, "row_count": len(rows), "results": rows},
        )
        atomic_write_text(root / "summary" / "aggregate_by_method_depth.csv", csv_text(list(aggregates[0]), aggregates))
        comparison = {
            "old_q2f_expectation_read_only": {
                f"p{depth}": {
                    "median_p_opt": float(np.median([row["final_p_opt"] for row in old_rows if row["depth"] == depth])),
                    "min_p_opt": float(np.min([row["final_p_opt"] for row in old_rows if row["depth"] == depth])),
                    "max_p_opt": float(np.max([row["final_p_opt"] for row in old_rows if row["depth"] == depth])),
                }
                for depth in (1,2,3)
            },
            "new_aggregates": aggregates,
        }
        atomic_write_json(root / "summary" / "comparison_with_old_q2f.json", comparison)
        relative_improvement = strongest["p_opt_median"] / config["success_rule"]["reference_p_opt"] - 1.0
        decision_payload = {
            "schema": "dtu-sciqis-q2f-final-improvement-decision",
            "version": "1.0",
            "predefined_rule": config["success_rule"],
            "improvement_found": improvement_found,
            "strongest_new_cell": strongest,
            "strongest_relative_improvement": relative_improvement,
            "selected_cell": selected if improvement_found else None,
            "recommendation": "ADOPT_IMPROVED_VARIANT" if improvement_found else "FREEZE_EXISTING_Q2F",
        }
        atomic_write_json(root / "summary" / "decision.json", decision_payload)
        atomic_write_text(
            root / "summary" / "Q2F_FINAL_IMPROVEMENT_RESULTS.md",
            result_report(
                identity, basis, threshold, aggregates, old_rows, improvement_found,
                selected, strongest, rows, figures,
            ),
        )
        validation = validate_results(
            rows, payloads, basis, costs, threshold, config,
            config_sha, source_sha, graph_sha, old_before,
        )
        if validation["status"] != "PASS":
            raise RuntimeError(f"postrun_validation_failed:{validation['checks']}")
        atomic_write_json(root / "validation" / "postrun_validation.json", validation)
        atomic_write_json(
            root / "execution_receipt.json",
            {
                "schema": "dtu-sciqis-q2f-final-improvement-execution-receipt",
                "version": "1.0",
                "run_identity": identity,
                "started_at_utc": started_at,
                "completed_at_utc": now(),
                "run_count": len(rows),
                "decision": decision_payload,
                "validation_status": validation["status"],
                "old_seals": validation["old_seals_after"],
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "matplotlib": matplotlib.__version__,
                },
                "git_head": git("rev-parse", "HEAD"),
                "git_status_porcelain": git("status", "--short"),
                "commit_created": False,
                "push_performed": False,
            },
        )
        manifest = result_manifest(root)
        manifest_path = root / "hashes" / "result_manifest.json"
        atomic_write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        print(
            json.dumps(
                {
                    "run_identity": identity,
                    "result_root": str(root),
                    "run_count": len(rows),
                    "better_route_count": threshold.better_route_count,
                    "strongest_new_cell": strongest,
                    "improvement_found": improvement_found,
                    "selected_cell": selected if improvement_found else None,
                    "final_decision": decision_payload["recommendation"],
                    "p_feas_range": [min(row["p_feas"] for row in rows), max(row["p_feas"] for row in rows)],
                    "optimizer_cap_hits": sum(row["optimizer_reason"] == "evaluation_budget_exhausted" for row in rows),
                    "validation": validation["status"],
                    "result_manifest_sha256": manifest_sha,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        failure = {
            "schema": "dtu-sciqis-q2f-final-improvement-infrastructure-failure",
            "version": "1.0",
            "failed_at_utc": now(),
            "active_run": active_run,
            "completed_scientific_runs": len(payloads),
            "completed_run_ids": [payload["result"]["run_id"] for payload in payloads],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path = root / "validation" / "infrastructure_failure.json"
        if not failure_path.exists():
            atomic_write_json(failure_path, failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    execute(arguments.preflight_only)


if __name__ == "__main__":
    main()
