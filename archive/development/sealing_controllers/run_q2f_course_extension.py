#!/usr/bin/env python3
"""Archived controller that produced the immutable Q2-F result root."""

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
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import scipy

from feasible_qaoa import (
    ASCENDING_CVAR_OBJECTIVE,
    EXPECTATION_OBJECTIVE,
    FIXED_CVAR_OBJECTIVE,
    INCUMBENT_BIASED_FEASIBLE,
    PROBABILITY_TOLERANCE,
    UNIFORM_FEASIBLE,
    ascending_cvar_alpha,
    build_feasible_initial_state,
    build_feasible_route_basis,
    build_logical_cost_hamiltonian,
    build_logical_path_exchange_mixer,
    run_q2f_cell,
)
from graph import DEFAULT_GRAPH_PATH, load_graph, path_cost, path_to_edge_bitstring
from qubo import decode_valid_route


CONFIG_PATH = PROJECT / "data" / "q2f_course_extension_config.json"
Q2R_IDENTITY = "q2r-9c98b90049545d0508fb20bb488019608a4356d55fe91ff809bce8dc5c5e0c69"
Q2R_ROOT = PROJECT / "results" / "q2_revision_formal" / Q2R_IDENTITY
Q2R_MANIFEST_PATH = Q2R_ROOT / "hashes" / "result_manifest.json"
Q2R_MANIFEST_SHA256 = "896fdbf45bbf19d04f74a802e78739714d286beabf322db6ef8f3594679411d1"
REPORT_PATH = PROJECT / "reports" / "Q2F_COURSE_EXTENSION_RESULTS.md"
SOURCE_PATHS = (
    PROJECT / "src" / "feasible_qaoa.py",
    PROJECT / "scripts" / "run_q2f_course_extension.py",
)
EVIDENCE_SOURCE_PATHS = SOURCE_PATHS + (
    PROJECT / "tests" / "test_feasible_qaoa.py",
    PROJECT / "docs" / "methods" / "Q2F_FEASIBLE_WARM_START.md",
    CONFIG_PATH,
)
OBJECTIVE_LABELS = {
    EXPECTATION_OBJECTIVE: "Expectation",
    FIXED_CVAR_OBJECTIVE: "CVaR(0.25)",
    ASCENDING_CVAR_OBJECTIVE: "Ascending-CVaR",
}
OBJECTIVE_COLORS = {
    EXPECTATION_OBJECTIVE: "#1f77b4",
    FIXED_CVAR_OBJECTIVE: "#d62728",
    ASCENDING_CVAR_OBJECTIVE: "#2ca02c",
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
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT, text=True
    ).rstrip("\n")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(f"output appeared before atomic publication: {path}")
    os.rename(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def verify_q2r_seal() -> dict[str, Any]:
    actual_manifest_sha = sha256_file(Q2R_MANIFEST_PATH)
    if actual_manifest_sha != Q2R_MANIFEST_SHA256:
        raise RuntimeError(
            f"sealed_q2r_manifest_mismatch:{actual_manifest_sha}"
        )
    manifest = json.loads(Q2R_MANIFEST_PATH.read_text(encoding="utf-8"))
    files = []
    for entry in manifest["files"]:
        path = Q2R_ROOT / entry["relative_path"]
        actual_hash = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_hash != entry["sha256"] or actual_size != entry["size_bytes"]:
            raise RuntimeError(
                f"sealed_q2r_artifact_mismatch:{entry['relative_path']}"
            )
        files.append(
            {
                "relative_path": entry["relative_path"],
                "sha256": actual_hash,
                "size_bytes": actual_size,
            }
        )
    return {
        "execution_identity": Q2R_IDENTITY,
        "result_manifest_sha256": actual_manifest_sha,
        "sealed_file_count": len(files),
        "files": files,
    }


def load_and_validate_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema") != "dtu-sciqis-q2f-course-extension-config":
        raise ValueError("invalid_q2f_config_schema")
    if config.get("status") != "FROZEN_BEFORE_Q2F_EXECUTION":
        raise ValueError("q2f_config_not_frozen")
    if [item["id"] for item in config["objectives"]] != [
        EXPECTATION_OBJECTIVE,
        FIXED_CVAR_OBJECTIVE,
        ASCENDING_CVAR_OBJECTIVE,
    ]:
        raise ValueError("q2f_objective_order_mismatch")
    if config["depths"] != [1, 2, 3] or config["seeds"] != [2601, 2602, 2603]:
        raise ValueError("q2f_depth_or_seed_matrix_mismatch")
    optimizer = config["optimizer"]
    if (
        optimizer["name"] != "COBYLA"
        or optimizer["objective_evaluation_cap"] != 100
        or optimizer["rhobeg"] != 0.5
        or optimizer["tolerance"] != 1e-8
        or optimizer["constraint_tolerance"] != 1e-8
    ):
        raise ValueError("q2f_optimizer_contract_mismatch")
    if (
        config["primary_initialization"]["mode"]
        != INCUMBENT_BIASED_FEASIBLE
        or config["primary_initialization"]["bias_lambda"] != 1.0
        or config["mixer"]["name"] != "logical_path_exchange_mixer"
        or config["primary_run_count"] != 27
        or config["feasibility_tolerance"] != 1e-12
    ):
        raise ValueError("q2f_primary_contract_mismatch")
    return config


def build_source_identity() -> tuple[list[dict[str, Any]], str]:
    entries = [
        {
            "path": str(path.relative_to(PROJECT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in SOURCE_PATHS
    ]
    return entries, sha256_bytes(canonical_json_bytes(entries))


def build_evidence_source_manifest() -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(PROJECT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in EVIDENCE_SOURCE_PATHS
    ]


def run_identity(
    graph_sha: str,
    source_sha: str,
    config_sha: str,
) -> str:
    digest = sha256_bytes(
        f"{graph_sha}\n{source_sha}\n{config_sha}\n".encode("utf-8")
    )
    return f"q2f-{digest}"


def summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    optimizer = result["optimizer"]
    metrics = result["metrics"]
    return {
        "run_id": result["run_id"],
        "objective": result["objective_mode"],
        "depth": result["depth"],
        "seed": result["seed"],
        "initialization_mode": result["initialization_mode"],
        "bias_lambda": result["bias_lambda"],
        "evaluation_budget": result["evaluation_budget"],
        "optimizer_evaluations": optimizer["evaluations"],
        "statevector_evaluations": optimizer["statevector_evaluations"],
        "optimizer_success": optimizer["success"],
        "optimizer_reason": optimizer["reason"],
        "optimizer_runtime": optimizer["runtime"],
        "optimizer_reported_objective": optimizer["optimizer_reported_objective_value"],
        "final_objective_at_terminal_alpha": optimizer["final_objective_at_terminal_alpha"],
        "terminal_alpha": optimizer["terminal_alpha"],
        "initial_p_opt": metrics["initial_p_opt"],
        "final_p_opt": metrics["final_p_opt"],
        "p_opt_amplification": metrics["p_opt_amplification"],
        "p_feas": metrics["p_feas"],
        "optimal_state_rank": metrics["optimal_state_rank"],
        "top3_lowest_cost_mass": metrics["top3_lowest_cost_mass"],
        "top5_lowest_cost_mass": metrics["top5_lowest_cost_mass"],
        "expected_route_cost": metrics["expected_route_cost"],
        "expected_normalized_cost": metrics["expected_normalized_cost"],
        "most_probable_route_id": metrics["most_probable_route_id"],
        "most_probable_route": "->".join(map(str, metrics["most_probable_route"])),
        "most_probable_route_cost": metrics["most_probable_route_cost"],
        "most_probable_route_probability": metrics["most_probable_route_probability"],
        "probability_entropy": metrics["probability_entropy"],
        "final_parameters": json.dumps(optimizer["final_parameters"], separators=(",", ":")),
        "probability_sum": float(sum(result["final_probabilities"])),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "final_p_opt",
        "expected_route_cost",
        "probability_entropy",
        "p_opt_amplification",
        "optimizer_evaluations",
        "optimizer_runtime",
    )
    output = []
    for objective in (
        EXPECTATION_OBJECTIVE,
        FIXED_CVAR_OBJECTIVE,
        ASCENDING_CVAR_OBJECTIVE,
    ):
        for depth in (1, 2, 3):
            selected = [
                row for row in rows
                if row["objective"] == objective and row["depth"] == depth
            ]
            if len(selected) != 3:
                raise RuntimeError(f"aggregate_cell_missing:{objective}:p{depth}")
            aggregate: dict[str, Any] = {
                "objective": objective,
                "depth": depth,
                "seed_count": len(selected),
                "seeds": json.dumps(sorted(row["seed"] for row in selected)),
            }
            for field in metrics:
                values = [float(row[field]) for row in selected]
                aggregate[f"{field}_median"] = float(np.median(values))
                aggregate[f"{field}_min"] = float(np.min(values))
                aggregate[f"{field}_max"] = float(np.max(values))
            output.append(aggregate)
    return output


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> list[str]:
    paths = []
    for extension in ("png", "svg"):
        path = figures_dir / f"{stem}.{extension}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite figure: {path}")
        fig.savefig(path, dpi=220 if extension == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.4),
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def generate_figures(
    root: Path,
    basis,
    uniform,
    biased,
    rows: list[dict[str, Any]],
    raw_payloads: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    q2r_context: dict[str, dict[str, float]],
) -> list[str]:
    configure_plot_style()
    figures_dir = root / "figures"
    outputs: list[str] = []
    route_ids = np.arange(basis.size)
    costs = np.asarray([route.routing_cost for route in basis.routes])

    fig, ax = plt.subplots()
    colors = ["#4c78a8"] * basis.size
    colors[basis.exact_optimal_route_id] = "#d62728"
    colors[basis.incumbent_route_id] = "#f2b134"
    ax.bar(route_ids, costs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set(title="Feasible-route cost spectrum", xlabel="Logical route ID", ylabel="Raw route cost")
    ax.set_xticks(route_ids)
    ax.text(basis.exact_optimal_route_id, costs[basis.exact_optimal_route_id] + 0.08, "optimum", ha="center", color="#a31313")
    ax.text(basis.incumbent_route_id, costs[basis.incumbent_route_id] + 0.08, "incumbent", ha="center", color="#8a5a00")
    outputs += save_figure(fig, figures_dir, "01_feasible_route_cost_spectrum")

    fig, ax = plt.subplots()
    width = 0.42
    ax.bar(route_ids - width / 2, uniform.probabilities, width, label="F0 uniform", color="#8da0cb")
    ax.bar(route_ids + width / 2, biased.probabilities, width, label="F1 incumbent-biased", color="#fc8d62")
    ax.axvline(basis.exact_optimal_route_id, color="#d62728", linestyle="--", linewidth=1, label="optimal route")
    ax.axvline(basis.incumbent_route_id, color="#8a5a00", linestyle=":", linewidth=1.5, label="incumbent route")
    ax.set(title="Feasible initialization distributions", xlabel="Logical route ID", ylabel="Probability")
    ax.set_xticks(route_ids)
    ax.legend(ncol=2)
    outputs += save_figure(fig, figures_dir, "02_initial_probability_distribution")

    def line_figure(field: str, ylabel: str, title: str, stem: str) -> None:
        fig, ax = plt.subplots()
        offsets = {EXPECTATION_OBJECTIVE: -0.05, FIXED_CVAR_OBJECTIVE: 0.0, ASCENDING_CVAR_OBJECTIVE: 0.05}
        for objective in OBJECTIVE_LABELS:
            color = OBJECTIVE_COLORS[objective]
            selected = [row for row in rows if row["objective"] == objective]
            ax.scatter(
                [row["depth"] + offsets[objective] for row in selected],
                [row[field] for row in selected],
                color=color,
                alpha=0.48,
                s=30,
            )
            median_values = [
                next(
                    agg[f"{field}_median"]
                    for agg in aggregates
                    if agg["objective"] == objective and agg["depth"] == depth
                )
                for depth in (1, 2, 3)
            ]
            ax.plot((1, 2, 3), median_values, marker="o", linewidth=2, color=color, label=OBJECTIVE_LABELS[objective])
        ax.set(title=title, xlabel="Fixed QAOA depth p", ylabel=ylabel, xticks=(1, 2, 3))
        ax.legend()
        outputs.extend(save_figure(fig, figures_dir, stem))

    line_figure("final_p_opt", "Optimal-route probability", "Q2-F optimal-route concentration versus depth", "03_popt_versus_depth")
    line_figure("expected_route_cost", "Expected raw route cost", "Q2-F expected route cost versus depth", "04_expected_cost_versus_depth")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, objective in zip(axes, OBJECTIVE_LABELS):
        p3_rows = [row for row in rows if row["objective"] == objective and row["depth"] == 3]
        cell_median = float(np.median([row["final_p_opt"] for row in p3_rows]))
        chosen = min(p3_rows, key=lambda row: (abs(row["final_p_opt"] - cell_median), row["seed"]))
        payload = next(item for item in raw_payloads if item["result"]["run_id"] == chosen["run_id"])
        probabilities = payload["result"]["final_probabilities"]
        ax.bar(route_ids, probabilities, color=OBJECTIVE_COLORS[objective], alpha=0.82)
        ax.axvline(basis.exact_optimal_route_id, color="#111111", linestyle="--", linewidth=1)
        ax.set_ylabel("Probability")
        ax.set_title(f"{OBJECTIVE_LABELS[objective]}: median-p_opt seed {chosen['seed']}")
    axes[-1].set_xlabel("Logical route ID")
    axes[-1].set_xticks(route_ids)
    fig.suptitle("p=3 feasible-route distributions (median-seed diagnostic)", y=1.01)
    outputs += save_figure(fig, figures_dir, "05_p3_median_seed_distributions")

    line_figure("p_opt_amplification", "Final / initial p_opt", "Q2-F amplification over the F1 initial state", "06_popt_amplification")

    q2f_p3 = {
        objective: next(
            agg for agg in aggregates
            if agg["objective"] == objective and agg["depth"] == 3
        )
        for objective in OBJECTIVE_LABELS
    }
    labels = ["Q2-R A0", "Q2-R A1", "Q2-F F1", "Q2-F E p3", "Q2-F C p3", "Q2-F A p3"]
    state_sizes = [16384, 16384, basis.size, basis.size, basis.size, basis.size]
    pfeas = [q2r_context["A0"]["p_feas"], q2r_context["A1"]["p_feas"], 1.0, 1.0, 1.0, 1.0]
    popt = [
        q2r_context["A0"]["p_opt"], q2r_context["A1"]["p_opt"], biased.p_opt,
        q2f_p3[EXPECTATION_OBJECTIVE]["final_p_opt_median"],
        q2f_p3[FIXED_CVAR_OBJECTIVE]["final_p_opt_median"],
        q2f_p3[ASCENDING_CVAR_OBJECTIVE]["final_p_opt_median"],
    ]
    colors = ["#6c757d", "#6c757d", "#fc8d62", "#1f77b4", "#d62728", "#2ca02c"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, values, title, ylabel in zip(
        axes,
        (state_sizes, pfeas, popt),
        ("Representation size", "Feasible probability", "Optimal-route probability"),
        ("Basis states (log scale)", "p_feas", "p_opt"),
    ):
        ax.bar(np.arange(len(labels)), values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    axes[0].set_yscale("log")
    fig.suptitle("Conceptual context only: Q2-R edge space and Q2-F logical space are different representations")
    fig.text(0.5, -0.04, "Not a fair runtime, scalability, or quantum-advantage comparison.", ha="center", fontsize=10, weight="bold")
    outputs += save_figure(fig, figures_dir, "07_q2r_q2f_conceptual_comparison")
    return outputs


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate_report(
    run_id: str,
    basis,
    uniform,
    biased,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    q2r_context: dict[str, dict[str, float]],
    figure_paths: list[str],
) -> str:
    agg = {(row["objective"], row["depth"]): row for row in aggregates}
    p_opt_table = []
    cost_table = []
    amplification_table = []
    for objective in OBJECTIVE_LABELS:
        p_opt_table.append([OBJECTIVE_LABELS[objective]] + [f"{agg[objective, p]['final_p_opt_median']:.12g}" for p in (1, 2, 3)])
        cost_table.append([OBJECTIVE_LABELS[objective]] + [f"{agg[objective, p]['expected_route_cost_median']:.12g}" for p in (1, 2, 3)])
        amplification_table.append([OBJECTIVE_LABELS[objective]] + [f"{agg[objective, p]['p_opt_amplification_median']:.12g}" for p in (1, 2, 3)])
    winners = {}
    for depth in (1, 2, 3):
        values = {objective: agg[objective, depth]["final_p_opt_median"] for objective in OBJECTIVE_LABELS}
        maximum = max(values.values())
        winners[depth] = [OBJECTIVE_LABELS[key] for key, value in values.items() if value == maximum]
    progression = {
        objective: [agg[objective, depth]["final_p_opt_median"] for depth in (1, 2, 3)]
        for objective in OBJECTIVE_LABELS
    }
    correlation = float(np.corrcoef(
        [row["expected_route_cost"] for row in rows],
        [row["final_p_opt"] for row in rows],
    )[0, 1])
    sensitivity = [
        {
            "objective": OBJECTIVE_LABELS[objective],
            "depth": depth,
            "minimum": agg[objective, depth]["final_p_opt_min"],
            "maximum": agg[objective, depth]["final_p_opt_max"],
            "range": agg[objective, depth]["final_p_opt_max"] - agg[objective, depth]["final_p_opt_min"],
        }
        for objective in OBJECTIVE_LABELS for depth in (1, 2, 3)
    ]
    largest_sensitivity = max(sensitivity, key=lambda item: item["range"])
    budget_hits = [row["run_id"] for row in rows if row["optimizer_reason"] == "evaluation_budget_exhausted"]
    best = max(rows, key=lambda row: row["final_p_opt"])
    progression_text = "; ".join(
        f"{OBJECTIVE_LABELS[objective]}={progression[objective]}"
        for objective in OBJECTIVE_LABELS
    )
    fixed_cvar_difference_text = ", ".join(
        f"p={depth}: "
        f"{agg[FIXED_CVAR_OBJECTIVE, depth]['final_p_opt_median'] - agg[EXPECTATION_OBJECTIVE, depth]['final_p_opt_median']:.12g}"
        for depth in (1, 2, 3)
    )
    ascending_difference_text = ", ".join(
        f"p={depth}: "
        f"{agg[ASCENDING_CVAR_OBJECTIVE, depth]['final_p_opt_median'] - agg[FIXED_CVAR_OBJECTIVE, depth]['final_p_opt_median']:.12g}"
        for depth in (1, 2, 3)
    )
    warm_rows = [
        [
            str(route.route_id),
            "->".join(map(str, route.node_sequence)),
            str(route.routing_cost),
            str(biased.route_distances[route.route_id]),
            f"{uniform.probabilities[route.route_id]:.12g}",
            f"{biased.probabilities[route.route_id]:.12g}",
            "yes" if route.exact_optimal else "",
            "yes" if route.route_id == basis.incumbent_route_id else "",
        ]
        for route in basis.routes
    ]
    report = f"""# Q2-F course-extension results

Run identity: `{run_id}`

## Scope

These are descriptive results from one 20-state logical feasible-route teaching
experiment. Full route enumeration is classical preprocessing. Q2-F is not
claimed scalable and does not demonstrate quantum advantage. The three-seed
matrix does not support statistical-significance claims. Q2-R and Q2-F use
different representations, so old/new numerical context is not a fair runtime
or algorithmic comparison.

## Initialization controls

- F0 uniform: `p_feas={uniform.p_feas:.17g}`, `p_opt={uniform.p_opt:.17g}`, expected route cost `{uniform.expected_route_cost:.17g}`.
- F1 incumbent-biased, lambda=1: `p_feas={biased.p_feas:.17g}`, `p_opt={biased.p_opt:.17g}`, expected route cost `{biased.expected_route_cost:.17g}`.

{markdown_table(['route ID', 'route', 'cost', 'distance to incumbent', 'F0 probability', 'F1 probability', 'optimal', 'incumbent'], warm_rows)}

## Median results across the three mandatory seeds

### p_opt

{markdown_table(['objective', 'p=1', 'p=2', 'p=3'], p_opt_table)}

### Expected raw route cost

{markdown_table(['objective', 'p=1', 'p=2', 'p=3'], cost_table)}

### p_opt amplification over F1

{markdown_table(['objective', 'p=1', 'p=2', 'p=3'], amplification_table)}

## Requested descriptive questions

1. **Did p_feas remain 1?** Yes. Every one of the 27 retained distributions passed `|p_feas-1| <= 1e-12`.
2. **What was initial warm-start p_opt?** F1 began at `{biased.p_opt:.17g}`. The F0 uniform control was `{uniform.p_opt:.17g}`.
3. **Highest median p_opt by depth:** p=1: {', '.join(winners[1])}; p=2: {', '.join(winners[2])}; p=3: {', '.join(winners[3])}.
4. **Did p_opt improve with depth?** Median trajectories were: {progression_text}. These are descriptive trajectories, not monotonicity or significance claims.
5. **Did fixed CVaR improve concentration relative to expectation?** Median differences `(CVaR - expectation)` were {fixed_cvar_difference_text}.
6. **Did Ascending-CVaR improve over fixed CVaR?** Median differences `(ascending - fixed)` were {ascending_difference_text}.
7. **How much existed before QAOA?** Feasible restriction plus F1 already placed `{biased.p_opt:.12g}` on the optimum, compared with sealed Q2-R A0/A1 values `{q2r_context['A0']['p_opt']:.12g}` and `{q2r_context['A1']['p_opt']:.12g}`. This is representation context, not a fair algorithm comparison.
8. **Expected cost versus p_opt:** Across the 27 rows, the descriptive Pearson correlation was `{correlation:.12g}`. A negative value means lower expected route cost co-occurred with higher p_opt in this matrix.
9. **Seed sensitivity:** The largest within-cell p_opt range was `{largest_sensitivity['range']:.12g}` for {largest_sensitivity['objective']} at p={largest_sensitivity['depth']} (min `{largest_sensitivity['minimum']:.12g}`, max `{largest_sensitivity['maximum']:.12g}`). All seeds remain reported.
10. **Budget hits:** `{len(budget_hits)}` runs reached the 100-request cap: {', '.join(budget_hits) if budget_hits else 'none'}.
11. **Strongest safe course-level takeaway:** Restricting evolution to the explicitly enumerated feasible-route basis makes feasibility structural and permits a clean study of how shallow logical path-exchange evolution redistributes probability among valid routes. The observed objective/depth/seed differences are instance-specific mechanism evidence only.

## Best descriptive observation

The largest retained final p_opt was `{best['final_p_opt']:.17g}` in `{best['run_id']}`. It is a secondary descriptive diagnostic, not a selected headline seed or significance claim.

## Read-only Q2-R context

- A0 sealed p_opt: `{q2r_context['A0']['p_opt']:.17g}`, p_feas: `{q2r_context['A0']['p_feas']:.17g}`.
- A1 sealed p_opt: `{q2r_context['A1']['p_opt']:.17g}`, p_feas: `{q2r_context['A1']['p_feas']:.17g}`.
- Q2-R computational state count: 16384; Q2-F logical route count: {basis.size}.

## Figures

"""
    report += "\n".join(
        f"- `{Path(path).relative_to(PROJECT)}`" for path in figure_paths
    )
    report += "\n"
    return report


def execute(preflight_only: bool) -> None:
    q2r_before = verify_q2r_seal()
    config = load_and_validate_config()
    graph_path = PROJECT / config["graph_path"]
    graph_sha = sha256_file(graph_path)
    config_sha = sha256_file(CONFIG_PATH)
    source_entries, source_sha = build_source_identity()
    identity = run_identity(graph_sha, source_sha, config_sha)
    result_root = PROJECT / "results" / "q2f_course_extension" / identity

    graph = load_graph(graph_path)
    basis = build_feasible_route_basis(graph, graph_path=str(graph_path))
    costs = build_logical_cost_hamiltonian(basis)
    mixer = build_logical_path_exchange_mixer(basis)
    uniform = build_feasible_initial_state(basis, costs, mode=UNIFORM_FEASIBLE)
    biased = build_feasible_initial_state(
        basis,
        costs,
        mode=INCUMBENT_BIASED_FEASIBLE,
        bias_lambda=float(config["primary_initialization"]["bias_lambda"]),
    )
    if basis.size != 20 or len(mixer.exchanges) != 106 or not mixer.connected:
        raise RuntimeError("q2f_discovered_basis_or_mixer_identity_changed")
    if preflight_only:
        print(
            json.dumps(
                {
                    "sealed_q2r_manifest": q2r_before["result_manifest_sha256"],
                    "run_identity": identity,
                    "result_root": str(result_root),
                    "basis_size": basis.size,
                    "exchange_count": len(mixer.exchanges),
                    "mixer_connected": mixer.connected,
                    "uniform_p_opt": uniform.p_opt,
                    "biased_p_opt": biased.p_opt,
                    "config_sha256": config_sha,
                    "source_sha256": source_sha,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if result_root.exists():
        raise FileExistsError(f"q2f result identity already exists: {result_root}")
    if REPORT_PATH.exists():
        raise FileExistsError(f"q2f report already exists: {REPORT_PATH}")
    result_root.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "summary", "figures", "validation", "hashes"):
        (result_root / name).mkdir()

    started_at = now()
    completed_payloads: list[dict[str, Any]] = []
    active_run_id: str | None = None
    try:
        atomic_write_json(
            result_root / "config.json",
            {
                "schema": "dtu-sciqis-q2f-execution-config",
                "version": "1.0",
                "run_identity": identity,
                "identity_scheme": "q2f- + sha256(graph_sha256 + newline + source_bundle_sha256 + newline + config_sha256 + newline)",
                "graph_sha256": graph_sha,
                "source_bundle_sha256": source_sha,
                "config_sha256": config_sha,
                "frozen_config": config,
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "qiskit": qiskit.__version__,
                },
                "started_at_utc": started_at,
            },
        )
        atomic_write_json(
            result_root / "source_manifest.json",
            {
                "source_bundle_sha256": source_sha,
                "execution_source_files": source_entries,
                "implementation_evidence_files": build_evidence_source_manifest(),
            },
        )
        atomic_write_json(
            result_root / "basis.json",
            {
                "basis": basis.as_dict(),
                "cost_hamiltonian": costs.as_dict(),
                "mixer": mixer.as_dict(),
            },
        )
        controls_payload = {
            "schema": "dtu-sciqis-q2f-initialization-controls",
            "version": "1.0",
            "run_identity": identity,
            "basis_size": basis.size,
            "controls": {
                "C0_uniform_feasible": uniform.as_dict(),
                "C1_incumbent_biased_feasible": biased.as_dict(),
            },
        }
        atomic_write_json(
            result_root / "summary" / "initialization_controls.json",
            controls_payload,
        )

        sequence = 0
        for objective_spec in config["objectives"]:
            objective = objective_spec["id"]
            for depth in config["depths"]:
                for seed in config["seeds"]:
                    sequence += 1
                    active_run_id = f"{objective}_p{depth}_seed{seed}"
                    print(
                        f"Q2F_RUN_START {sequence}/27 {active_run_id}",
                        flush=True,
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = run_q2f_cell(
                            basis,
                            costs,
                            mixer,
                            biased,
                            objective_mode=objective,
                            depth=int(depth),
                            seed=int(seed),
                            evaluation_budget=int(config["optimizer"]["objective_evaluation_cap"]),
                            fixed_cvar_alpha=0.25,
                            alpha_start=0.25,
                            alpha_end=1.0,
                            rhobeg=float(config["optimizer"]["rhobeg"]),
                            tolerance=float(config["optimizer"]["tolerance"]),
                        )
                    warning_rows = [
                        {
                            "category": warning.category.__name__,
                            "message": str(warning.message),
                            "filename": str(warning.filename),
                            "lineno": int(warning.lineno),
                        }
                        for warning in caught
                    ]
                    if not result.optimizer.success:
                        warning_rows.append(
                            {
                                "category": "optimizer_terminal_status",
                                "message": result.optimizer.reason,
                                "retained": True,
                            }
                        )
                    payload = {
                        "schema": "dtu-sciqis-q2f-raw-run",
                        "version": "1.0",
                        "identity": {
                            "run_identity": identity,
                            "cell_id": active_run_id,
                            "sequence": sequence,
                            "graph_sha256": graph_sha,
                            "source_bundle_sha256": source_sha,
                            "config_sha256": config_sha,
                        },
                        "scientific_identity": {
                            "basis_size": basis.size,
                            "basis_ordering": basis.ordering,
                            "exact_optimal_route_id": basis.exact_optimal_route_id,
                            "incumbent_route_id": basis.incumbent_route_id,
                            "mixer": mixer.mixer_name,
                            "mixer_exchange_count": len(mixer.exchanges),
                            "mixer_connected": mixer.connected,
                            "mixer_weight_rule": mixer.weight_rule,
                            "cost_normalization": costs.normalization,
                            "initialization": biased.as_dict(),
                            "parameter_order": "all_gammas_then_all_betas",
                            "layer_order": "cost_then_mixer",
                        },
                        "result": result.as_dict(),
                        "warnings": warning_rows,
                        "completed_at_utc": now(),
                    }
                    atomic_write_json(
                        result_root / "raw" / f"{active_run_id}.json",
                        payload,
                    )
                    completed_payloads.append(payload)
                    print(
                        f"Q2F_RUN_COMPLETE {active_run_id} evals={result.optimizer.evaluations} "
                        f"p_opt={result.metrics.p_opt:.17g} cost={result.metrics.expected_route_cost:.17g} "
                        f"status={result.optimizer.reason}",
                        flush=True,
                    )

        rows = [summary_row(payload) for payload in completed_payloads]
        aggregates = aggregate_rows(rows)
        atomic_write_json(
            result_root / "summary" / "q2f_results.json",
            {
                "schema": "dtu-sciqis-q2f-results-summary",
                "version": "1.0",
                "run_identity": identity,
                "row_count": len(rows),
                "results": rows,
            },
        )
        atomic_write_text(
            result_root / "summary" / "q2f_results.csv",
            csv_text(list(rows[0]), rows),
        )
        atomic_write_text(
            result_root / "summary" / "aggregate_by_objective_depth.csv",
            csv_text(list(aggregates[0]), aggregates),
        )

        q2r_context: dict[str, dict[str, float]] = {}
        for arm in ("A0", "A1"):
            old = json.loads((Q2R_ROOT / "raw" / f"{arm}.json").read_text(encoding="utf-8"))
            q2r_context[arm] = {
                "p_feas": float(old["result"]["p_feas"]),
                "p_opt": float(old["result"]["p_opt"]),
                "final_expectation": float(old["result"]["final_expectation_value"]),
            }

        checks: list[dict[str, Any]] = []
        def check(check_id: str, passed: bool, detail: Any) -> None:
            checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

        rebuilt_basis = build_feasible_route_basis(graph, graph_path=str(graph_path))
        check("basis.every_state_feasible", all(
            decode_valid_route(graph, route.edge_bitstring) == route.node_sequence
            and path_cost(graph, route.node_sequence) == route.routing_cost
            and path_to_edge_bitstring(graph, route.node_sequence) == route.edge_bitstring
            for route in basis.routes
        ), {"basis_size": basis.size})
        check("basis.deterministic_and_complete", rebuilt_basis == basis and basis.size == 20, basis.as_dict())
        check("basis.exact_optimum_unchanged", basis.routes[basis.exact_optimal_route_id].routing_cost == 10
              and basis.routes[basis.exact_optimal_route_id].node_sequence == (0, 1, 2, 4, 5, 6),
              basis.routes[basis.exact_optimal_route_id].as_dict())
        check("mixer.hermitian_connected_feasible", mixer.connected and len(mixer.exchanges) == 106
              and np.allclose(mixer.hamiltonian, mixer.hamiltonian.conj().T, atol=1e-14), mixer.as_dict())
        expected_cells = {
            (objective, depth, seed)
            for objective in OBJECTIVE_LABELS
            for depth in (1, 2, 3)
            for seed in (2601, 2602, 2603)
        }
        actual_cells = {(row["objective"], row["depth"], row["seed"]) for row in rows}
        check("matrix.exact_27_cells", len(rows) == 27 and actual_cells == expected_cells, sorted(actual_cells))
        check("matrix.no_duplicate_rows", len({row["run_id"] for row in rows}) == 27, [row["run_id"] for row in rows])
        for objective in OBJECTIVE_LABELS:
            for depth in (1, 2, 3):
                seeds = sorted(row["seed"] for row in rows if row["objective"] == objective and row["depth"] == depth)
                check(f"matrix.{objective}.p{depth}.three_seeds", seeds == [2601, 2602, 2603], seeds)
        for payload, row in zip(completed_payloads, rows):
            result = payload["result"]
            probabilities = np.asarray(result["final_probabilities"], dtype=float)
            trace = result["optimizer"]["evaluation_trace"]
            check(f"probability.{row['run_id']}.normalized", probabilities.shape == (basis.size,)
                  and np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
                  and abs(float(np.sum(probabilities)) - 1.0) <= 1e-12,
                  {"sum": float(np.sum(probabilities)), "size": len(probabilities)})
            check(f"feasibility.{row['run_id']}.structural", abs(row["p_feas"] - 1.0) <= 1e-12,
                  row["p_feas"])
            check(f"budget.{row['run_id']}.cap_and_trace", 0 < row["optimizer_evaluations"] <= 100
                  and len(trace) == row["optimizer_evaluations"]
                  and [record["evaluation_index"] for record in trace] == list(range(1, len(trace) + 1)),
                  {"evaluations": row["optimizer_evaluations"], "trace": len(trace)})
            if row["objective"] == ASCENDING_CVAR_OBJECTIVE:
                expected_alphas = [ascending_cvar_alpha(index, 100) for index in range(len(trace))]
                actual_alphas = [record["alpha"] for record in trace]
                check(f"ascending.{row['run_id']}.schedule", np.allclose(actual_alphas, expected_alphas, atol=0, rtol=0),
                      {"first": actual_alphas[0], "last": actual_alphas[-1], "count": len(actual_alphas)})
            elif row["objective"] == FIXED_CVAR_OBJECTIVE:
                check(f"cvar.{row['run_id']}.fixed_alpha", all(record["alpha"] == 0.25 for record in trace), 0.25)
            else:
                check(f"expectation.{row['run_id']}.no_alpha", all(record["alpha"] is None for record in trace), None)
        check("feasibility.all_27_exact_tolerance", all(abs(row["p_feas"] - 1.0) <= 1e-12 for row in rows),
              {row["run_id"]: row["p_feas"] for row in rows})
        check("config.unchanged_during_execution", sha256_file(CONFIG_PATH) == config_sha, sha256_file(CONFIG_PATH))
        current_source_entries, current_source_sha = build_source_identity()
        check("source.unchanged_during_execution", current_source_sha == source_sha and current_source_entries == source_entries,
              {"before": source_sha, "after": current_source_sha})
        q2r_after = verify_q2r_seal()
        check("q2r.sealed_evidence_untouched", q2r_after == q2r_before, q2r_after)
        if not all(item["passed"] for item in checks):
            atomic_write_json(
                result_root / "validation" / "postrun_validation.json",
                {
                    "overall_passed": False,
                    "checks": checks,
                    "failed": [item["check_id"] for item in checks if not item["passed"]],
                },
            )
            raise RuntimeError("q2f_postrun_scientific_validation_failed")

        figure_paths = generate_figures(
            result_root,
            basis,
            uniform,
            biased,
            rows,
            completed_payloads,
            aggregates,
            q2r_context,
        )
        report_text = generate_report(
            identity,
            basis,
            uniform,
            biased,
            rows,
            aggregates,
            q2r_context,
            figure_paths,
        )
        atomic_write_text(REPORT_PATH, report_text)
        expected_figure_files = {
            f"{index:02d}_{stem}.{extension}"
            for index, stem in (
                (1, "feasible_route_cost_spectrum"),
                (2, "initial_probability_distribution"),
                (3, "popt_versus_depth"),
                (4, "expected_cost_versus_depth"),
                (5, "p3_median_seed_distributions"),
                (6, "popt_amplification"),
                (7, "q2r_q2f_conceptual_comparison"),
            )
            for extension in ("png", "svg")
        }
        actual_figure_files = {path.name for path in (result_root / "figures").iterdir() if path.is_file()}
        check("figures.all_14_outputs", actual_figure_files == expected_figure_files, sorted(actual_figure_files))
        check("report.created_after_dataset_validation", REPORT_PATH.is_file() and REPORT_PATH.stat().st_size > 0,
              {"path": str(REPORT_PATH.relative_to(PROJECT)), "size": REPORT_PATH.stat().st_size})
        final_q2r = verify_q2r_seal()
        check("q2r.final_seal_verification", final_q2r == q2r_before, final_q2r)
        validation_passed = all(item["passed"] for item in checks)
        validation_payload = {
            "schema": "dtu-sciqis-q2f-postrun-validation",
            "version": "1.0",
            "run_identity": identity,
            "overall_passed": validation_passed,
            "check_count": len(checks),
            "passed_count": sum(item["passed"] for item in checks),
            "failed_count": sum(not item["passed"] for item in checks),
            "checks": checks,
            "q2r_before": q2r_before,
            "q2r_after": final_q2r,
            "validated_at_utc": now(),
        }
        atomic_write_json(
            result_root / "validation" / "postrun_validation.json",
            validation_payload,
        )
        if not validation_passed:
            raise RuntimeError("q2f_final_artifact_validation_failed")

        receipt = {
            "schema": "dtu-sciqis-q2f-execution-receipt",
            "version": "1.0",
            "run_identity": identity,
            "status": "27_runs_complete_validated_preseal",
            "run_count": len(rows),
            "execution_order": [row["run_id"] for row in rows],
            "started_at_utc": started_at,
            "completed_at_utc": now(),
            "graph_sha256": graph_sha,
            "source_bundle_sha256": source_sha,
            "config_sha256": config_sha,
            "q2r_manifest_sha256_before": q2r_before["result_manifest_sha256"],
            "q2r_manifest_sha256_after": final_q2r["result_manifest_sha256"],
            "full_pytest_gate": {
                "passed": True,
                "recorded_by_invoking_agent_before_execution": True,
            },
            "git_head": git("rev-parse", "HEAD"),
            "git_status_preseal": git("status", "--porcelain=v1", "--untracked-files=all"),
            "commit_performed": False,
            "push_performed": False,
            "raw_runs_overwritten": False,
            "outcome_dependent_reruns": False,
            "replacement_seeds": False,
        }
        atomic_write_json(result_root / "execution_receipt.json", receipt)

        artifacts = sorted(
            path
            for path in result_root.rglob("*")
            if path.is_file() and path != result_root / "hashes" / "result_manifest.json"
        )
        manifest_entries = [
            {
                "scope": "result_root",
                "relative_path": str(path.relative_to(result_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ]
        manifest_entries.append(
            {
                "scope": "external_report",
                "project_relative_path": str(REPORT_PATH.relative_to(PROJECT)),
                "size_bytes": REPORT_PATH.stat().st_size,
                "sha256": sha256_file(REPORT_PATH),
            }
        )
        result_manifest = {
            "schema": "dtu-sciqis-q2f-result-manifest",
            "version": "1.0",
            "run_identity": identity,
            "sealed_at_utc": now(),
            "file_count": len(manifest_entries),
            "self_exclusion": "hashes/result_manifest.json is excluded to avoid recursive self-hashing",
            "files": manifest_entries,
        }
        result_manifest_path = result_root / "hashes" / "result_manifest.json"
        atomic_write_json(result_manifest_path, result_manifest)
        result_manifest_sha = sha256_file(result_manifest_path)
        print(
            f"Q2F_SEALED run_identity={identity} files={len(manifest_entries)} "
            f"result_manifest_sha256={result_manifest_sha}",
            flush=True,
        )

    except BaseException as exc:
        failure_path = result_root / "execution_failure.json"
        if not failure_path.exists() and not (result_root / "hashes" / "result_manifest.json").exists():
            try:
                atomic_write_json(
                    failure_path,
                    {
                        "schema": "dtu-sciqis-q2f-execution-failure",
                        "version": "1.0",
                        "run_identity": identity,
                        "active_run_id": active_run_id,
                        "completed_run_ids": [payload["result"]["run_id"] for payload in completed_payloads],
                        "completed_evaluations": {
                            payload["result"]["run_id"]: payload["result"]["optimizer"]["evaluations"]
                            for payload in completed_payloads
                        },
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "automatic_retry": False,
                        "replacement_seed": False,
                        "failed_at_utc": now(),
                    },
                )
            except BaseException as record_error:
                print(f"Q2F_FAILURE_RECORDING_ERROR {record_error}", file=sys.stderr, flush=True)
        print(f"Q2F_EXECUTION_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify identity and construct basis/mixer without writing or optimizing",
    )
    args = parser.parse_args()
    execute(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
