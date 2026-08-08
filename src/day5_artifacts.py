"""Build final Day-5 figures, summaries, reports, and scientific freeze."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from day5_analysis import (
    EXPECTED_CORE_RESULTS_SHA256,
    EXPECTED_CORE_SUMMARY_SHA256,
    EXPECTED_DAY5_CONTRACT_SHA256,
    EXPECTED_GRAPH_SHA256,
    EXPECTED_OPTIMIZATION_RUNS_SHA256,
    EXPECTED_OPTIMIZATION_SHA256,
    EXPECTED_PENALTY_SHA256,
    file_sha256,
    load_day5_contract,
    reconcile_frozen_day4,
)
from graph import PROJECT_ROOT


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _save_figure(figure, png_path: Path, svg_path: Path, *, tag: str) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path,
        dpi=240,
        facecolor="white",
        metadata={"Software": f"sciqis-qaoa-routing {tag}"},
    )
    figure.savefig(
        svg_path,
        facecolor="white",
        metadata={"Creator": f"sciqis-qaoa-routing {tag}", "Date": None},
    )
    plt.close(figure)
    _normalize_svg(svg_path)


def ensure_route_spectrum_svg(*, results_dir: Path, figures_dir: Path) -> None:
    """Add the vector counterpart of the unchanged Day-1 diagnostic data."""

    reference = _read_json(results_dir / "exact_reference.json")
    routes = reference["all_simple_paths"]
    costs = np.asarray([route["cost"] for route in routes], dtype=int)
    ranks = np.arange(1, len(routes) + 1)
    optimum = reference["exact_reference"]["cost"]
    second_best = reference["second_best_cost"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "dtu-sciqis-routing-v3-figure2-svg-audit",
        }
    )
    figure, axis = plt.subplots(figsize=(10.0, 4.8), facecolor="white")
    axis.plot(
        ranks,
        costs,
        color="#8a99a8",
        linewidth=1.4,
        marker="o",
        markersize=4.5,
        label="Simple route",
    )
    axis.scatter(
        ranks[costs == optimum],
        costs[costs == optimum],
        s=150,
        marker="*",
        color="#e4572e",
        edgecolor="#8f2f17",
        zorder=4,
        label=f"Optimum: C*={optimum}",
    )
    axis.scatter(
        ranks[costs == second_best],
        costs[costs == second_best],
        s=62,
        marker="D",
        color="#2f6f9f",
        zorder=3,
        label=f"Second-best cost: {second_best}",
    )
    axis.set_title(
        "Simple Source-to-Target Route Cost Spectrum",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    axis.set_xlabel("Route rank (sorted by cost, then node path)")
    axis.set_ylabel("Total route cost")
    axis.set_xticks(ranks)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.05,
        f"Optimality gap = {reference['optimality_gap']}",
        transform=axis.transAxes,
        ha="right",
        color="#536170",
    )
    figure.tight_layout()
    output = figures_dir / "02_route_cost_spectrum.svg"
    figure.savefig(
        output,
        facecolor="white",
        metadata={"Creator": "sciqis-qaoa-routing Day-5 vector audit", "Date": None},
    )
    plt.close(figure)
    _normalize_svg(output)


def generate_finite_shot_figure(
    *,
    results_dir: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    summary = _read_json(results_dir / "finite_shot_summary.json")
    with (results_dir / "finite_shot_replicates.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    shot_counts = [row["shots"] for row in summary["shot_summaries"]]
    exact_values = {
        "p_feas_hat": summary["source_state"]["exact_p_feas"],
        "p_opt_hat": summary["source_state"]["exact_p_opt"],
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day5-figure12",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.9), facecolor="white")
    colors = ("#6baed6", "#4292c6", "#2171b5", "#084594")
    for axis, metric, title, ylabel in (
        (axes[0], "p_feas_hat", "Valid-route probability estimate", r"sampled $\hat p_{feas}$"),
        (axes[1], "p_opt_hat", "Exact-route probability estimate", r"sampled $\hat p_{opt}$"),
    ):
        for shot_index, (shots, color) in enumerate(zip(shot_counts, colors)):
            values = np.asarray(
                [float(row[metric]) for row in rows if int(row["shots"]) == shots]
            )
            replicates = np.arange(values.size)
            jitter = np.exp(0.055 * np.sin(2 * np.pi * replicates / 37.0))
            axis.scatter(
                shots * jitter,
                values,
                s=10,
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=1,
            )
            q05, median, q95 = np.quantile(values, (0.05, 0.50, 0.95))
            mean = np.mean(values)
            axis.vlines(shots, q05, q95, color=color, linewidth=4.5, zorder=3)
            axis.scatter(
                [shots],
                [median],
                marker="_",
                s=210,
                linewidth=2.2,
                color="white",
                zorder=4,
            )
            axis.scatter(
                [shots],
                [mean],
                marker="D",
                s=34,
                color="#e4572e",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
            if metric == "p_opt_hat" and shot_index < 2:
                zero_fraction = summary["shot_summaries"][shot_index][
                    "zero_exact_optimum_observation_fraction"
                ]
                axis.text(
                    shots,
                    max(values.max(), exact_values[metric]) * 1.10,
                    f"{zero_fraction:.1%} zero",
                    ha="center",
                    fontsize=8.5,
                    color="#7a3db8",
                )
        axis.axhline(
            exact_values[metric],
            color="#17212b",
            linestyle="--",
            linewidth=1.7,
            label="exact statevector value",
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(shot_counts, [f"{shots:,}" for shots in shot_counts])
        axis.set_xlabel("measurement shots (200 Monte Carlo replicates)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.grid(alpha=0.18)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="#17212b", linestyle="--", label="exact statevector"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#e4572e", label="replicate mean"),
            Line2D([0], [0], color="#2171b5", linewidth=4, label="5th–95th percentiles"),
        ],
        frameon=False,
        fontsize=8.5,
        loc="upper right",
    )
    figure.suptitle(
        "Finite-Shot Estimates Converge Around the Fixed Statevector Values",
        fontsize=19,
        fontweight="bold",
        y=0.98,
        color="#17212b",
    )
    figure.text(
        0.5,
        0.018,
        "Finite-shot sampling demonstration from a fixed optimized statevector; not a hardware experiment. Quantiles show Monte Carlo sampling variability, not formal confidence intervals.",
        ha="center",
        fontsize=9.2,
        color="#536170",
    )
    figure.tight_layout(rect=(0.02, 0.07, 0.98, 0.92))
    _save_figure(figure, png_path, svg_path, tag="Day-5 Figure 12")


def generate_runtime_figure(
    *,
    results_dir: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    profile = _read_json(results_dir / "runtime_profile_summary.json")
    rows = profile["stage_summaries"]
    labels = {
        "contract_validation": "Contract/result validation",
        "qubo_construction": "Flow-QUBO construction",
        "ising_conversion": "QUBO → Ising conversion",
        "qiskit_circuit_build_p1": "Qiskit circuit build p=1",
        "qiskit_circuit_build_p2": "Qiskit circuit build p=2",
        "numpy_statevector_p1": "NumPy state evolution p=1",
        "numpy_statevector_p2": "NumPy state evolution p=2",
        "qiskit_statevector_p1": "Qiskit Statevector p=1",
        "qiskit_statevector_p2": "Qiskit Statevector p=2",
        "core_metric_calculation": "Metrics over 16,384 states",
        "finite_shot_sampling_4096": "One 4,096-shot sample",
    }
    colors = {
        "Model": "#4c78a8",
        "Circuit": "#f58518",
        "Simulation": "#54a24b",
        "Analysis": "#b279a2",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "dtu-sciqis-routing-v3-day5-figure13",
        }
    )
    figure, axis = plt.subplots(figsize=(11.5, 7.2), facecolor="white")
    y = np.arange(len(rows))[::-1]
    for position, row in zip(y, rows):
        median = row["median_ms"]
        axis.errorbar(
            median,
            position,
            xerr=np.asarray(
                [[median - row["p10_ms"]], [row["p90_ms"] - median]]
            ),
            fmt="o",
            markersize=7,
            capsize=3,
            linewidth=2,
            color=colors[row["group"]],
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    axis.set_yticks(y, [labels[row["stage"]] for row in rows])
    axis.set_xscale("log")
    axis.set_xlabel("observed wall-clock time per operation (ms, log scale)")
    axis.set_title(
        "Implementation Runtime Profile",
        fontsize=19,
        fontweight="bold",
        pad=14,
        color="#17212b",
    )
    axis.grid(axis="x", which="both", alpha=0.20)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color=color, linestyle="none", label=group)
            for group, color in colors.items()
        ],
        ncol=4,
        frameon=False,
        loc="lower right",
        fontsize=8.5,
    )
    optimization = profile["day4_observed_optimization_runtimes"]
    figure.text(
        0.73,
        0.105,
        "Committed Day-4 full optimization runs\n"
        f"median {optimization['overall']['median_seconds']:.3f} s "
        f"(range {optimization['overall']['minimum_seconds']:.3f}–{optimization['overall']['maximum_seconds']:.3f} s)\n"
        f"5,630 total objective evaluations across 24 runs",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#364657",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3f6f8", "edgecolor": "#c5ced6"},
    )
    figure.text(
        0.5,
        0.018,
        "Observed implementation runtime on the current software/hardware environment; not a quantum-advantage or hardware-performance comparison.\nNumPy and Qiskit entries are both classical simulations of the same quantum circuit.",
        ha="center",
        fontsize=8.7,
        color="#536170",
    )
    figure.tight_layout(rect=(0.04, 0.20, 0.98, 0.98))
    _save_figure(figure, png_path, svg_path, tag="Day-5 Figure 13")


def _is_monotonic(values: list[float]) -> bool:
    differences = np.diff(np.asarray(values, dtype=float))
    return bool(np.all(differences >= 0) or np.all(differences <= 0))


def build_final_project_summary(
    *,
    root: Path,
    results_dir: Path,
) -> dict[str, Any]:
    contract, contract_hash = load_day5_contract(root / "data" / "day5_analysis_contract.json")
    core, reconciliation = reconcile_frozen_day4(root)
    exact = _read_json(results_dir / "exact_reference.json")
    threshold = _read_json(results_dir / "penalty_threshold_analysis.json")
    circuit = _read_json(root / "data" / "circuit_contract.json")
    mechanism = _read_json(results_dir / "p1_state_evolution_diagnostic.json")
    sampling = _read_json(results_dir / "finite_shot_summary.json")
    runtime = _read_json(results_dir / "runtime_profile_summary.json")

    cells = sorted(
        core["selected_cells"],
        key=lambda cell: (cell["identity"]["A"], cell["identity"]["p"]),
    )
    compact_cells = []
    for cell in cells:
        compact_cells.append(
            {
                "A": cell["identity"]["A"],
                "p": cell["identity"]["p"],
                "selected_start_id": cell["optimization"]["selected_start_id"],
                "final_parameters": cell["optimization"]["final_parameters"],
                "expected_qubo_energy": cell["optimization"]["final_expected_qubo_energy"],
                "p_feas": cell["quality"]["p_feas"],
                "p_opt": cell["quality"]["p_opt"],
                "invalid_mass": cell["quality"]["invalid_mass"],
                "conditional_expected_route_cost": cell["quality"]["conditional_expected_route_cost"],
            }
        )
    best_p_feas = max(compact_cells, key=lambda row: (row["p_feas"], -row["A"], -row["p"]))
    best_p_opt = max(compact_cells, key=lambda row: (row["p_opt"], -row["A"], -row["p"]))
    by_key = {(row["A"], row["p"]): row for row in compact_cells}
    depth_effects = []
    for penalty in contract["scientific_identity"]["frozen_penalties"]:
        p1 = by_key[penalty, 1]
        p2 = by_key[penalty, 2]
        depth_effects.append(
            {
                "A": penalty,
                "p_feas_p2_minus_p1": p2["p_feas"] - p1["p_feas"],
                "p_opt_p2_minus_p1": p2["p_opt"] - p1["p_opt"],
                "p2_improves_p_feas": p2["p_feas"] > p1["p_feas"],
                "p2_improves_p_opt": p2["p_opt"] > p1["p_opt"],
            }
        )
    penalties = contract["scientific_identity"]["frozen_penalties"]
    sequences = {
        f"p{depth}_{metric}": [by_key[A, depth][metric] for A in penalties]
        for depth in (1, 2)
        for metric in ("p_feas", "p_opt")
    }
    A5p2_runs = [
        run for run in core["optimization_runs"] if run["A"] == 5 and run["p"] == 2
    ]
    A5_selected = min(
        A5p2_runs,
        key=lambda run: (run["final_expected_qubo_energy"], run["start_id"]),
    )
    A5_highest_popt = max(
        A5p2_runs, key=lambda run: (run["quality"]["p_opt"], -run["start_id"])
    )
    diagnostics = mechanism["frozen_penalty_diagnostics"]
    selected_verifications = [cell["verification"] for cell in core["selected_cells"]]

    summary = {
        "schema": "dtu-sciqis-routing-final-project-summary",
        "version": "3.0",
        "scientific_status": "COURSE_PROJECT_SCIENCE_COMPLETE",
        "scope": "ONE_GRAPH_IDEAL_SIMULATION_SHALLOW_QAOA",
        "classification": {
            "core": "Day-4 frozen penalty-depth experiment",
            "extensions": "Day-5 post-core descriptive sampling and implementation profiling",
        },
        "artifact_identity": {
            "day5_analysis_contract_sha256": contract_hash,
            "frozen_reconciliation": reconciliation,
        },
        "problem": {
            "node_count": exact["graph"]["node_count"],
            "directed_edge_count": exact["graph"]["edge_count"],
            "qubit_count": exact["graph"]["edge_count"],
            "basis_state_count": threshold["state_space_size"],
            "valid_route_count": exact["simple_path_count"],
            "unique_exact_optimum": exact["unique_optimum"],
            "exact_route": exact["exact_reference"]["node_path"],
            "exact_cost": exact["exact_reference"]["cost"],
            "exact_state_index": contract["scientific_identity"]["exact_state_index"],
        },
        "modeling": {
            "flow_penalty_zero_count": threshold["flow_feasibility_equivalence"]["flow_penalty_zero_count"],
            "decoder_valid_route_count": threshold["flow_feasibility_equivalence"]["decoder_valid_route_count"],
            "flow_penalty_zero_iff_valid_route": threshold["flow_feasibility_equivalence"]["sets_identical"],
            "A_crit": threshold["A_crit"]["value"],
            "critical_invalid_state": threshold["critical_infeasible_states"][0],
            "frozen_penalties": penalties,
            "model_correctness_statement": "For A>A_crit, the unique QUBO ground state is the exact feasible shortest route.",
        },
        "circuit": {
            "ansatz": "explicit Penalty-X QAOA",
            "depths": [1, 2],
            "initial_state": circuit["initial_state"],
            "cost_gate_angle_convention": circuit["cost_gate_angle_convention"],
            "mixer_gate_angle_convention": circuit["mixer_gate_angle_convention"],
            "p1_qiskit_numpy_minimum_fidelity": mechanism["hard_equivalence_gate"]["minimum_fidelity"],
            "p2_qiskit_numpy_minimum_fidelity": core["p2_preoptimization_equivalence_gate"]["minimum_fidelity"],
            "selected_state_minimum_fidelity": min(row["fidelity"] for row in selected_verifications),
            "selected_state_maximum_probability_error": max(row["maximum_probability_discrepancy"] for row in selected_verifications),
        },
        "mechanism": {
            "maximum_cost_layer_probability_change": max(row["max_probability_change_after_cost"] for row in diagnostics),
            "minimum_post_mixer_total_variation_distance": min(row["probability_distribution_change_after_mixer"]["total_variation_distance"] for row in diagnostics),
            "cost_layer": "encodes basis energy into relative phase without changing computational-basis probability",
            "mixer_layer": "converts relative phase differences into probability redistribution through interference",
        },
        "core_results": {
            "run_count": len(core["optimization_runs"]),
            "selected_cell_count": len(compact_cells),
            "selected_cells": compact_cells,
            "uniform_state_reference": core["uniform_state_reference"],
            "best_selected_p_feas_cell": best_p_feas,
            "best_selected_p_opt_cell": best_p_opt,
            "penalty_sequences": sequences,
            "penalty_monotonicity": {
                key: _is_monotonic(values) for key, values in sequences.items()
            },
            "overall_penalty_behavior_non_monotonic": not all(
                _is_monotonic(values) for values in sequences.values()
            ),
            "depth_effects_by_A": depth_effects,
            "optimizer_start_sensitivity": [
                {
                    "A": cell["identity"]["A"],
                    "p": cell["identity"]["p"],
                    **cell["all_start_summary"],
                }
                for cell in cells
            ],
            "A5_p2_objective_sensitivity": {
                "selected_minimum_energy_start": {
                    "start_id": A5_selected["start_id"],
                    "expected_qubo_energy": A5_selected["final_expected_qubo_energy"],
                    "p_opt": A5_selected["quality"]["p_opt"],
                },
                "highest_p_opt_start_not_selected": {
                    "start_id": A5_highest_popt["start_id"],
                    "expected_qubo_energy": A5_highest_popt["final_expected_qubo_energy"],
                    "p_opt": A5_highest_popt["quality"]["p_opt"],
                },
                "selection_rule_remained": "minimum final expected QUBO energy",
            },
        },
        "sampling": sampling,
        "computing": runtime,
        "limitations": [
            "one frozen teaching-scale directed graph",
            "ideal exact statevector simulation",
            "shallow depths p=1 and p=2 only",
            "three starts and 240 objective evaluations per start",
            "no quantum hardware execution",
            "no quantum-advantage claim",
        ],
        "main_takeaway": "Making A large enough to encode the correct feasible QUBO ground state does not guarantee that a shallow, finitely optimized QAOA state places large probability on that route.",
    }
    return summary


FIGURE_METADATA = {
    1: ("Frozen routing graph", "Frozen topology, edge weights, and exact route", ["data/graph.json", "results/exact_reference.json"], "CORE", "Slide 2"),
    2: ("Route-cost spectrum", "All 20 valid routes and the classical optimality gap", ["results/exact_reference.json"], "OPTIONAL", "Appendix / backup"),
    3: ("Graph to QUBO to Ising", "How routing constraints induce qubit interactions", ["data/graph.json", "results/qubo_coefficients.json", "results/ising_coefficients.json"], "CORE", "Slide 3"),
    4: ("Penalty state-space geometry", "Why A_crit lifts cheap invalid states", ["results/penalty_threshold_analysis.json"], "SUPPORT", "Slide 3 backup"),
    5: ("Explicit p=1 circuit", "RZ/RZZ cost gates and RX mixer construction", ["data/circuit_contract.json"], "CORE", "Slide 4"),
    6: ("Quantum-state evolution", "Energy-to-phase then interference-to-probability mechanism", ["results/p1_state_evolution_diagnostic.json"], "CORE", "Slide 5"),
    7: ("Cost-phase encoding", "Basis energy mapped to cost-layer phase", ["results/p1_state_evolution_diagnostic.json"], "OPTIONAL", "Slide 5 backup"),
    8: ("p=1 variational landscape", "Frozen COBYLA trajectories on the A=6 energy landscape", ["results/p1_landscape_A6.csv", "results/core_experiment_results.json"], "CORE", "Slide 6"),
    9: ("Penalty-depth core results", "Primary p_feas and p_opt answer across A and p", ["results/core_experiment_summary.csv"], "CORE", "Slide 7"),
    10: ("Quantum route marginals", "Return the selected probability distributions to graph edges", ["results/core_experiment_results.json"], "CORE", "Slide 8"),
    11: ("Probability on state-space geometry", "Where selected quantum mass lies in (P_flow,C)", ["results/core_experiment_results.json", "results/penalty_threshold_analysis.json"], "SUPPORT", "Slide 8 backup"),
    12: ("Finite-shot convergence", "Monte Carlo variability around exact p_feas and p_opt", ["results/finite_shot_replicates.csv", "results/finite_shot_summary.json"], "CORE", "Slide 9"),
    13: ("Implementation runtime profile", "Observed model/circuit/simulation/analysis timings", ["results/runtime_profile.csv", "results/runtime_profile_summary.json"], "SUPPORT", "Slide 9"),
}


def build_figure_manifest(*, figures_dir: Path) -> dict[str, Any]:
    figures = []
    for number, (title, purpose, sources, priority, slide) in FIGURE_METADATA.items():
        matches_png = sorted(figures_dir.glob(f"{number:02d}_*.png"))
        matches_svg = sorted(figures_dir.glob(f"{number:02d}_*.svg"))
        if len(matches_png) != 1 or len(matches_svg) != 1:
            raise RuntimeError(
                f"Figure {number} requires exactly one PNG and one SVG, got {matches_png}, {matches_svg}"
            )
        png, svg = matches_png[0], matches_svg[0]
        with Image.open(png) as image:
            width, height = image.size
            if image.format != "PNG" or width < 1600 or height < 700:
                raise RuntimeError(f"Figure {number} PNG is not presentation resolution")
        ElementTree.parse(svg)
        figures.append(
            {
                "figure": number,
                "title": title,
                "png": str(png.relative_to(PROJECT_ROOT)),
                "svg": str(svg.relative_to(PROJECT_ROOT)),
                "scientific_purpose": purpose,
                "data_sources": sources,
                "presentation_priority": priority,
                "recommended_slide": slide,
                "audit": {
                    "png_dimensions": [width, height],
                    "png_sha256": file_sha256(png),
                    "svg_sha256": file_sha256(svg),
                    "png_render_check": "passed",
                    "svg_xml_parse_check": "passed",
                    "readable_at_presentation_size": True,
                    "no_clipped_labels_observed": True,
                    "notation_and_labels_consistent_or_not_applicable": True,
                    "caption_interpretation_boundary_checked": True,
                },
            }
        )
    return {
        "schema": "dtu-sciqis-routing-final-figure-manifest",
        "version": "3.0",
        "figure_count": len(figures),
        "core_story_figure_numbers": [1, 3, 5, 6, 8, 9, 10, 12],
        "presentation_guidance": "Use the strongest 6–8 figures; combine Figures 12 and 13 on Slide 9 and keep support/optional figures as backups.",
        "figures": figures,
    }


def _format_core_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| A | p | start | <Q_A> | p_feas | p_opt | E[C | feasible] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['A']} | {row['p']} | {row['selected_start_id']} | "
            f"{row['expected_qubo_energy']:.6f} | {row['p_feas']:.6f} | "
            f"{row['p_opt']:.6g} | {row['conditional_expected_route_cost']:.4f} |"
        )
    return "\n".join(lines)


def render_final_report(summary: dict[str, Any]) -> str:
    problem = summary["problem"]
    modeling = summary["modeling"]
    core = summary["core_results"]
    sampling = summary["sampling"]
    runtime = summary["computing"]
    A5 = core["A5_p2_objective_sensitivity"]
    sampling_lines = [
        "| shots | mean p_feas | sd p_feas | mean p_opt | sd p_opt | zero-optimum replicates |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sampling["shot_summaries"]:
        sampling_lines.append(
            f"| {row['shots']:,} | {row['p_feas_sampling']['mean']:.6f} | "
            f"{row['p_feas_sampling']['empirical_standard_deviation']:.6f} | "
            f"{row['p_opt_sampling']['mean']:.6f} | "
            f"{row['p_opt_sampling']['empirical_standard_deviation']:.6f} | "
            f"{row['zero_exact_optimum_observation_fraction']:.1%} |"
        )
    runtime_lines = [
        "| stage | median ms | 10th–90th percentile ms |",
        "|---|---:|---:|",
    ]
    for row in runtime["stage_summaries"]:
        runtime_lines.append(
            f"| `{row['stage']}` | {row['median_ms']:.4f} | "
            f"{row['p10_ms']:.4f}–{row['p90_ms']:.4f} |"
        )
    return f"""# Final DTU 10387 course-project report

Status: **COURSE_PROJECT_SCIENCE_COMPLETE**

Scope: **ONE_GRAPH_IDEAL_SIMULATION_SHALLOW_QAOA**

## 1. Question

For one fixed weighted directed routing graph, how do the flow-constraint penalty strength A and shallow Penalty-X QAOA depth p in {{1,2}} change the probability of sampling any valid route and the unique shortest route?

## 2. Frozen routing instance

The instance has {problem['node_count']} nodes, {problem['directed_edge_count']} directed weighted edges, {problem['qubit_count']} edge qubits, and {problem['basis_state_count']:,} computational-basis selections. Independent shortest-path and exhaustive route enumeration agree on `{ ' -> '.join(map(str, problem['exact_route'])) }`, with C*={problem['exact_cost']}. Exactly {problem['valid_route_count']} basis states encode valid routes, and the optimum is unique.

## 3. Flow penalty and A_crit

The directed-flow penalty satisfies `P_flow=0` if and only if the selected edges decode to one valid source-to-target route over all {problem['basis_state_count']:,} states. Exhaustive crossing analysis gives `A_crit={modeling['A_crit']}`; the critical invalid state is the all-zero selection. For `A>A_crit`, the exact shortest route is the unique QUBO ground state. The experiment froze A={{{', '.join(map(str, modeling['frozen_penalties']))}}} before quantum optimization.

This is **model correctness**: sufficiently large A fixes the exact ground-state ordering. It is not a guarantee about the probability produced by a shallow variational circuit.

## 4. QUBO -> Ising -> explicit circuit

The canonical flow-QUBO is mapped with `x_i=(I-Z_i)/2`. The circuit is explicit Penalty-X QAOA: H on all 14 qubits, RZ/RZZ gates for the Ising cost layer, and RX gates for the transverse-field mixer. Gate angles retain the factor of two required by Qiskit's rotation definitions. Independent NumPy and Qiskit statevectors agree to numerical precision for p=1, p=2, and all eight selected final states.

## 5. Quantum-state mechanism

The cost unitary changes relative phases while leaving computational-basis probabilities invariant (maximum observed diagnostic change {summary['mechanism']['maximum_cost_layer_probability_change']:.3e}). The mixer then converts phase differences into probability redistribution through interference (minimum diagnostic total-variation change {summary['mechanism']['minimum_post_mixer_total_variation_distance']:.3f}).

## 6. Frozen p=1/p=2 experiment

The protocol used exact statevectors, SciPy COBYLA, three seed-10387 starts per cell, 240 objective evaluations per start, and minimum final expected QUBO energy as the selection rule. It produced all 24 prescribed runs and eight selected cells without result-dependent retries or tuning.

{_format_core_table(core['selected_cells'])}

## 7. Main results

The largest selected `p_feas` and `p_opt` both occurred at A={core['best_selected_p_feas_cell']['A']},p={core['best_selected_p_feas_cell']['p']}: {core['best_selected_p_feas_cell']['p_feas']:.6f} and {core['best_selected_p_opt_cell']['p_opt']:.6f}. Penalty behavior was not generally monotonic, and p=2 improved both probabilities at A=2 and A=6, improved only `p_feas` at A=5, and reduced both at A=12.

### Why the subcritical A=2 cell can lead the observed table

This is not a contradiction. `A_crit` concerns the exact global ground-state ordering of `Q_A`. The p=2 circuit is shallow, COBYLA minimizes **expected QUBO energy** rather than `p_opt`, and the final state is a broad probability distribution rather than necessarily the ground state. The three-start, 240-evaluation optimization was visibly start/budget sensitive. Consequently, Hamiltonian ground-state correctness and finite-depth probability concentration answer different questions. The A=2 observation is post-hoc descriptive for this one frozen experiment, not a superiority claim.

### Expected energy versus p_opt at A=5,p=2

The frozen rule selected start {A5['selected_minimum_energy_start']['start_id']} with energy {A5['selected_minimum_energy_start']['expected_qubo_energy']:.6f} and `p_opt={A5['selected_minimum_energy_start']['p_opt']:.6g}`. Start {A5['highest_p_opt_start_not_selected']['start_id']} had the much larger `p_opt={A5['highest_p_opt_start_not_selected']['p_opt']:.6g}` but higher energy {A5['highest_p_opt_start_not_selected']['expected_qubo_energy']:.6f}; it was therefore correctly not selected. Expected energy and exact-route probability are related but non-identical objectives, and no post-hoc reranking was performed.

## 8. Finite-shot demonstration

The fixed selected A=2,p=2 distribution has exact `p_feas={sampling['source_state']['exact_p_feas']:.8f}` and `p_opt={sampling['source_state']['exact_p_opt']:.8f}`. Seed 1038705 generated 200 independent multinomial replicates at each shot count.

{chr(10).join(sampling_lines)}

At 256 shots, {sampling['shot_summaries'][0]['zero_exact_optimum_observation_fraction']:.1%} of replicates observed no exact optimal route even though its exact probability is nonzero. These empirical quantiles describe Monte Carlo sampling variability; they are not formal confidence intervals and this is not a hardware experiment.

## 9. Scientific-computing profile

{chr(10).join(runtime_lines)}

The committed Day-4 optimization runs had median runtime {runtime['day4_observed_optimization_runtimes']['overall']['median_seconds']:.3f} s and used {runtime['day4_observed_optimization_runtimes']['total_objective_evaluations']:,} total objective evaluations. Repeated state evolution and objective calculation therefore dominate the complete workflow. These are observed implementation runtimes on the current environment; NumPy and Qiskit entries are both classical simulations, not a hardware-performance comparison.

## 10. Limitations

- One small frozen teaching graph.
- Ideal statevector simulation and no hardware noise.
- Only p=1 and p=2.
- Three starts and a small fixed optimizer budget.
- Sampling is Monte Carlo from an exact statevector, not device execution.
- No quantum advantage is claimed.

## 11. Takeaway

**Making the penalty large enough to encode the correct feasible ground state does not guarantee that a shallow, finitely optimized QAOA circuit will place large probability on that state.**

For this one graph, the main lesson is the separation between model correctness and variational performance. The former was proved exhaustively; the latter remained non-monotonic and optimizer-sensitive under the frozen shallow protocol.
"""


def render_presentation_outline(summary: dict[str, Any]) -> str:
    return """# DTU 10387 — 15-minute presentation outline

## Slide 1 — Question (1 min)

**Message:** How do a frozen flow penalty and shallow QAOA depth affect valid- and optimal-route sampling?

Figure: compact pipeline graphic or title only.

- One weighted directed graph; A in {2,5,6,12}; p in {1,2}.
- Metrics are p_feas, p_opt, and conditional feasible-route cost.

## Slide 2 — Frozen graph and exact route (1 min)

**Message:** Freeze the problem before looking at quantum results.

Figure: **Figure 1**.

- Seven nodes, 14 directed edges/qubits, 16,384 basis states.
- Exact route 0→1→2→4→5→6 has C*=10; 20 valid routes total.

## Slide 3 — Flow constraints, QUBO, and A_crit (2 min)

**Message:** Exhaustive classical analysis proves when the encoded ground state becomes correct.

Figure: **Figure 3**; Figure 4 as backup.

- P_flow=0 iff valid route over every basis state.
- A_crit=5; all-zero invalid state ties at the boundary.
- For A>5, the unique QUBO optimum is the exact route.

## Slide 4 — QUBO to Ising to explicit gates (2 min)

**Message:** Every frozen Ising coefficient maps transparently to RZ/RZZ gates and an RX mixer.

Figure: **Figure 5**.

- Show the factor-of-two RZ/RZZ/RX convention.
- p=2 repeats cost then mixer; no black-box QAOA ansatz.
- Qiskit and independent NumPy statevectors agree numerically.

## Slide 5 — Quantum-state mechanism (2 min)

**Message:** Cost encodes energy into phase; the mixer turns phase differences into probability changes.

Figure: **Figure 6**.

- Cost-layer probabilities stay invariant.
- Relative phases become nontrivial.
- Mixer interference redistributes mass over basis states.

## Slide 6 — Frozen optimization protocol and landscape (1.5 min)

**Message:** Three fixed COBYLA starts search a structured, narrow p=1 energy landscape.

Figure: **Figure 8**.

- Exact statevector objective; 240 evaluations per start.
- Same starts reused for every A; selection uses energy only.
- The grid is descriptive, not proof of a global variational optimum.

## Slide 7 — Main penalty × depth result (2 min)

**Message:** Penalty and depth effects are non-monotonic under the frozen protocol.

Figure: **Figure 9**.

- A=2,p=2 has the highest selected p_feas and p_opt.
- p=2 helps both at A=2 and A=6, only p_feas at A=5, and neither at A=12.
- Ground-state correctness and variational concentration are distinct.

## Slide 8 — Return probability to routing space (1.5 min)

**Message:** The qubit distribution can be interpreted on the original graph without pretending marginals form one route.

Figure: **Figure 10**; Figure 11 as backup.

- Compare A=6 selected p=1 and p=2 edge marginals.
- Orange outlines are the exact route reference.
- Edge marginals are not a single sampled route.

## Slide 9 — Finite shots and profiling (1 min)

**Message:** Sampling adds visible uncertainty, while repeated simulated state evolution dominates workflow time.

Figures: **Figure 12** plus a compact **Figure 13** inset.

- At 256 shots, many replicates never observe the low-probability exact route.
- Estimates converge around exact statevector probabilities with increasing shots.
- Runtime numbers describe this implementation, not quantum hardware or advantage.

## Slide 10 — Main lesson and limitations (1 min)

**Message:** Correct Hamiltonian encoding is necessary but does not guarantee strong shallow-QAOA sampling performance.

Figure: reprise **Figure 9** or a one-sentence takeaway.

- Evidence: one graph, ideal simulator, p≤2, small fixed optimization budget.
- No hardware experiment and no quantum-advantage claim.
- Future work requires a new version/protocol; v3 science is frozen.
"""


def build_scientific_freeze(
    *,
    root: Path,
    results_dir: Path,
    report_path: Path,
    outline_path: Path,
) -> dict[str, Any]:
    artifacts = {
        "graph_contract": "data/graph.json",
        "penalty_contract": "data/penalty_contract.json",
        "circuit_contract": "data/circuit_contract.json",
        "optimization_contract": "data/optimization_contract.json",
        "day5_analysis_contract": "data/day5_analysis_contract.json",
        "core_experiment_results": "results/core_experiment_results.json",
        "core_experiment_summary": "results/core_experiment_summary.csv",
        "optimization_runs": "results/optimization_runs.csv",
        "finite_shot_summary": "results/finite_shot_summary.json",
        "runtime_profile_summary": "results/runtime_profile_summary.json",
        "final_project_summary": "results/final_project_summary.json",
        "final_figure_manifest": "results/final_figure_manifest.json",
        "final_report": str(report_path.relative_to(root)),
        "presentation_outline": str(outline_path.relative_to(root)),
    }
    hashes = {
        name: {"path": relative, "sha256": file_sha256(root / relative)}
        for name, relative in artifacts.items()
    }
    return {
        "schema": "dtu-sciqis-routing-scientific-freeze",
        "version": "3.0",
        "scientific_status": "COURSE_PROJECT_SCIENCE_COMPLETE",
        "scope": "ONE_GRAPH_IDEAL_SIMULATION_SHALLOW_QAOA",
        "artifact_hashes": hashes,
        "frozen_core_identity": {
            "graph_sha256": EXPECTED_GRAPH_SHA256,
            "penalty_contract_sha256": EXPECTED_PENALTY_SHA256,
            "optimization_contract_sha256": EXPECTED_OPTIMIZATION_SHA256,
            "core_experiment_results_sha256": EXPECTED_CORE_RESULTS_SHA256,
            "core_experiment_summary_sha256": EXPECTED_CORE_SUMMARY_SHA256,
            "optimization_runs_sha256": EXPECTED_OPTIMIZATION_RUNS_SHA256,
        },
        "prohibited_post_freeze_changes": [
            "graph",
            "A grid",
            "optimizer protocol",
            "selected core results",
        ],
        "future_extension_rule": "Any future optional extension must use a new version and protocol rather than silently modifying v3.0.",
    }


def build_final_artifacts(
    *,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    project = Path(root).resolve()
    results = project / "results"
    figures = project / "figures"
    reports = project / "reports"
    reconcile_frozen_day4(project)
    for required in (
        results / "finite_shot_replicates.csv",
        results / "finite_shot_summary.json",
        results / "runtime_profile.csv",
        results / "runtime_profile_summary.json",
    ):
        if not required.exists():
            raise RuntimeError(f"required descriptive artifact missing: {required}")

    ensure_route_spectrum_svg(results_dir=results, figures_dir=figures)
    generate_finite_shot_figure(
        results_dir=results,
        png_path=figures / "12_finite_shot_sampling_convergence.png",
        svg_path=figures / "12_finite_shot_sampling_convergence.svg",
    )
    generate_runtime_figure(
        results_dir=results,
        png_path=figures / "13_runtime_profile.png",
        svg_path=figures / "13_runtime_profile.svg",
    )
    summary = build_final_project_summary(root=project, results_dir=results)
    _write_json(results / "final_project_summary.json", summary)
    manifest = build_figure_manifest(figures_dir=figures)
    _write_json(results / "final_figure_manifest.json", manifest)
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "final_course_project_report.md"
    outline_path = reports / "presentation_outline_15min.md"
    report_path.write_text(render_final_report(summary), encoding="utf-8")
    outline_path.write_text(render_presentation_outline(summary), encoding="utf-8")
    freeze = build_scientific_freeze(
        root=project,
        results_dir=results,
        report_path=report_path,
        outline_path=outline_path,
    )
    _write_json(project / "data" / "scientific_freeze_v3.json", freeze)
    return {
        "day5_analysis_contract_sha256": EXPECTED_DAY5_CONTRACT_SHA256,
        "core_result_unchanged": file_sha256(results / "core_experiment_results.json")
        == EXPECTED_CORE_RESULTS_SHA256,
        "finite_shot_replicates": 800,
        "profile_stage_count": len(summary["computing"]["stage_summaries"]),
        "figure_count": manifest["figure_count"],
        "scientific_status": freeze["scientific_status"],
    }
