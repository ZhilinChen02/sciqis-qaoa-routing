"""Draw the 17 report figures from saved experiment tables."""

import csv
import json
from math import pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from experiments.dynamics_study import (
    ALGORITHMS,
    DISPLAY_NAMES,
    GROVER_FEASIBLE,
    GROVER_GLOBAL,
    PENALTY_X,
)
from graph import DEFAULT_GRAPH_PATH, get_edge_order, load_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    PENALTY_X: "#4C78A8",
    GROVER_GLOBAL: "#F58518",
    GROVER_FEASIBLE: "#54A24B",
}
MARKERS = {"initial": "o", "cost": "s", "mixer": "D"}


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(row, name):
    return float(row[name])


def _true(value):
    return value is True or str(value).lower() == "true"


def _save(figure, folder, filename):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _portable(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _figure_graph(folder):
    graph = load_graph(DEFAULT_GRAPH_PATH)
    route = (0, 1, 2, 4, 5, 6)
    optimal_edges = set(zip(route, route[1:]))
    positions = {
        0: (0.0, 0.2), 1: (1.0, 1.0), 2: (2.0, 1.25),
        3: (2.15, 0.15), 4: (3.25, 1.05), 5: (4.15, 0.35), 6: (5.25, 0.75),
    }
    figure, axis = plt.subplots(figsize=(10, 4.6))
    for edges, color, width in (
        ([edge for edge in get_edge_order(graph) if edge not in optimal_edges], "#9AA0A6", 1.5),
        (list(optimal_edges), "#D62728", 3.4),
    ):
        nx.draw_networkx_edges(
            graph, positions, edgelist=edges, edge_color=color, width=width,
            arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.03", ax=axis,
        )
    node_colors = ["#2E86AB" if node == 0 else "#7A5195" if node == 6 else "#F2C14E" for node in graph]
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=850, edgecolors="white", linewidths=2, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_color="white", font_weight="bold", ax=axis)
    nx.draw_networkx_edge_labels(
        graph, positions,
        edge_labels={(u, v): graph.edges[u, v]["weight"] for u, v in graph.edges},
        rotate=False, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8}, ax=axis,
    )
    axis.set_title("Fixed routing graph\noptimal route 0→1→2→4→5→6, cost 10")
    axis.axis("off")
    return _save(figure, folder, "01_fixed_weighted_routing_graph.png")


def _figure_pipeline(folder):
    labels = (
        "weighted\ngraph", "14 edge\nbits", "routing\nQUBO", "Ising\n$H_C$",
        "cost\nphase", "mixer", "measure\nand decode",
    )
    colors = ("#DDEBF7", "#FFF2CC", "#FCE4D6", "#E2F0D9", "#E4DFEC", "#D9EAD3", "#DDEBF7")
    figure, axis = plt.subplots(figsize=(13, 3.4))
    x_values = np.linspace(0.07, 0.93, len(labels))
    for index, (x, label, color) in enumerate(zip(x_values, labels, colors)):
        axis.text(x, 0.55, label, ha="center", va="center", fontsize=10,
                  bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "#555"})
        if index:
            axis.annotate("", (x - 0.06, 0.55), (x_values[index - 1] + 0.06, 0.55),
                          arrowprops={"arrowstyle": "->", "color": "#555"})
    axis.text(0.65, 0.2, "repeat p layers: phase → interference", ha="center", fontweight="bold")
    axis.set_title("Routing QUBO → Ising → QAOA pipeline")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    return _save(figure, folder, "02_qubo_ising_qaoa_pipeline.png")


def _figure_spaces(folder):
    descriptions = {
        PENALTY_X: ("16,384 bitstrings", "$H_X=\\sum_j X_j$", "local mixing"),
        GROVER_GLOBAL: ("16,384 bitstrings", "$|s_{all}\\rangle\\langle s_{all}|$", "global mixing"),
        GROVER_FEASIBLE: ("20 valid routes", "$|s_F\\rangle\\langle s_F|$", "feasible mixing"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for axis, algorithm in zip(axes, ALGORITHMS):
        circle = plt.Circle((0.5, 0.58), 0.3, color=COLORS[algorithm], alpha=0.18)
        axis.add_patch(circle)
        count = 20 if algorithm == GROVER_FEASIBLE else 90
        random = np.random.default_rng(12 + ALGORITHMS.index(algorithm))
        angles = random.uniform(0, 2 * pi, count)
        radii = 0.27 * np.sqrt(random.random(count))
        axis.scatter(0.5 + radii * np.cos(angles), 0.58 + radii * np.sin(angles), s=8, color=COLORS[algorithm])
        space, mixer, note = descriptions[algorithm]
        axis.text(0.5, 0.95, DISPLAY_NAMES[algorithm], ha="center", fontweight="bold")
        axis.text(0.5, 0.58, space, ha="center", fontweight="bold")
        axis.text(0.5, 0.18, mixer, ha="center", fontsize=11)
        axis.text(0.5, 0.08, note, ha="center", color="#555")
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
    figure.suptitle("Three search-space and mixer geometries")
    return _save(figure, folder, "03_three_search_spaces_and_mixers.png")


def _bar_comparison(summaries, folder, metric, filename, title):
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    x_values = np.arange(2)
    width = 0.24
    for offset, algorithm in enumerate(ALGORITHMS):
        rows = sorted((row for row in summaries if row["algorithm"] == algorithm), key=lambda row: row["p"])
        values = [float(row[metric]) for row in rows]
        bars = axis.bar(x_values + (offset - 1) * width, values, width, color=COLORS[algorithm], label=DISPLAY_NAMES[algorithm])
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], fontsize=8)
    axis.set(xticks=x_values, xticklabels=["p=1", "p=2"], ylim=(0, 1.08), ylabel="probability mass", title=title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=3)
    return _save(figure, folder, filename)


def _figure_mass(summaries, folder):
    rows = sorted(summaries, key=lambda row: (ALGORITHMS.index(row["algorithm"]), row["p"]))
    labels = [f"{row['algorithm'].replace('_', ' ')}\np={row['p']}" for row in rows]
    optimum = np.array([row["p_opt"] for row in rows])
    other = np.array([row["p_feas"] - row["p_opt"] for row in rows])
    invalid = np.array([row["invalid_mass"] for row in rows])
    figure, axis = plt.subplots(figsize=(11, 5.3))
    x_values = np.arange(len(rows))
    axis.bar(x_values, optimum, color="#2CA02C", label="optimal feasible")
    axis.bar(x_values, other, bottom=optimum, color="#9ED17B", label="other feasible")
    axis.bar(x_values, invalid, bottom=optimum + other, color="#D95F5F", label="infeasible")
    axis.set(xticks=x_values, xticklabels=labels, ylim=(0, 1.02), ylabel="probability mass", title="Final probability-mass decomposition")
    axis.tick_params(axis="x", rotation=18)
    axis.legend(ncol=3)
    axis.grid(axis="y", alpha=0.22)
    return _save(figure, folder, "06_probability_mass_decomposition.png")


def _trace_figure(rows, folder, metric, filename, title, ylabel):
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for algorithm in ALGORITHMS:
        selected = sorted(
            (row for row in rows if row["algorithm"] == algorithm and int(row["p"]) == 2),
            key=lambda row: int(row["checkpoint_index"]),
        )
        values = [_number(row, metric) for row in selected]
        axis.plot(range(5), values, color=COLORS[algorithm], lw=2, label=DISPLAY_NAMES[algorithm])
        for index, row in enumerate(selected):
            axis.scatter(index, values[index], marker=MARKERS[row["operation"]], s=60, color=COLORS[algorithm])
    axis.set(xticks=range(5), xticklabels=["Initial", "Cost-1", "Mixer-1", "Cost-2", "Mixer-2"], ylabel=ylabel, title=title)
    if metric in ("p_feas", "p_opt"):
        axis.set_ylim(-0.02, 1.04)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    return _save(figure, folder, filename)


def _figure_phase(result_root, folder):
    """Show phase, probability and complex-amplitude changes for each method."""

    figure, axes = plt.subplots(3, 3, figsize=(15, 12))
    for row_index, algorithm in enumerate(ALGORITHMS):
        rows = _read_csv(result_root / f"phase_analysis/{algorithm}_p2_seed2601.csv")
        cost = [row for row in rows if row["checkpoint"] == "Cost-1"]
        mixed = [row for row in rows if row["checkpoint"] == "Mixer-1"]

        axes[row_index, 0].scatter([_number(row, "energy") for row in cost], [_number(row, "phase") for row in cost], s=9, alpha=0.4, color=COLORS[algorithm])
        axes[row_index, 0].set(title=f"{DISPLAY_NAMES[algorithm]}\nafter cost", ylabel="wrapped phase")

        for data, color, label in ((cost, "#777", "after cost"), (mixed, COLORS[algorithm], "after mixer")):
            axes[row_index, 1].scatter([_number(row, "energy") for row in data], [max(_number(row, "probability"), 1e-16) for row in data], s=8, alpha=0.4, color=color, label=label)
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 1].set(title="probability redistribution", ylabel="probability")
        axes[row_index, 1].legend(fontsize=7)

        # A few high-probability states keep this panel readable for the full space.
        count = len(cost) if algorithm == GROVER_FEASIBLE else min(20, len(cost))
        indices = np.argsort([-_number(row, "probability") for row in mixed])[:count]
        for index in indices:
            left, right = cost[int(index)], mixed[int(index)]
            a = np.sqrt(_number(left, "probability")) * np.exp(1j * _number(left, "phase"))
            b = np.sqrt(_number(right, "probability")) * np.exp(1j * _number(right, "phase"))
            axes[row_index, 2].plot([a.real, b.real], [a.imag, b.imag], "-o", ms=2, lw=0.7, alpha=0.6, color=COLORS[algorithm])
        axes[row_index, 2].axhline(0, color="#bbb", lw=0.7)
        axes[row_index, 2].axvline(0, color="#bbb", lw=0.7)
        axes[row_index, 2].set(title="complex amplitudes", xlabel="Re", ylabel="Im")
        for axis in axes[row_index]:
            axis.grid(alpha=0.18)
    figure.suptitle("Cost phase and mixer interference (optimized p=2)")
    figure.tight_layout()
    return _save(figure, folder, "10_phase_and_interference_evolution.png")


def _figure_energy(result_root, folder):
    rows = _read_csv(result_root / "energy_landscape.csv")
    total = np.array([_number(row, "total_qubo_energy") for row in rows])
    route = np.array([_number(row, "routing_term") for row in rows])
    penalty = np.array([_number(row, "flow_penalty") for row in rows])
    feasible = np.array([_true(row["feasible"]) for row in rows])
    optimal = np.array([_true(row["exact_optimal"]) for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.linspace(total.min(), np.percentile(total, 99.5), 50)
    axes[0, 0].hist(total[~feasible], bins=bins, color="#D95F5F", alpha=0.65, label="infeasible")
    axes[0, 0].hist(total[feasible], bins=bins, color="#2CA02C", alpha=0.85, label="feasible")
    axes[0, 0].axvline(total[optimal][0], color="black", ls="--", label="optimum")
    axes[0, 0].set(title="Penalty-QUBO energy", xlabel="energy", ylabel="state count")
    axes[0, 0].legend()
    values, counts = np.unique(penalty, return_counts=True)
    axes[0, 1].bar(values, counts, color="#8064A2")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Flow penalties", xlabel="penalty", ylabel="count")
    values, counts = np.unique(route[feasible], return_counts=True)
    axes[1, 0].bar(values, counts, color="#54A24B")
    axes[1, 0].set(title="Costs of 20 feasible routes", xlabel="route cost", ylabel="count")
    order = np.argsort(total, kind="stable")[:120]
    axes[1, 1].scatter(range(len(order)), total[order], c=np.where(feasible[order], "#2CA02C", "#D95F5F"), s=18)
    axes[1, 1].set(title="120 lowest-energy states", xlabel="rank", ylabel="energy")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Full 2¹⁴-state energy landscape, penalty A=6")
    figure.tight_layout()
    return _save(figure, folder, "11_energy_landscape.png")


def _figure_geometry(folder):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    cube = nx.hypercube_graph(3)
    nx.draw_networkx(cube, nx.spring_layout(cube, seed=2601), ax=axes[0], node_size=300, font_size=7)
    axes[0].set_title("X mixer\nHamming-neighbor edges")
    complete = nx.complete_graph(8)
    positions = nx.circular_layout(complete)
    nx.draw_networkx(complete, positions, ax=axes[1], node_size=300, font_size=7, edge_color="#F58518", alpha=0.55)
    axes[1].set_title("Global Grover mixer")
    nx.draw_networkx_nodes(complete, positions, ax=axes[2], node_color="#ddd", node_size=260)
    feasible = complete.subgraph([0, 3, 7])
    nx.draw_networkx(feasible, positions, ax=axes[2], node_color="#8FD175", edge_color="#54A24B", node_size=380)
    axes[2].set_title("Feasible Grover mixer")
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Small q=3 illustration of mixer geometry")
    return _save(figure, folder, "12_conceptual_search_space_geometry.png")


def _figure_top_states(result_root, folder):
    rows = _read_csv(result_root / "top_state_evolution.csv")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    for axis, algorithm in zip(axes, ALGORITHMS):
        grouped = {}
        for row in rows:
            if row["algorithm"] == algorithm:
                grouped.setdefault(int(row["basis_index"]), []).append(row)
        indices = sorted(grouped, key=lambda index: -max(_number(row, "probability") for row in grouped[index]))[:6]
        for index in indices:
            sequence = sorted(grouped[index], key=lambda row: int(row["checkpoint_index"]))
            first = sequence[0]
            color = "#2CA02C" if _true(first["exact_optimal"]) else COLORS[algorithm] if _true(first["feasible"]) else "#D95F5F"
            label = "optimum" if _true(first["exact_optimal"]) else f"E={float(first['energy']):g}"
            axis.plot([int(row["checkpoint_index"]) for row in sequence], [_number(row, "probability") for row in sequence], "-o", color=color, alpha=0.75, label=label)
        axis.set(xticks=range(5), xticklabels=["Init", "C1", "M1", "C2", "M2"], title=DISPLAY_NAMES[algorithm], xlabel="checkpoint")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=7)
    axes[0].set_ylabel("state probability")
    figure.suptitle("Representative-state probability evolution")
    figure.tight_layout()
    return _save(figure, folder, "13_top_state_probability_evolution.png")


def _figure_runtime(result_root, folder):
    rows = _read_csv(result_root / "runtime_profile.csv")
    setup = [row for row in rows if row["scope"] == "setup"]
    runs = [row for row in rows if row["scope"] == "run"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = [row["component"].replace("_seconds", "").replace("_", " ") for row in setup]
    values = [_number(row, "seconds") for row in setup]
    axes[0].barh(labels, values, color="#7A9E9F")
    axes[0].set_xscale("log")
    axes[0].set(title="Preprocessing runtime", xlabel="seconds (log)")
    run_ids = list(dict.fromkeys(row["run_id"] for row in runs))
    component = lambda run_id, name: next(_number(row, "seconds") for row in runs if row["run_id"] == run_id and row["component"] == name)
    optimize = np.array([component(run_id, "optimize_evaluate_seconds") for run_id in run_ids])
    analysis = np.array([component(run_id, "decode_analysis_trace_seconds") for run_id in run_ids])
    x_values = np.arange(len(run_ids))
    axes[1].bar(x_values, optimize, label="optimize", color="#4C78A8")
    axes[1].bar(x_values, analysis, bottom=optimize, label="analysis", color="#F2CF5B")
    axes[1].set(xticks=x_values, xticklabels=[name.replace("_seed2601", "") for name in run_ids], title="Per-run runtime", ylabel="seconds")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()
    figure.tight_layout()
    return _save(figure, folder, "14_runtime_breakdown.png")


def _parameter_landscape(result_root, folder, summaries, algorithm, number):
    rows = _read_csv(result_root / f"parameter_landscapes/{algorithm}_p1.csv")
    gammas = sorted({_number(row, "gamma") for row in rows})
    betas = sorted({_number(row, "beta") for row in rows})
    summary = next(row for row in summaries if row["algorithm"] == algorithm and int(row["p"]) == 1)
    gamma_star, beta_star = map(float, summary["optimized_parameters"])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for axis, (metric, title) in zip(axes, (("expected_hc", "expected energy"), ("p_feas", "$p_{feas}$"), ("p_opt", "$p_{opt}$"))):
        values = np.array([_number(row, metric) for row in rows]).reshape(len(gammas), len(betas))
        image = axis.imshow(values, origin="lower", aspect="auto", extent=(0, pi, 0, 2 * pi), cmap="viridis")
        axis.scatter(beta_star, gamma_star, marker="*", s=150, color="white", edgecolor="black")
        axis.set(title=title, xlabel="β", ylabel="γ")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(f"p=1 parameter landscape — {DISPLAY_NAMES[algorithm]}")
    figure.tight_layout()
    return _save(figure, folder, f"{number:02d}_p1_parameter_landscape_{algorithm}.png")


def generate_figures_from_saved(result_root: Path, figure_root: Path) -> list[Path]:
    """Regenerate all figures without rerunning an optimizer."""

    result_root, figure_root = Path(result_root), Path(figure_root)
    summaries = _read_json(result_root / "final_summary.json")
    traces = _read_csv(result_root / "dynamics_trace.csv")
    paths = [
        _figure_graph(figure_root),
        _figure_pipeline(figure_root),
        _figure_spaces(figure_root),
        _bar_comparison(summaries, figure_root, "p_feas", "04_final_pfeas_comparison.png", "Final feasible-route probability"),
        _bar_comparison(summaries, figure_root, "p_opt", "05_final_popt_comparison.png", "Final optimal-route probability"),
        _figure_mass(summaries, figure_root),
        _trace_figure(traces, figure_root, "expected_hc", "07_layer_by_layer_expected_hc.png", "Layer-by-layer expected energy", "$\\langle H_C\\rangle$"),
        _trace_figure(traces, figure_root, "p_feas", "08_layer_by_layer_pfeas.png", "Layer-by-layer feasible probability", "$p_{feas}$"),
        _trace_figure(traces, figure_root, "p_opt", "09_layer_by_layer_popt.png", "Layer-by-layer optimal probability", "$p_{opt}$"),
        _figure_phase(result_root, figure_root),
        _figure_energy(result_root, figure_root),
        _figure_geometry(figure_root),
        _figure_top_states(result_root, figure_root),
        _figure_runtime(result_root, figure_root),
    ]
    for number, algorithm in enumerate(ALGORITHMS, start=15):
        paths.append(_parameter_landscape(result_root, figure_root, summaries, algorithm, number))

    manifest = {
        "source_result_root": _portable(result_root),
        "figures": [_portable(path) for path in paths],
        "regeneration": "python scripts/make_qaoa_dynamics_figures.py",
    }
    (figure_root / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths
