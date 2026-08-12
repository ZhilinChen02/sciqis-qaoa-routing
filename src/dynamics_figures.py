"""Figures regenerated entirely from saved QAOA-dynamics study data."""

from __future__ import annotations

import csv
import json
from math import pi
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np

from dynamics_study import (
    ALGORITHMS,
    DISPLAY_NAMES,
    GROVER_FEASIBLE,
    GROVER_GLOBAL,
    PENALTY_X,
    write_json,
)
from graph import DEFAULT_GRAPH_PATH, get_edge_order, load_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    PENALTY_X: "#4C78A8",
    GROVER_GLOBAL: "#F58518",
    GROVER_FEASIBLE: "#54A24B",
}
CHECKPOINT_MARKERS = {"initial": "o", "cost": "s", "mixer": "D"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def _save(fig: plt.Figure, figure_root: Path, filename: str) -> Path:
    figure_root.mkdir(parents=True, exist_ok=True)
    path = figure_root / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _portable_path(path: Path) -> str:
    """Prefer repository-relative paths while preserving external overrides."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _figure_1_graph(figure_root: Path) -> Path:
    graph = load_graph(DEFAULT_GRAPH_PATH)
    optimal_route = (0, 1, 2, 4, 5, 6)
    optimal_edges = set(zip(optimal_route[:-1], optimal_route[1:]))
    positions = {
        0: (0.0, 0.2),
        1: (1.0, 1.0),
        2: (2.0, 1.25),
        3: (2.15, 0.15),
        4: (3.25, 1.05),
        5: (4.15, 0.35),
        6: (5.25, 0.75),
    }
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ordinary = [edge for edge in get_edge_order(graph) if edge not in optimal_edges]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=ordinary,
        edge_color="#9AA0A6",
        width=1.5,
        arrows=True,
        arrowsize=17,
        connectionstyle="arc3,rad=0.03",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=list(optimal_edges),
        edge_color="#D62728",
        width=3.4,
        arrows=True,
        arrowsize=19,
        connectionstyle="arc3,rad=0.03",
        ax=ax,
    )
    node_colors = ["#2E86AB" if node == 0 else "#7A5195" if node == 6 else "#F2C14E" for node in graph]
    nx.draw_networkx_nodes(
        graph, positions, node_color=node_colors, node_size=850, edgecolors="white", linewidths=2, ax=ax
    )
    nx.draw_networkx_labels(graph, positions, font_size=12, font_weight="bold", font_color="white", ax=ax)
    labels = {(u, v): graph.edges[u, v]["weight"] for u, v in graph.edges}
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=labels,
        font_size=9,
        label_pos=0.52,
        rotate=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        ax=ax,
    )
    ax.text(0.0, -0.35, "source", ha="center", color="#2E86AB", fontweight="bold")
    ax.text(5.25, 0.15, "target", ha="center", color="#7A5195", fontweight="bold")
    ax.set_title("Fixed 7-node directed routing graph\noptimal route 0→1→2→4→5→6, cost 10", fontsize=14)
    ax.axis("off")
    return _save(fig, figure_root, "01_fixed_weighted_routing_graph.png")


def _box(ax: plt.Axes, x: float, text: str, color: str, *, width: float = 0.15) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, 0.37),
        width,
        0.27,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x, 0.505, text, ha="center", va="center", fontsize=9.4, wrap=True)


def _figure_2_pipeline(figure_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 3.7))
    xs = np.linspace(0.08, 0.92, 7)
    texts = (
        "weighted\ndirected graph",
        "14 edge bits\n$x_e\\in\\{0,1\\}$",
        "route cost\n+$A\\sum_v f_v^2$",
        "QUBO → Ising\ndiagonal $H_C$",
        "cost phase\n$e^{-i\\gamma H_C}$",
        "chosen mixer\n$e^{-i\\beta H_M}$",
        "measure → decode\n$p_{feas},p_{opt}$",
    )
    colors = ("#DDEBF7", "#FFF2CC", "#FCE4D6", "#E2F0D9", "#E4DFEC", "#D9EAD3", "#DDEBF7")
    for x, label, color in zip(xs, texts, colors):
        _box(ax, float(x), label, color, width=0.125)
    for left, right in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left + 0.067, 0.505),
                (right - 0.067, 0.505),
                arrowstyle="-|>",
                mutation_scale=13,
                color="#555555",
                linewidth=1.3,
            )
        )
    ax.text(0.64, 0.17, "repeat p layers: objective energy → relative phase → mixer interference", ha="center", fontsize=11, fontweight="bold", color="#444444")
    ax.set_title("Routing QUBO → Ising → alternating-operator QAOA pipeline", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return _save(fig, figure_root, "02_qubo_ising_qaoa_pipeline.png")


def _figure_3_three_spaces(figure_root: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    panels = (
        (
            PENALTY_X,
            "all 16,384 edge bitstrings",
            "$H_X=\\sum_j X_j$",
            "local Hamming-neighbor mixing",
        ),
        (
            GROVER_GLOBAL,
            "all 16,384 edge bitstrings",
            "$H_{G,all}=|s_{all}\\rangle\\langle s_{all}|$",
            "global rank-one projector mixing",
        ),
        (
            GROVER_FEASIBLE,
            "20 feasible routes only",
            "$H_{G,F}=|s_F\\rangle\\langle s_F|$",
            "projector mixing inside feasible basis",
        ),
    )
    for ax, (algorithm, space, mixer, interpretation) in zip(axes, panels):
        ax.add_patch(plt.Circle((0.5, 0.58), 0.31, color=COLORS[algorithm], alpha=0.16, ec=COLORS[algorithm], lw=2))
        if algorithm == GROVER_FEASIBLE:
            for angle in np.linspace(0, 2 * np.pi, 20, endpoint=False):
                ax.plot(0.5 + 0.24 * np.cos(angle), 0.58 + 0.24 * np.sin(angle), ".", color=COLORS[algorithm], ms=5)
        else:
            rng = np.random.default_rng(12 if algorithm == PENALTY_X else 13)
            points = rng.normal(size=(90, 2))
            points /= np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
            radii = 0.27 * np.sqrt(rng.random((90, 1)))
            points = np.asarray([0.5, 0.58]) + points * radii
            ax.scatter(points[:, 0], points[:, 1], s=7, color=COLORS[algorithm], alpha=0.45)
        ax.text(0.5, 0.95, DISPLAY_NAMES[algorithm], ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(0.5, 0.59, space, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(0.5, 0.20, mixer, ha="center", fontsize=11)
        ax.text(0.5, 0.09, interpretation, ha="center", fontsize=9, color="#555555")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle("Same routing objective, three search-space/mixer geometries", fontsize=15)
    return _save(fig, figure_root, "03_three_search_spaces_and_mixers.png")


def _bar_comparison(
    summaries: list[dict[str, Any]], figure_root: Path, metric: str, filename: str, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(2)
    width = 0.24
    for offset, algorithm in enumerate(ALGORITHMS):
        rows = sorted((row for row in summaries if row["algorithm"] == algorithm), key=lambda row: row["p"])
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(x + (offset - 1) * width, values, width, color=COLORS[algorithm], label=DISPLAY_NAMES[algorithm])
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=8)
    ax.set_xticks(x, ["p=1", "p=2"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("probability mass")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    return _save(fig, figure_root, filename)


def _figure_6_mass(summaries: list[dict[str, Any]], figure_root: Path) -> Path:
    ordered = sorted(summaries, key=lambda row: (ALGORITHMS.index(row["algorithm"]), row["p"]))
    labels = [f"{row['algorithm'].replace('_', ' ')}\np={row['p']}" for row in ordered]
    optimum = np.asarray([row["p_opt"] for row in ordered], dtype=float)
    feasible = np.asarray([row["p_feas"] - row["p_opt"] for row in ordered], dtype=float)
    invalid = np.asarray([row["invalid_mass"] for row in ordered], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5.3))
    x = np.arange(len(ordered))
    ax.bar(x, optimum, color="#2CA02C", label="optimal feasible")
    ax.bar(x, feasible, bottom=optimum, color="#9ED17B", label="other feasible")
    ax.bar(x, invalid, bottom=optimum + feasible, color="#D95F5F", label="infeasible")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("probability mass")
    ax.set_title("Final probability-mass decomposition")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.22)
    return _save(fig, figure_root, "06_probability_mass_decomposition.png")


def _trace_figure(
    rows: list[dict[str, str]], figure_root: Path, metric: str, filename: str, title: str, ylabel: str
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for algorithm in ALGORITHMS:
        chosen = sorted(
            (row for row in rows if row["algorithm"] == algorithm and int(row["p"]) == 2),
            key=lambda row: int(row["checkpoint_index"]),
        )
        x = np.arange(len(chosen))
        y = np.asarray([float(row[metric]) for row in chosen])
        ax.plot(x, y, color=COLORS[algorithm], lw=2.1, label=DISPLAY_NAMES[algorithm])
        for index, row in enumerate(chosen):
            ax.scatter(index, y[index], marker=CHECKPOINT_MARKERS[row["operation"]], s=65, color=COLORS[algorithm], edgecolor="white", zorder=3)
    checkpoint_labels = ["Initial", "Cost-1", "Mixer-1", "Cost-2", "Mixer-2"]
    ax.set_xticks(np.arange(5), checkpoint_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    if metric in ("p_feas", "p_opt"):
        ax.set_ylim(-0.02, 1.04)
    if metric == "expected_hc":
        ax.text(0.5, 0.03, "Cost markers overlay the preceding energy: only mixer steps change ⟨H_C⟩", transform=ax.transAxes, ha="center", fontsize=9, color="#444444")
    return _save(fig, figure_root, filename)


def _phase_data(result_root: Path, algorithm: str) -> list[dict[str, str]]:
    return _read_csv(result_root / "phase_analysis" / f"{algorithm}_p2_seed2601.csv")


def _figure_10_phase(result_root: Path, figure_root: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    top_rows = _read_csv(result_root / "top_state_evolution.csv")
    for row_index, algorithm in enumerate(ALGORITHMS):
        data = _phase_data(result_root, algorithm)
        # Use the first complete cost/mixer pair saved for every method.  At the
        # retained Penalty-X point gamma_1≈0, so that row honestly shows a
        # near-identity cost step rather than substituting post-hoc parameters.
        layer = 1
        cost = [row for row in data if row["checkpoint"] == f"Cost-{layer}"]
        mixed = [row for row in data if row["checkpoint"] == f"Mixer-{layer}"]
        ax = axes[row_index, 0]
        for feasible_value, color, label in ((False, "#D95F5F", "infeasible"), (True, "#2CA02C", "feasible")):
            selected = [row for row in cost if _bool(row["feasible"]) == feasible_value]
            if selected:
                ax.scatter(
                    [_float(row, "energy") for row in selected],
                    [_float(row, "phase") for row in selected],
                    s=8 if algorithm != GROVER_FEASIBLE else 24,
                    alpha=0.22 if algorithm != GROVER_FEASIBLE else 0.8,
                    color=color,
                    label=label,
                )
        ax.set_ylabel("wrapped phase [rad]")
        ax.set_title(f"{DISPLAY_NAMES[algorithm]}\nafter Cost-{layer}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.18)

        ax = axes[row_index, 1]
        ax.scatter(
            [_float(row, "energy") for row in cost],
            [max(_float(row, "probability"), 1e-16) for row in cost],
            s=7 if algorithm != GROVER_FEASIBLE else 26,
            alpha=0.3,
            color="#777777",
            label="after cost",
        )
        ax.scatter(
            [_float(row, "energy") for row in mixed],
            [max(_float(row, "probability"), 1e-16) for row in mixed],
            s=7 if algorithm != GROVER_FEASIBLE else 26,
            alpha=0.45,
            color=COLORS[algorithm],
            label="after mixer",
        )
        ax.set_yscale("log")
        ax.set_ylabel("basis probability")
        maximum_delta = max(
            abs(_float(left, "probability") - _float(right, "probability"))
            for left, right in zip(cost, mixed)
        )
        ax.set_title(
            "phase → probability redistribution"
            if maximum_delta > 1e-10
            else "retained mixer angle: no probability change"
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.18)

        ax = axes[row_index, 2]
        if algorithm == GROVER_FEASIBLE:
            cost_by_index = {int(row["basis_index"]): row for row in cost}
            mixed_by_index = {int(row["basis_index"]): row for row in mixed}
            indices: Iterable[int] = sorted(cost_by_index)
        else:
            selected_top = [
                row
                for row in top_rows
                if row["algorithm"] == algorithm
                and row["checkpoint"] in (f"Cost-{layer}", f"Mixer-{layer}")
            ]
            cost_by_index = {
                int(row["basis_index"]): row
                for row in selected_top
                if row["checkpoint"] == f"Cost-{layer}"
            }
            mixed_by_index = {
                int(row["basis_index"]): row
                for row in selected_top
                if row["checkpoint"] == f"Mixer-{layer}"
            }
            indices = sorted(set(cost_by_index) & set(mixed_by_index))
        for index in indices:
            left, right = cost_by_index[index], mixed_by_index[index]
            left_complex = np.sqrt(_float(left, "probability")) * np.exp(1j * _float(left, "phase"))
            right_complex = np.sqrt(_float(right, "probability")) * np.exp(1j * _float(right, "phase"))
            color = "#2CA02C" if _bool(left["exact_optimal"]) else "#D95F5F" if not _bool(left["feasible"]) else COLORS[algorithm]
            ax.annotate("", xy=(right_complex.real, right_complex.imag), xytext=(left_complex.real, left_complex.imag), arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.55, "lw": 0.8})
            ax.scatter([left_complex.real], [left_complex.imag], marker="x", color=color, s=25)
            ax.scatter([right_complex.real], [right_complex.imag], marker="o", color=color, s=20)
        ax.axhline(0, color="#BBBBBB", lw=0.7)
        ax.axvline(0, color="#BBBBBB", lw=0.7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"complex amplitudes: Cost-{layer} × → Mixer-{layer} ○")
        ax.set_xlabel("Re amplitude")
        ax.set_ylabel("Im amplitude")
        ax.grid(alpha=0.18)
    for ax in axes[-1, :2]:
        ax.set_xlabel("raw basis energy")
    fig.suptitle("Cost-dependent phase and mixer interference (optimized p=2)", fontsize=15, y=1.01)
    fig.tight_layout()
    return _save(fig, figure_root, "10_phase_and_interference_evolution.png")


def _figure_11_energy(result_root: Path, figure_root: Path) -> Path:
    rows = _read_csv(result_root / "energy_landscape.csv")
    total = np.asarray([_float(row, "total_qubo_energy") for row in rows])
    route = np.asarray([_float(row, "routing_term") for row in rows])
    penalty = np.asarray([_float(row, "flow_penalty") for row in rows])
    feasible = np.asarray([_bool(row["feasible"]) for row in rows])
    optimal = np.asarray([_bool(row["exact_optimal"]) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.linspace(float(total.min()), float(np.percentile(total, 99.5)), 50)
    axes[0, 0].hist(total[~feasible], bins=bins, alpha=0.65, color="#D95F5F", label=f"infeasible ({np.sum(~feasible):,})")
    axes[0, 0].hist(total[feasible], bins=bins, alpha=0.85, color="#2CA02C", label=f"feasible ({np.sum(feasible)})")
    axes[0, 0].axvline(total[optimal][0], color="black", ls="--", lw=1.5, label="optimum E=10")
    axes[0, 0].set(title="Total Penalty-QUBO energy", xlabel="QUBO energy", ylabel="state count")
    axes[0, 0].legend(fontsize=8)

    values, counts = np.unique(penalty, return_counts=True)
    axes[0, 1].bar(values, counts, color="#8064A2", width=0.75)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Flow-penalty distribution", xlabel="$P_{flow}$", ylabel="state count (log)")

    feasible_route = route[feasible]
    values, counts = np.unique(feasible_route, return_counts=True)
    axes[1, 0].bar(values, counts, color="#54A24B", width=0.75)
    axes[1, 0].axvline(10, color="black", ls="--", lw=1.3)
    axes[1, 0].set(title="Routing-cost distribution of the 20 feasible routes", xlabel="route cost", ylabel="route count")

    order = np.argsort(total, kind="stable")[:120]
    colors = np.where(feasible[order], "#2CA02C", "#D95F5F")
    axes[1, 1].scatter(np.arange(len(order)), total[order], c=colors, s=18)
    axes[1, 1].axhline(10, color="black", ls="--", lw=1.1)
    axes[1, 1].set(title="120 lowest-energy bitstrings", xlabel="sorted state rank", ylabel="QUBO energy")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Full 2¹⁴-state energy landscape at frozen penalty A=6", fontsize=15)
    fig.tight_layout()
    return _save(fig, figure_root, "11_energy_landscape.png")


def _figure_12_geometry(figure_root: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    cube = nx.hypercube_graph(3)
    positions_3d = nx.spring_layout(cube, seed=2601)
    nx.draw_networkx(cube, positions_3d, ax=axes[0], node_size=310, node_color="#9CC2E5", edge_color="#4C78A8", labels={node: "".join(map(str, node)) for node in cube}, font_size=7)
    axes[0].set_title("X mixer\nlocal edges at Hamming distance 1")

    complete = nx.complete_graph(8)
    positions = nx.circular_layout(complete)
    nx.draw_networkx_edges(complete, positions, ax=axes[1], alpha=0.18, width=0.8, edge_color="#F58518")
    nx.draw_networkx_nodes(complete, positions, ax=axes[1], node_size=320, node_color="#FBC58A")
    nx.draw_networkx_labels(complete, positions, ax=axes[1], labels={index: f"{index:03b}" for index in complete}, font_size=7)
    axes[1].set_title("Global Grover projector\nconceptual global support")

    nx.draw_networkx_nodes(complete, positions, ax=axes[2], node_size=270, node_color="#DDDDDD", alpha=0.45)
    feasible_nodes = [0, 3, 7]
    feasible_graph = complete.subgraph(feasible_nodes)
    nx.draw_networkx_edges(feasible_graph, positions, ax=axes[2], width=2, edge_color="#54A24B")
    nx.draw_networkx_nodes(feasible_graph, positions, ax=axes[2], node_size=390, node_color="#8FD175")
    nx.draw_networkx_labels(complete, positions, ax=axes[2], labels={index: f"{index:03b}" for index in complete}, font_size=7)
    axes[2].set_title("Feasible Grover projector\nlogical support restricted to F")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Conceptual small-q mixer geometry (q=3 illustration, not the 14-qubit state graph)", fontsize=14)
    return _save(fig, figure_root, "12_conceptual_search_space_geometry.png")


def _figure_13_top_states(result_root: Path, figure_root: Path) -> Path:
    rows = _read_csv(result_root / "top_state_evolution.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=False)
    for ax, algorithm in zip(axes, ALGORITHMS):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        by_index: dict[int, list[dict[str, str]]] = {}
        for row in selected:
            by_index.setdefault(int(row["basis_index"]), []).append(row)
        ordered_indices = sorted(
            by_index,
            key=lambda index: (
                not _bool(by_index[index][0]["exact_optimal"]),
                not _bool(by_index[index][0]["feasible"]),
                float(by_index[index][0]["energy"]),
                index,
            ),
        )[:6]
        for index in ordered_indices:
            sequence = sorted(by_index[index], key=lambda row: int(row["checkpoint_index"]))
            first = sequence[0]
            if _bool(first["exact_optimal"]):
                label, color, lw = "exact optimum", "#2CA02C", 2.8
            elif _bool(first["feasible"]):
                label, color, lw = f"feasible E={float(first['energy']):g}", COLORS[algorithm], 1.4
            else:
                label, color, lw = f"infeasible E={float(first['energy']):g}", "#D95F5F", 1.2
            ax.plot(
                [int(row["checkpoint_index"]) for row in sequence],
                [_float(row, "probability") for row in sequence],
                marker="o",
                color=color,
                alpha=0.95 if _bool(first["exact_optimal"]) else 0.65,
                lw=lw,
                label=label,
            )
        ax.set_xticks(range(5), ["Init", "C1", "M1", "C2", "M2"])
        ax.set_title(DISPLAY_NAMES[algorithm])
        ax.set_xlabel("checkpoint")
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7)
    axes[0].set_ylabel("basis-state probability")
    fig.suptitle("Selected low-energy and representative-state probability evolution (optimized p=2)", fontsize=14)
    fig.tight_layout()
    return _save(fig, figure_root, "13_top_state_probability_evolution.png")


def _figure_14_runtime(result_root: Path, figure_root: Path) -> Path:
    rows = _read_csv(result_root / "runtime_profile.csv")
    setup = [row for row in rows if row["scope"] == "setup"]
    run_rows = [row for row in rows if row["scope"] == "run"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = [row["component"].replace("_seconds", "").replace("_", " ") for row in setup]
    values = [_float(row, "seconds") for row in setup]
    y_positions = np.arange(len(labels))
    left_limit = min(values) / 2.0
    axes[0].hlines(y_positions, left_limit, values, color="#7A9E9F", linewidth=3)
    axes[0].scatter(values, y_positions, color="#3D6667", s=42, zorder=3)
    for y_value, seconds in zip(y_positions, values):
        axes[0].annotate(
            f"{seconds:.3g} s",
            (seconds, y_value),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlim(left_limit, max(values) * 2.2)
    axes[0].set_yticks(y_positions, labels, fontsize=8)
    axes[0].set_xlabel("measured seconds (log scale)")
    axes[0].set_title("Measured model/state/mixer preprocessing")
    axes[0].grid(axis="x", alpha=0.2)

    run_ids = list(dict.fromkeys(row["run_id"] for row in run_rows))
    optimize = np.asarray([
        next(_float(row, "seconds") for row in run_rows if row["run_id"] == run_id and row["component"] == "optimize_evaluate_seconds")
        for run_id in run_ids
    ])
    analysis = np.asarray([
        next(_float(row, "seconds") for row in run_rows if row["run_id"] == run_id and row["component"] == "decode_analysis_trace_seconds")
        for run_id in run_ids
    ])
    x = np.arange(len(run_ids))
    axes[1].bar(x, optimize, color="#4C78A8", label="optimize/evaluate")
    axes[1].bar(x, analysis, bottom=optimize, color="#F2CF5B", label="decode/analysis trace")
    axes[1].set_xticks(x, [run_id.replace("_seed2601", "") for run_id in run_ids], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("measured seconds")
    axes[1].set_title("Per-run measured runtime")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Runtime breakdown (no fabricated or zero-filled components)", fontsize=14)
    fig.tight_layout()
    return _save(fig, figure_root, "14_runtime_breakdown.png")


def _parameter_landscape(
    result_root: Path, figure_root: Path, summaries: list[dict[str, Any]], algorithm: str, number: int
) -> Path:
    rows = _read_csv(result_root / "parameter_landscapes" / f"{algorithm}_p1.csv")
    gammas = sorted(set(_float(row, "gamma") for row in rows))
    betas = sorted(set(_float(row, "beta") for row in rows))
    metrics = (
        ("expected_hc", "expected raw $\\langle H_C\\rangle$"),
        ("p_feas", "$p_{feas}$"),
        ("p_opt", "$p_{opt}$"),
    )
    optimized = next(row for row in summaries if row["algorithm"] == algorithm and int(row["p"]) == 1)
    parameter = optimized["optimized_parameters"]
    gamma_star, beta_star = float(parameter[0]), float(parameter[1])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, (metric, title) in zip(axes, metrics):
        values = np.asarray([_float(row, metric) for row in rows]).reshape(len(gammas), len(betas))
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=(0.0, pi, 0.0, 2.0 * pi),
            cmap="viridis",
        )
        ax.scatter([beta_star], [gamma_star], marker="*", s=150, color="white", edgecolor="black", linewidth=0.8, label="optimizer result")
        ax.set_xlabel("β")
        ax.set_ylabel("γ")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.82)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Frozen exploratory p=1 landscape — {DISPLAY_NAMES[algorithm]}", fontsize=14)
    fig.tight_layout()
    return _save(fig, figure_root, f"{number:02d}_p1_parameter_landscape_{algorithm}.png")


def generate_figures_from_saved(result_root: Path, figure_root: Path) -> list[Path]:
    """Generate every required figure without rerunning optimization."""

    summaries = _read_json(result_root / "final_summary.json")
    traces = _read_csv(result_root / "dynamics_trace.csv")
    paths = [
        _figure_1_graph(figure_root),
        _figure_2_pipeline(figure_root),
        _figure_3_three_spaces(figure_root),
        _bar_comparison(summaries, figure_root, "p_feas", "04_final_pfeas_comparison.png", "Final feasible-route probability mass"),
        _bar_comparison(summaries, figure_root, "p_opt", "05_final_popt_comparison.png", "Final exact-optimal-route probability mass"),
        _figure_6_mass(summaries, figure_root),
        _trace_figure(traces, figure_root, "expected_hc", "07_layer_by_layer_expected_hc.png", "Layer-by-layer expected raw cost-Hamiltonian energy (optimized p=2)", "$\\langle H_C\\rangle$"),
        _trace_figure(traces, figure_root, "p_feas", "08_layer_by_layer_pfeas.png", "Layer-by-layer feasible probability mass (optimized p=2)", "$p_{feas}$"),
        _trace_figure(traces, figure_root, "p_opt", "09_layer_by_layer_popt.png", "Layer-by-layer optimal probability mass (optimized p=2)", "$p_{opt}$"),
        _figure_10_phase(result_root, figure_root),
        _figure_11_energy(result_root, figure_root),
        _figure_12_geometry(figure_root),
        _figure_13_top_states(result_root, figure_root),
        _figure_14_runtime(result_root, figure_root),
    ]
    for number, algorithm in enumerate(ALGORITHMS, start=15):
        paths.append(_parameter_landscape(result_root, figure_root, summaries, algorithm, number))
    write_json(
        figure_root / "figure_manifest.json",
        {
            "source_result_root": _portable_path(result_root),
            "figures": [_portable_path(path) for path in paths],
            "regeneration": "python scripts/make_qaoa_dynamics_figures.py",
        },
    )
    return paths
