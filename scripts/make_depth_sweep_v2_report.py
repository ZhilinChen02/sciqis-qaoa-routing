"""Generate the English report from the saved Depth-110 main-track files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "depth110_ext" / "v2"


def load_rows(result_root: Path) -> dict[str, list[dict[str, str]]]:
    grouped = {"penalty_x": [], "grover_cost_phase": []}
    with (result_root / "canonical_rows.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            grouped[row["algorithm"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["depth"]))
    return grouped


def shown(value) -> str:
    return "NOT REACHED" if value is None else str(value)


def make_report(result_root: Path) -> str:
    grouped = load_rows(result_root)
    summary = json.loads(
        (result_root / "depth_summary.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (result_root / "experiment_config.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (result_root / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    penalty = summary["penalty_x"]
    grover = summary["grover_cost_phase"]
    penalty_last = grouped["penalty_x"][-1]
    grover_last = grouped["grover_cost_phase"][-1]
    plan_crossover = shown(summary["first_grover_above_penalty_depth"])
    penalty_overtake = shown(summary["first_penalty_x_above_grover_depth"])

    lines = [
        "# Depth-110 QAOA Main Track",
        "",
        "## Research question",
        "",
        "This study compares Penalty-X QAOA with a feasible-subspace Grover-Mixer "
        "QAOA on the fixed course routing instance for every depth from p=1 to p=110.",
        "",
        "The Grover cost layer follows the route-cost operator used in the project plan:",
        "",
        r"\[H_C^G=\sum_i C(P_i)|P_i\rangle\langle P_i|,\qquad U_C(\gamma)|P_i\rangle=e^{-i\gamma C(P_i)}|P_i\rangle.\]",
        "",
        "The incumbent-threshold mask is used only to calculate the auxiliary "
        "better-solution probability (BSP). It is not used in the QAOA evolution.",
        "",
        "## Frozen protocol",
        "",
        f"- Depths: p=1,...,110 for both algorithms ({provenance['v2_main_track_rows']} canonical rows).",
        f"- Base seed: {config['base_seed']}; depth RNG seed: `base_seed + 7919 * p`.",
        "- Classical optimizer: COBYLA with layerwise continuation.",
        f"- Function-evaluation budget: `{config['objective_evaluation_cap_formula']}`.",
        f"- Penalty-X rows retained unchanged: {provenance['penalty_x_rows_reused']}.",
        f"- Route-cost Grover-Mixer rows optimized for this main track: {provenance['cost_phase_grover_rows_rerun']}.",
        "- An exhausted function-evaluation budget is labelled **BUDGET_LIMITED**. "
        "It is not reported as a wall-clock timeout or a crash.",
        "",
        "## Main results",
        "",
        "| Metric | Penalty-X | Route-cost Grover-Mixer |",
        "| --- | ---: | ---: |",
        f"| best depth | {penalty['best_depth']} | {grover['best_depth']} |",
        f"| best p_opt | {penalty['best_p_opt']:.6f} | {grover['best_p_opt']:.6f} |",
        f"| first p_opt >= 0.50 | {shown(penalty['p50_depth'])} | {shown(grover['p50_depth'])} |",
        f"| first p_opt >= 0.90 | {shown(penalty['p90_depth'])} | {shown(grover['p90_depth'])} |",
        f"| first p_opt >= 0.99 | {shown(penalty['p99_depth'])} | {shown(grover['p99_depth'])} |",
        f"| observed saturation-criterion onset | {shown(penalty['saturation_depth'])} | {shown(grover['saturation_depth'])} |",
        f"| p=110 p_opt | {float(penalty_last['p_opt']):.6f} | {float(grover_last['p_opt']):.6f} |",
        f"| p=110 p_feas | {float(penalty_last['p_feas']):.6f} | {float(grover_last['p_feas']):.6f} |",
        f"| optimizer wall time | {penalty['total_optimizer_wall_time_seconds']:.2f} s | {grover['total_optimizer_wall_time_seconds']:.2f} s |",
        f"| budget-limited rows | {penalty['status_counts'].get('BUDGET_LIMITED', 0)} | {grover['status_counts'].get('BUDGET_LIMITED', 0)} |",
        "",
        f"The plan-defined crossover, the first depth at which Grover-Mixer has higher p_opt than Penalty-X, is: **{plan_crossover}**.",
        f"The first later depth at which Penalty-X overtakes the Grover-Mixer is: **{penalty_overtake}** over the saved depth range.",
        "",
        "Saturation follows the pre-registered definition: the onset is the first "
        "depth p for which L=10 consecutive changes satisfy "
        "|p_opt(k)-p_opt(k-1)| < 1e-3 for k=p,...,p+L-1. Under this definition, "
        "the saved trajectories give observed criterion onsets p_sat=2 for "
        "Penalty-X and p_sat=92 for the route-cost Grover-Mixer under the frozen "
        "optimization budget. The Penalty-X value indicates no measurable early "
        "improvement; it is not evidence of successful convergence at p=2.",
        "",
        "## Figures",
        "",
        "![Optimal-route probability](../../../figures/depth110_ext/v2/figure_1_p_opt_vs_depth.png)",
        "",
        "Figure 1 shows optimal-route probability. Hollow markers are budget-limited "
        "observations retained as end-to-end results. The plot marks p50, p90, p99, "
        "both saturation-criterion onsets, the best observed depths, and the "
        "plan-defined crossover at p=1. Penalty-X never overtakes the Grover-Mixer "
        "over p=1,...,110.",
        "",
        "![Feasible-route probability](../../../figures/depth110_ext/v2/figure_2_p_feas_vs_depth.png)",
        "",
        "Figure 2 shows feasible-route probability. The Grover-Mixer has p_feas=1 "
        "because its state is constructed in the 20-route feasible subspace. This is "
        "a structural property of the representation, not evidence of quantum advantage.",
        "",
        "![Normalized regret](../../../figures/depth110_ext/v2/figure_3_regret_vs_depth.png)",
        "",
        "Figure 3 shows normalized regret. Penalty-X uses the expected cost over the "
        "full bitstring space, while the Grover-Mixer uses the expected cost over the "
        "20 feasible routes. The two curves must therefore be interpreted together "
        "with feasibility and invalid probability mass.",
        "",
        "## Interpretation and limitations",
        "",
        "- The route-cost Grover-Mixer remains structurally feasible and reaches "
        "p_opt=0.50, 0.90, and 0.99 at depths 58, 69, and 88, respectively.",
        "- Penalty-X remains dominated by invalid probability mass under this frozen protocol.",
        "- The 218 budget-limited rows are observed results under a fixed evaluation cap. "
        "They do not separate ansatz expressivity from classical optimization difficulty.",
        "- This is an ideal statevector/logical-subspace simulation on one small instance.",
        "- The study does not claim quantum advantage, hardware performance, or scalable routing.",
        "",
        "## Rebuild the analysis from saved data",
        "",
        "These commands recalculate the summary, regenerate the three figures, rebuild "
        "this report, and validate all saved artifacts without rerunning optimization:",
        "",
        "```powershell",
        "python scripts/make_depth_sweep_v2_figures.py",
        "python scripts/make_depth_sweep_v2_report.py",
        "python scripts/verify_depth_sweep_v2.py",
        "```",
        "",
        "To rerun only the route-cost Grover trajectory in a new result directory:",
        "",
        "```powershell",
        "python scripts/run_cost_phase_grover_sweep.py --max-depth 110 --result-root results/depth110_ext/new_grover_run",
        "```",
        "",
        "For a full repository-level reproduction in new directories, first generate "
        "the Penalty-X source rows, then the route-cost Grover rows, and finally build "
        "a new 220-row main track. The two source runners write only the two "
        "algorithms used in the main comparison.",
        "",
        "```powershell",
        "python src/depth_sweep.py --max-depth 110 --result-root results/depth110_ext/reproduction_penalty",
        "python scripts/run_cost_phase_grover_sweep.py --max-depth 110 --result-root results/depth110_ext/reproduction_grover",
        "python scripts/build_depth110_v2.py --penalty-root results/depth110_ext/reproduction_penalty --cost-phase-root results/depth110_ext/reproduction_grover --v2-root results/depth110_ext/reproduction_main",
        "```",
        "",
        "This full rerun is computationally expensive. It is not required to rebuild "
        "the figures, report, or validation checks from the saved v2 artifacts.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    output = args.result_root / "REPORT.md"
    output.write_text(make_report(args.result_root), encoding="utf-8")
    print(f"[report] wrote {output}")


if __name__ == "__main__":
    main()
