"""Validate the corrected 220-row Depth-110 v2 main track."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "depth110_ext" / "v2"
DEFAULT_FIGURE_ROOT = PROJECT_ROOT / "figures" / "depth110_ext" / "v2"
ALGORITHMS = ("penalty_x", "grover_cost_phase")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def saturation_depth(
    rows: list[dict[str, str]],
    tolerance: float = 1e-3,
    window: int = 10,
) -> int | None:
    """Calculate the Section 9.2 saturation onset from saved rows."""

    values = np.asarray([float(row["p_opt"]) for row in rows])
    depths = [int(row["depth"]) for row in rows]
    for start in range(len(values) - window):
        changes = np.abs(np.diff(values[start:start + window + 1]))
        if float(np.max(changes)) < tolerance:
            return depths[start + 1]
    return None


def first_depth_above(
    first: list[dict[str, str]],
    second: list[dict[str, str]],
) -> int | None:
    """Return the first depth where the first p_opt is strictly larger."""

    for first_row, second_row in zip(first, second, strict=True):
        if float(first_row["p_opt"]) > float(second_row["p_opt"]):
            return int(first_row["depth"])
    return None


def verify(result_root: Path, figure_root: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    required = [
        "canonical_rows.csv", "optimized_parameters.json", "runtime_profile.csv",
        "failure_taxonomy.json", "final_distributions.npz", "experiment_config.json",
        "environment.json", "grover_cost_phase_metadata.json", "PROVENANCE.json",
        "depth_summary.json", "REPORT.md", "README-QUICKSTART.md",
    ]
    for name in required:
        check(f"required file {name}", (result_root / name).is_file())
    if not all(item["passed"] for item in checks):
        return {"passed": False, "checks": checks}

    rows = load_rows(result_root / "canonical_rows.csv")
    check("canonical row count is 220", len(rows) == 220, f"got {len(rows)}")
    grouped = {algorithm: [row for row in rows if row["algorithm"] == algorithm] for algorithm in ALGORITHMS}
    check("algorithm set is exact", set(row["algorithm"] for row in rows) == set(ALGORITHMS))
    for algorithm in ALGORITHMS:
        depths = [int(row["depth"]) for row in grouped[algorithm]]
        check(f"{algorithm} has 110 rows", len(depths) == 110, f"got {len(depths)}")
        check(f"{algorithm} depths are contiguous 1..110", depths == list(range(1, 111)))

    numeric_fields = (
        "p_feas", "p_opt", "invalid_mass", "expected_cost_full",
        "expected_cost_feasible", "feasible_conditional_p_opt",
        "normalized_regret", "optimizer_wall_time_seconds",
        "evaluation_budget", "evaluation_budget_used",
    )
    finite = all(
        math.isfinite(float(row[field]))
        for row in rows for field in numeric_fields if row[field] not in ("", "nan")
    )
    check("all required numeric values are finite", finite)
    check("all probabilities are within [0,1]", all(
        -1e-12 <= float(row[field]) <= 1.0 + 1e-9
        for row in rows for field in ("p_feas", "p_opt", "invalid_mass", "feasible_conditional_p_opt")
    ))
    check("p_opt never exceeds p_feas", all(float(row["p_opt"]) <= float(row["p_feas"]) + 1e-12 for row in rows))
    check("Grover p_feas is structurally one", all(abs(float(row["p_feas"]) - 1.0) < 1e-10 for row in grouped["grover_cost_phase"]))
    check("evaluation use never exceeds frozen budget", all(float(row["evaluation_budget_used"]) <= float(row["evaluation_budget"]) for row in rows))
    check("optimizer wall time is nonnegative", all(float(row["optimizer_wall_time_seconds"]) >= 0.0 for row in rows))

    config = json.loads((result_root / "experiment_config.json").read_text(encoding="utf-8"))
    check("v2 algorithm identities are exact", config["algorithms"] == list(ALGORITHMS))
    check("threshold phase is excluded", config["threshold_used_in_evolution"] is False)
    metadata = json.loads((result_root / "grover_cost_phase_metadata.json").read_text(encoding="utf-8"))
    check("Grover phase operator is route-cost diagonal", metadata["phase_operator"] == "route_cost_diagonal")
    check("route-cost phase has several cost values", len(metadata["route_cost_unique_values"]) > 1)
    check("threshold is not used in Grover evolution", metadata["incumbent_threshold_used_in_evolution"] is False)

    summary = json.loads((result_root / "depth_summary.json").read_text(encoding="utf-8"))
    for algorithm in ALGORITHMS:
        calculated = saturation_depth(grouped[algorithm])
        reported = summary[algorithm]["saturation_depth"]
        check(
            f"{algorithm} saturation depth matches Section 9.2",
            reported == calculated,
            f"reported {reported}, calculated {calculated}",
        )
    check("Penalty-X saturation onset is p=2", summary["penalty_x"]["saturation_depth"] == 2)
    check("Grover saturation onset is p=92", summary["grover_cost_phase"]["saturation_depth"] == 92)
    plan_crossover = first_depth_above(
        grouped["grover_cost_phase"], grouped["penalty_x"]
    )
    penalty_overtake = first_depth_above(
        grouped["penalty_x"], grouped["grover_cost_phase"]
    )
    check(
        "plan-defined Grover-over-Penalty crossover matches saved rows",
        summary["first_grover_above_penalty_depth"] == plan_crossover,
        f"reported {summary['first_grover_above_penalty_depth']}, calculated {plan_crossover}",
    )
    check("plan-defined crossover is p=1", plan_crossover == 1)
    check(
        "Penalty-X overtake observation matches saved rows",
        summary["first_penalty_x_above_grover_depth"] == penalty_overtake,
        f"reported {summary['first_penalty_x_above_grover_depth']}, calculated {penalty_overtake}",
    )
    check("Penalty-X never overtakes Grover over p=1..110", penalty_overtake is None)

    parameters = json.loads((result_root / "optimized_parameters.json").read_text(encoding="utf-8"))
    check("parameter file has 220 entries", len(parameters) == 220, f"got {len(parameters)}")
    expected_keys = {f"{algorithm}_p{depth}_seed2601" for algorithm in ALGORITHMS for depth in range(1, 111)}
    check("parameter keys match all main-track rows", set(parameters) == expected_keys)
    check("each parameter vector has length 2p", all(
        len(payload["parameters"]) == 2 * int(payload["depth"])
        for payload in parameters.values()
    ))

    failure = json.loads((result_root / "failure_taxonomy.json").read_text(encoding="utf-8"))
    check("failure taxonomy has 220 entries", len(failure) == 220, f"got {len(failure)}")
    check("evaluation caps are named BUDGET_LIMITED", all(
        (item["status"] == "BUDGET_LIMITED")
        if item["optimizer_reason"].startswith("evaluation_budget_exhausted") else True
        for item in failure
    ))
    check("no evaluation cap is mislabeled TIMEOUT", not any(
        item["status"] == "TIMEOUT" and item["optimizer_reason"].startswith("evaluation_budget_exhausted")
        for item in failure
    ))

    with np.load(result_root / "final_distributions.npz") as payload:
        check("distribution archive has 220 arrays", len(payload.files) == 220, f"got {len(payload.files)}")
        check("distribution keys match main track", set(payload.files) == expected_keys)
        max_norm_error = max(abs(float(np.sum(payload[key])) - 1.0) for key in payload.files)
        check("every saved distribution is normalized", max_norm_error < 1e-9, f"max error {max_norm_error:.3e}")
        check("Penalty-X distributions have 2^14 entries", all(payload[key].shape == (16384,) for key in payload.files if key.startswith("penalty_x_")))
        check("Grover distributions have 20 feasible entries", all(payload[key].shape == (20,) for key in payload.files if key.startswith("grover_cost_phase_")))

    figure_names = (
        "figure_1_p_opt_vs_depth.png",
        "figure_2_p_feas_vs_depth.png",
        "figure_3_regret_vs_depth.png",
    )
    for name in figure_names:
        check(f"figure exists and is nonempty: {name}", (figure_root / name).is_file() and (figure_root / name).stat().st_size > 10_000)

    provenance = json.loads((result_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    check("old GM-Th rows are excluded", provenance["old_gm_th_rows_included"] == 0)
    check(
        "provenance contains no local absolute path",
        all(
            not str(value).startswith(("C:\\", "D:\\"))
            for value in provenance.get("source_sha256", {}).keys()
        ),
    )
    passed = all(bool(item["passed"]) for item in checks)
    return {
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(not bool(item["passed"]) for item in checks),
        "maximum_distribution_normalization_error": max_norm_error,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    args = parser.parse_args()
    print(f"[verify-v2] checking {args.result_root}")
    summary = verify(args.result_root, args.figure_root)
    output = args.result_root / "VALIDATION_SUMMARY.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[verify-v2] wrote {output}")
    if not summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
