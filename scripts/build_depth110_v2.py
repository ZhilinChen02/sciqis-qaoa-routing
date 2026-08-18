"""Build the 220-row Depth-110 main track from two source trajectories.

The inputs are a Penalty-X trajectory and a route-cost Grover-Mixer trajectory.
Only these two algorithm identities are accepted in the combined result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PENALTY = PROJECT_ROOT / "results" / "depth110_ext" / "penalty_x_raw"
DEFAULT_COST_PHASE = PROJECT_ROOT / "results" / "depth110_ext" / "v2_grover_cost_phase_raw"
DEFAULT_V2 = PROJECT_ROOT / "results" / "depth110_ext" / "v2"
PENALTY_X = "penalty_x"
GROVER_COST_PHASE = "grover_cost_phase"
SEED = 2601


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_half(rows: list[dict[str, str]], algorithm: str) -> None:
    selected = [row for row in rows if row["algorithm"] == algorithm]
    depths = [int(row["depth"]) for row in selected]
    if depths != list(range(1, 111)):
        raise ValueError(f"{algorithm}_depths_are_not_1_to_110")
    if len(selected) != 110:
        raise ValueError(f"{algorithm}_row_count_is_not_110")


def classify_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for row in rows:
        reason = row["optimizer_reason"]
        if reason.startswith("evaluation_budget_exhausted"):
            status = "BUDGET_LIMITED"
        elif reason.startswith("optimizer_failed"):
            status = "OPTIMIZER_FAILED"
        else:
            status = "COMPLETED"
        payload.append(
            {
                "algorithm": row["algorithm"],
                "depth": row["depth"],
                "status": status,
                "optimizer_reason": reason,
            }
        )
    return payload


def build(penalty_source: Path, cost_phase: Path, v2: Path) -> dict[str, object]:
    if v2.exists() and any(v2.iterdir()):
        raise FileExistsError(f"v2_output_must_be_empty_or_new:{v2}")

    canonical_fields, penalty_source_rows = read_csv(
        penalty_source / "canonical_rows.csv"
    )
    cost_fields, new_rows = read_csv(cost_phase / "canonical_rows.csv")
    if canonical_fields != cost_fields:
        raise ValueError("canonical_csv_schema_mismatch")

    penalty_rows = [
        row for row in penalty_source_rows if row["algorithm"] == PENALTY_X
    ]
    grover_rows = [row for row in new_rows if row["algorithm"] == GROVER_COST_PHASE]
    validate_half(penalty_rows, PENALTY_X)
    validate_half(grover_rows, GROVER_COST_PHASE)
    combined_rows = penalty_rows + grover_rows

    v2.mkdir(parents=True, exist_ok=True)
    write_csv(v2 / "canonical_rows.csv", canonical_fields, combined_rows)

    runtime_fields, penalty_runtime = read_csv(
        penalty_source / "runtime_profile.csv"
    )
    new_runtime_fields, new_runtime = read_csv(cost_phase / "runtime_profile.csv")
    if runtime_fields != new_runtime_fields:
        raise ValueError("runtime_csv_schema_mismatch")
    combined_runtime = (
        [row for row in penalty_runtime if row["algorithm"] == PENALTY_X]
        + [row for row in new_runtime if row["algorithm"] == GROVER_COST_PHASE]
    )
    write_csv(v2 / "runtime_profile.csv", runtime_fields, combined_runtime)

    penalty_params = json.loads(
        (penalty_source / "optimized_parameters.json").read_text(encoding="utf-8")
    )
    new_params = json.loads((cost_phase / "optimized_parameters.json").read_text(encoding="utf-8"))
    parameters = {
        **{
            key: value
            for key, value in penalty_params.items()
            if value["algorithm"] == PENALTY_X
        },
        **{key: value for key, value in new_params.items() if value["algorithm"] == GROVER_COST_PHASE},
    }
    if len(parameters) != 220:
        raise ValueError(f"parameter_entry_count_is_not_220:{len(parameters)}")
    (v2 / "optimized_parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    failure_rows = classify_rows(combined_rows)
    (v2 / "failure_taxonomy.json").write_text(
        json.dumps(failure_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    distributions: dict[str, np.ndarray] = {}
    with np.load(penalty_source / "final_distributions.npz") as penalty_npz:
        for key in penalty_npz.files:
            if key.startswith(f"{PENALTY_X}_"):
                distributions[key] = np.asarray(
                    penalty_npz[key], dtype=np.float64
                )
    with np.load(cost_phase / "final_distributions.npz") as new_npz:
        for key in new_npz.files:
            if key.startswith(f"{GROVER_COST_PHASE}_"):
                distributions[key] = np.asarray(new_npz[key], dtype=np.float64)
    if len(distributions) != 220:
        raise ValueError(f"distribution_count_is_not_220:{len(distributions)}")
    np.savez_compressed(v2 / "final_distributions.npz", **distributions)

    run_metadata = json.loads((cost_phase / "run_metadata.json").read_text(encoding="utf-8"))
    if run_metadata.get("phase_operator") != "route_cost_diagonal":
        raise ValueError("cost_phase_metadata_missing")
    if run_metadata.get("incumbent_threshold_used_in_evolution") is not False:
        raise ValueError("threshold_phase_still_used_in_evolution")
    (v2 / "grover_cost_phase_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    config = {
        "schema": "dtu-sciqis-qaoa-depth110-main-track",
        "version": "2.0",
        "title": "Depth-110 QAOA Main Track: Penalty-X vs Cost-Phase Grover-Mixer",
        "algorithms": [PENALTY_X, GROVER_COST_PHASE],
        "depths": list(range(1, 111)),
        "base_seed": SEED,
        "depth_rng_seed_formula": "base_seed + 7919 * p",
        "optimizer": "COBYLA",
        "objective_evaluation_cap_formula": "80 + 1.5 * (2p), clamped to [60, 400]",
        "initialization": (
            "deterministic depth-derived initialization from base seed; "
            "p>1 uses layerwise continuation"
        ),
        "grover_cost_operator": "H_C^G = sum_i C(P_i) |P_i><P_i|",
        "grover_phase": "exp(-i * gamma * C(P_i))",
        "threshold_used_in_evolution": False,
        "threshold_mask_use": "BSP diagnostic only",
        "penalty_x_source": "verified Penalty-X source trajectory",
        "old_threshold_phase_grover_included": False,
    }
    (v2 / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "optimizer": "scipy.optimize.minimize(method=COBYLA)",
        "statevector_mode": "ideal_exact_numpy_complex128",
        "depth_range_frozen": "1..110 inclusive",
        "algorithms_frozen": [PENALTY_X, GROVER_COST_PHASE],
    }
    (v2 / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source_files = [
        penalty_source / "canonical_rows.csv",
        penalty_source / "optimized_parameters.json",
        penalty_source / "final_distributions.npz",
        cost_phase / "canonical_rows.csv",
        cost_phase / "optimized_parameters.json",
        cost_phase / "final_distributions.npz",
        cost_phase / "run_metadata.json",
    ]
    provenance = {
        "v2_main_track_rows": 220,
        "penalty_x_rows_reused": 110,
        "cost_phase_grover_rows_rerun": 110,
        "old_gm_th_rows_included": 0,
        "old_gm_th_policy": "historical method; excluded from v2 outputs",
        "source_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in source_files
        },
    }
    (v2 / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = """# Depth-110 Main Track v2

This directory contains the 220-row main track:

- 110 verified Penalty-X rows from the saved source trajectory.
- 110 newly optimized Grover-Mixer rows using the route-cost phase.

The Grover cost layer is `exp(-i * gamma * C(P_i))`.  The incumbent-threshold
mask is used only for the auxiliary BSP metric.  The earlier threshold-phase
Grover trajectory is not part of this directory or the v2 package.

`evaluation_budget_exhausted` is reported as `BUDGET_LIMITED` in
`failure_taxonomy.json`; it does not mean the process crashed or exceeded its
wall-clock guard.

Run the figure, report, and validation scripts to rebuild the analysis from the
saved files without rerunning either optimizer.

The repository also contains every module used by the two sweep runners and by
the v2 builder. See `REPORT.md` for the commands that reproduce a new 220-row
main track in separate result directories.
"""
    (v2 / "README-QUICKSTART.md").write_text(readme, encoding="utf-8")

    return {
        "v2_root": str(v2),
        "canonical_rows": len(combined_rows),
        "parameter_entries": len(parameters),
        "distribution_entries": len(distributions),
        "budget_limited_rows": sum(row["status"] == "BUDGET_LIMITED" for row in failure_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--penalty-root", type=Path, default=DEFAULT_PENALTY)
    parser.add_argument("--cost-phase-root", type=Path, default=DEFAULT_COST_PHASE)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.penalty_root, args.cost_phase_root, args.v2_root),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
