#!/usr/bin/env python3
"""Run the frozen three-geometry p=1/p=2 dynamics study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dynamics_figures import generate_figures_from_saved  # noqa: E402
from dynamics_study import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    GROVER_FEASIBLE,
    PENALTY_X,
    load_study_config,
    postrun_scientific_validation,
    prepare_study_context,
    regression_comparison,
    run_primary_matrix,
    save_study_artifacts,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--figure-root", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()
    config = load_study_config(arguments.config)
    result_root = arguments.result_root or PROJECT_ROOT / config["result_namespace"]
    figure_root = arguments.figure_root or PROJECT_ROOT / config["figure_namespace"]
    if result_root.exists() and any(result_root.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty result namespace: {result_root}"
        )
    context = prepare_study_context(penalty=float(config["penalty"]))
    runs = run_primary_matrix(context, config)
    postrun = postrun_scientific_validation(runs)
    artifact_summary = save_study_artifacts(
        context, runs, config, result_root=result_root
    )
    preserved = [
        regression_comparison(context, run)
        for run in runs
        if run.algorithm in (PENALTY_X, GROVER_FEASIBLE)
    ]
    write_json(result_root / "preserved_algorithm_regressions.json", preserved)
    write_json(result_root / "postrun_scientific_validation.json", postrun)
    figure_paths = (
        []
        if arguments.no_figures
        else generate_figures_from_saved(result_root, figure_root)
    )
    print(f"wrote {artifact_summary['summary_rows']} primary result rows to {result_root}")
    print(f"wrote {len(figure_paths)} figures to {figure_root}")


if __name__ == "__main__":
    main()
