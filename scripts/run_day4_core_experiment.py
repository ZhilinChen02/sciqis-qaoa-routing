"""Run the expensive frozen 24-start Day-4 core experiment exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph import load_graph  # noqa: E402
from optimization import run_core_experiment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXPENSIVE: execute all 24 frozen Day-4 COBYLA runs."
    )
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / "data/optimization_contract.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing Day-4 results using the same frozen contract.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    primary_output = args.results_dir / "core_experiment_results.json"
    if primary_output.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite {primary_output}; pass --overwrite explicitly"
        )
    payload = run_core_experiment(
        load_graph(PROJECT_ROOT / "data" / "graph.json"),
        contract_path=args.contract,
        results_dir=args.results_dir,
    )
    summary = {
        "optimization_contract_sha256": payload["optimization_contract"]["sha256"],
        "run_count": payload["experiment_completeness"]["actual_run_count"],
        "selected_cell_count": payload["experiment_completeness"]["actual_selected_cell_count"],
        "selected_cells": [
            {
                "A": cell["identity"]["A"],
                "p": cell["identity"]["p"],
                "selected_start_id": cell["optimization"]["selected_start_id"],
                "expected_qubo_energy": cell["optimization"]["final_expected_qubo_energy"],
                "p_feas": cell["quality"]["p_feas"],
                "p_opt": cell["quality"]["p_opt"],
            }
            for cell in payload["selected_cells"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
