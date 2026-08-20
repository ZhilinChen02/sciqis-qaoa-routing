#!/usr/bin/env python3
"""Archived entry point for the superseded comparison pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment import DEFAULT_CONFIG_PATH, DEFAULT_RESULTS_DIR, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    arguments = parser.parse_args()
    summary = run_experiment(
        config_path=arguments.config,
        results_dir=arguments.results_dir,
    )
    print(
        "Exact route",
        "->".join(map(str, summary["exact_optimal_route"])),
        "cost",
        summary["exact_optimal_cost"],
    )
    for row in summary["results"]:
        print(
            f"{row['solver']} p={row['p']}: "
            f"p_feas={row['p_feas']:.6f}, p_opt={row['p_opt']:.6f}"
        )


if __name__ == "__main__":
    main()
