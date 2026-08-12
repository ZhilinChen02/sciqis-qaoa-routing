#!/usr/bin/env python3
"""Regenerate QAOA-dynamics figures from saved tables only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dynamics_figures import generate_figures_from_saved  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "qaoa_dynamics_deep_dive" / "v1",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=PROJECT_ROOT / "figures" / "qaoa_dynamics_deep_dive" / "v1",
    )
    arguments = parser.parse_args()
    paths = generate_figures_from_saved(arguments.result_root, arguments.figure_root)
    print(f"generated {len(paths)} figures in {arguments.figure_root}")


if __name__ == "__main__":
    main()
