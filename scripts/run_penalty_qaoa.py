#!/usr/bin/env python3
"""Compatibility command for the small two-mixer course demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main import print_summary, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the readable QAOA core demo.")
    parser.add_argument("--penalty", type=float, default=6.0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--budget", type=int, default=100)
    arguments = parser.parse_args()
    result = run(
        penalty=arguments.penalty,
        depth=arguments.depth,
        seed=arguments.seed,
        evaluation_budget=arguments.budget,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
