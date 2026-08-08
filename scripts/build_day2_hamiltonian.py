"""Regenerate the authorized Day-2 QUBO, Ising, and figure artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from day2_artifacts import build_day2_artifacts  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build only the DTU v3 Day-2 QUBO/Ising artifacts."
    )
    parser.add_argument("--graph", type=Path, default=PROJECT_ROOT / "data/graph.json")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_day2_artifacts(
        graph_path=args.graph,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
