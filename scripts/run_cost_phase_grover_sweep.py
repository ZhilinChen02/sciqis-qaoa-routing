"""Run only the route-cost-phase Grover-Mixer depth trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth110_cost_phase import write_cost_phase_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=110)
    parser.add_argument("--depths", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--penalty", type=float, default=6.0)
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    depths = (
        tuple(args.depths)
        if args.depths
        else tuple(range(1, int(args.max_depth) + 1))
    )
    summary = write_cost_phase_run(
        depths=depths,
        seed=args.seed,
        penalty=args.penalty,
        result_root=args.result_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
