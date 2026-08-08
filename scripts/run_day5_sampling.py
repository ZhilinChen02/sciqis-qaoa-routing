"""Run the frozen Day-5 finite-shot Monte Carlo demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from day5_analysis import run_finite_shot_sampling  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="POST-CORE DESCRIPTIVE: generate 800 frozen finite-shot replicates."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = (
        PROJECT_ROOT / "results" / "finite_shot_replicates.csv",
        PROJECT_ROOT / "results" / "finite_shot_summary.json",
    )
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise SystemExit("finite-shot artifacts already exist; pass --overwrite explicitly")
    result = run_finite_shot_sampling(root=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "replicates": result["raw_replicate_count"],
                "shot_counts": [row["shots"] for row in result["shot_summaries"]],
                "source_probability_sha256": result["source_state"]["probability_vector_float64_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
