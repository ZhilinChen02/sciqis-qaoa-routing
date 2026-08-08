"""Run the frozen Day-5 implementation profiling protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from profiling import run_runtime_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="POST-CORE DESCRIPTIVE: profile the current implementation."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = (
        PROJECT_ROOT / "results" / "runtime_profile.csv",
        PROJECT_ROOT / "results" / "runtime_profile_summary.json",
    )
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise SystemExit("runtime-profile artifacts already exist; pass --overwrite explicitly")
    result = run_runtime_profile(root=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "stage_count": len(result["stage_summaries"]),
                "raw_measurements": result["raw_measurement_count"],
                "day4_runs_reexecuted": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
