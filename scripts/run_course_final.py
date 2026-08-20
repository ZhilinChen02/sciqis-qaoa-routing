#!/usr/bin/env python3
"""Run the final incumbent-threshold GM-Th-QAOA course configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.course_final import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    course_final_payload,
    load_course_final_config,
    run_course_final,
)


def _write_output(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run teaching-scale feasible-subspace incumbent-threshold GM-Th-QAOA."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="development override; repeat for multiple seeds",
    )
    parser.add_argument(
        "--depth", type=int, help="development override in {1,2,3,4}"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional non-results JSON path; default execution writes nothing",
    )
    arguments = parser.parse_args()

    config = load_course_final_config(arguments.config)
    context, runs = run_course_final(
        config,
        seeds=arguments.seed,
        depth=arguments.depth,
    )
    print(
        f"basis={context.basis.size} feasible routes | "
        f"incumbent_cost={context.threshold.incumbent_raw_cost:g} | "
        f"marked_routes={context.threshold.better_route_count}"
    )
    print("seed   p_feas              p_opt               expected_cost       evaluations")
    for run in runs:
        print(
            f"{run.seed:<6d} {run.p_feas:<19.16g} {run.p_opt:<19.16g} "
            f"{run.expected_cost:<19.12g} {run.evaluations}"
        )
    values = sorted(run.p_opt for run in runs)
    middle = len(values) // 2
    median_p_opt = (
        values[middle]
        if len(values) % 2
        else 0.5 * (values[middle - 1] + values[middle])
    )
    print(f"median_p_opt={median_p_opt:.16g}")
    if arguments.output is not None:
        _write_output(
            arguments.output,
            course_final_payload(config, context, runs),
        )
        print(f"output={arguments.output.resolve()}")


if __name__ == "__main__":
    main()
