#!/usr/bin/env python3
"""Run the basic graph→QUBO→Ising→Penalty-X course demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from penalty_demo import run_penalty_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the readable Penalty-X QAOA core demo.")
    parser.add_argument("--penalty", type=float, default=6.0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--budget", type=int, default=100)
    arguments = parser.parse_args()
    result = run_penalty_demo(
        penalty=arguments.penalty,
        depth=arguments.depth,
        seed=arguments.seed,
        evaluation_budget=arguments.budget,
    )
    route = (
        "none"
        if result.most_probable_feasible_route is None
        else "->".join(map(str, result.most_probable_feasible_route))
    )
    print(f"graph: {result.node_count} nodes, {result.edge_count} edges")
    print(f"encoding: {result.state_count} edge-basis states, penalty={result.penalty:g}")
    print(
        f"QUBO/Ising max error={result.qubo_ising_max_error:g}; "
        f"exact ground states={result.ground_state_count}"
    )
    print(
        f"Penalty-X p={result.depth}, seed={result.seed}, "
        f"evaluations={result.optimizer.evaluations}, termination={result.optimizer.reason}"
    )
    print(f"p_feas={result.p_feas:.16g}")
    print(f"p_opt={result.p_opt:.16g}")
    print(
        f"most_probable_feasible_route={route}; "
        f"cost={result.most_probable_feasible_cost}"
    )


if __name__ == "__main__":
    main()
