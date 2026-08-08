# From a Weighted Routing Graph to an Explicit QAOA Circuit

DTU 10387 — Scientific Computing in Quantum Information Science. This repository
implements the deliberately small Project v3.0 workflow:

```text
frozen graph
  → edge variables
  → QUBO
  → Ising Hamiltonian
  → QAOA
  → measurement
  → decoded route
```

Only the first stage is complete. Later boxes describe the planned teaching
narrative; they are not implemented or claimed here.

## Project v3.0 — Day 1 freeze

**CURRENT COMPLETED STAGE: Frozen graph + exact reference.**

The canonical input is [`data/graph.json`](data/graph.json): seven nodes,
fourteen positive-integer directed edges, source 0, and target 6. Edge variables
are frozen lexicographically by `(u, v)` and assigned `qubit_index=0,...,13`.
Future stages must reuse this order rather than relying on NetworkX iteration.

The exact route is checked in two independent ways:

1. NetworkX weighted shortest path;
2. a deterministic custom DFS that enumerates every simple directed `0→6`
   path and sums its edge weights.

Both must agree before `results/exact_reference.json` is written. The reference
is a correctness/evaluation oracle, not a competing algorithm.

**NEXT STAGE: Flow-constraint QUBO construction.**

No QUBO, Ising, QAOA, circuit, optimizer, warm-start, RCSP, path-exchange,
GPU/TN, or benchmark result is part of this Day-1 freeze.

## Reproduce Day 1

With the project environment installed, run:

```bash
python scripts/build_day1_reference.py
PYTHONPATH=src pytest -q tests/test_graph.py tests/test_exact_reference.py
```

The build command deterministically regenerates:

- `results/exact_reference.json`
- `results/all_simple_paths.csv`
- `figures/01_frozen_weighted_graph.png`
- `figures/01_frozen_weighted_graph.svg`
- `figures/02_route_cost_spectrum.png` (diagnostic)

The short teaching notebook is
[`notebooks/01_graph_and_reference.ipynb`](notebooks/01_graph_and_reference.ipynb).

## Day-1 code

- `src/graph.py` — canonical loader, strict validation, edge order, bit mapping,
  and path-cost utilities;
- `src/exact_reference.py` — independent exact methods and JSON/CSV records;
- `src/day1_artifacts.py` — deterministic presentation figures and the small
  artifact build;
- `scripts/build_day1_reference.py` — one public regeneration command;
- `tests/test_graph.py`, `tests/test_exact_reference.py` — focused scientific
  correctness tests.
