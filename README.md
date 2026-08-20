# QAOA Routing

A teaching-scale implementation of the Quantum Approximate Optimization
Algorithm (QAOA) for a directed, weighted routing problem. The project follows
the complete path from a graph model to a QUBO, an Ising Hamiltonian, optimized
QAOA states, decoded routes, metrics, and browser visualizations.

It uses ideal statevector simulation and does not claim quantum advantage.

## Problem

The bundled instance has 7 nodes and 14 directed edges. Each edge has a binary
selection variable. A valid solution must carry one unit of flow from source to
target while minimizing total edge cost.

The exact reference route is:

```text
0 -> 1 -> 2 -> 4 -> 5 -> 6, cost = 10
```

This route is used to evaluate output distributions; it is not supplied to the
optimizer.

## Graph -> QUBO -> Ising -> QAOA

The routing QUBO combines edge cost with squared flow-balance penalties:

```text
sum_i weight_i x_i
+ A * sum_nodes(outgoing - incoming - supply)^2
```

The source supply is `+1`, the target supply is `-1`, and every other supply is
zero. Substituting `x_i = (1 - z_i) / 2` produces the Ising coefficients used by
the phase operator.

The implementation includes:

- full-space Penalty-X and Global-Grover QAOA;
- a 20-route feasible representation with path-exchange and Grover mixers;
- Expectation and fractional-cutoff CVaR objectives;
- incumbent-threshold GM-Th-QAOA;
- deterministic COBYLA optimization;
- `p_feas`, `p_opt`, conditional route cost, entropy, and related metrics.

## Installation

Python 3.11-3.13 is supported.

```bash
python -m pip install -e ".[dev]"
```

## Core code

- `src/main.py` - the shortest complete graph-to-QAOA demonstration.
- `src/graph.py` - graph loading, path utilities, and edge encoding.
- `src/qubo.py` - flow constraints, QUBO/Ising mapping, and route decoding.
- `src/qaoa.py` - state evolution, X/Grover mixers, objectives, and COBYLA.
- `src/metrics.py` - distribution and routing metrics.
- `src/feasible_qaoa.py` - the logical feasible-route basis.
- `src/feasible_experiments.py` - feasible mixers, Expectation, and GM-Th logic.
- `data/graph.json` - the routing instance.
- `src/experiments/` - reusable experiment orchestration.
- `src/support/` - visualizer data adapters and scientific checks.

## How to run

Run the smallest graph-to-QAOA example:

```bash
python -m src.main
```

Run the full-space Penalty-X demonstration or the feasible GM-Th course
command:

```bash
python scripts/run_penalty_qaoa.py
python scripts/run_course_final.py --seed 2601 --depth 1
```

For a faster Penalty-X smoke run:

```bash
python scripts/run_penalty_qaoa.py --budget 4
```

The course command writes nothing unless `--output` is supplied.

## Results

`results/` contains the retained experiment evidence, including the formal Q2
revision, feasible-course extensions, dynamics deep dive, CVaR studies, and the
full depth-1-through-110 Penalty-X/Global-Grover sweep used by the Evolution
Microscope. These files are read-only evidence; the visualizer never reruns the
optimizer.

The most useful entry points are each study's configuration, summary,
validation, and result manifest. Raw traces and saved distributions remain
available for reproducibility and numerical cross-checks.

## Web visualizers

The general routing visualizer remains a compact demonstration. The QAOA
dynamics visualizer is the read-only **Evolution Microscope**: it replays the
frozen Penalty-X and Global-Grover optimized parameters at every cost and mixer
checkpoint for depths `p=1,...,110`. Intermediate statevectors are reconstructed
deterministically; no optimization is rerun.

```bash
python scripts/run_qaoa_visualizer.py
python scripts/run_qaoa_dynamics_visualizer.py
```

Run the microscope without opening a browser, or execute its complete p=110
scientific validation:

```bash
python scripts/run_qaoa_visualizer.py --check
python scripts/run_qaoa_dynamics_visualizer.py --no-browser
python scripts/run_qaoa_dynamics_visualizer.py --check
```

The Evolution Microscope reads `results/global_depth110/` directly, exposes all
depths `p=1,...,110`, and validates replayed final states against saved
checkpoint metrics and distributions.

## Tests

```bash
python -m compileall src scripts
python -m pytest -q
```

The suite checks deterministic graph encoding, all 16,384 QUBO/Ising basis
energies, QAOA normalization, Penalty-X and Grover mixers, feasible QAOA,
Expectation, CVaR, GM-Th behavior, route decoding, metrics, COBYLA, and both web
visualizers.
