# QAOA Routing — DTU SCIQIS Course Project

This repository is a teaching implementation of QAOA for one directed,
weighted routing problem. It follows the full chain from edge variables and a
routing QUBO to an Ising Hamiltonian, statevector evolution, classical COBYLA
optimization, route decoding, and probability metrics. It also compares the
full 14-qubit search space with a 20-route feasible basis and includes the
browser visualizations used in the presentation. The project makes no quantum
advantage claim.

## Start here

Read these files in order:

1. `data/graph.json` — the 7-node, 14-edge course instance.
2. `src/main.py` — the shortest complete graph-to-QAOA demonstration.
3. `src/graph.py` — deterministic graph loading, paths, and edge encoding.
4. `src/qubo.py` — routing constraints, QUBO/Ising mapping, and decoding.
5. `src/qaoa.py` — state evolution, X/Grover mixers, objectives, and COBYLA.
6. `src/metrics.py` — `p_feas`, `p_opt`, and distribution summaries.
7. `src/feasible_qaoa.py` — the verified 20-route logical basis.
8. `src/feasible_experiments.py` — feasible mixers and retained objectives.
9. `src/experiments/course_final.py` — the final course comparison.

Run the small end-to-end example from the repository root:

```bash
python src/main.py
```

It prints the exact reference route plus QAOA metrics and writes no results.

## Core scientific path

Each directed edge has a binary variable `x_i`. The objective is the selected
edge cost, and the QUBO adds a squared flow-balance constraint at every node:

```text
sum_i weight_i x_i
+ A * sum_nodes(outgoing - incoming - supply)^2
```

The source supply is `+1`, the target supply is `-1`, and all other supplies
are zero. Substituting `x_i = (1 - z_i) / 2` gives the Ising coefficients. A
QAOA layer then applies a cost phase followed by a mixer, and COBYLA chooses
the `gamma` and `beta` angles.

The retained optimization objectives are Expectation and CVaR. CVaR includes
the fractional probability mass at the cutoff. The saved course extension also
records its historical ascending-CVaR schedule and remains readable without
rerunning it. The retained mixers are Penalty-X, full-space Global-Grover, path
exchange, feasible Grover, and the threshold phase used by the final GM-Th
course method.

The main metrics are:

- `p_feas`: total probability on valid source-to-target routes;
- `p_opt`: total probability on the exact optimal route;
- `p_opt_given_feas = p_opt / p_feas` when `p_feas > 0`;
- feasible-conditional expected route cost.

Independent shortest-path and exhaustive route checks agree on:

```text
0 -> 1 -> 2 -> 4 -> 5 -> 6, cost = 10
```

This exact solution is used for evaluation, not supplied to the optimizer.

## Interactive visualization

Launch the two local browser demonstrations with:

```bash
python scripts/run_qaoa_visualizer.py
python scripts/run_qaoa_dynamics_visualizer.py
```

Their non-browser validation modes are:

```bash
python scripts/run_qaoa_visualizer.py --check
python scripts/run_qaoa_dynamics_visualizer.py --check
```

## Course experiments

These are the two recommended experiment commands:

```bash
python scripts/run_penalty_qaoa.py --budget 4
python scripts/run_course_final.py --seed 2601 --depth 1
```

The first is a small full-space Penalty-QAOA smoke run. The second exercises
the retained final course methods with a deliberately small presentation-safe
configuration. Neither command silently reruns historical long experiments.

## Results, figures, and reports

The student-facing result report is
`reports/FINAL_EXPERIMENT_RESULTS_CN.md` (with a matching PDF). The course
figures are in `figures/course/` and can be regenerated only from saved data by
running `python scripts/make_course_figures.py`.

Saved course evidence is organized as follows:

- `results/q2f_final_improvement/` — final method comparison;
- `results/qaoa_dynamics_deep_dive/` — layer-by-layer dynamics;
- `results/q2f_course_extension/` — Expectation/CVaR course extension;
- `results/q2_revision_formal/` — historical course experiments.

`archive/development/` is historical evidence and is not imported by the
installed runtime or normal test path.

## Install and validate

Python 3.11–3.13 is supported.

```bash
python -m pip install -e ".[dev]"
python -m compileall src scripts
python -m pytest -q
```

The tests cover deterministic graph encoding, exact route identity, all 16,384
QUBO/Ising basis energies, QAOA normalization, full-space and feasible mixers,
Expectation and fractional-cutoff CVaR, route decoding, metrics, visualizer
check modes, and the course commands.

## Limitations

- This is one teaching-scale graph under ideal statevector simulation.
- The feasible representation classically enumerates all 20 valid routes.
- Optimizer comparisons depend on finite budgets and a small seed set.
- The project does not claim scalable hardware execution, a depth law, or
  quantum advantage over classical shortest-path algorithms.
