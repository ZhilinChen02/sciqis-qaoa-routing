# QAOA Routing — DTU SCIQIS Course Project

This repository is a teaching implementation of QAOA for one directed,
weighted routing problem. It follows the complete scientific path:

```text
weighted graph
  -> edge variables
  -> routing QUBO
  -> Ising Hamiltonian
  -> QAOA state evolution and COBYLA optimization
  -> measurement probabilities
  -> decoded routes and p_feas / p_opt
```

The project compares the full 14-qubit bitstring space with a 20-route feasible
basis and includes the browser visualizations used in the presentation. It uses
ideal statevector simulation and makes no quantum-advantage claim.

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
python -m src.main
```

It prints the exact reference route plus QAOA metrics and writes no results.

## Scientific model

Each directed edge has a binary variable `x_i`. The objective is the selected
edge cost, and the QUBO adds a squared flow-balance constraint at every node:

```text
sum_i weight_i x_i
+ A * sum_nodes(outgoing - incoming - supply)^2
```

The source supply is `+1`, the target supply is `-1`, and all other supplies
are zero. Substituting `x_i = (1 - z_i) / 2` gives the Ising coefficients. A
QAOA layer applies a cost phase followed by a mixer, and COBYLA chooses the
`gamma` and `beta` angles.

The retained objectives are Expectation and fractional-cutoff CVaR. Saved
course data also records the historical ascending-CVaR schedule. The retained
mixers are Penalty-X, full-space Global-Grover, path exchange, feasible Grover,
and the threshold phase used by the final GM-Th course method.

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

## Course experiments

```bash
python scripts/run_penalty_qaoa.py
python scripts/run_course_final.py
```

For a quick smoke run:

```bash
python scripts/run_penalty_qaoa.py --budget 4
python scripts/run_course_final.py --seed 2601 --depth 1
```

The optional two-qubit circuit example shows how simple Ising `Z` and `ZZ`
terms become `RZ` and `CX-RZ-CX` gates:

```bash
python scripts/show_simple_ising_circuit.py
```

## Interactive visualization

Launch the two English-language browser demonstrations with:

```bash
python scripts/run_qaoa_visualizer.py
python scripts/run_qaoa_dynamics_visualizer.py
```

Validate their saved inputs without opening a browser:

```bash
python scripts/run_qaoa_visualizer.py --check
python scripts/run_qaoa_dynamics_visualizer.py --check
```

## Results, figures, presentation, and report

The student-facing result report is
`reports/FINAL_EXPERIMENT_RESULTS_CN.md`, with a matching PDF. The editable
presentation, PDF, notes, and data audit are under `presentation/`. Course
figures are in `figures/course/` and can be regenerated only from saved data:

```bash
python scripts/make_course_figures.py
```

Saved course evidence is organized as follows:

- `results/q2f_final_improvement/` — final method comparison;
- `results/qaoa_dynamics_deep_dive/` — layer-by-layer dynamics;
- `results/q2f_course_extension/` — Expectation/CVaR course extension;
- `results/q2_revision_formal/` — historical course experiments.

`archive/development/` preserves historical evidence but is not imported by
the installed runtime or normal test path.

## Install and validate

Python 3.11–3.13 is supported.

```bash
python -m pip install -e ".[dev]"
python -m compileall src scripts
python -m pytest -q
```

The tests cover deterministic graph encoding, exact route identity, all 16,384
QUBO/Ising basis energies, QAOA normalization, full-space and feasible mixers,
Expectation and fractional-cutoff CVaR, route decoding, metrics, browser input
validation, and the course commands.

## Limitations

- This is one teaching-scale graph under ideal statevector simulation.
- The feasible representation classically enumerates all 20 valid routes.
- Optimizer comparisons depend on finite budgets and a small seed set.
- The project does not claim scalable hardware execution, a depth law, or
  quantum advantage over classical shortest-path algorithms.
