# QAOA Routing — DTU SCIQIS Course Project

## Project question

This project studies how QAOA encodes and samples routes in a small weighted
directed graph, and how a constraint-preserving representation changes
feasibility and optimal-route probability.

The repository has two readable paths:

1. a course-core Penalty-X QAOA using one binary variable per directed edge;
2. a teaching-scale logical extension whose basis contains only valid routes.

The final recommended extension is feasible-subspace incumbent-threshold
GM-Th-QAOA at depth `p=3`.

## Core pipeline

```text
weighted graph
    -> edge variables
    -> flow-constrained QUBO
    -> Ising Hamiltonian
    -> Penalty-X QAOA
    -> statevector measurement probabilities
    -> route decoding and p_feas / p_opt
```

The fixed graph has 7 nodes and 14 directed weighted edges. Flow conservation
is enforced with penalty `A=6`. Independent shortest-path and exhaustive
simple-route calculations agree on the unique cost-10 reference route:

```text
0 -> 1 -> 2 -> 4 -> 5 -> 6
```

The reference is used to evaluate `p_opt`; it does not enter QAOA parameter
optimization.

## Final extension

The edge representation has

```text
2^14 = 16,384 computational states.
```

The logical feasible representation classically enumerates the actual

```text
20 valid simple source-to-target routes.
```

Every logical coordinate is therefore feasible. The uniform logical initial
state has `p_feas=1` structurally and places `1/20 = 5%` probability on each
route.

The deterministic greedy incumbent is

```text
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6, cost 11.
```

The final threshold is constructed without an optimum label:

```text
marked(P) = 1 when raw_cost(P) < incumbent_cost = 11.
```

Exactly one of the 20 routes satisfies this threshold on the course instance.
Only after threshold construction is that route compared with the classical
reference for reporting `p_opt`.

GM-Th-QAOA starts in the uniform feasible state, applies the Boolean threshold
phase, and mixes with the rank-one Grover feasible mixer

\[
U_G(\beta)=e^{-i\beta|F\rangle\langle F|}
=I+(e^{-i\beta}-1)|F\rangle\langle F|.
\]

The evolution never leaves the 20-dimensional feasible basis.

## Final results

The immutable three-seed descriptive study gives:

| Configuration | Median p_opt | p_feas |
|---|---:|---:|
| Uniform feasible initialization | 5.00% | ≈100% |
| GM-Th-QAOA p=1 | 39.20% | ≈100% |
| GM-Th-QAOA p=2 | 81.608% | ≈100% |
| GM-Th-QAOA p=3 | **99.8712%** | ≈100% |

This behavior is similar to amplitude amplification: alternating selective
phases and global feasible-space mixing concentrates mass on the marked set.
It is an instance-specific mechanism demonstration, not a scaling result.

## Limitations

- All feasible routes are enumerated classically before logical simulation.
- The logical Hilbert space has only 20 states.
- Exactly one route lies below the incumbent threshold on this instance.
- Structural `p_feas=1` is not empirical quantum advantage.
- This is not a scalable routing algorithm or hardware implementation.
- No quantum-advantage or statistical-significance claim is made.
- The shortest-path problem itself is not claimed to be NP-hard.

## Installation

Python 3.11–3.13 is supported.

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Run the final three-seed course method. It prints results and does not write to
the immutable result roots:

```bash
python scripts/run_course_final.py
```

Optional development overrides are explicit:

```bash
python scripts/run_course_final.py --seed 2601 --depth 3 --output /tmp/q2f-smoke.json
```

Run the readable Penalty-X course core:

```bash
python scripts/run_penalty_qaoa.py
```

## Tests

```bash
pytest -q
```

The retained tests cover graph/reference consistency, QUBO/Ising equivalence,
statevector normalization, Penalty-X evolution, route decoding, feasible-basis
ordering, path-exchange connectivity, Grover/threshold unitaries, incumbent-only
threshold construction, deterministic seeds, and the `p_feas=1` invariant.

## Repository structure

```text
configs/                    final explicit course configuration
data/                       fixed graph and frozen reference contracts
src/                        core and feasible/logical scientific modules
scripts/run_course_final.py final GM-Th-QAOA command
scripts/run_penalty_qaoa.py Penalty-X core command
scripts/make_course_figures.py
tests/                      focused scientific tests
figures/course/             five presentation figures
results/                    three immutable sealed evidence roots
docs/methods/               concise method descriptions
archive/development/        non-runtime historical development material
```

## Reproduce figures

The plotter verifies the three sealed manifests and reads their existing CSV,
JSON, and probability vectors. It does not rerun an optimizer.

```bash
python scripts/make_course_figures.py
```

Use a temporary output directory for a smoke test:

```bash
python scripts/make_course_figures.py --output /tmp/sciqis-course-figures
```

The immutable result roots are identified and checked by
`src/sealed_results.py`. Detailed feasible-space and final-threshold definitions
are in `docs/methods/Q2F_FEASIBLE_WARM_START.md` and
`docs/methods/Q2F_FINAL_IMPROVEMENT.md`.
