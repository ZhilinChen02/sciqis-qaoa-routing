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

The first two mathematical-model stages are complete. Later boxes describe the
planned teaching narrative; they are not implemented or claimed here.

## Project v3.0 progress

**COMPLETED — Day 1: Frozen graph + exact reference.**

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

**COMPLETED — Day 2: Flow constraints + explicit QUBO + penalty threshold +
Ising Hamiltonian.**

The directed balance residual at node `v` is
`f_v(x) = outgoing(v) − incoming(v) − b_v`, with `b_0=+1`, `b_6=−1`,
and zero elsewhere. Day 2 expands
`Q_A(x)=C(x)+A Σ_v f_v(x)²` directly in the unambiguous polynomial convention
`c + Σ_i q_i x_i + Σ_{i<j} q_ij x_i x_j`.

Exhaustive inspection of all `2^14 = 16,384` edge selections gives
`A_crit = 5`. Below this value a cheap infeasible state beats the true route;
at `A=5` the empty selection ties `C*=10`; and for every `A>5` the exact route
is the unique global minimizer. The penalty grid is therefore frozen before
any QAOA result as:

| Label | A | Meaning |
|---|---:|---|
| weak | 2 | clearly subcritical |
| critical | 5 | exact tie boundary |
| just-supercritical | 6 | smallest teaching integer above the boundary |
| strong | 12 | twice the just-supercritical value |

The QUBO is mapped explicitly with `x_i=(I−Z_i)/2`, including the identity
constant. QUBO and Ising energies agree exactly on all 16,384 basis states for
all four frozen penalties.

**NEXT — Day 3: Explicit p=1 Penalty-X QAOA circuit.**

No QAOA circuit, mixer, optimizer, execution, probability, or QAOA result is
implemented or claimed in the Day-2 freeze.

## Frozen bit-order convention

Three representations are always named explicitly:

- `edge_vector`: `[x0, x1, ..., x13]`;
- `canonical_bitstring`: text in `q0 → q13` order, for example
  `10010001000101`;
- `qiskit_display_bitstring`: text in `q13 → q0` order when Qiskit-style
  classical display is needed, for example `10100010001001`.

Conversion helpers perform every reversal explicitly. Integer basis-state
indices use `q0` as the least-significant bit.

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

## Reproduce Day 2

```bash
python scripts/build_day2_hamiltonian.py
PYTHONPATH=src pytest -q \
  tests/test_graph.py tests/test_exact_reference.py tests/test_bit_order.py \
  tests/test_qubo.py tests/test_ising.py
```

This regenerates the penalty threshold/contract, canonical QUBO and Ising
coefficient files, and Figures 3–4. The teaching notebook is
[`notebooks/02_qubo_and_ising.ipynb`](notebooks/02_qubo_and_ising.ipynb).

## Reusable code

- `src/graph.py` — canonical loader, strict validation, edge order, bit mapping,
  and path-cost utilities;
- `src/exact_reference.py` — independent exact methods and JSON/CSV records;
- `src/day1_artifacts.py` — deterministic presentation figures and the small
  artifact build;
- `scripts/build_day1_reference.py` — one public regeneration command;
- `tests/test_graph.py`, `tests/test_exact_reference.py` — focused scientific
  correctness tests.
- `src/qubo.py` — named bit states, directed-flow residuals, explicit canonical
  polynomial expansion, full-state decoder checks, and penalty threshold;
- `src/ising.py` — exact `x_i=(I−Z_i)/2` coefficient transformation and basis
  energy evaluation;
- `src/day2_artifacts.py` — deterministic Day-2 JSON and scientific figures;
- `scripts/build_day2_hamiltonian.py` — one public Day-2 regeneration command;
- `tests/test_bit_order.py`, `tests/test_qubo.py`, `tests/test_ising.py` — bit
  convention and exhaustive mathematical-model gates.
