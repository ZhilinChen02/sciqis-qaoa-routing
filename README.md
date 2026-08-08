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

Days 1–4 now cover the frozen mathematical model, explicit shallow circuits,
and the core exact-statevector penalty/depth experiment. Measurement and route
decoding remain part of the teaching flow rather than a quantum-advantage claim.

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

**COMPLETED — Day 3: Explicit p=1 Penalty-X circuit + independent
statevector validation.**

The circuit is constructed directly from primitive gates rather than a QAOA
ansatz class:

```text
H on q0,...,q13
  → COST: RZ(2γ₁hᵢ), RZZ(2γ₁Jᵢⱼ)
  → MIXER: RX(2β₁) on every qubit
```

The identity term `c₀I` is omitted as a physical gate because it changes only
global phase. An independent NumPy implementation validates the Qiskit
statevector at the initial, post-cost, and post-mixer checkpoints:

- `U_C`: basis energy → relative phase, with probabilities unchanged;
- `U_M`: phase differences/interference → probability redistribution.

The fixed diagnostic pair `γ=π/7`, `β=π/11` is labelled
`DIAGNOSTIC_ONLY_NOT_OPTIMIZED`. It is used only to expose the mechanism and is
not a performance, tuning, or optimization result.

**COMPLETED — Day 4: Frozen p=1/p=2 penalty-depth numerical experiment.**

The protocol in `data/optimization_contract.json` fixed SciPy COBYLA, three
seeded starts per cell, the same 240-evaluation budget at both depths, and
minimum final expected-QUBO energy as the selection rule before the 24 runs.
The repeated objective uses the independently validated NumPy statevector;
every one of the eight selected states is checked again with the explicit
Qiskit primitive-gate circuit.

Observed under that frozen rule, the result is deliberately not a simple
monotone success story. Depth p=2 increased both `p_feas` and `p_opt` at A=2
and A=6, increased only `p_feas` at A=5, and decreased both at A=12. Across A,
valid-route probability was non-monotone for p=1 and decreased over the four
prescribed points for p=2; exact-route probability was non-monotone at both
depths. Most selected runs exhausted the fixed budget, and the all-start tables
retain this optimizer sensitivity rather than hiding it.

**NEXT — Day 5: Profiling + optional finite-shot demonstration + final
presentation/report polish.**

No warm start, path-exchange mixer, RCSP, p=3 rescue run, or quantum-advantage
claim is included in the core experiment.

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

## Reproduce Day 3

With Qiskit available in the active environment:

```bash
python scripts/build_day3_circuit.py
PYTHONPATH=src pytest -q \
  tests/test_day3_circuit.py tests/test_statevector_reference.py
```

This regenerates the circuit contract, the state-evolution diagnostic, and
Figures 5–7. The teaching notebook is
[`notebooks/03_explicit_p1_qaoa.ipynb`](notebooks/03_explicit_p1_qaoa.ipynb).

## Reproduce Day 4

The expensive command reruns all 24 frozen COBYLA starts and requires an
explicit overwrite flag once results exist:

```bash
python scripts/run_day4_core_experiment.py --overwrite
```

The cheap command rebuilds Figures 8–11 and the 121×81 p=1 landscape from the
saved core result (reusing the saved landscape grid when present):

```bash
python scripts/build_day4_figures.py
PYTHONPATH=src pytest -q \
  tests/test_p2_circuit.py tests/test_optimization_contract.py \
  tests/test_core_metrics.py
```

The teaching notebook loads saved optimization results by default and does not
silently rerun the expensive experiment:
[`notebooks/04_penalty_depth_experiment.ipynb`](notebooks/04_penalty_depth_experiment.ipynb).

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
- `src/circuit.py` — explicit H, RZ/RZZ cost, and transverse-field RX mixer
  construction for parameterized p=1 and p=2;
- `src/statevector_reference.py` — independent NumPy cost-phase and pairwise
  X-mixer evolution without a dense matrix;
- `src/day3_artifacts.py` — hard statevector equivalence gate, diagnostic JSON,
  and deterministic Figures 5–7;
- `scripts/build_day3_circuit.py` — one public Day-3 regeneration command;
- `tests/test_day3_circuit.py`, `tests/test_statevector_reference.py` — explicit
  angle, parameterization, normalization, phase, interference, and independent
  equivalence tests.
- `src/optimization.py` — frozen exact-statevector objective, all-start COBYLA
  execution, primary route metrics, selection, and final explicit-circuit check;
- `src/day4_artifacts.py` — saved-result validation and deterministic Figures
  8–11;
- `scripts/run_day4_core_experiment.py` — explicit expensive 24-run entry point;
- `scripts/build_day4_figures.py` — cheap saved-result figure/table rebuild;
- `tests/test_p2_circuit.py`, `tests/test_optimization_contract.py`,
  `tests/test_core_metrics.py` — p=2, protocol, completeness, metric, selection,
  and selected-state verification gates.
