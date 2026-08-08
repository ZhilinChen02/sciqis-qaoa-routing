# Frozen Routing QUBO to Explicit Shallow QAOA

This DTU 10387 course project follows one teaching-scale weighted routing
instance from an exact classical specification to a fully explicit, independently
validated p=1/p=2 QAOA experiment and a finite-shot measurement demonstration.

Scientific status: **COURSE_PROJECT_SCIENCE_COMPLETE**

Evidence scope: **ONE_GRAPH_IDEAL_SIMULATION_SHALLOW_QAOA**

## Research question

For one fixed weighted directed graph, how do the frozen flow-constraint penalty
strength `A ∈ {2,5,6,12}` and shallow Penalty-X QAOA depth `p ∈ {1,2}` change:

- the probability `p_feas` of sampling any valid source-to-target route;
- the probability `p_opt` of sampling the unique shortest route; and
- the expected route cost conditional on a valid measurement?

## Scientific pipeline

```text
weighted directed graph
  → edge variables / qubits
  → flow-constraint QUBO
  → Ising Hamiltonian
  → explicit RZ/RZZ/RX QAOA
  → exact statevector
  → frozen-budget optimization
  → finite measurement sampling
  → decoded route
```

## Frozen instance and contracts

- 7 nodes, 14 directed weighted edges, 14 edge qubits.
- `2^14 = 16,384` computational-basis edge selections.
- Exactly 20 valid source-to-target routes.
- Unique exact route: `0 → 1 → 2 → 4 → 5 → 6`, with `C*=10`.
- Canonical exact bitstring (`q0 → q13`): `10010001000101`.
- Qiskit display string (`q13 → q0`): `10100010001001`.
- Exact-route state index: `10377`, with q0 as the least-significant bit.

The immutable inputs and protocols are:

- [`data/graph.json`](data/graph.json) — graph and edge/qubit order;
- [`data/penalty_contract.json`](data/penalty_contract.json) — `A_crit=5` and
  frozen grid `{2,5,6,12}`;
- [`data/circuit_contract.json`](data/circuit_contract.json) — explicit gate
  and bit-order conventions;
- [`data/optimization_contract.json`](data/optimization_contract.json) — three
  seed-10387 COBYLA starts and 240 evaluations per start;
- [`data/day5_analysis_contract.json`](data/day5_analysis_contract.json) —
  post-core sampling and profiling only;
- [`data/scientific_freeze_v3.json`](data/scientific_freeze_v3.json) — final
  artifact hashes and prohibited post-freeze changes.

Exhaustive inspection proves `P_flow(x)=0` if and only if `x` is one valid route.
At `A=5`, the all-zero invalid state ties the exact route; for `A>5`, the exact
route is the unique QUBO ground state.

## Frozen core result

The 24-run Day-4 experiment used exact statevectors and selected each cell only
by minimum final expected QUBO energy. No result-dependent retry or retuning was
performed.

| A | p | selected start | ⟨Q_A⟩ | p_feas | p_opt | E[C \| feasible] |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 33.279167 | 0.005512 | 9.09902e-5 | 12.0521 |
| 2 | 2 | 2 | 20.099583 | 0.051449 | 0.00311885 | 12.1657 |
| 5 | 1 | 1 | 41.738372 | 0.016754 | 0.000153198 | 11.5809 |
| 5 | 2 | 0 | 37.912717 | 0.045736 | 2.88239e-6 | 13.4609 |
| 6 | 1 | 1 | 66.972737 | 0.002994 | 1.39051e-6 | 12.5840 |
| 6 | 2 | 0 | 37.765753 | 0.012000 | 0.000133445 | 11.9538 |
| 12 | 1 | 2 | 69.010509 | 0.019542 | 0.000185044 | 11.1408 |
| 12 | 2 | 0 | 66.622962 | 0.010075 | 4.45792e-5 | 11.9843 |

Uniform-state references are `p_feas=20/16384≈0.0012207` and
`p_opt=1/16384≈0.0000610`.

## Main scientific findings

1. Model correctness and variational performance are distinct. Although
   `A>A_crit` gives the correct exact QUBO ground state, shallow optimized QAOA
   need not concentrate much probability on it.
2. Penalty behavior was non-monotonic overall. A larger valid-state energy
   separation did not imply monotonically larger `p_feas` or `p_opt`.
3. p=2 improved both probabilities at A=2 and A=6, improved only `p_feas` at
   A=5, and reduced both at A=12.
4. The subcritical A=2,p=2 cell led the selected table. This is not a
   contradiction: `A_crit` concerns exact ground-state ordering, while the
   shallow optimizer minimizes expected energy and returns a distribution.
5. At A=5,p=2, the minimum-energy selected start had very low `p_opt`; another
   start had much higher `p_opt` but also higher expected energy and was
   correctly not selected under the frozen rule.

The evidence-driven takeaway is:

> Making the penalty large enough to encode the correct feasible ground state
> does not guarantee that a shallow, finitely optimized QAOA circuit will place
> large probability on that state.

## Finite shots and profiling

The post-core demonstration samples the already selected A=2,p=2 exact
distribution at 256, 1,024, 4,096, and 16,384 shots, with 200 replicates per
level and seed 1038705. At 256 shots, 45% of replicates observed no exact route,
despite its nonzero exact probability. This is Monte Carlo sampling variability,
not a hardware experiment or a formal confidence-interval study.

Implementation profiling uses `perf_counter_ns`, three warm-ups, and 31 timed
repeats per operation. It reports current-environment software timings only.
NumPy and Qiskit are both classical simulations here; their timings are not a
classical-versus-quantum or hardware-performance comparison. Repeated state
evolution/objective evaluation dominates the complete optimization workflow.

## Reports and presentation assets

- [Final course-project report](reports/final_course_project_report.md)
- [15-minute presentation outline](reports/presentation_outline_15min.md)
- [Final machine-readable summary](results/final_project_summary.json)
- [Figure 1–13 manifest](results/final_figure_manifest.json)
- [Primary core result figure](figures/09_penalty_depth_core_results.svg)
- [Finite-shot figure](figures/12_finite_shot_sampling_convergence.svg)
- [Runtime figure](figures/13_runtime_profile.svg)

## Reproduce saved-result artifacts

Cheap final rebuild—does not rerun the 24 optimizations:

```bash
python scripts/build_final_report.py
```

Recreate the deterministic finite-shot extension:

```bash
python scripts/run_day5_sampling.py --overwrite
python scripts/build_final_report.py
```

Profiling is environment-dependent and is therefore run explicitly:

```bash
python scripts/run_day5_profiling.py --overwrite
python scripts/build_final_report.py
```

Expensive core reproduction—reruns all 24 frozen COBYLA starts using the
unchanged contract:

```bash
python scripts/run_day4_core_experiment.py --overwrite
```

Official tracked tests:

```bash
pytest -q $(git ls-files 'tests/test_*.py')
```

## Repository structure

- `data/` — immutable graph, penalty, circuit, optimization, analysis, and
  final scientific-freeze contracts.
- `src/` — graph/QUBO/Ising logic, explicit circuits, independent statevector,
  optimization metrics, sampling, profiling, and artifact builders.
- `results/` — exact references, all-start optimization data, finite-shot raw
  replicates, runtime measurements, and final summaries.
- `figures/` — deterministic PNG and SVG scientific figures 1–13.
- `notebooks/` — short teaching notebooks that load reusable source logic and
  saved results.
- `reports/` — final written interpretation and presentation outline.
- `scripts/` — explicit expensive experiment runners and cheap artifact builds.
- `tests/` — focused identity, exhaustive-model, circuit, result, sampling,
  profiling, and reporting checks.

## Evidence boundary

This is one small graph studied with ideal statevector simulation, shallow
p=1/p=2 circuits, three optimizer starts, and a fixed evaluation budget. The
finite-shot section samples an exact probability vector rather than hardware.
No quantum advantage is claimed, and the results should not be generalized
beyond this frozen course-project scope without a new version and protocol.
