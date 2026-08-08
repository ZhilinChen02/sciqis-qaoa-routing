# Final DTU 10387 course-project report

Status: **COURSE_PROJECT_SCIENCE_COMPLETE**

Scope: **ONE_GRAPH_IDEAL_SIMULATION_SHALLOW_QAOA**

## 1. Question

For one fixed weighted directed routing graph, how do the flow-constraint penalty strength A and shallow Penalty-X QAOA depth p in {1,2} change the probability of sampling any valid route and the unique shortest route?

## 2. Frozen routing instance

The instance has 7 nodes, 14 directed weighted edges, 14 edge qubits, and 16,384 computational-basis selections. Independent shortest-path and exhaustive route enumeration agree on `0 -> 1 -> 2 -> 4 -> 5 -> 6`, with C*=10. Exactly 20 basis states encode valid routes, and the optimum is unique.

## 3. Flow penalty and A_crit

The directed-flow penalty satisfies `P_flow=0` if and only if the selected edges decode to one valid source-to-target route over all 16,384 states. Exhaustive crossing analysis gives `A_crit=5`; the critical invalid state is the all-zero selection. For `A>A_crit`, the exact shortest route is the unique QUBO ground state. The experiment froze A={2, 5, 6, 12} before quantum optimization.

This is **model correctness**: sufficiently large A fixes the exact ground-state ordering. It is not a guarantee about the probability produced by a shallow variational circuit.

## 4. QUBO -> Ising -> explicit circuit

The canonical flow-QUBO is mapped with `x_i=(I-Z_i)/2`. The circuit is explicit Penalty-X QAOA: H on all 14 qubits, RZ/RZZ gates for the Ising cost layer, and RX gates for the transverse-field mixer. Gate angles retain the factor of two required by Qiskit's rotation definitions. Independent NumPy and Qiskit statevectors agree to numerical precision for p=1, p=2, and all eight selected final states.

## 5. Quantum-state mechanism

The cost unitary changes relative phases while leaving computational-basis probabilities invariant (maximum observed diagnostic change 2.711e-20). The mixer then converts phase differences into probability redistribution through interference (minimum diagnostic total-variation change 0.373).

## 6. Frozen p=1/p=2 experiment

The protocol used exact statevectors, SciPy COBYLA, three seed-10387 starts per cell, 240 objective evaluations per start, and minimum final expected QUBO energy as the selection rule. It produced all 24 prescribed runs and eight selected cells without result-dependent retries or tuning.

| A | p | start | <Q_A> | p_feas | p_opt | E[C | feasible] |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 33.279167 | 0.005512 | 9.09902e-05 | 12.0521 |
| 2 | 2 | 2 | 20.099583 | 0.051449 | 0.00311885 | 12.1657 |
| 5 | 1 | 1 | 41.738372 | 0.016754 | 0.000153198 | 11.5809 |
| 5 | 2 | 0 | 37.912717 | 0.045736 | 2.88239e-06 | 13.4609 |
| 6 | 1 | 1 | 66.972737 | 0.002994 | 1.39051e-06 | 12.5840 |
| 6 | 2 | 0 | 37.765753 | 0.012000 | 0.000133445 | 11.9538 |
| 12 | 1 | 2 | 69.010509 | 0.019542 | 0.000185044 | 11.1408 |
| 12 | 2 | 0 | 66.622962 | 0.010075 | 4.45792e-05 | 11.9843 |

## 7. Main results

The largest selected `p_feas` and `p_opt` both occurred at A=2,p=2: 0.051449 and 0.003119. Penalty behavior was not generally monotonic, and p=2 improved both probabilities at A=2 and A=6, improved only `p_feas` at A=5, and reduced both at A=12.

### Why the subcritical A=2 cell can lead the observed table

This is not a contradiction. `A_crit` concerns the exact global ground-state ordering of `Q_A`. The p=2 circuit is shallow, COBYLA minimizes **expected QUBO energy** rather than `p_opt`, and the final state is a broad probability distribution rather than necessarily the ground state. The three-start, 240-evaluation optimization was visibly start/budget sensitive. Consequently, Hamiltonian ground-state correctness and finite-depth probability concentration answer different questions. The A=2 observation is post-hoc descriptive for this one frozen experiment, not a superiority claim.

### Expected energy versus p_opt at A=5,p=2

The frozen rule selected start 0 with energy 37.912717 and `p_opt=2.88239e-06`. Start 2 had the much larger `p_opt=0.00169498` but higher energy 49.170820; it was therefore correctly not selected. Expected energy and exact-route probability are related but non-identical objectives, and no post-hoc reranking was performed.

## 8. Finite-shot demonstration

The fixed selected A=2,p=2 distribution has exact `p_feas=0.05144857` and `p_opt=0.00311885`. Seed 1038705 generated 200 independent multinomial replicates at each shot count.

| shots | mean p_feas | sd p_feas | mean p_opt | sd p_opt | zero-optimum replicates |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.049629 | 0.012682 | 0.002988 | 0.003252 | 45.0% |
| 1,024 | 0.051045 | 0.006831 | 0.002983 | 0.001650 | 3.5% |
| 4,096 | 0.051655 | 0.003625 | 0.003174 | 0.000974 | 0.0% |
| 16,384 | 0.051280 | 0.001779 | 0.003140 | 0.000456 | 0.0% |

At 256 shots, 45.0% of replicates observed no exact optimal route even though its exact probability is nonzero. These empirical quantiles describe Monte Carlo sampling variability; they are not formal confidence intervals and this is not a hardware experiment.

## 9. Scientific-computing profile

| stage | median ms | 10th–90th percentile ms |
|---|---:|---:|
| `contract_validation` | 25.0905 | 23.7837–26.8317 |
| `qubo_construction` | 0.4685 | 0.4581–0.4842 |
| `ising_conversion` | 0.2200 | 0.2160–0.2265 |
| `qiskit_circuit_build_p1` | 0.8282 | 0.7728–0.8595 |
| `qiskit_circuit_build_p2` | 1.2991 | 1.2835–1.3955 |
| `numpy_statevector_p1` | 4.3944 | 4.3180–4.6521 |
| `numpy_statevector_p2` | 8.3617 | 7.7303–8.8736 |
| `qiskit_statevector_p1` | 9.4706 | 8.5043–10.5561 |
| `qiskit_statevector_p2` | 13.5070 | 13.3251–15.0812 |
| `core_metric_calculation` | 27.6091 | 27.5435–27.6545 |
| `finite_shot_sampling_4096` | 0.7286 | 0.7249–0.7332 |

The committed Day-4 optimization runs had median runtime 1.794 s and used 5,630 total objective evaluations. Repeated state evolution and objective calculation therefore dominate the complete workflow. These are observed implementation runtimes on the current environment; NumPy and Qiskit entries are both classical simulations, not a hardware-performance comparison.

## 10. Limitations

- One small frozen teaching graph.
- Ideal statevector simulation and no hardware noise.
- Only p=1 and p=2.
- Three starts and a small fixed optimizer budget.
- Sampling is Monte Carlo from an exact statevector, not device execution.
- No quantum advantage is claimed.

## 11. Takeaway

**Making the penalty large enough to encode the correct feasible ground state does not guarantee that a shallow, finitely optimized QAOA circuit will place large probability on that state.**

For this one graph, the main lesson is the separation between model correctness and variational performance. The former was proved exhaustively; the latter remained non-monotonic and optimizer-sensitive under the frozen shallow protocol.
