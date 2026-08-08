# DTU 10387 — 15-minute presentation outline

## Slide 1 — Question (1 min)

**Message:** How do a frozen flow penalty and shallow QAOA depth affect valid- and optimal-route sampling?

Figure: compact pipeline graphic or title only.

- One weighted directed graph; A in {2,5,6,12}; p in {1,2}.
- Metrics are p_feas, p_opt, and conditional feasible-route cost.

## Slide 2 — Frozen graph and exact route (1 min)

**Message:** Freeze the problem before looking at quantum results.

Figure: **Figure 1**.

- Seven nodes, 14 directed edges/qubits, 16,384 basis states.
- Exact route 0→1→2→4→5→6 has C*=10; 20 valid routes total.

## Slide 3 — Flow constraints, QUBO, and A_crit (2 min)

**Message:** Exhaustive classical analysis proves when the encoded ground state becomes correct.

Figure: **Figure 3**; Figure 4 as backup.

- P_flow=0 iff valid route over every basis state.
- A_crit=5; all-zero invalid state ties at the boundary.
- For A>5, the unique QUBO optimum is the exact route.

## Slide 4 — QUBO to Ising to explicit gates (2 min)

**Message:** Every frozen Ising coefficient maps transparently to RZ/RZZ gates and an RX mixer.

Figure: **Figure 5**.

- Show the factor-of-two RZ/RZZ/RX convention.
- p=2 repeats cost then mixer; no black-box QAOA ansatz.
- Qiskit and independent NumPy statevectors agree numerically.

## Slide 5 — Quantum-state mechanism (2 min)

**Message:** Cost encodes energy into phase; the mixer turns phase differences into probability changes.

Figure: **Figure 6**.

- Cost-layer probabilities stay invariant.
- Relative phases become nontrivial.
- Mixer interference redistributes mass over basis states.

## Slide 6 — Frozen optimization protocol and landscape (1.5 min)

**Message:** Three fixed COBYLA starts search a structured, narrow p=1 energy landscape.

Figure: **Figure 8**.

- Exact statevector objective; 240 evaluations per start.
- Same starts reused for every A; selection uses energy only.
- The grid is descriptive, not proof of a global variational optimum.

## Slide 7 — Main penalty × depth result (2 min)

**Message:** Penalty and depth effects are non-monotonic under the frozen protocol.

Figure: **Figure 9**.

- A=2,p=2 has the highest selected p_feas and p_opt.
- p=2 helps both at A=2 and A=6, only p_feas at A=5, and neither at A=12.
- Ground-state correctness and variational concentration are distinct.

## Slide 8 — Return probability to routing space (1.5 min)

**Message:** The qubit distribution can be interpreted on the original graph without pretending marginals form one route.

Figure: **Figure 10**; Figure 11 as backup.

- Compare A=6 selected p=1 and p=2 edge marginals.
- Orange outlines are the exact route reference.
- Edge marginals are not a single sampled route.

## Slide 9 — Finite shots and profiling (1 min)

**Message:** Sampling adds visible uncertainty, while repeated simulated state evolution dominates workflow time.

Figures: **Figure 12** plus a compact **Figure 13** inset.

- At 256 shots, many replicates never observe the low-probability exact route.
- Estimates converge around exact statevector probabilities with increasing shots.
- Runtime numbers describe this implementation, not quantum hardware or advantage.

## Slide 10 — Main lesson and limitations (1 min)

**Message:** Correct Hamiltonian encoding is necessary but does not guarantee strong shallow-QAOA sampling performance.

Figure: reprise **Figure 9** or a one-sentence takeaway.

- Evidence: one graph, ideal simulator, p≤2, small fixed optimization budget.
- No hardware experiment and no quantum-advantage claim.
- Future work requires a new version/protocol; v3 science is frozen.
