# 15-minute DTU presentation story

## 0:00-1:30 — The fixed routing problem

Use Figure 1 (`01_fixed_weighted_routing_graph.png`).

- “We study one frozen seven-node, fourteen-edge directed graph.”
- Point out source 0, target 6, and the highlighted unique cost-10 route.
- “The aim is not to beat Dijkstra. It is to understand shallow QAOA dynamics.”

## 1:30-4:00 — Edge bits, flow constraints, QUBO, and Ising

Use Figure 2 (`02_qubo_ising_qaoa_pipeline.png`).

- “Each directed edge is one bit, so the full space has 2^14 = 16,384 states.”
- Write `f_v = outgoing - incoming - b_v` and
  `Q_6 = sum_e w_e x_e + 6 sum_v f_v^2`.
- State the mapping `x=(1-Z)/2` and note that QUBO and Ising energies match
  exactly on all states.
- “Only 20 bitstrings decode as valid routes; one is optimal.”

## 4:00-7:00 — What one QAOA layer actually does

Use Figures 7 and 10.

- “The cost operator multiplies amplitude `a_x` by `exp(-i gamma E_x)`.”
- “Magnitude is unchanged, so probability and expected cost are unchanged at
  the cost checkpoint.”
- Point to the horizontal Initial→Cost-1 and Mixer-1→Cost-2 segments.
- “The mixer recombines phase-tagged amplitudes. Interference—not the cost phase
  alone—redistributes probability.”
- Mention that a mixer can temporarily increase energy; Global-Grover does so
  at its first p=2 mixer before the second mixer decreases it.

Exact sentence to emphasize:

> The cost Hamiltonian encodes objective information into relative phases; the
> mixer converts those phase differences into interference and probability
> redistribution.

## 7:00-10:00 — Three constructions

Use Figures 3 and 12.

1. Penalty-X: all 16,384 bitstrings, same Penalty-QUBO, local hypercube walk.
2. Global-Grover: all 16,384 bitstrings, byte-identical cost diagonal, global
   rank-one projector update.
3. Feasible-Grover: 20 enumerated routes, restricted route costs, projector
   mixing inside the logical feasible space.

Key comparison statements:

- “Penalty-X versus Global-Grover changes the mixer but not representation,
  initial state, or cost Hamiltonian.”
- “Global-Grover versus Feasible-Grover keeps projector geometry but changes
  support and baseline probability.”
- “Full-space initial p_opt is 1/16,384; feasible-space initial p_opt is 1/20.”

## 10:00-13:00 — p=1/p=2 results and dynamics

Use Figures 4-9 and the compact table:

| Method | p | p_feas | p_opt | `<H_C>` |
|---|---:|---:|---:|---:|
| Penalty-X | 1 | 0.001221 | 0.000061 | 86.000 |
| Penalty-X | 2 | 0.001221 | 0.000061 | 86.000 |
| Global-Grover | 1 | 0.001221 | 0.000061 | 86.000 |
| Global-Grover | 2 | 0.004174 | 0.000213 | 61.943 |
| Feasible-Grover | 1 | 1.000000 | 0.050000 | 12.200 |
| Feasible-Grover | 2 | 1.000000 | 0.170927 | 11.307 |

Talking points:

- “With this fixed single start, p=1 retained the uniform baselines.”
- “At p=2, both Grover constructions found nontrivial final concentration.”
- “The raw feasible and full-space energies are not directly comparable as
  computational workloads because their basis distributions differ.”
- “Both p=2 Grover optimizations reached the 100-evaluation cap, so these are
  retained finite results, not convergence guarantees.”

## 13:00-14:00 — Scientific computing evidence

Use Figures 11 and 14.

- 103 tests pass after implementation.
- Dense `16,384 x 16,384` Grover matrices are forbidden by construction and
  checked in tests; the update is linear in statevector size.
- All saved distributions normalize; every checkpoint norm is one.
- Existing Penalty-X and feasible-Grover simulators match the new traced paths
  to at worst `5.1e-18` in amplitude.
- Feasible-path enumeration took 0.01335 s and cost-vector construction
  0.000147 s on this run; these are reported rather than hidden.

## 14:00-15:00 — Conclusions and limitations

- “Cost phase is information encoding, not direct probability concentration.”
- “Mixer geometry changes how that phase information becomes probability.”
- “Restriction guarantees logical feasibility and changes the initial baseline.”
- “This is one teaching-scale ideal simulation. It makes no quantum-advantage,
  scalability, hardware-efficiency, or general mixer-ranking claim.”
- “Because feasible routes and costs were explicitly enumerated, the optimum
  was already classically visible by `argmin`; that construction is pedagogical.”
