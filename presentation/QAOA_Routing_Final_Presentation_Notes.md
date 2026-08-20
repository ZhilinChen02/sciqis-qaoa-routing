# QAOA Routing Final Presentation — Speaker Notes

**Target main-deck timing:** approximately 14:20–15:00, excluding backup slides.

**Presentation boundary:** one teaching-scale routing instance, ideal simulation, no quantum-advantage or standard-Grover speedup claim.

## Slide 1 — From Routing to QAOA: How Mixer Design and Circuit Depth Shape Feasibility and Optimality

**Timing:** 0:50

[~0:50] We use a small routing problem as a transparent test case for understanding how a constrained optimization problem is encoded and explored by QAOA. The goal is not to compete with classical shortest-path algorithms and we make no quantum-advantage claim. The question is narrower: when the cost problem is fixed, how do the mixer and the circuit depth change feasibility and optimality?

## Slide 2 — A small graph, a large binary space

**Timing:** 1:10

[~1:10] This is the exact directed graph stored in data/graph.json: seven nodes and fourteen positive-weight edges. Node 0 is the source and node 6 is the target. Exhaustive route enumeration and NetworkX agree on one optimal path, 0–1–2–4–5–6, with cost 10. The graph is intentionally small, but fourteen edge bits already give 16,384 states. Only twenty decode as valid routes, so most of the binary space is invalid. That leads directly to the constraint encoding.

## Slide 3 — From edges to a routing QUBO

**Timing:** 1:20

[~1:20] Each directed edge gets a binary variable. The ordinary route cost is just the weighted sum of the selected edges. Validity is expressed with one flow-balance equation per node: the source creates one unit, intermediate nodes conserve it, and the target absorbs it. We move everything to the left, subtract b_v, and square the residual. The final QUBO is route cost plus A times the flow penalty; this instance fixes A at 6. The useful mental model is that the objective says what is cheap, while the penalty says what is valid. Next we translate that classical polynomial into quantum operators.

## Slide 4 — QUBO → Ising → QAOA

**Timing:** 1:15

[~1:15] Expanding the squared constraints gives a constant, linear terms, and pairwise binary terms. A QUBO matrix and this polynomial are two equivalent representations after collecting coefficients. We then substitute x_i equals one minus Z_i over two. That produces an Ising Hamiltonian with identity, Z, and ZZ terms. The repository checks the QUBO and Ising energies exactly on every basis state. The cost unitary encodes energy into phase; it does not directly change probabilities. The mixer is what converts those phase differences into interference. The cost Hamiltonian is shared by both experiments, so the next slide isolates the mixer choice.

## Slide 5 — Same cost landscape, two mixer geometries

**Timing:** 1:25

[~1:25] Both methods start uniformly over the same 16,384 bitstrings and use the same penalized cost Hamiltonian. Penalty-X applies independent X rotations. Its geometry is local: it connects Hamming-neighbor bitstrings through single-bit flips. Global-Grover instead uses the rank-one projector onto the uniform full-space state. The exact implementation is I plus e to the minus i beta minus one times |s><s|, so it redistributes amplitude globally without building a dense matrix. It is not feasibility-preserving. It is also not standard Grover Search: no oracle directly identifies the optimal route. The cost Hamiltonian still defines quality, and this Grover-style operation is only the QAOA mixer.

## Slide 6 — The hybrid loop: COBYLA and CVaR

**Timing:** 1:15

[~1:15] QAOA is hybrid. For each proposed angle vector, the simulator returns an objective value and COBYLA proposes another vector. COBYLA is derivative-free and it optimizes the gamma and beta angles, not a route directly. At depth p there are 2p parameters, so p=110 means 220 angles. The frozen depth sweep gives both mixers the same depth-specific budget, max of 120 and 4p plus 64, which is 504 evaluations at p=110. The main sweep optimizes expectation. CVaR is a separate study: it averages only the lowest-energy alpha fraction of probability mass. It changes the classical objective, not the mixer. In the repository's fresh-seed p=110 study, the Penalty-X result was mixed at 7 of 10 wins, while Global-Grover showed 10 of 10 paired wins; absolute probabilities remained small.

## Slide 7 — What we measure — and how

**Timing:** 1:10

[~1:10] We need three probabilities. p_feas is all mass on valid decoded routes. p_opt is mass on the unique optimal route. Their ratio is the probability of the optimum conditioned on having sampled something feasible. The identity in the center is the key to the result: total success is feasible-region entry times selection quality inside that region. The frozen experiment compares two full-space mixers at every depth from 1 to 110. Each method-depth has its own COBYLA run, but the initialization follows one deterministic layerwise-continuation trajectory per mixer, so these are not independent random replicates. At each depth the cost, decoder, representation, and evaluation budget are matched, giving 220 method-depth rows. With the metrics defined, we can now ask where the probability goes.

## Slide 8 — Depth 1–110: where does probability go?

**Timing:** 1:35

[~1:35] The large plot is total optimal-route probability. Penalty-X rises quickly and reaches the largest value in the entire sweep at p=21: about 0.2833 percent. It then drops sharply and oscillates. Global-Grover stays near the uniform baseline at shallow depth, then improves in steps. Its first total-probability crossover is at p=22, and in this frozen continuation trajectory Penalty-X never overtakes it again at later depths. Global-Grover reaches its own maximum at p=110, about 0.1030 percent, still below the earlier Penalty-X peak. The smaller plots explain why: Penalty-X has more feasible mass at all 110 depths, while Global-Grover has the higher conditional optimal fraction at 89 of 110 depths.

## Slide 9 — Conditional quality is not total success

**Timing:** 1:25

[~1:25] At p=110, Penalty-X puts almost 14.7 percent of the state on feasible routes, but only 0.2435 percent of that feasible mass is the optimum. Global-Grover reaches only 2.19 percent feasible mass, yet 4.69 percent of its feasible mass is optimal. That conditional concentration is 19.3 times larger, but it comes with 6.7 times less feasible mass. Multiplying the two factors, Global-Grover has 2.88 times higher total success at the same depth. So it is better at selecting the optimum once probability reaches the feasible region, but worse at getting there. Across the whole sweep, however, Penalty-X's earlier p=21 peak remains 2.75 times larger than Global-Grover's best observed value.

## Slide 10 — Deeper is not automatically better

**Timing:** 1:15

[~1:15] The natural depth question is whether something dramatic happens near p around one hundred. In the frozen data, it does not. The p=90 to 110 window remains oscillatory, with six direction changes for Penalty-X and nine for Global-Grover. Neither method reaches even ten percent total optimal probability. More importantly, the classical problem grows with depth: p=110 has 220 variables and only 504 objective evaluations. Every one of the 220 sweep cells exhausted its declared budget. We therefore cannot attribute poor high-depth performance only to the quantum ansatz or mixer; the classical optimization also becomes harder. The cautious conclusion is limited to this instance and protocol and does not disprove Grover scaling.

## Slide 11 — Scientific computing and validation

**Timing:** 0:55

[~0:55] The scientific-computing workflow is explicit and reproducible: build the model, simulate, optimize, save immutable artifacts, validate, and plot. The graph reference uses NetworkX, while NumPy performs exact statevector evolution and SciPy provides COBYLA. The current working tree passes 167 tests, including 23 visualization tests. The depth sweep and CVaR studies each retain 43 passing artifact-level validation checks. QUBO and Ising energies agree exactly across all 16,384 states, and the core smoke command still reproduces the unique cost-10 route. The important point is not software complexity; it is that every plotted number has a checkable route back to a saved artifact.

## Slide 12 — Takeaways and limitations

**Timing:** 0:45

[~0:45] The complete pipeline works end to end. The mixer choice matters because it changes how probability is transported through the same cost landscape. Penalty-X is much better at entering the feasible region; Global-Grover more often concentrates feasible mass toward the optimum. The product identity explains why improving one factor may not improve total success. These conclusions are deliberately narrow: one small graph, ideal simulation, finite exhausted optimization budgets, and no hardware or quantum-advantage claim. The final message is that feasibility, conditional quality, and classical optimization difficulty must be analyzed separately.

## Slide 13 — Backup A — Why subtract bᵥ?

**Timing:** backup

Backup explanation of the flow residual and one valid/invalid example.

## Slide 14 — Backup B — QUBO matrix and Ising mapping

**Timing:** backup

Backup derivation of the matrix/polynomial equivalence and x-to-Z mapping.

## Slide 15 — Backup C — What is COBYLA?

**Timing:** backup

Backup explanation of COBYLA, the 2p parameter count, and the frozen evaluation rule.

## Slide 16 — Backup D — Global-Grover is not standard Grover Search

**Timing:** backup

Backup comparison separating the repository's Grover-style mixer from standard oracle-based Grover Search.

## Slide 17 — Backup E — CVaR robustness study

**Timing:** backup

Backup summary of the separate fresh-seed CVaR robustness study and its claim boundary.

## Suggested transitions

- Slide 2 → 3: “The graph is small, but the binary state space already contains many invalid edge combinations. So the next question is how to encode route validity.”
- Slide 4 → 5: “The cost Hamiltonian tells us which solutions are good, but the mixer determines how amplitude moves between them. This is where our two QAOA designs differ.”
- Slide 7 → 8: “With these three metrics defined, we can now ask where the probability actually goes as depth increases.”
- Slide 8 → 9: “The three curves appear to tell different stories, and the product identity resolves the apparent contradiction.”
- Slide 10 → 11: “Because high-depth conclusions depend on the numerical protocol, validation and provenance are part of the scientific result.”
