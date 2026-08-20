# QAOA Routing Final Presentation — Speaker Notes

**Boundary:** course experiments only; one teaching-scale graph; ideal simulation; no quantum-advantage claim.

## Slide 1 — QAOA Routing

We show the complete scientific chain for one transparent routing instance. The aim is to understand the encoding and probability dynamics, not to outperform classical shortest-path algorithms.

## Slide 2 — The routing problem

The graph is fixed in data/graph.json. NetworkX shortest path and exhaustive simple-route enumeration agree on the same unique cost-10 route.

## Slide 3 — Edge variables become a routing QUBO

The objective says which route is cheap, while the flow penalty says whether the selected edges form a valid source-to-target route.

## Slide 4 — QUBO → Ising → QAOA

The cost layer alone does not change measurement probabilities. Interference appears after the mixer, so the mixer geometry is scientifically important.

## Slide 5 — Two search spaces, three mixers

Restricting the logical basis to valid routes structurally removes infeasible probability mass, but it does not by itself guarantee concentration on the optimum.

## Slide 6 — What happens to probability mass?

The full-space methods leave most probability on infeasible bitstrings. The feasible basis removes that mass by construction and lets us study concentration among valid routes.

## Slide 7 — Final course method comparison

The table uses the saved three-seed median. Evaluation caps differ between the historical Q2-F baseline and the final matrix, so this is a course-result summary rather than a strict equal-budget leaderboard.

## Slide 8 — The final distribution is concentrated on the optimum

This is an ideal statevector probability distribution, not finite-shot frequency. The optimum label is used for evaluation and presentation, not supplied directly to COBYLA.

## Slide 9 — Why GM-Th works on this instance

The near-unit success probability depends on instance-specific threshold information. This is a clear mechanism demonstration, not a scalable quantum-routing or quantum-advantage result.

## Slide 10 — CVaR course extension: a useful negative result

Some shallow cells improve, but the saved three-seed depth-three comparison does not show stable superiority over expectation. We report this as a legitimate negative result.

## Slide 11 — Run it and watch it

The circuit replay focuses on the final feasible methods. The dynamics dashboard compares Penalty-X, Global-Grover, and Feasible-Grover layer by layer.

## Slide 12 — What we learned

The main scientific message is the separation of validity from optimality. Representation controls where amplitude is allowed; the phase and mixer control where probability concentrates inside that representation.

## Slide 13 — Backup: a two-qubit Ising gate example

This optional Qiskit example connects the Ising formula to familiar gates without claiming that the logical feasible-space operators have a shallow hardware decomposition.
