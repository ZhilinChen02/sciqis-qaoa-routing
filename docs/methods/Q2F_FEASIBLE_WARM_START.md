# Q2-F Feasible Warm-Start QAOA

## Scope and claim boundary

Q2-F is an optional logical/mechanism extension for the DTU SCIQIS course
project. It is separate from the course-core Penalty-X experiment and from the
sealed Q2-R formal execution. Q2-F is an exact logical feasible-subspace
teaching experiment, not a hardware-scalable routing implementation.

Full feasible-route enumeration is classical preprocessing. The resulting
complete route basis and cost vector expose the exact optimum classically, so
this experiment is mechanism analysis, not quantum computational advantage.
The structural identity `p_feas = 1` is likewise not an empirical quantum
advantage. Its purpose is to isolate probability concentration among feasible
routes after infeasible edge assignments have been removed.

No claim is made for scalability, hardware efficiency, runtime advantage,
noise robustness, general routing performance, or statistical significance.
Ascending-CVaR is a development/course-extension objective, not a formally
preregistered research result.

## Feasible logical basis

The frozen directed graph has 14 ordered edge variables, so edge-space Q2-R
evolves over `2^14 = 16384` computational states. Q2-F instead uses the basis

\[
\mathcal F=\{|P_0\rangle,\ldots,|P_{19}\rangle\},
\]

where the actual independently verified basis size is `r = 20`. Every basis
element is one simple directed route from source 0 to target 6.

`build_feasible_route_basis` reuses the existing independent deterministic DFS
enumerator, and independently compares its route set with
`networkx.all_simple_paths`. Each route is verified with the current graph,
edge ordering, raw edge-weight sum, edge-bitstring encoder, and independent
selected-edge route decoder. Duplicate node paths and bitstrings are rejected.
The deterministic order is `(routing cost, lexicographic node sequence)`.

The unique exact route remains

```text
0 -> 1 -> 2 -> 4 -> 5 -> 6, cost 10.
```

The existing historical greedy incumbent remains

```text
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6, cost 11.
```

The optimum is used only for final evaluation fields. It is not supplied to
the feasible warm-start probability constructor or optimizer callback.

## Logical cost Hamiltonian

The raw logical cost is

\[
H_C^F=\operatorname{diag}(C(P_0),\ldots,C(P_{19})).
\]

Raw route costs range from 10 to 14. Optimization and cost phases use the
transparent fixed normalization

\[
\widetilde C(P)=\frac{C(P)-10}{14-10}=\frac{C(P)-10}{4}.
\]

Every raw cost is retained. The positive affine normalization does not change
the optimal route or the definition of `p_opt`.

## Validated path-exchange mixer

The repository originally had no path-exchange implementation. Q2-F uses a
tested deterministic divergence-reconvergence rule on the current DiGraph and
existing edge representation.

For two feasible routes, remove their maximal common edge prefix and suffix.
They are adjacent when the two remaining nonempty subpaths have the same start
and end nodes and have disjoint internal nodes. This represents one valid
alternative-subpath exchange, not an arbitrary Hamming-neighbor rule.

On the frozen 20-route basis the rule produces 106 undirected exchanges and a
connected exchange graph. No complete-graph fallback or artificial bridge is
used. With uniform deterministic weights,

\[
H_M^F=\sum_{\{i,j\}\in E_F}
(|P_i\rangle\langle P_j|+|P_j\rangle\langle P_i|).
\]

The implementation name is `logical_path_exchange_mixer`. It is a true
logical path-exchange reference according to the validated local construction;
it is not presented as a new literature mixer or a hardware circuit.
Hermiticity, graph connectivity, unitarity, norm preservation, and `beta=0`
identity are tested. Exact Hermitian eigendecomposition applies
`exp(-i beta H_M^F)` inside the 20-dimensional basis. No amplitude coordinate
for an infeasible route exists.

## Feasible initial states

The zero-optimization controls are:

- F0, uniform feasible:

  \[
  |\psi_{F0}\rangle=\frac1{\sqrt{20}}\sum_i|P_i\rangle.
  \]

- F1, incumbent-biased feasible:

  \[
  d_i=d_H(x(P_i),x(P_{\mathrm{inc}})),\quad
  w_i=e^{-d_i},\quad
  p_i=\frac{w_i}{\sum_jw_j},\quad
  a_i=\sqrt{p_i}.
  \]

The fixed bias is `lambda = 1.0`. Distances are edge-bit Hamming distances.
The constructor receives route bitstrings and the historical incumbent
bitstring only—no route cost, optimality flag, exact route, or `p_opt`.
The primary 27-run matrix uses F1. F0 and F1 probabilities, distances, initial
`p_opt`, and expected route costs are retained in the result artifacts.

## Logical QAOA evolution

Parameters use the existing grouped convention

```text
[gamma_1, ..., gamma_p, beta_1, ..., beta_p]
```

with `gamma in [0, 2*pi]`, `beta in [0, pi]`, and depths `p=1,2,3`. Each
layer applies cost then mixer:

\[
|\psi_p\rangle=\prod_{\ell=p}^{1}
e^{-i\beta_\ell H_M^F}e^{-i\gamma_\ell\widetilde H_C^F}|\psi_{F1}\rangle.
\]

The exact NumPy logical state is checked after every cost and mixer layer. A
probability normalization error or `|p_feas-1| > 1e-12` is a hard failure.
No power-of-two padding is used for simulation.

## Optimization objectives

All objective losses use normalized logical route costs.

1. `expectation`:

   \[
   L=\sum_i p_i\widetilde C(P_i).
   \]

2. `cvar`: the existing validated lower-tail, fractional-cutoff CVaR with
   fixed `alpha=0.25`.

3. `ascending_cvar`: for zero-based objective-request index `t` and declared
   cap `T=100`,

   \[
   \alpha(t)=0.25+(1.0-0.25)\frac{t}{T-1}.
   \]

   It is clamped to `[0.25,1.0]`; `T=1` uses 0.25. The schedule advances on
   every declared optimizer request, including a bounded penalty request, and
   never reads objective outcomes. Every effective alpha is stored. If COBYLA
   converges before request 100, the realized trajectory ends before alpha 1;
   the denominator remains the prospectively declared cap.

Ascending-CVaR is deliberately nonstationary. Its terminal SciPy parameters
are retained; if the hard cap interrupts before SciPy returns, the last
in-bounds parameters are retained. Stationary expectation/CVaR runs retain the
best finite in-bounds objective point. The retention policy is recorded.

## Primary course-extension matrix

The frozen matrix is:

```text
initialization: F1, lambda=1.0
mixer:          logical_path_exchange_mixer, uniform weights
objectives:     expectation, CVaR(0.25), Ascending-CVaR(0.25 -> 1.0)
depths:         1, 2, 3
seeds:          2601, 2602, 2603
cap:            100 objective requests per run
optimizer:      bounded COBYLA, rhobeg=0.5, tol=catol=1e-8
total:          27 independently initialized fixed-depth runs
```

All gamma coordinates are drawn before all beta coordinates from
`numpy.random.default_rng(seed)`. Every cell retains convergence, budget
exhaustion, and poor finite results. No adaptive depth, seed replacement,
outcome-dependent rerun, or post-result lambda change is permitted.

## Metrics and interpretation

Each run retains the full probability vector and reports `p_feas`, `p_opt`,
optimal rank, top-3/top-5 lowest-cost route mass, expected raw and normalized
cost, most probable route/cost/probability, Shannon entropy, initial and final
`p_opt`, amplification, all optimizer requests and alphas, evaluation count,
statevector count, runtime, termination, and layer norms.

Aggregates across all three mandatory seeds use median, minimum, and maximum.
The best seed is only a secondary diagnostic. Three seeds do not support a
statistical-significance claim.

The sealed Q2-R A0/A1 values may appear as read-only conceptual context. Since
Q2-R and Q2-F use different representations, their state-space sizes,
feasibility, success probabilities, and runtimes are not fair algorithmic or
quantum-advantage comparisons.
