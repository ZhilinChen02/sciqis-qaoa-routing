# Archived Q2-R preregistration contract

Status: **FROZEN_NOT_EXECUTED**  
Frozen: 2026-08-10  
Formal execution authorized: **no**  
Smoke-data status: **DEVELOPMENT_ONLY_NOT_CONFIRMATORY**

The machine-readable contract is
`data/q2_revision_preregistration.json`. Its SHA-256 is recorded in the
verification section below. This document explains that contract; if prose is
ambiguous, the hashed JSON controls. Freezing this contract does not authorize
the formal run.

## Scientific question and scope

On the one frozen directed weighted routing instance, how do a lower-tail CVaR
optimization objective and observable incremental depth, separately and
jointly, change final solution concentration under the existing
incumbent-product aligned-state/aligned-mixer Q2 algorithm?

This is one Q2-R algorithm with two factorial controls, not four different
quantum algorithms. All inference is instance-specific. No population-level,
general routing, quantum-advantage, or ingredient-novelty claim is authorized.

The preregistered non-directional hypotheses are:

1. Holding depth policy fixed, CVaR(0.25) changes final `p_opt` and `p_feas`
   relative to expectation optimization.
2. Holding objective mode fixed and the total objective-evaluation budget at
   `3B`, incremental `p=1 -> 2 -> 3` changes final `p_opt` and `p_feas`
   relative to fixed `p=3`.
3. The change associated with CVaR(0.25) differs between fixed and incremental
   depth policies.

Every started arm and failure is reported. There is no winner-selection rule.

## Frozen historical Q2 identity

The source of historical values is the unchanged
`results/baseline/experiment_config.json`, whose SHA-256 is
`4273d0d5ed5f4f16221d14ae2005b5e7c132a30ca157ae68d6e62a53a86ebbd2`.
The graph is `data/graph.json`, SHA-256
`451a6e4cffd01552c1cb6e4290b5ee8d50fc97514aab8ed47a6b33a111f5e54e`.

The following are frozen across every arm:

- epsilon `0.1` and the historical deterministic greedy incumbent;
- incumbent-product initial state and incumbent-dependent aligned mixer;
- historical mixer evolution scaling `beta / q`;
- directed-flow QUBO, fixed penalty `6.0`, Ising mapping, and affine energy
  normalization;
- parameter order `[gamma_1,...,gamma_p,beta_1,...,beta_p]` and cost-then-mixer
  layer order;
- graph, edge/qubit order, decoder, and final metric definitions.

Q2-R is an incumbent-product adaptation inspired by the aligned construction
of Egger et al. It is not the complete Egger method. The aligned mixer is not
replaced by an ordinary X mixer.

## Historical A0 equivalence gates

Two regression gates must pass before authorization:

1. An optimizer-independent `p=2` gate compares historical Q2 with A0 using
   one identical parameter vector. It covers the prepared state, statevector,
   probability vector, expectation, raw expected energy, `p_feas`, `p_opt`,
   invalid mass, and decoded-route metrics at tolerance `1e-12` or tighter.
2. An optimizer-trajectory gate compares historical `p=2` Q2 with fixed-depth,
   expectation-based `p=2` Q2-R using identical initial parameters, COBYLA settings,
   bounds, evaluation budget, epsilon, graph, Hamiltonian, and parameter
   convention. It compares the complete ordered evaluation trace, evaluation
   count, final parameters, objective, probability vector, `p_feas`, and
   `p_opt` at tolerance `1e-12` or tighter.

These tests are intended to fail if A0 changes the aligned state/mixer,
`beta / q` scaling, parameter ordering, bounds, optimizer trajectory, or final
metric definitions. The gate establishes implementation equivalence at
matched depth; the preregistered primary A0 below is the same method run at
fixed `p=3` under its declared budget.

## Primary 2x2 factorial

Let `B=100`, the historical Q2 per-depth objective-evaluation budget.

| arm | objective | depth policy | allocation | total cap |
|---|---|---|---:|---:|
| A0 | expectation | fixed `p=3` | one fixed-depth episode with `3B` | 300 |
| A1 | CVaR, `alpha=0.25` | fixed `p=3` | one fixed-depth episode with `3B` | 300 |
| A2 | expectation | incremental `p=1 -> 2 -> 3` | `B` per attempted depth | 300 |
| A3 | CVaR, `alpha=0.25` | incremental `p=1 -> 2 -> 3` | `B` per attempted depth | 300 |

The primary comparison is this full A0--A3 factorial. Fixed `p=1` controls
receive `B=100`, and fixed `p=2` controls receive `2B=200`, separately for
expectation and CVaR. Those four fixed-depth runs are budget-matched diagnostic
controls, not extra primary arms.

Unused incremental budget after an early stop is not moved to a later depth,
another arm, or another seed.

## Objective and alpha

Expectation optimization minimizes

\[
L_{\mathrm{EE}}=\sum_i p_i E_i.
\]

CVaR optimization uses the frozen `alpha=0.25`. States are sorted by increasing
energy and exactly the lowest alpha probability mass is consumed. If state `k`
crosses the discrete cutoff,

\[
L_{\alpha}=\frac{1}{\alpha}\left[
\sum_{i<k}p_iE_i+
\left(\alpha-\sum_{i<k}p_i\right)E_k
\right].
\]

The final atom is therefore partial. `alpha=1` is retained only as a
mathematical and optimizer-wrapper equivalence test. There is no primary alpha
sweep, and the development smoke run was not used to choose alpha.

## Optimizer, initialization, and seed policy

Every arm uses the historical bounded COBYLA wrapper:

- `gamma_j` bounds `[0, 2*pi]` and `beta_j` bounds `[0, pi]`;
- `rhobeg=0.5`, optimizer tolerance and constraint tolerance `1e-8`;
- source-uniform initialization from `numpy.random.default_rng`;
- all gamma coordinates drawn before all beta coordinates;
- the single historical seed `2601`;
- exactly one start per arm.

No replacement or additional seed is allowed after observing results.

## Objective-call accounting

One counted unit is one optimization-objective request. Fixed and incremental
arms use identical semantics:

- every COBYLA objective request counts, including an out-of-bounds penalty
  request;
- the initial state diagnostic is computed once, cached as COBYLA's first
  request, and charged exactly once;
- at transferred depth, the neutral-transfer diagnostic is likewise computed
  once, cached as the first optimizer request, and charged exactly once;
- the objective is not recomputed after COBYLA merely for reporting;
- final ordinary expectations and final scientific metrics are separate
  reporting calculations, not free optimizer-objective calls.

The trace records per-depth and cumulative objective counts, each objective
request and role, and per-depth and cumulative statevector evaluations. A
formal result is invalid if reported cumulative objective calls differ from
the sum of all per-depth trace entries.

## Neutral transfer and incremental stopping

For a completed depth `p`, the deterministic transfer is

```text
[old gammas, 0, old betas, 0].
```

The appended cost and mixer evolutions are the identity. Before optimizing any
transferred depth, the implementation requires

\[
\left|L_{p,\mathrm{neutral}}-L_{p-1,\mathrm{final}}\right|\le10^{-12}.
\]

At `p=1`, seed-to-final optimizer progress is recorded but is not called a
depth gain. A finite retained `p=1` result always proceeds to `p=2` when budget
remains. For transferred `p>1`, marginal gain is

\[
G_{p,\mathrm{abs}}=L_{p-1,\mathrm{final}}-L_{p,\mathrm{final}},
\qquad
G_{p,\mathrm{rel}}=
\frac{G_{p,\mathrm{abs}}}
{\max(|L_{p-1,\mathrm{final}}|,10^{-12})}.
\]

The `p=2` marginal gain determines whether `p=3` is attempted. Continuation
requires both strict conditions

```text
marginal_depth_gain_relative > 0.01
marginal_depth_gain_absolute > 1e-12
```

plus remaining budget. The hard maximum is `p=3`. The trigger cannot read the
known optimum, optimal bitstring, `p_opt`, decoder success label, or post-hoc
best depth.

## Endpoints and reporting

The co-primary final endpoints are `p_opt` and `p_feas`. Diagnostic endpoints
are invalid probability mass, ordinary normalized expectation, raw expected
energy, feasibility-conditional cost, decoded-route metrics, incumbent
probability, optimal-state rank, circuit resources, objective/statevector
counts, and runtime.

The optimized expectation or CVaR value remains a training quantity. CVaR does
not replace `p_opt`, `p_feas`, invalid mass, ordinary expectation, raw expected
energy, or route decoding. Exact-optimum information may be used only in final
evaluation and cannot affect alpha, epsilon, depth, stopping, optimizer, seed,
or retention decisions.

## Failure retention and stopping

Fixed arms stop on historical COBYLA convergence/failure or their declared
hard cap. Incremental arms stop after `p=3`, after the frozen `p=2` marginal
gain gate fails, when cumulative budget is exhausted, or on a hard non-finite,
neutral-equivalence, or unrecoverable execution failure.

Every started arm is retained. A finite best point is retained when COBYLA
reports non-success or exhausts the cap. A hard failure retains the completed
trace, consumed budget, and stop reason. Runs are not replaced, excluded, or
rerun with a new seed based on their outcome.

## Prohibited post-hoc changes

After freezing, it is prohibited to change alpha, epsilon, incumbent, `p_max`,
transfer, thresholds, budgets, budget reallocation, seed/multistart policy,
optimizer, bounds, initialization, `rhobeg`, tolerance, endpoints, or retention
rules in response to outcome data. It is also prohibited to select an arm or
depth with exact-optimum knowledge, drop an unfavorable arm, promote smoke
values to confirmatory evidence, or claim the combination of published
ingredients as novel.

Any necessary implementation correction must be documented before formal
execution and creates a new version and new hash; the present contract must not
be silently edited.

## Verification and execution gate

Preregistration JSON SHA-256:

```text
273304603e2357e35e958048b51fb8e58eae8dc02d9ae3c42090ac5b5456258f
```

The formal experiment has not been run. Development smoke values are excluded
from confirmatory tables and interpretation. A formal run requires passing the
state, optimizer-trajectory, neutral-transfer, objective-accounting, focused,
and full-suite tests, recording the final JSON hash, and receiving separate
user authorization.
