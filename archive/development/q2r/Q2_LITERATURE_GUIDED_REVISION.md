# Archived Q2 literature-guided revision

## Scope

Q2-R is an opt-in revision of the repository's existing Q2 Warm-Start QAOA. It
keeps one aligned warm-start algorithm and makes two additions independently
switchable:

1. expectation or lower-tail CVaR as the classical optimization loss;
2. fixed or observable incremental depth with deterministic parameter transfer.

The historical `run_experiment` path, its configuration, result schema, and
saved results are unchanged. Q2-R is exposed through `Q2RevisionConfig` and
`run_q2_revision` in `src/q2_revision.py`. The compact development-only example
is `data/q2_revision_config.example.json`; its alpha and trigger thresholds are
not final scientific choices.

## 1. Existing Q2 retained by Q2-R

Let the deterministic greedy feasible incumbent be

\[
b=(b_1,\ldots,b_q), \qquad q=14.
\]

For `0 < epsilon < 0.5`, the existing implementation defines

\[
c_i=\begin{cases}
1-\epsilon,&b_i=1,\\
\epsilon,&b_i=0,
\end{cases}
\]

and prepares

\[
|\phi_0\rangle=\bigotimes_i
\left(\sqrt{1-c_i}|0\rangle+\sqrt{c_i}|1\rangle\right).
\]

The mixer is not an ordinary X mixer. Its single-qubit Hamiltonian is

\[
H_{M,i}=-2\sqrt{c_i(1-c_i)}X_i-(1-2c_i)Z_i,
\]

and the prepared qubit is its eigenstate with eigenvalue `-1`. Q2-R calls the
same `simulate_qaoa_state` and `build_qaoa_circuit` functions, so it preserves:

- the directed graph, edge order, flow QUBO, Ising mapping, and normalization;
- the greedy incumbent and epsilon construction;
- the incumbent-dependent aligned initial state and mixer;
- cost-then-mixer layer order;
- mixer-angle scaling by `1 / num_qubits`;
- the independent route decoder and final distribution metrics.

### Historical A0 equivalence gate

`test_historical_q2_and_a0_are_equivalent_without_optimizer` evaluates one
fixed `p=2` parameter vector through both the historical Q2 entry point and the
Q2-R A0 direct-evaluation entry point. With the same graph, incumbent, epsilon,
Hamiltonian, and grouped parameter vector, the regression compares the prepared
initial state, complex statevector, probability vector, expectation loss,
`p_feas`, `p_opt`, invalid mass, raw expected energy, feasibility-conditioned
cost, decoded route summaries, and state ranks at tolerance `1e-12` or tighter.
This gate is optimizer-independent and currently passes. It is designed to fail
if A0 changes aligned state preparation, the aligned mixer, grouped parameter
ordering, or the historical `beta / q` mixer scaling.

`test_historical_q2_and_a0_optimizer_trajectories_are_equivalent` adds the
optimizer-level `p=2` gate. With identical historical initialization, COBYLA bounds,
`rhobeg`, tolerance, evaluation cap, epsilon, graph, and Hamiltonian, it compares
every ordered objective request and value, the evaluation count, final
parameters/objective/probabilities, `p_feas`, and `p_opt` at `1e-12` or tighter.

### Relationship to Egger et al.

The state/mixer pair is an incumbent-product adaptation of the aligned
state/mixer construction described by Egger, Mareček, and Woerner. The project
uses a binary greedy route clipped away from the poles. It does not implement
their continuous QP/SDP-relaxation preprocessing, rounded-MaxCut variants, or
the complete published workflow. Code and reports must not call Q2-R the
"complete Egger algorithm."

## 2. Objective abstraction and discrete CVaR

`src/objectives.py` provides the common entry point
`evaluate_optimization_objective(probabilities, energies, mode, cvar_alpha)`.
The `expectation` mode retains the existing loss:

\[
L_{\mathrm{EE}}=\sum_i p_i E_i.
\]

For minimization, `cvar` sorts by increasing energy and consumes exactly the
lowest alpha probability mass. If `k` is the first ordered state whose
cumulative mass reaches alpha, Q2-R computes

\[
L_{\alpha}=\frac{1}{\alpha}\left(
\sum_{i<k}p_iE_i+
\left(\alpha-\sum_{i<k}p_i\right)E_k
\right), \qquad 0<\alpha\le1.
\]

Only the required fraction of the cutoff atom is used. The implementation does
not retain a fixed number of bitstrings and does not include all tied cutoff
mass. Tied-energy ordering is immaterial because every portion of the tie has
the same energy. At `alpha=1`, CVaR returns the ordinary expectation.

Inputs must be finite vectors of equal nonzero length. Probabilities must sum to
one within `1e-10`. Negative values below `-1e-12` are rejected; artifacts in
`[-1e-12,0)` are clipped to zero and the distribution is renormalized. A
partial CVaR tail normalizes any remaining accepted roundoff drift before
consuming alpha mass. The expectation path preserves an already non-negative
validated vector verbatim so A0 remains numerically identical to the historical
objective after `state_probabilities` has normalized it. Other malformed
distributions are rejected.

CVaR is only the optimizer's scalar loss. It does not replace success,
feasibility, route cost, or ordinary expected energy.

## 3. Incremental-depth controller

Fixed mode optimizes only `initial_depth`. Incremental mode performs:

```text
optimize at p
  -> evaluate loss improvement and remaining budget
  -> stop, or neutrally transfer to p+1
  -> optimize all p+1 layers
```

The first depth and transferred depths have different semantics.

For `p=1`, let

\[
P_{1,\mathrm{abs}}=L_{1,\mathrm{seed}}-L_{1,\mathrm{final}},
\qquad
P_{1,\mathrm{rel}}=
\frac{P_{1,\mathrm{abs}}}
{\max(|L_{1,\mathrm{seed}}|,s_{\min})}.
\]

This is baseline optimizer progress from a seeded initialization. It is not a
depth gain. Accordingly, all `inherited_parent_*`, `neutral_transfer_*`, and
`marginal_depth_gain_*` fields are `None` for `p=1`.

For each transferred `p>1`, the parent is the completed depth `p-1`. Before the
new optimizer starts, Q2-R numerically evaluates the neutrally transferred
parameters and records

\[
e_p=\left|L_{p,\mathrm{neutral}}-L_{p-1,\mathrm{final}}\right|.
\]

The run aborts if `e_p > 1e-12`. The marginal contribution of optimizing the
new depth is then defined as

\[
G_{p,\mathrm{abs}}=
L_{p-1,\mathrm{final}}-L_{p,\mathrm{final}},
\qquad
G_{p,\mathrm{rel}}=
\frac{G_{p,\mathrm{abs}}}
{\max(|L_{p-1,\mathrm{final}}|,s_{\min})}.
\]

This—not progress from an arbitrary initialization—is the reported marginal
depth gain. The preregistered controller records baseline progress but always
attempts the first neutral transfer from `p=1` to `p=2` when budget remains.
After each transferred depth it uses marginal depth gain and continues only
when all conditions hold:

- depth mode is incremental;
- current depth is below `max_depth`;
- cumulative objective evaluations are below the total budget;
- the applicable absolute gain is strictly above its configured tolerance; and
- the applicable relative gain is strictly above its configured tolerance.

Otherwise it records one of `fixed_depth_complete`, `maximum_depth_reached`,
`cumulative_evaluation_budget_exhausted`, or
`marginal_depth_gain_plateau`. The
trigger receives only optimization losses, depths, budgets, and its frozen
thresholds. It has no argument for the exact optimum, optimum bitstring,
`p_opt`, decoder label, or post-hoc winning depth.

## 4. Neutral parameter transfer

The repository stores parameters as

```text
[gamma_1, ..., gamma_p, beta_1, ..., beta_p]
```

rather than interleaved layer pairs. Q2-R therefore transfers

```text
[gamma_1, ..., gamma_p, 0, beta_1, ..., beta_p, 0].
```

All old gamma and beta values are copied exactly. The new cost angle and mixer
angle are both zero, so the appended cost-then-mixer layer is exactly the
identity before reoptimization. No random draw changes inherited values. The
pre-optimizer neutral objective is cached and supplies COBYLA's first counted
request, so numerical inheritance verification does not create an unreported
optimization opportunity.

This is not Saini et al.'s DDQAOA transfer: Q2-R does not use their `1.2/0.8`
scaling, linear/cubic interpolation switch, plus-state/X-mixer ansatz, Adam
loop, or CSPP QUBO. Zero-appended full reoptimization is also not claimed as
novel; Truger et al. used an incremental full-optimization construction.

## 5. Configuration and A0--A3 isolation

The four implementation identities are:

| identity | aligned state/mixer | objective | depth policy | transfer |
|---|---|---|---|---|
| A0 | existing | expectation | fixed | inactive |
| A1 | existing | CVaR | fixed | inactive |
| A2 | existing | expectation | incremental | neutral zero append |
| A3 | existing | CVaR | incremental | neutral zero append |

`ablation_configurations` constructs all four from common settings. A0 to A1
changes only `objective_mode` and `cvar_alpha`. A0 to A2 changes only
`depth_mode` and `max_depth`; transfer semantics are already frozen in every
configuration. A0 to A3 changes those two groups together. Every configuration
validates the same conservative identifiers:

```text
warm_start_mode = incumbent_product_aligned_existing
mixer_mode      = incumbent_dependent_aligned_existing
mixer_scale_mode = inverse_num_qubits_existing
transfer_mode   = neutral_zero_append_grouped
```

The default `Q2RevisionConfig()` is expectation-based, fixed at `p=1`, seeded
with the existing source-uniform policy, and limited to the historical
per-depth budget. The old experiment code does not invoke this new API, so its
behavior remains unchanged unless Q2-R is selected explicitly.

## 6. Budget accounting and depth history

Each depth receives at most `per_depth_budget` objective calls and at most the
remaining `total_cumulative_budget`. If the total budget is omitted, the
effective cap is the per-depth budget times the number of permitted attempts.
No depth receives unrecorded optimization work.

### Budget-matched fixed controls

`budget_matched_fixed_depth_config` constructs a fixed-depth comparator from an
incremental reference. If incremental optimization starts at `p0` with
per-depth cap `B`, a fixed control at depth `p` receives

\[
B_{\mathrm{fixed}}(p)=
\min\left(B_{\mathrm{total}},(p-p_0+1)B\right).
\]

For an incremental `1 -> 2` run this supports the preregisterable comparison:

```text
fixed p=1:       budget B
fixed p=2:       budget 2B
incremental 1->2 cumulative budget 2B
```

The construction extends analogously through `p_max`. A fixed comparator uses
one optimization episode at its target depth with the full matched cumulative
cap; the incremental run records each per-depth allocation and its cumulative
total. This capability is implemented but no formal comparison has been run.

The result records:

- objective evaluations and actual statevector evaluations per depth;
- cumulative objective and statevector evaluations;
- optimizer runtime per depth and cumulative optimization runtime;
- input/output parameters and before/after losses;
- parent depth/objective, neutral-transfer objective, and equivalence error;
- baseline optimizer progress at `p=1` and marginal depth gain for `p>1`;
- ordinary normalized and raw expectations at every completed depth;
- continuation decision and reason;
- realized maximum depth and final stop reason.

Objective evaluations count every COBYLA request, including any bounded penalty
request. Statevector evaluations are counted independently and include the
post-optimization distribution evaluation at each depth. `overall_runtime` also
captures revision-controller overhead; `cumulative_runtime` is the sum of the
per-depth optimizer runtimes.

## 7. Final metric separation

`run_q2_optimization` performs optimization and depth control without exact
route information. Only after it stops does `run_q2_revision` attach the exact
evaluation oracle. The result keeps separate fields for:

- `optimization_objective_mode`, `cvar_alpha`, and
  `optimized_objective_value`;
- ordinary normalized `final_expectation_value` and raw QUBO
  `final_expected_energy`;
- `p_feas`, `p_opt`, and `invalid_probability_mass`;
- feasibility-conditioned route cost and decoded-route summaries;
- initial/final incumbent and optimum probabilities;
- optimal rank and circuit resources.

Thus CVaR cannot overwrite or be mislabeled as `p_feas`, `p_opt`, expected
energy, or decoded route quality.

## 8. Smoke-test status

The previously run four-path smoke used deliberately tiny development budgets
only to verify execution, result schema, CVaR plumbing, and transfer. Its values
are non-scientific: they must not select an arm, alpha, epsilon, depth trigger,
or `p_max`, and they are not evidence that any mechanism improves performance.

## 9. Literature boundaries

Cain et al. analyze a good computational-basis string followed by ordinary,
incumbent-independent QAOA operators. Q2-R has a full-support biased product
state and an incumbent-dependent aligned mixer. Their work motivates an
incumbent-lock diagnosis but does not directly prove this Q2 implementation is
stuck.

Truger et al. already combine warm starts, CVaR, epsilon studies, and
incremental depth for MaxCut. Q2-R therefore exists to support clean routing
ablations, not to claim novelty from combining published ingredients.

## 10. Parameters still requiring preregistration

Before any formal experiment, freeze without consulting final `p_opt`:

- CVaR alpha and whether `alpha=1` is included as a mathematical control;
- epsilon;
- absolute and relative continuation thresholds and scale floor;
- initial depth and `p_max`;
- per-depth and total cumulative objective-evaluation budgets;
- the budget-matched fixed-depth comparator set;
- optimizer seeds/multistart policy and deterministic run-selection rule;
- exact-statevector versus sampled diagnostic policy;
- final reporting contract and result destination.

## 11. What is not claimed

- Q2-R is not the complete Egger algorithm.
- Q2-R is not Saini's DDQAOA.
- CVaR is not success probability and does not guarantee higher `p_opt`.
- Allowing a deeper circuit does not establish a benefit from incremental depth.
- The neutral transfer rule and the combination of mechanisms are not novel.
- Cain et al.'s negative theorem does not directly cover this aligned mixer.
- Development tests or the four-path smoke are not scientific evidence.
- No alpha, trigger tolerance, maximum depth, or formal experimental budget is
  selected by this implementation task.
