# `optimizer_budget_robustness_v1` Follow-up Protocol

## Status: draft only — do not execute

This document proposes a small follow-up experiment. It is not a frozen manifest, does not authorize computation, and must not create or modify scientific artifacts until separately reviewed, approved, and frozen. The completed `cvar_robustness_v1` root remains immutable and is used only as a read-only reference.

## Sole scientific purpose

Test whether the Global-Grover CVaR 0.10 effect at p=110 persists under a larger optimizer budget, then perform one limited alternative-optimizer sensitivity check. The follow-up must not expand into alpha tuning, depth exploration, mixer comparison, seed selection, or broader algorithm search.

## Invariants

Hold fixed the routing instance, Hamiltonian, Global-Grover mixer, p=110 depth, direct parameterization, simulator, probability definitions, expectation and CVaR 0.10 objectives, and the original ten primary seeds (8601–8610). For each seed, reuse the exact frozen initialization angles and initialization hash from `cvar_robustness_v1`; do not continue from the endpoint of a completed optimization.

The existing 5,500-evaluation cells are immutable baselines. Reference them by canonical cell identity and hash. Do not rerun, copy into a new scientific row, repair, or overwrite them.

## Block A: larger-budget COBYLA test

Run Global-Grover at p=110 for expectation and CVaR 0.10 across all ten frozen primary seeds, using COBYLA with `max_evals=11,000`. All other optimizer settings must match the frozen primary contract. This produces 20 new cells.

The predeclared persistence rule reuses the original thresholds:

- `PERSISTS_UNDER_LARGER_BUDGET` only if CVaR 0.10 has higher `p_opt` for at least 8/10 paired seeds and the median paired ratio is at least 2.0;
- `MIXED_UNDER_LARGER_BUDGET` if exactly one condition passes;
- `NOT_PERSISTENT_UNDER_LARGER_BUDGET` if neither condition passes.

Also report absolute medians, win count, median and geometric-mean paired ratios, a paired bootstrap 95% interval, each termination status, and paired changes from 5,500 to 11,000 evaluations for each objective. If the 11,000-evaluation cells also exhaust their budgets, do not describe them as converged.

## Block B: limited alternative-optimizer sensitivity

Use exactly one alternative optimizer: SciPy COBYQA. Before freezing the manifest, pin the exact SciPy version and record every supported option, bound treatment, stopping tolerance, and objective-call counting rule. Do not tune those settings after observing results.

Run expectation and CVaR 0.10 for the first five numerically ordered frozen seeds (8601–8605), with their exact original initialization hashes and a maximum of 11,000 objective evaluations. The subset is determined by seed order, not outcomes. This produces 10 new cells.

Treat this block as descriptive sensitivity evidence. Report paired wins, medians, paired ratios, uncertainty, termination status, and evaluation counts, but do not let a five-seed alternative-optimizer result override the Block A persistence classification or support universal optimizer claims.

## Size and storage

The protocol contains 30 new cells: 20 larger-budget COBYLA cells plus 10 COBYQA sensitivity cells. If authorized, store them in a new root:

```text
results/optimizer_budget_robustness_v1/
```

Never write follow-up rows, manifests, checkpoints, or distributions into `results/cvar_robustness_v1/`.

## Required freeze and preservation controls

Before any future execution:

1. Create and review a new manifest containing all 30 exact cell identities, frozen settings, original seed values, and original initialization hashes.
2. Pin the canonical hash of the 140-cell baseline and hashes for its manifest, checkpoints, and distributions.
3. Implement atomic per-cell persistence and fail-closed resume checks in the new root.
4. Record objective calls consistently across optimizers; do not substitute iteration counts for evaluation counts.
5. Verify paired initialization hashes, normalized distributions, `p_opt <= p_feas`, and `p_opt_given_feas = p_opt / p_feas`.
6. Recheck the immutable baseline hashes before and after the follow-up.
7. Preserve all scientific failures; do not replace failed seeds or extend budgets adaptively.

No extra alpha, depth, mixer, initialization, optimizer, or seed block may be added after results are observed. Any such extension requires a separately named and separately frozen protocol.

## Interpretation boundaries

Block A isolates whether the observed objective effect persists at twice the original p=110 evaluation budget on the same seeds and initializations; it is not an independent-instance replication. Block B provides limited optimizer sensitivity only. Neither block can establish convergence unless the recorded termination evidence supports it, and neither can establish instance generality, hardware robustness, a universally superior optimizer, quantum advantage, or a depth scaling law.

## Execution gate

No command in this protocol has been executed. Execution requires separate authorization after the 30-cell manifest, dependency versions, optimizer options, hashes, and reporting rules have been reviewed and frozen.
