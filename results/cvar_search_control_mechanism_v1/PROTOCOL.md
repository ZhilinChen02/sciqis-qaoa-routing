# CVaR Search-Control Mechanism Study v1 — Frozen Protocol

## Prospective status

This protocol is prospective. It was written before any new scientific cell was executed or inspected. The 140-cell `results/cvar_robustness_v1` study is immutable historical evidence and may only be read and hash-verified. The precursor optimizer-budget document in that root does not govern this broader study.

The new namespace is `results/cvar_search_control_mechanism_v1`. It contains exactly 60 planned scientific cells and may not be expanded after results are observed.

## Scientific objective and invariants

The study asks why CVaR 0.10 produced reproducible `p_opt` recovery for Global-Grover but only mixed recovery for Penalty-X. It separates finite-budget behavior, persistent objective effects, objective-by-optimizer interaction, basin transfer, and mixer-associated probability transport.

All arms use the unchanged routing graph, p=110, direct 220-angle parameterization, raw continuous penalized-cost Hamiltonian with penalty 6.0, full 16,384-state ideal simulator, decoder, feasibility definition, route ordering, optimal-route identity, and physical parameter bounds. Only expectation and CVaR 0.10 are used. There are no new depths, alphas, mixers, graphs, Hamiltonians, decoding rules, or scaling fits.

The inherited optimal route is `[0,1,2,4,5,6]`, cost 10, canonical bitstring `10010001000101`, state index 10377. Feasibility means that the selected edges decode to exactly one source-to-target path with no remaining selected edges.

## Inherited metrics and optimizer semantics

All metrics call the existing robustness-study implementations without redefinition:

- `p_opt`: total probability on the exact optimal-route mask.
- `p_feas`: total probability on decoder-valid routes.
- `p_opt_given_feas`: `p_opt / p_feas` when `p_feas > 0`.
- Top-3 and Top-5 mass: absolute probability on the best 3 or 5 feasible routes ordered by true route cost, with state index as the deterministic tie-break.
- low-energy top-100 mass: absolute probability on the 100 lowest penalized-energy basis states, ordered by energy and state index.
- full entropy: Shannon entropy with natural logarithms over the full distribution.
- feasible entropy: Shannon entropy with natural logarithms after conditioning on feasibility.

COBYLA uses SciPy bounds, `rhobeg=0.5`, `tol=1e-8`, `catol=1e-8`, and `disp=False`. Its maximum evaluation count is passed as `maxiter`, matching the historical runner. Only calls through the optimizer callback are charged; the explicit start evaluation is index 0 and is not charged. The retained incumbent is the lowest actual training objective encountered, with the same out-of-bounds penalty convention as the frozen campaign.

Fresh direct angles use `numpy.default_rng(SeedSequence([seed, 110]))`; 110 gammas are sampled uniformly on `[0,2π]`, followed by 110 betas on `[0,π]`. Paired objectives and both mixers share the exact seed-derived vector and hash.

## Seeds and evidence classes

The fresh primary seeds are exactly 8801–8810. They are the deterministic first ten seeds selected from 8801 and do not overlap seeds in any prior result root. The sensitivity block is exactly the first five: 8801–8805.

Experiment A's B=11,000 endpoint is the confirmatory primary comparison. A's intermediate checkpoints are secondary mechanistic evidence. Experiments B, C, and D are pre-registered sensitivity/mechanism arms and remain descriptive because they use five seed pairs.

## Experiment A — budget response curve

Run one Global-Grover COBYLA trajectory for each objective and each of the ten primary seeds: 20 cells, each capped at 11,000 optimizer objective evaluations. Reconstruct the incumbent at budgets 1,000, 2,500, 5,500, 8,000, and 11,000 from the single trajectory. Checkpoints are not independent cells.

At every checkpoint retain the incumbent parameter vector, its evaluation index, both objective values, all inherited mechanism metrics, distribution hash, and optimizer status. If an optimizer terminates before a later checkpoint, carry the terminal incumbent forward with `TERMINATED_BEFORE_CHECKPOINT`; do not fabricate evaluations.

The primary endpoint is `p_opt` at B=11,000. Apply the historical rule verbatim: `ROBUST_CVAR_RECOVERY` requires at least 8/10 CVaR wins and median paired ratio at least 2.0; exactly one condition gives `MIXED_CVAR_RECOVERY`; neither gives `NOT_REPLICATED`. Report individual zero-safe ratios, win counts, medians, geometric means when all ratios are finite and positive, paired bootstrap intervals, absolute `p_opt`, and all ten paired trajectories at every budget.

For zero-safe ratios, a positive denominator uses the ordinary ratio; 0/0 is assigned ratio 1; positive/0 is an extended positive infinity recorded with an explicit status. Geometric means are withheld unless all ratios are finite and positive.

## Experiment B — optimizer sensitivity

Run Global-Grover with Nelder-Mead for expectation and CVaR 0.10 on seeds 8801–8805: 10 cells capped by `maxfev=5,500`. Freeze `xatol=fatol=1e-8`, `adaptive=False`, `disp=False`, the default SciPy initial simplex, and the same physical bounds. No option may be tuned after results are observed.

Use the same initial vectors as Experiment A. Compare within-Nelder-Mead CVaR versus expectation and compare its paired log-ratios against A's COBYLA checkpoints at B=5,500. A five-seed directional effect requires at least 4/5 wins and median paired ratio at least 2.0.

Classify `OPTIMIZER_ROBUST_EFFECT` when both optimizers satisfy that rule. Classify `OBJECTIVE_OPTIMIZER_INTERACTION` only when exactly one satisfies it and the optimizer difference in log-ratios has the same sign in at least 4/5 seeds with absolute median at least log(2). Otherwise classify `OPTIMIZER_SENSITIVITY_UNRESOLVED`. Two optimizers never establish optimizer generality.

## Experiment C — objective switch and basin transfer

For seeds 8801–8805, independently verify and retrieve Experiment A's B=5,500 expectation and CVaR incumbents. Evaluate each source vector under both objectives and all common metrics. From each source, start new, matched COBYLA invocations targeting expectation and CVaR, yielding `E_TO_E`, `E_TO_C`, `C_TO_E`, and `C_TO_C`: 20 cells, each with 5,500 additional evaluations.

Every branch has identical restart semantics and is compared only with another restarted branch. Save start and finish distributions, both objective values, metric deltas, feasible-route mass redistribution, raw Euclidean parameter displacement, and the evaluation trace.

For same-target paired contrasts, a material transfer uses at least 4/5 wins and a median `p_opt` ratio of at least 2.0. Define CVaR-to-expectation transfer by `C_TO_E / E_TO_E`. Define expectation-basin rescue as `E_TO_C / C_TO_C`: rescue noninferiority requires a median ratio at least 0.5 and at least 4/5 seed ratios at least 0.5. Objective pressure is supported when both `E_TO_C / E_TO_E` and `C_TO_C / C_TO_E` satisfy the material-transfer rule.

Decision order is frozen:

1. `BIDIRECTIONAL_TRANSFER` if CVaR-to-expectation transfer and expectation-basin rescue both hold.
2. `BASIN_DISCOVERY_SUPPORTED` if CVaR-to-expectation transfer alone holds.
3. `OBJECTIVE_PRESSURE_REQUIRED` if transfer does not hold but the objective-pressure criterion holds.
4. `BASIN_EFFECT_UNRESOLVED` otherwise.

These labels describe five paired restarts and do not establish landscape topology.

## Experiment D — Penalty-X negative control

Run one Penalty-X COBYLA trajectory for expectation and CVaR 0.10 on seeds 8801–8805: 10 cells capped at 11,000 evaluations, with the same five checkpoint budgets and metrics as Experiment A.

At each checkpoint compute each mixer's seed-paired CVaR-minus-expectation change. The final transport gate requires Global-Grover to show positive `p_feas` and low-energy-top-100 changes in at least 4/5 seeds, Global-minus-Penalty changes to be positive in at least 4/5 seeds for both metrics, and median cross-mixer differences of at least 0.01 probability in `p_feas` and 0.02 in low-energy-top-100 mass.

If every gate holds, classify `MIXER_SPECIFIC_TRANSPORT_SUPPORTED`. If Global transport is consistent and at least one cross-mixer metric passes both its consistency and magnitude gate, or if three of the four aggregate gates hold, classify `PARTIAL_MIXER_INTERACTION`. If neither cross-mixer consistency gate holds and both median differences are below their magnitude thresholds, classify `NO_CLEAR_MIXER_DIFFERENCE`. Otherwise classify `MIXER_MECHANISM_UNRESOLVED`. Mixer dynamics are not called causal solely from this association.

## Mechanism trajectory analysis

For Global-Grover, define the earliest checkpoint event as follows: at least 4/5 paired changes have the predicted sign and the median change reaches 0.01 for low-energy-top-100 mass, 0.005 for `p_feas`, or 0.005 for `p_opt_given_feas`; the `p_opt` event instead uses at least 4/5 wins and median ratio at least 2.0.

Classify `PROGRESSIVE_FEASIBLE_AMPLIFICATION` only when the low-energy event is no later than the feasible-mass event, the feasible event is no later than the optimal-mass event, at least one ordering is separated by a checkpoint, and within-feasible sharpening does not precede those events. If qualifying events first appear in the same checkpoint, use `SIMULTANEOUS_WITHIN_CHECKPOINT_RESOLUTION`. Otherwise use `TEMPORAL_ORDER_UNRESOLVED`. These temporal labels are descriptive and not causal.

## Persistence, failures, and verification

Every scientific identity has an atomic JSON completion marker committed after its trace, distributions, and any trajectory artifacts. Resume skips valid COMPLETE or FAILED identities, rejects invalid or duplicate identities, reruns only missing/incomplete identities, and never replaces a retained scientific failure.

The maximum is exactly 60 cells and 495,000 planned optimizer objective evaluations before failures; no retry may expand it. The historical evidence inventory, metrics, paired initialization hashes, parameter bounds, normalization, route masks, checkpoint provenance, and Experiment C restart references must verify before execution and after completion.

All repository tests must pass before execution. The protocol and manifest hashes are printed in the pre-execution review. No classification is computed until its complete required paired coverage exists. Negative and unresolved outcomes are valid and do not permit protocol changes.

## Prohibited claims

This study cannot establish a depth scaling law, optimal alpha, general optimizer superiority, general mixer superiority, causal landscape topology, new-instance generality, noisy-device robustness, quantum advantage, or fully converged superiority when evaluation budgets are exhausted.
