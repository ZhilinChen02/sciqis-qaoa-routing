# Final Scientific Interpretation

## Central result

Global-Grover's CVaR 0.10 recovery persisted when the COBYLA budget doubled. At B=11,000, CVaR won 10/10 fresh pairs; expectation and CVaR median `p_opt` were 0.000542 and 0.002165, the median paired ratio was 2.889, the geometric mean was 3.744, and the paired bootstrap interval was [2.618, 5.766]. The frozen classification is `ROBUST_CVAR_RECOVERY`.

The effect passed the same rule at every trajectory checkpoint from B=1,000 through B=11,000. It is therefore not explained by the original B=5,500 cap, although every new cell also exhausted its budget and the comparison remains a finite-budget effect rather than a converged-optimum result. Absolute `p_opt` remained small: the Global CVaR median was about 0.216%.

## What changed in the search

The strongest mechanistic signature is probability transport. At B=11,000, Global-Grover's median CVaR-minus-expectation changes were +0.02827 in `p_feas` and +0.07944 in low-energy-top-100 mass, versus +0.00472 and +0.01169 for Penalty-X. Global `p_opt_given_feas` changed by only +0.00092 and never met the trajectory sharpening criterion. The frozen mixer result is `MIXER_SPECIFIC_TRANSPORT_SUPPORTED`.

Low-energy, feasible, and optimal-mass effects all qualified at the first B=1,000 checkpoint, while within-feasible sharpening never qualified. The correct temporal label is `SIMULTANEOUS_WITHIN_CHECKPOINT_RESOLUTION`: feasible amplification is visible during optimization, but the checkpoint grid cannot establish which of the first three changes happened first.

## Optimizer and basin evidence

Nelder-Mead reproduced the positive direction in 5/5 pairs but had a median ratio of 1.768, below the frozen 2× gate; matched COBYLA at B=5,500 had 5/5 wins and ratio 3.219. The result is `OBJECTIVE_OPTIMIZER_INTERACTION` in effect magnitude, not optimizer-independent robustness or an optimizer leaderboard.

The restart result is `OBJECTIVE_PRESSURE_REQUIRED`. `C_TO_E / E_TO_E` had 4/5 wins but median ratio 1.578, missing the 2× transfer gate. In contrast, `E_TO_C / E_TO_E` and `C_TO_C / C_TO_E` both had 5/5 wins and median ratios 3.395 and 2.420. CVaR-discovered parameters already had higher `p_opt`, but were worse under the expectation objective in all five seeds. Continued CVaR pressure is therefore better supported than a favorable-basin-discovery-only account.

## Penalty-X nuance

The five-seed Penalty-X control also showed 5/5 wins and a median ratio of 4.822 at B=11,000. This descriptive new-seed result does not overwrite the historical 10-seed `MIXED_CVAR_RECOVERY` classification. It shows that Penalty-X is not a simple no-effect control: its feasible/low-energy transport was weaker, but its median `p_opt_given_feas` improvement (+0.04087) was larger than Global-Grover's.

## Defensible answer

CVaR changes this finite-budget search primarily through rapid objective-pressure-driven migration into low-energy and feasible regions. That transport is substantially stronger for Global-Grover and persists with twice the COBYLA budget. The direction survives a Nelder-Mead sensitivity check, but its magnitude interacts with the classical optimizer. Matched restarts do not meet the frozen basin-transfer criterion; continued CVaR pressure is required. The data do not resolve causal temporal order within the first 1,000 evaluations or prove that mixer dynamics alone cause the difference.

## Claim boundaries

Supported: persistent Global-Grover recovery under the doubled finite budget; an optimizer interaction in magnitude; objective-pressure evidence; stronger Global feasible/low-energy transport; and an optimization-process form of feasible-region amplification.

Unsupported: converged-optimum superiority, universal optimizer or alpha generality, causal landscape or mixer claims, depth scaling, instance or hardware generality, high absolute success probability, and quantum advantage.

All 60/60 cells completed, all 60 exhausted their budgets, final verification passed 836/836 checks, the full repository suite passed 176 tests, and the immutable historical result root remained byte-for-byte unchanged.
