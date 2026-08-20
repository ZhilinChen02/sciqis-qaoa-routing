# CVaR Search-Control Mechanism Study v1

## Status and preservation

All 60/60 frozen scientific cells completed: A=20, B=10, C=20, D=10. There were 0 failed, 0 censored, and 0 duplicate cells. All 60 optimizations exhausted their declared evaluation budgets; none converged early.

Independent final verification passed 836/836 checks. The full repository suite passed 176 tests before execution and again after analysis.

The 312-file `results/cvar_robustness_v1` inventory remained byte-for-byte unchanged. Its canonical inventory digest is `c0bc2821ee625c53da11308a630a75d56e78bcdf1e5ad778f1c5233485ec6dee`; its frozen manifest remains `f468f300e51a82e42e88e9821ed6eb672ce97983b495c5efc1bc4949e0416a11`, and its canonical CSV remains `1472b97860f8b417fcb09ebe18c65b6d6fb2fcbd481eb507119938d02b3c9710`.

## RQ1 — budget persistence

Global-Grover CVaR 0.10 satisfied the historical robustness rule at every observed budget on the 10 new seed pairs.

| Budget | Expectation median `p_opt` | CVaR median `p_opt` | Wins | Median paired ratio | Bootstrap 95% interval | Classification |
|---:|---:|---:|---:|---:|---:|---|
| 1,000 | 4.928e-04 | 1.159e-03 | 10/10 | 2.447 | [2.050, 4.240] | `ROBUST_CVAR_RECOVERY` |
| 2,500 | 5.458e-04 | 1.928e-03 | 10/10 | 3.285 | [2.299, 6.163] | `ROBUST_CVAR_RECOVERY` |
| 5,500 | 5.223e-04 | 2.059e-03 | 10/10 | 3.145 | [2.551, 5.934] | `ROBUST_CVAR_RECOVERY` |
| 8,000 | 5.440e-04 | 2.111e-03 | 10/10 | 2.912 | [2.582, 5.853] | `ROBUST_CVAR_RECOVERY` |
| 11,000 | 5.422e-04 | 2.165e-03 | 10/10 | 2.889 | [2.618, 5.766] | `ROBUST_CVAR_RECOVERY` |

At the confirmatory B=11,000 endpoint, the geometric-mean paired ratio was 3.744. The effect therefore persisted when the evaluation cap doubled; it did not disappear as the budget rose from 5,500 to 11,000. The ratio peaked earlier and then moderated, so the trajectory does not imply monotonic growth with optimizer budget.

Absolute performance remained small: median CVaR `p_opt=0.002165`, about 0.216%, versus 0.054% for expectation.

## RQ2 — optimizer sensitivity

The pre-registered result is `OBJECTIVE_OPTIMIZER_INTERACTION`.

- COBYLA at B=5,500: 5/5 wins, median ratio 3.219, geometric mean 4.106; the five-seed directional gate passed.
- Nelder-Mead at B=5,500: 5/5 wins, median ratio 1.768, geometric mean 1.932; the 2× magnitude gate failed.
- The paired COBYLA-minus-Nelder-Mead median log-ratio difference was 0.759, exceeded log(2), and had the same direction in 5/5 seeds.

Thus CVaR's qualitative direction reproduced under Nelder-Mead, but the magnitude depended materially on the classical optimizer under the frozen rule. This is not an optimizer leaderboard and does not establish generality beyond these two methods.

## RQ3 — basin transfer

The pre-registered classification is `OBJECTIVE_PRESSURE_REQUIRED`.

| Contrast | Wins | Median `p_opt` ratio | Frozen interpretation |
|---|---:|---:|---|
| `C_TO_E / E_TO_E` | 4/5 | 1.578 | Missed the 2× basin-transfer threshold |
| `E_TO_C / C_TO_C` | 1/5 | 0.673 | Missed rescue-noninferiority consistency |
| `E_TO_C / E_TO_E` | 5/5 | 3.395 | CVaR pressure improved the expectation source basin |
| `C_TO_C / C_TO_E` | 5/5 | 2.420 | Continued CVaR pressure improved the CVaR source basin |

Before continuation, `theta_C` had higher `p_opt` than `theta_E` in 5/5 seeds, with median ratio 3.219, but it had a lower (better) expectation objective in 0/5 seeds. A CVaR-discovered point was therefore not already a better expectation-objective solution. Restarted expectation optimization retained some `p_opt` advantage from the CVaR source, but not enough for the frozen 2× transfer gate. The stronger evidence is that continued CVaR pressure improved `p_opt` from either source basin.

This five-seed restart arm supports an objective-pressure interpretation, not a claim about global landscape topology or an absence of basin effects.

## RQ4 — mixer-associated transport

The pre-registered classification is `MIXER_SPECIFIC_TRANSPORT_SUPPORTED`.

At B=11,000, using the same five seeds for both mixers:

| Paired CVaR-minus-expectation median change | Global-Grover | Penalty-X | Global-minus-Penalty |
|---|---:|---:|---:|
| `p_feas` | +0.02827 | +0.00472 | +0.02476 |
| Low-energy top-100 mass | +0.07944 | +0.01169 | +0.06558 |
| `p_opt_given_feas` | +0.00092 | +0.04087 | — |
| Top-3 absolute feasible mass | +0.00435 | +0.00083 | — |
| Top-5 absolute feasible mass | +0.00724 | +0.00188 | — |

All frozen Global transport consistency and magnitude gates passed. Global-Grover's recovery was associated with substantially stronger movement into low-energy and feasible regions, while its within-feasible optimal fraction changed little. Penalty-X showed smaller feasible/low-energy transport but stronger within-feasible sharpening in this descriptive subset.

The Penalty-X control itself had 5/5 `p_opt` wins and a median ratio of 4.822 at B=11,000. This new five-seed, larger-budget result does not replace the historical 10-seed `MIXED_CVAR_RECOVERY` classification. It shows that the cross-mixer distinction is not simply “CVaR helps one mixer and never the other”; the more defensible distinction is the strength and channel of probability transport, together with seed and budget sensitivity.

## RQ5 — mechanism dynamics

The trajectory classification is `SIMULTANEOUS_WITHIN_CHECKPOINT_RESOLUTION`.

For the matched five-seed Global-Grover mechanism block, low-energy-top-100 mass, feasible mass, and optimal mass all passed their event criteria at the first checkpoint, B=1,000. At that checkpoint their median CVaR effects were +0.0591, +0.0151, and a 2.462× `p_opt` ratio. The `p_opt_given_feas` event never passed at any checkpoint; its median change was only +0.00092 at B=11,000.

The trajectories therefore show that feasible-region amplification is already an optimization-process phenomenon by B=1,000, not merely a terminal artifact. The checkpoint grid cannot resolve whether low-energy migration preceded feasible or optimal mass within the first 1,000 evaluations, so no causal temporal ordering is claimed.

## Integrated scientific conclusion

The robust Global-Grover recovery is not explained by the original 5,500-evaluation cap: it persisted in 10/10 fresh seeds through 11,000 evaluations. It is qualitatively visible under Nelder-Mead but materially stronger under COBYLA, supporting an objective-by-optimizer interaction in magnitude. Matched restarts favor continued CVaR objective pressure over a basin-discovery-only explanation. The most distinctive mixer-associated signature is Global-Grover's much stronger migration into low-energy and feasible probability mass, without a qualifying within-feasible-sharpening event.

The most defensible answer is therefore: CVaR changes the finite-budget search by applying objective pressure that rapidly redistributes Global-Grover probability toward low-energy and feasible regions; the effect persists with more COBYLA evaluations, its magnitude interacts with the classical optimizer, and it is not adequately explained by favorable-basin discovery alone. Penalty-X can also benefit, but in this small control its transport is weaker and its conditional feasible sharpening is stronger.

## Supported claims

- Global-Grover CVaR 0.10 recovery persisted under a doubled COBYLA cap on 10 fresh paired seeds.
- The recovery direction reproduced under Nelder-Mead, but its magnitude showed a pre-registered objective-by-optimizer interaction.
- Continued CVaR pressure, rather than basin transfer alone, was required by the frozen five-seed decision rule.
- CVaR-induced feasible and low-energy probability transport was materially stronger for Global-Grover than Penalty-X in the matched control.
- Feasible-region amplification was present by the first B=1,000 checkpoint, while Global within-feasible sharpening never met its event criterion.
- All conclusions are matched finite-budget effects.

## Unsupported or prohibited claims

- Fully converged variational-optimum superiority; all 60 cells exhausted their budgets.
- Causal attribution to mixer dynamics alone.
- A universal optimizer-independent CVaR magnitude.
- A universal best alpha or any new alpha conclusion.
- A depth scaling law.
- Generality across routing instances, hardware, noise models, or optimizers beyond the two tested.
- Global landscape topology or proof that basin effects are absent.
- A resolved causal temporal order within the first 1,000 evaluations.
- That Penalty-X never benefits from CVaR.
- High absolute optimal-solution probability or quantum advantage.

## Limitations

The study uses one fixed routing instance, ideal statevectors, p=110 only, ten primary seeds, and five seeds in B/C/D. Nelder-Mead is a sensitivity arm, not a broad benchmark. The first trajectory checkpoint occurs at B=1,000, limiting temporal resolution. Absolute `p_opt` remains below one percent. Budget exhaustion prevents convergence claims.

## Artifacts

- [`PROTOCOL.md`](PROTOCOL.md) and [`frozen_manifest.json`](frozen_manifest.json)
- [`canonical_rows.csv`](canonical_rows.csv) and [`canonical_rows.xlsx`](canonical_rows.xlsx)
- [`trajectory_checkpoints.csv`](trajectory_checkpoints.csv)
- [`basin_transfer_rows.csv`](basin_transfer_rows.csv)
- [`budget_response_summary.csv`](budget_response_summary.csv)
- [`analysis_summary.json`](analysis_summary.json)
- [`FINAL_SCIENTIFIC_INTERPRETATION.md`](FINAL_SCIENTIFIC_INTERPRETATION.md)
- [`verification/final_validation.json`](verification/final_validation.json)
- [`figures/`](figures/)

The nine main figures cover the Global budget response in `p_opt`, `p_feas`, low-energy mass, and `p_opt_given_feas`; optimizer sensitivity; basin transfer; cross-mixer feasible and low-energy transport; and entropy trajectories.
