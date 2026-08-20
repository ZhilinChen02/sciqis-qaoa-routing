# Final Scientific Interpretation

## Evidence status

This interpretation is based exclusively on the 140/140 verified canonical cells in `cvar_robustness_v1`. Those cells, their distributions, and the frozen manifest are immutable evidence. No scientific cell was rerun, repaired, overwritten, or otherwise modified while preparing this document.

The frozen experiment supports a mixer-specific, matched finite-budget conclusion: CVaR 0.10 produced reproducible recovery for Global-Grover, but the confirmatory Penalty-X robustness rule was not satisfied. Across both mixers, the observed mechanism is best classified as `FEASIBLE_REGION_AMPLIFICATION`.

## Presentation-ready summary

| Evidence class | Algorithm and depth | Seeds | Expectation median `p_opt` | CVaR 0.10 median `p_opt` | CVaR wins | Median paired ratio | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Confirmatory, frozen | Penalty-X, p=110 | 10 | 5.509e-05 | 2.727e-04 | 7/10 | 2.971 | `MIXED_CVAR_RECOVERY`; frozen robustness rule not satisfied |
| Descriptive secondary | Global-Grover, p=110 | 10 | 5.464e-04 | 2.264e-03 | 10/10 | 3.929 | Reproducible CVaR 0.10 recovery |
| Depth replication | Penalty-X, p=50 | 5 | 5.684e-05 | 4.481e-05 | 1/5 | 0.788 | No recovery at this depth |
| Depth replication | Penalty-X, p=100 | 5 | 2.783e-05 | 1.173e-04 | 3/5 | 1.260 | Mixed descriptive result |
| Depth replication | Global-Grover, p=50 | 5 | 3.911e-04 | 7.706e-04 | 5/5 | 2.283 | Directional replication |
| Depth replication | Global-Grover, p=100 | 5 | 3.208e-04 | 1.422e-03 | 5/5 | 4.104 | Directional replication |

Absolute optimal-solution probabilities remain small. Even the largest central value in the primary comparison—Global-Grover CVaR 0.10 at p=110—is only about 0.00226, or 0.226%. The results therefore show relative recovery, not high absolute solution probability.

## Confirmatory frozen result

For Penalty-X at p=110, CVaR 0.10 beat expectation in 7 of 10 paired fresh seeds. The median paired ratio was 2.971 and the geometric-mean paired ratio was 2.766. The paired bootstrap 95% interval for the median ratio was 0.613 to 9.097.

The frozen confirmatory rule required both at least 8/10 wins and a median paired ratio of at least 2.0. Only the ratio condition passed. The correct classification is therefore `MIXED_CVAR_RECOVERY`, not robust recovery. The earlier single-cell result remains prior/pilot evidence and does not alter this classification.

## Descriptive secondary results

For Global-Grover at p=110, CVaR 0.10 beat expectation in 10/10 paired seeds. The median paired ratio was 3.929, the geometric-mean ratio was 4.005, and the paired bootstrap 95% interval for the median ratio was 3.115 to 4.910. This is reproducible CVaR 0.10 recovery within the frozen problem and finite-budget protocol.

The Global-Grover direction replicated descriptively at p=50 and p=100, with 5/5 wins at each depth and median paired ratios of 2.283 and 4.104, respectively. Penalty-X did not show the same consistency across depths. These three depths are discrete replication checks; no depth scaling law is fitted or implied.

## Exploratory alpha findings

The alpha response is exploratory. Penalty-X had a non-monotonic response, with median `p_opt` values of 5.509e-05, 9.303e-05, 7.939e-05, 2.727e-04, and 3.186e-04 for expectation/alpha=1, 0.50, 0.25, 0.10, and 0.05. Its observed best alpha was 0.05.

Global-Grover had median `p_opt` values of 5.464e-04, 1.186e-03, 1.303e-03, 2.264e-03, and 2.207e-03 over the same sequence. Its observed best alpha was 0.10, narrowly above 0.05.

These are `EXPLORATORY_BEST_ALPHA` findings only. They do not establish alpha=0.10, alpha=0.05, or any other alpha as universally optimal.

## Mechanism evidence

The mechanism comparison uses paired median changes from expectation to CVaR 0.10 at p=110.

| Metric | Penalty-X paired change | Global-Grover paired change | Interpretation |
|---|---:|---:|---|
| `p_feas` | +0.003806 | +0.031455 | Feasible-region mass increased for both mixers |
| `p_opt_given_feas` | -0.004951 | -0.002311 | No consistent improvement conditional on feasibility |
| Feasible top-3 absolute mass | +0.000603 | +0.004625 | More absolute probability reached top feasible routes |
| Feasible top-5 absolute mass | +0.001046 | +0.007679 | More absolute probability reached top feasible routes |
| Feasible-conditional top-3 mass | +0.009802 | -0.007808 | Conditional concentration was mixed across mixers |
| Feasible-conditional top-5 mass | -0.007197 | -0.011561 | Conditional concentration did not consistently improve |
| Low-energy top-10 mass | +0.001680 | +0.015571 | Low-energy probability mass increased |
| Low-energy top-100 mass | +0.009841 | +0.092148 | Broad low-energy probability mass increased |
| Full entropy | -0.002681 | +0.138680 | No common entropy-concentration direction |
| Feasible-conditional entropy | +0.025130 | +0.000952 | No evidence of lower feasible-conditional entropy |

The medians make the distinction quantitative. Penalty-X `p_feas` rose from 0.001448 to 0.005136, while median `p_opt_given_feas` moved from 0.04396 to 0.05438 but had a negative paired median change because the ratio of marginal medians is not the median of paired changes. Global-Grover `p_feas` rose from 0.01014 to 0.04265, while `p_opt_given_feas` fell from 0.05365 to 0.05108.

Thus CVaR increased feasible and low-energy probability mass, including the absolute mass assigned to top routes, without a consistent improvement in the fraction of feasible mass placed on the optimum or in feasible-conditional concentration. The supported mechanism classification is `FEASIBLE_REGION_AMPLIFICATION`, not `WITHIN_FEASIBLE_OPTIMAL_CONCENTRATION`.

## Optimization caveat

Every one of the 140 cells exhausted its declared COBYLA evaluation budget, and none converged early. The depth-specific budgets were 5,500 evaluations for the 100 p=110 cells, 5,000 for the 20 p=100 cells, and 2,500 for the 20 p=50 cells. There were no failed scientific cells.

Accordingly, the supported result is a matched finite-budget optimization effect. It is not proof that CVaR reaches a superior fully converged variational optimum. In particular, “all cells exhausted 5,500 evaluations” would be inaccurate for the secondary depths; all cells exhausted the budget frozen for their depth.

## Limitations and prohibited claims

The evidence does not support any of the following claims:

- robust Penalty-X CVaR recovery under the frozen confirmatory rule;
- a universally optimal CVaR alpha;
- superiority at fully converged variational optima;
- a depth scaling law or asymptotic trend;
- broad superiority of CVaR, Global-Grover, or either mixer across routing instances;
- generalization beyond the single fixed routing problem and frozen initialization blocks;
- robustness to hardware noise, sampling noise, or physical-device execution;
- quantum advantage;
- high absolute optimal-solution probability;
- a causal claim that mixer choice alone explains the observed difference.

## Strongest existing figures

The following six existing figures give the most compact presentation of the result:

1. [`01_penaltyx_p110_seed_pairs_popt.png`](figures/01_penaltyx_p110_seed_pairs_popt.png) — the confirmatory paired result and its 7/10 split.
2. [`02_globalgrover_p110_seed_pairs_popt.png`](figures/02_globalgrover_p110_seed_pairs_popt.png) — the clearest view of the 10/10 Global-Grover recovery.
3. [`05_popt_vs_pfeas_alpha_tradeoff.png`](figures/05_popt_vs_pfeas_alpha_tradeoff.png) — connects optimal mass to feasible-region amplification.
4. [`06_popt_given_feas_alpha_response.png`](figures/06_popt_given_feas_alpha_response.png) — shows why within-feasible optimal concentration is not the primary mechanism.
5. [`08_penalized_energy_cumulative_mass.png`](figures/08_penalized_energy_cumulative_mass.png) — displays the increase in low-energy cumulative mass.
6. [`10_depth_replication_expectation_vs_cvar.png`](figures/10_depth_replication_expectation_vs_cvar.png) — summarizes the p=50, p=100, and p=110 replication pattern without fitting a scaling law.

## Scientific conclusion

CVaR did not meet the frozen robustness rule for Penalty-X: the result is `MIXED_CVAR_RECOVERY` with 7/10 wins. In contrast, Global-Grover showed reproducible CVaR 0.10 recovery with 10/10 wins at p=110 and directional replication at p=50 and p=100. The most consistent explanation is `FEASIBLE_REGION_AMPLIFICATION`: CVaR moved more probability into feasible and low-energy regions, while concentration on the optimum conditional on feasibility did not consistently improve. These conclusions apply to matched, budget-exhausted optimization runs on the frozen instance; absolute `p_opt` remained small, and claims about convergence, universal alpha choice, depth scaling, or broader generality remain unsupported.
