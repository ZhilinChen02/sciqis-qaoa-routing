# CVaR robustness and probability-concentration mechanism study

## 1. Executive conclusion

**MIXED_CVAR_RECOVERY**. In the prospective Penalty-X p=110 fresh-seed contrast, CVaR 0.10 beat expectation in 7/10 pairs and the median paired ratio was 2.970995322. The frozen rule required at least 8 wins and a median ratio of at least 2.0.

This experiment measures optimizer-initialization robustness on one fixed routing instance. It does not establish task-level generalization or algorithm superiority.

## 2. Prior evidence

The motivating `PRIOR_OBSERVED_CELL` is kept separate from validation: Penalty-X p=110 direct expectation p_opt=0.0003727607654 and direct CVaR 0.10 p_opt=0.00774114095. Those historical rows are not among the 10 fresh seeds. The analogous Global-Grover pilot values were expectation=0.001115215212 and CVaR 0.10=0.002185719509.

## 3. Penalty-X seed robustness

Median expectation p_opt=5.508996535e-05; median CVaR 0.10 p_opt=0.0002726654993. The geometric-mean ratio is 2.766293235, range 0.2603790218–43.6006589, with paired-seed bootstrap interval [0.6127836987, 9.097101096] for the median ratio.

| Seed | Expectation p_opt | CVaR 0.10 p_opt | Ratio | Difference |
|---:|---:|---:|---:|---:|
| 8601 | 5.440879598e-05 | 0.0005614108535 | 10.31838407 | 0.0005070020575 |
| 8602 | 0.0002592422151 | 6.750123436e-05 | 0.2603790218 | -0.0001917409807 |
| 8603 | 0.0004335385675 | 0.000553788694 | 1.277368925 | 0.0001202501265 |
| 8604 | 8.503431079e-05 | 3.896927055e-05 | 0.4582770201 | -4.606504024e-05 |
| 8605 | 0.0001776502487 | 0.0001088611765 | 0.6127836987 | -6.878907223e-05 |
| 8606 | 4.435980077e-05 | 0.0004028594895 | 9.08163433 | 0.0003584996888 |
| 8607 | 6.638303385e-06 | 1.766672779e-05 | 2.661331784 | 1.102842441e-05 |
| 8608 | 4.342771227e-05 | 0.000142471509 | 3.28065886 | 9.904379677e-05 |
| 8609 | 5.577113472e-05 | 0.0004392433137 | 7.875818125 | 0.000383472179 |
| 8610 | 1.1213519e-05 | 0.000488916817 | 43.6006589 | 0.000477703298 |

## 4. Global-Grover seed robustness

`SECONDARY_ROBUSTNESS_ANALYSIS`: CVaR 0.10 won 10/10 pairs. Median expectation p_opt=0.0005464230655; median CVaR p_opt=0.00226372078; median ratio=3.929006272; geometric mean=4.005428305; range=2.060930109–10.64456643. Its descriptive frozen-rule label is ROBUST_CVAR_RECOVERY.

| Seed | Expectation p_opt | CVaR 0.10 p_opt | Ratio | Difference |
|---:|---:|---:|---:|---:|
| 8601 | 0.0005398857217 | 0.00190667578 | 3.53162846 | 0.001366790058 |
| 8602 | 0.0006014280696 | 0.002374729613 | 3.948484836 | 0.001773301543 |
| 8603 | 0.0004784259569 | 0.001877187762 | 3.923674572 | 0.001398761805 |
| 8604 | 0.0005529604093 | 0.002715292242 | 4.910464107 | 0.002162331833 |
| 8605 | 0.0006420433158 | 0.002526015397 | 3.934337971 | 0.001883972081 |
| 8606 | 0.0008478606682 | 0.00174738158 | 2.060930109 | 0.0008995209114 |
| 8607 | 0.000797782617 | 0.002152711948 | 2.698369083 | 0.001354929331 |
| 8608 | 0.0003436169976 | 0.001105338056 | 3.21677351 | 0.000761721058 |
| 8609 | 0.0003324331205 | 0.003538606433 | 10.64456643 | 0.003206173313 |
| 8610 | 0.0005259118685 | 0.002777134113 | 5.280607416 | 0.002251222244 |

The cross-mixer pattern is classified as **STRONGER_IN_GLOBAL_GROVER**. This compares response shape without pooling the two mixers.

## 5. Alpha response

Penalty-X: **NONMONOTONIC_RESPONSE**; `EXPLORATORY_BEST_ALPHA`=0.05. Global-Grover: **UNRESOLVED**; `EXPLORATORY_BEST_ALPHA`=0.1. These are observed seed-aggregated response descriptions, not claims of a universally optimal alpha.

## 6. Probability-concentration mechanism

For Penalty-X, the median paired CVaR-minus-expectation changes were: p_feas=0.003806433808, p_opt|feas=-0.004951097373, feasible top-3 mass=0.0006028825173, feasible top-5 mass=0.001045725983, lowest-100-energy mass=0.009840930317, full entropy=-0.002681060076.

For Global-Grover the corresponding changes were: p_feas=0.03145507496, p_opt|feas=-0.002310805126, feasible top-3 mass=0.004624697826, feasible top-5 mass=0.007679349852, lowest-100-energy mass=0.09214828575, full entropy=0.1386802853.

Mechanism interpretation: **FEASIBLE_REGION_AMPLIFICATION**. Entropy is used only as concentration evidence; lower entropy is not automatically equated with better optimization.

## 7. Depth replication

Penalty-X increases across the three observed depths; Global-Grover is irregular across depth. The exact p=50/100/110 paired medians and ratios are in `depth_replication_summary.csv`. With only three depths and different fresh seed blocks, no scaling law is fitted.

## 8. Optimizer status

Across 140 frozen cells, 0 converged early and 140 exhausted the declared budget; 0 scientific rows failed. Budget exhaustion is retained as an observed termination state, not used to invalidate quality metrics. Objective-specific counts are in `optimizer_status_summary.csv`.

## 9. Scientific interpretation

**MIXER_SPECIFIC_OBJECTIVE_EFFECT**. This label follows the prospective paired result and the separate mixer response. The alpha sweep and secondary depths are descriptive mechanism/replication analyses and do not redefine the primary criterion.

## 10. Limitations

- Single fixed routing problem.
- Simulator-only ideal statevectors.
- Optimizer-seed robustness is not task-level generalization.
- Alpha choices, especially 0.10, were motivated by prior observed results.
- Ten seeds quantify initialization sensitivity but do not establish broad algorithm superiority.
- No quantum-advantage claim is made.
