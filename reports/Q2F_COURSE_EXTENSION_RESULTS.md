# Q2-F course-extension results

Run identity: `q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7`

## Scope

These are descriptive results from one 20-state logical feasible-route teaching
experiment. Full route enumeration is classical preprocessing. Q2-F is not
claimed scalable and does not demonstrate quantum advantage. The three-seed
matrix does not support statistical-significance claims. Q2-R and Q2-F use
different representations, so old/new numerical context is not a fair runtime
or algorithmic comparison.

## Initialization controls

- F0 uniform: `p_feas=1.0000000000000002`, `p_opt=0.050000000000000003`, expected route cost `12.200000000000001`.
- F1 incumbent-biased, lambda=1: `p_feas=1.0000000000000002`, `p_opt=0.037645444636975889`, expected route cost `11.16276967033334`.

| route ID | route | cost | distance to incumbent | F0 probability | F1 probability | optimal | incumbent |
|---|---|---|---|---|---|---|---|
| 0 | 0->1->2->4->5->6 | 10 | 3 | 0.05 | 0.037645444637 | yes |  |
| 1 | 0->1->2->3->4->5->6 | 11 | 0 | 0.05 | 0.756128968246 |  | yes |
| 2 | 0->1->2->3->5->6 | 11 | 3 | 0.05 | 0.037645444637 |  |  |
| 3 | 0->1->2->4->6 | 11 | 6 | 0.05 | 0.00187425632588 |  |  |
| 4 | 0->2->4->5->6 | 11 | 6 | 0.05 | 0.00187425632588 |  |  |
| 5 | 0->1->2->3->4->6 | 12 | 3 | 0.05 | 0.037645444637 |  |  |
| 6 | 0->1->2->5->6 | 12 | 4 | 0.05 | 0.0138489851357 |  |  |
| 7 | 0->1->3->4->5->6 | 12 | 3 | 0.05 | 0.037645444637 |  |  |
| 8 | 0->1->3->5->6 | 12 | 6 | 0.05 | 0.00187425632588 |  |  |
| 9 | 0->2->3->4->5->6 | 12 | 3 | 0.05 | 0.037645444637 |  |  |
| 10 | 0->2->3->5->6 | 12 | 6 | 0.05 | 0.00187425632588 |  |  |
| 11 | 0->2->4->6 | 12 | 9 | 0.05 | 9.33137278355e-05 |  |  |
| 12 | 0->1->3->4->6 | 13 | 6 | 0.05 | 0.00187425632588 |  |  |
| 13 | 0->1->4->5->6 | 13 | 4 | 0.05 | 0.0138489851357 |  |  |
| 14 | 0->2->3->4->6 | 13 | 6 | 0.05 | 0.00187425632588 |  |  |
| 15 | 0->2->5->6 | 13 | 7 | 0.05 | 0.000689500369777 |  |  |
| 16 | 0->3->4->5->6 | 13 | 4 | 0.05 | 0.0138489851357 |  |  |
| 17 | 0->3->5->6 | 13 | 7 | 0.05 | 0.000689500369777 |  |  |
| 18 | 0->1->4->6 | 14 | 7 | 0.05 | 0.000689500369777 |  |  |
| 19 | 0->3->4->6 | 14 | 7 | 0.05 | 0.000689500369777 |  |  |

## Median results across the three mandatory seeds

### p_opt

| objective | p=1 | p=2 | p=3 |
|---|---|---|---|
| Expectation | 0.0711933070863 | 0.0445594158188 | 0.264900939435 |
| CVaR(0.25) | 0.144809279491 | 0.150787885158 | 0.258176188036 |
| Ascending-CVaR | 0.139572723793 | 0.0866439609652 | 0.263190459077 |

### Expected raw route cost

| objective | p=1 | p=2 | p=3 |
|---|---|---|---|
| Expectation | 11.8677995049 | 11.6902134328 | 11.3859055151 |
| CVaR(0.25) | 11.6757940714 | 11.902816026 | 11.6468440273 |
| Ascending-CVaR | 11.5980874638 | 12.0615152743 | 11.6318752408 |

### p_opt amplification over F1

| objective | p=1 | p=2 | p=3 |
|---|---|---|---|
| Expectation | 1.89115330614 | 1.18366023429 | 7.03673291655 |
| CVaR(0.25) | 3.84666141912 | 4.00547494158 | 6.85809904825 |
| Ascending-CVaR | 3.70755944415 | 2.30157889755 | 6.99129633385 |

## Requested descriptive questions

1. **Did p_feas remain 1?** Yes. Every one of the 27 retained distributions passed `|p_feas-1| <= 1e-12`.
2. **What was initial warm-start p_opt?** F1 began at `0.037645444636975889`. The F0 uniform control was `0.050000000000000003`.
3. **Highest median p_opt by depth:** p=1: CVaR(0.25); p=2: CVaR(0.25); p=3: Expectation.
4. **Did p_opt improve with depth?** Median trajectories were: Expectation=[0.07119330708632546, 0.04455941581880775, 0.2649009394352171]; CVaR(0.25)=[0.1448092794905909, 0.15078788515810426, 0.2581761880356989]; Ascending-CVaR=[0.1395727237932293, 0.08664396096516934, 0.26319045907666766]. These are descriptive trajectories, not monotonicity or significance claims.
5. **Did fixed CVaR improve concentration relative to expectation?** Median differences `(CVaR - expectation)` were p=1: 0.0736159724043, p=2: 0.106228469339, p=3: -0.00672475139952.
6. **Did Ascending-CVaR improve over fixed CVaR?** Median differences `(ascending - fixed)` were p=1: -0.00523655569736, p=2: -0.0641439241929, p=3: 0.00501427104097.
7. **How much existed before QAOA?** Feasible restriction plus F1 already placed `0.037645444637` on the optimum, compared with sealed Q2-R A0/A1 values `0.00228945652616` and `0.00308457540338`. This is representation context, not a fair algorithm comparison.
8. **Expected cost versus p_opt:** Across the 27 rows, the descriptive Pearson correlation was `-0.438391592534`. A negative value means lower expected route cost co-occurred with higher p_opt in this matrix.
9. **Seed sensitivity:** The largest within-cell p_opt range was `0.20764941752` for Expectation at p=3 (min `0.25022247917`, max `0.45787189669`). All seeds remain reported.
10. **Budget hits:** `15` runs reached the 100-request cap: expectation_p1_seed2601, expectation_p1_seed2602, expectation_p1_seed2603, expectation_p2_seed2601, expectation_p2_seed2602, expectation_p2_seed2603, expectation_p3_seed2601, expectation_p3_seed2602, expectation_p3_seed2603, cvar_p1_seed2601, cvar_p1_seed2602, cvar_p1_seed2603, cvar_p2_seed2601, cvar_p2_seed2603, cvar_p3_seed2603.
11. **Strongest safe course-level takeaway:** Restricting evolution to the explicitly enumerated feasible-route basis makes feasibility structural and permits a clean study of how shallow logical path-exchange evolution redistributes probability among valid routes. The observed objective/depth/seed differences are instance-specific mechanism evidence only.

## Best descriptive observation

The largest retained final p_opt was `0.45787189669012895` in `expectation_p3_seed2602`. It is a secondary descriptive diagnostic, not a selected headline seed or significance claim.

## Read-only Q2-R context

- A0 sealed p_opt: `0.002289456526158069`, p_feas: `0.76793913435815275`.
- A1 sealed p_opt: `0.0030845754033760578`, p_feas: `0.73541689726594339`.
- Q2-R computational state count: 16384; Q2-F logical route count: 20.

## Figures

- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/01_feasible_route_cost_spectrum.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/01_feasible_route_cost_spectrum.svg`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/02_initial_probability_distribution.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/02_initial_probability_distribution.svg`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/03_popt_versus_depth.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/03_popt_versus_depth.svg`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/04_expected_cost_versus_depth.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/04_expected_cost_versus_depth.svg`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/05_p3_median_seed_distributions.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/05_p3_median_seed_distributions.svg`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/06_popt_amplification.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/06_popt_amplification.svg`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/07_q2r_q2f_conceptual_comparison.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/07_q2r_q2f_conceptual_comparison.svg`
