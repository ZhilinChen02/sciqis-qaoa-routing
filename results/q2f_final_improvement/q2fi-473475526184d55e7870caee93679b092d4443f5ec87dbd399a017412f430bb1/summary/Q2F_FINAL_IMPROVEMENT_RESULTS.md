# Q2-F final bounded improvement results

Run identity: `q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1`

This is a descriptive, teaching-scale mechanism study. Full feasible-route
enumeration is classical preprocessing; structural `p_feas=1` is not quantum
advantage; and three seeds do not support statistical-significance claims.

The incumbent route is `0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6` with raw
cost `11`. The strict incumbent-derived set
`B={P_i:C(P_i)<C(P_incumbent)}` contains `1` route.
Only after constructing B was it compared with the evaluation-only optimum:
`BSP == p_opt` on this instance is `true`.

## Median p_opt

| method | p=1 | p=2 | p=3 | p=4 |
|---|---|---|---|---|
| Old Q2-F expectation | 0.0711933070863 | 0.0445594158188 | 0.264900939435 | N/A |
| BSP path-exchange | 0.152230638106 | 0.136081542503 | 0.444655601258 | 0.374338668208 |
| GM-QAOA expectation | 0.05 | 0.17092725087 | 0.182410370885 | 0.175615223588 |
| GM-Th-QAOA | 0.392 | 0.816079949418 | 0.998712003756 | 0.998428837277 |

## Seed min–max p_opt

| method | p=1 | p=2 | p=3 | p=4 |
|---|---|---|---|---|
| BSP path-exchange | 0.064487–0.153462 | 0.0940256–0.154849 | 0.255823–0.462094 | 0.233242–0.449236 |
| GM-QAOA expectation | 0.05–0.18209 | 0.0518303–0.187649 | 0.1218–0.291155 | 0.171962–0.189656 |
| GM-Th-QAOA | 0.392–0.392 | 0.81608–0.81608 | 0.816034–0.999804 | 0.963891–0.999706 |

The strongest new median was `0.99871200375606495` for
`GM-Th-QAOA` at `p=3`, a relative
change of `277.013%` from `0.2649009394`. The predefined 10% rule
passed: `TRUE`. Selected final course variant:
`GM-Th-QAOA at p=3`. Final decision: `ADOPT_IMPROVED_VARIANT`.

All 36 required rows were retained. The observed p_feas range was
`[0.99999999999999978, 1.0000000000000002]`; all values satisfy the 1e-12 invariant.
`23` optimizer runs reached the 150-request cap and remain retained.

## Interpretation boundary

BSP and the GM-Th phase use only the incumbent threshold, never an optimum
label. The Grover variants use the exact rank-one mixer
`exp(-i beta |F><F|)` with uniform feasible initialization. GM-Th-QAOA here is
an incumbent-threshold demonstration in a 20-state logical route space, not a
scalable routing claim or quantum-advantage claim.

## Figures

- `results/q2f_final_improvement/q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1/figures/01_median_popt_versus_depth.png`
- `results/q2f_final_improvement/q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1/figures/02_strongest_distribution_vs_old_q2f.png`
- `results/q2f_final_improvement/q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1/figures/03_popt_pfeas_summary.png`
