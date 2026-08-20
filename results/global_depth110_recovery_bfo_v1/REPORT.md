# Depth-110 QAOA recovery experiment: budget × Fourier × CVaR

## Executive summary

The preregistered analysis classifies the poor historical **depth-110** result as most consistent with **expectation-objective mismatch**. At p=110, 0 budget comparisons, 0 Fourier-expectation comparisons, and 3 direct-CVaR comparisons met the materiality rule. A material p_opt improvement was defined before execution as both a recovery factor of at least 2.0 and an absolute increase of at least 0.0001.

The preregistered primary Global-Grover p=110 cell reached p_opt=0.000795002259, versus the immutable historical 0.001029783423: R_opt=0.7720091832. Its p_feas changed from 0.02194609806 to 0.01564604966 (factor 0.712930819, absolute change -0.006300048394), and p_opt|feas changed by 0.003888388529.

## Historical reference

No file under `results/global_depth110/` was modified. These values are read directly from its canonical CSV and protected by the recovery root's 468-file SHA-256 manifest.

| Algorithm | p | Historical p_opt | Historical p_feas | Historical p_opt|feas |
|---|---:|---:|---:|---:|
| Global-Grover | 21 | 0.0002146538325 | 0.004247473224 | 0.05053683005 |
| Global-Grover | 22 | 0.0002153104546 | 0.004261535978 | 0.05052414335 |
| Global-Grover | 50 | 0.0005116045803 | 0.01006551572 | 0.05082745828 |
| Global-Grover | 100 | 0.0008439592738 | 0.0183336698 | 0.04603329737 |
| Global-Grover | 110 | 0.001029783423 | 0.02194609806 | 0.04692330363 |
| Penalty-X | 21 | 0.002833153994 | 0.01794636948 | 0.157867807 |
| Penalty-X | 22 | 7.448970898e-05 | 0.02120802542 | 0.003512335897 |
| Penalty-X | 50 | 4.566694964e-05 | 0.07444995443 | 0.000613391237 |
| Penalty-X | 100 | 0.0001348414793 | 0.1775980655 | 0.0007592508337 |
| Penalty-X | 110 | 0.0003578671392 | 0.1469702273 | 0.002434963502 |

## Budget findings (RQ1)

At fixed direct parameterization and expectation objective, the largest 10×→50× p_opt factor occurred for Penalty-X p=100: 0.000192240578→0.0003098777594, factor 1.611926902, with p_feas change 0.00126781797 and p_opt|feas change 0.0006619455359. 0 of 10 matched comparisons met the preregistered materiality rule. Optimizer termination and every exact quality metric are preserved in `budget_sweep.csv`; the diagnosis is based on p_opt, p_feas, and p_opt|feas, not status alone.

## Fourier findings (RQ2)

At p=100/110, direct uses 200/220 active variables, Fourier q=5 uses 10, and Fourier q=10 uses 20. All receive the same absolute 25×2p evaluation budget. The largest expectation-objective Fourier-vs-direct p_opt factor was Penalty-X p=100 Fourier q=10: 0.0002306294106→0.003099986937, factor 13.44142072. 1 of 8 matched comparisons were material. The tables report objective attainment and conditional concentration, exposing the tradeoff: fewer variables receive far more evaluations per variable but restrict schedules to the chosen Fourier basis.

## CVaR findings (RQ3)

The largest matched CVaR-vs-expectation p_opt factor was Penalty-X p=110, Direct, CVaR 0.10: 0.0003727607654→0.00774114095, factor 20.76704865. Its p_feas changed by 0.0551765189 and p_opt|feas by 0.03317394984. 15 matched comparisons were material. CVaR was evaluated on the full penalized-energy distribution with fractional cutoff and no feasible renormalization.

Across the 74 retained cells, 17 showed lower final training objective but lower p_opt than initialization. Figure 09 and `optimizer_trace_summary.csv` therefore directly document objective/p_opt alignment or decoupling rather than assuming it.

## Combined recovery (RQ4)

Preregistered primary first: Global-Grover, p=110, 25×2p evaluations, Fourier q=10, CVaR 0.10. p_opt=0.000795002259, R_opt=0.7720091832; p_feas=0.01564604966; p_opt|feas=0.05081169216. This result is not replaced by a maximum selected after execution.

Separately, `EXPLORATORY_BEST_CELL` is bfo-4df21966fd8ad7d401bad12a3419c58ad68467dfb4c3683a061b6c2f64c341b8: Penalty-X, p=110, 25×, Direct, CVaR 0.10, p_opt=0.00774114095, historical factor=21.63132655.

The matched budget, Fourier, and objective sections above are the required ablations; no combined gain is assigned to a single component without those controls.

## Penalty-X p=21/p=22 diagnosis (RQ5)

Under the preregistered ratio rule, the collapse **disappears**. This needs a crucial qualification: the larger-budget expectation rows still show a severe p=21→22 drop, while CVaR 0.10 removes the ratio collapse by lowering p=21 to roughly the already-poor p=22 level, not by recovering p=22 to the historical p=21 peak.

- expectation, 10×: p21=0.001555312524, p22=5.871301497e-05, ratio=0.03774997891
- expectation, 25×: p21=0.001414981234, p22=1.92695333e-05, ratio=0.01361822534
- expectation, 50×: p21=0.001414974692, p22=1.949576194e-05, ratio=0.01377817006
- cvar_0.10, 25×: p21=2.2698165e-05, p22=1.850551113e-05, ratio=0.8152866601
- cvar_0.25, 25×: p21=3.64275287e-05, p22=5.478261008e-06, ratio=0.1503879402

## Negative-result branch

At least one controlled intervention met the preregistered materiality rule. The attribution remains limited to the matched ablations reported above.

## Reproducibility and scope

The experiment contains exactly 74 unique canonical cells. It uses the historical direct schedules as immutable starts, deterministic least-squares Fourier fits, COBYLA with unchanged tolerance/rhobeg semantics, and the unchanged full 14-qubit simulator, penalty Hamiltonian, Global-Grover mixer, Penalty-X mixer, decoder, and probability definitions. Cell checkpoints, distributions, and evaluation traces are atomic and independently hash-verified.
