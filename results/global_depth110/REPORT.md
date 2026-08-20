# Full-Space Global QAOA Depth-110 Study

## Research question

We compare Penalty-X QAOA and Full-Space Global Grover-Mixer QAOA on exactly the same \(2^{14}=16384\)-state domain and the same penalized routing cost operator for every QAOA depth \(p=1,\ldots,110\).

The old 20-state feasible-subspace Grover study is a separate experiment. None of its trajectories or distributions is included here.

## Algorithm definitions

For edge bits \(x\), both algorithms use

\[
H_C(x)=C_{route}(x)+6\sum_v\left(\operatorname{out}_v(x)-\operatorname{in}_v(x)-b_v\right)^2.
\]

The shared initial state is

\[
|s\rangle=|+\rangle^{\otimes 14}=\frac{1}{\sqrt{16384}}\sum_x|x\rangle.
\]

Penalty-X uses \(H_M^X=\sum_{j=1}^{14}X_j\). Full-Space Global-Grover uses

\[
H_M^G=|s\rangle\langle s|,\qquad
U_M^G(\beta)=I+(e^{-i\beta}-1)|s\rangle\langle s|.
\]

Every layer is cost then mixer. Both algorithms optimize the same raw expectation \(\langle H_C\rangle\). The Global-Grover mixer mixes amplitudes globally over the full 16,384-state computational domain. Feasibility is not enforced structurally; it must emerge through the shared penalized cost landscape and variational optimization.

## Frozen protocol

- Fixed graph: 7 nodes, 14 directed weighted edges, source 0, destination 6.
- State index: q0 is the least-significant index bit; displayed bitstrings are q0 through q13.
- Penalty: coefficient 6 multiplying the repository's squared flow residuals.
- Feasibility: the repository's exact single-source-to-target route decoder; independently enumerated count is 20.
- Exact reference: unique feasible cost-10 optimum.
- Parameters: \((\gamma_1,\ldots,\gamma_p,\beta_1,\ldots,\beta_p)\), exactly \(2p\) values.
- Bounds: \(\gamma\in[0,2\pi]\), \(\beta\in[0,\pi]\).
- Seeds: \(2601+7919p\). Layerwise continuation appends a seeded new layer to the retained depth-\(p-1\) parameters.
- Optimizer: COBYLA, rhobeg 0.5, tolerance \(10^{-8}\).
- Evaluation budget: \(\max(120,2(2p)+64)\), identical for both algorithms; p=110 receives 504 evaluations for 220 parameters.
- Normalized regret: \((E[C\mid F]-C_*)/(C_{max,F}-C_*)\).
- Simulation: ideal NumPy complex128 statevectors; all saved distributions have 16,384 entries.

## Main results

| Result | Penalty-X | Full-Space Global-Grover |
|---|---:|---:|
| best observed depth | 21 | 110 |
| best observed p_opt | 0.002833153994 | 0.001029783423 |
| p=110 p_opt | 0.0003578671392 | 0.001029783423 |
| p=110 p_feas | 0.1469702273 | 0.02194609806 |
| p=110 p_opt\|feas | 0.002434963502 | 0.04692330363 |
| p=110 expected penalized cost | 18.36925468 | 33.27459465 |
| budget-limited rows | 110 | 110 |
| total wall time (s) | 2284.649 | 539.004 |

Total recorded cell wall time was 2823.653 seconds.

### Trajectory landmarks

| Mixer | p | p_opt | p_feas | p_opt\|feas | expected penalized cost |
|---|---:|---:|---:|---:|---:|
| Penalty-X | 1 | 0.0003267283648 | 0.005251685228 | 0.06221400381 | 69.04069492 |
| Penalty-X | 10 | 0.001633890668 | 0.01554630606 | 0.1050983212 | 59.18394627 |
| Penalty-X | 50 | 4.566694964e-05 | 0.07444995443 | 0.000613391237 | 25.77495498 |
| Penalty-X | 90 | 0.0001592635458 | 0.1356734324 | 0.001173874229 | 19.51958315 |
| Penalty-X | 110 | 0.0003578671392 | 0.1469702273 | 0.002434963502 | 18.36925468 |
| Global-Grover | 1 | 6.480843533e-05 | 0.001223207673 | 0.05298236492 | 85.97100418 |
| Global-Grover | 10 | 6.488244153e-05 | 0.00120511378 | 0.05383926613 | 85.94168337 |
| Global-Grover | 50 | 0.0005116045803 | 0.01006551572 | 0.05082745828 | 44.88302208 |
| Global-Grover | 90 | 0.0007387539558 | 0.01523359574 | 0.04849504796 | 37.75421583 |
| Global-Grover | 110 | 0.001029783423 | 0.02194609806 | 0.04692330363 | 33.27459465 |

### Threshold depths

| p_opt threshold | Penalty-X first p | Global-Grover first p |
|---:|---:|---:|
| 0.10 | NOT_REACHED | NOT_REACHED |
| 0.25 | NOT_REACHED | NOT_REACHED |
| 0.50 | NOT_REACHED | NOT_REACHED |
| 0.75 | NOT_REACHED | NOT_REACHED |
| 0.90 | NOT_REACHED | NOT_REACHED |
| 0.95 | NOT_REACHED | NOT_REACHED |
| 0.99 | NOT_REACHED | NOT_REACHED |

### Saturation, crossover, and oscillation

- Penalty-X saturation criterion: p=12 (p_opt=0.00194102, p_feas=0.0147086; successful-convergence flag=False; later ≥10^-3 transition violations=2).
- Global-Grover saturation criterion: p=11 (p_opt=6.18174e-05, p_feas=0.0012691; successful-convergence flag=False; later ≥10^-3 transition violations=0).
- First p with Global-Grover p_opt > Penalty-X p_opt: 22.
- First later Penalty-X overtake: NOT_REACHED.
- p_opt direction changes across all depths: Penalty-X 52, Global-Grover 50.

A flat trajectory at small p_opt is not called successful convergence. The absolute tolerance criterion is especially permissive when all probabilities are tiny. The saturation record therefore includes optimizer statuses and later violations; Penalty-X's early criterion hit is transient if later violations are nonzero.

## Every-depth results

The complete 110-row paired table is in [depth_by_depth.md](depth_by_depth.md) and [depth_by_depth.csv](depth_by_depth.csv). The canonical 220-row data are in [canonical_rows.csv](canonical_rows.csv) and [canonical_rows.xlsx](canonical_rows.xlsx).

## Behavior near p≈100

The complete p=90..110 metrics are in [near_p100.md](near_p100.md) and [near_p100.csv](near_p100.csv), with p=95..105 explicitly flagged. Penalty-X has 6 p_opt direction changes and Global-Grover has 9 over p=90..110. Their respective p_opt ranges over that interval are 0.000267767 and 0.000322769. The best p=90..110 points occur at p=105 for Penalty-X (p_opt=0.000402608) and p=110 for Global-Grover (p_opt=0.00102978). From p=90 to p=110, Global-Grover rises from 0.000738754 to 0.00102978, while Penalty-X moves from 0.000159264 to 0.000357867 non-monotonically.

The standard one-marked-state binary-oracle reference \(\frac{\pi}{4}\sqrt{16384}\approx100.53\) is only exploratory context. This experiment uses a continuous penalized cost phase and variational angles, not a binary marked-state oracle, so no peak at p≈100 is implied.

## Interpretation and answers to the scientific questions

1. **Optimal probability:** Penalty-X grows from 0.000326728 at p=1 to its global observed maximum 0.00283315 at p=21, then falls and oscillates, ending at 0.000357867. Global-Grover stays near the uniform baseline at shallow depth, improves in several step-like episodes, and reaches its observed maximum 0.00102978 at p=110.
2. **Feasible probability:** Penalty-X moves from 0.00525169 to 0.14697; Global-Grover moves from 0.00122321 to 0.0219461. Feasibility remains measured, never structural. Global-Grover has the larger measured feasible mass at 0 depths; Penalty-X does so at 110 depths.
3. **Feasibility discovery:** under this frozen run, Penalty-X puts more mass in feasible routes at every one of the 110 depths. This is an instance-and-optimizer result, not a general mixer ranking.
4. **Concentration after feasibility:** Conditioned on feasibility, Global-Grover has the larger p_opt|feas at 89 depths, while Penalty-X does so at 21 depths. At p=110 specifically, the values are 0.00243496 for Penalty-X and 0.0469233 for Global-Grover.
5. **Best depth:** Penalty-X p=21; Global-Grover p=110.
6. **Saturation or oscillation:** the explicit 10-transition saturation test and direction-change counts are reported above. A budget-limited flat curve cannot isolate ansatz saturation.
7. **Near p≈100:** the dedicated saved table shows the measured peaks, ranges, and direction changes; any resemblance to \(\sqrt N\) Grover scaling is exploratory only.
8. **Ansatz versus optimizer:** 220 of 220 rows are budget-limited. Consequently, depth trends may reflect both ansatz geometry and finite-budget COBYLA behavior; this single protocol cannot cleanly separate them.

The distribution metrics distinguish global concentration (entropy/effective support), feasible concentration (p_feas), and optimal concentration (p_opt and p_opt|feas). Figures are generated only from the saved canonical rows.

## Limitations

- One fixed small routing instance.
- Ideal statevector simulation and no hardware noise.
- No quantum-advantage claim and no scalability claim.
- Optimization difficulty may confound ansatz expressivity.
- The p≈100 standard Grover scale is only a theoretical reference because the main experiment uses a continuous penalized cost phase rather than a binary marked-state oracle.
- Deterministic single-start layerwise continuation measures one frozen optimizer trajectory per mixer, not global parameter optimality.
