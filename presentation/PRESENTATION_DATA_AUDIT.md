# Presentation Data Audit

**Generated:** 2026-08-20 (Europe/Copenhagen project context)

**Git HEAD:** `cd04027e30020936ff1c8f03d24faf9a0c1f8c7d`

**Working-tree note:** The repository already contained uncommitted scientific/code changes before this presentation was generated. The deck therefore cites the current working tree and frozen result artifacts, not HEAD alone.

## Numerical and protocol claims used

| Slide | Claim | Source of truth |
|---|---|---|
| 1 | Course / title identity: DTU 10387 SCIQIS; author handle ZhilinChen02; repository URL | `README.md; pyproject.toml; git remote; git log` |
| 2 | 7 nodes, 14 directed positive-weight edges, source 0, target 6 | `data/graph.json; src/graph.py` |
| 2 | 2^14 = 16,384 bitstrings | `data/graph.json; src/graph.py; results/global_depth110/reference_solution.json` |
| 2 | 20 valid decoded routes = 0.001220703125 of the full space | `results/global_depth110/reference_solution.json` |
| 2 | Unique optimum 0→1→2→4→5→6, routing cost 10; state 10377; bitstring 10010001000101 | `results/global_depth110/reference_solution.json; src/graph.py` |
| 3 | Penalty coefficient A=6 | `configs/global_depth110.json; results/global_depth110/experiment_config.json` |
| 4/B | Expanded QUBO has 14 linear coefficients and 46 nonzero pair coefficients; QUBO constant 12; Ising energy error 0 | `src/qubo.py; read-only exhaustive calculation used for deck` |
| 5 | Penalty-X is exp(-i beta sum X_j) without beta/n scaling in the depth-110 study | `configs/global_depth110.json; src/global_depth_sweep/simulator.py; src/qaoa.py` |
| 5 | Global-Grover update is I plus (exp(-i beta)-1) times the rank-one projector onto the uniform state over all 16,384 states; no feasible projection or threshold phase | `src/qaoa.py; configs/global_depth110.json; results/global_depth110/validation_summary.json` |
| 6/7 | 2p parameters; p=110 gives 220 parameters | `configs/global_depth110.json; src/global_depth_sweep/experiment.py` |
| 6/10/C | Evaluation budget max(120,4p+64); p=110 gives 504 evaluations | `configs/global_depth110.json; src/global_depth_sweep/experiment.py` |
| 7 | p=1…110; 2 mixers × 110 depths = 220 rows; deterministic layerwise continuation | `results/global_depth110/canonical_rows.csv; configs/global_depth110.json` |
| 8 | Penalty-X best p=21, p_opt=0.00283315399358155 | `results/global_depth110/analysis_summary.json` |
| 8 | Global-Grover best p=110, p_opt=0.00102978342271465 | `results/global_depth110/analysis_summary.json` |
| 8 | First Global-Grover total-p_opt crossover p=22; no later Penalty-X overtake in the frozen trajectory | `results/global_depth110/analysis_summary.json` |
| 8/12 | Penalty-X has higher p_feas at 110/110 depths; Global-Grover higher p_opt/feas at 89/110 depths | `results/global_depth110/analysis_summary.json` |
| 9 | Penalty-X p110: p_feas=0.146970227261608, p_opt/feas=0.00243496350177599, p_opt=0.000357867139229737 | `results/global_depth110/analysis_summary.json` |
| 9 | Global-Grover p110: p_feas=0.0219460980572834, p_opt/feas=0.0469233036335992, p_opt=0.00102978342271465 | `results/global_depth110/analysis_summary.json` |
| 9 | Derived ratios: conditional=19.270639×; Penalty-X feasible mass=6.696873× Global; Global p110 total=2.877558× Penalty-X; Penalty-X best=2.751213× Global best | `Derived directly from the preceding frozen p110 and best-p_opt values` |
| 10 | p_opt direction changes p90–p110: Penalty-X 6, Global-Grover 9; across all depths: 52 and 50 | `results/global_depth110/analysis_summary.json` |
| 10 | Neither method reaches p_opt ≥ 0.10; all 220 rows are budget-limited | `results/global_depth110/analysis_summary.json; REPORT.md` |
| 11 | 167 tests pass; 23 targeted visualization tests pass | `Commands recorded below, run 2026-08-20` |
| 11 | Depth artifact validation 43/43; CVaR artifact validation 43/43 | `results/global_depth110/validation_summary.json; results/cvar_robustness_v1/validation_summary.json` |
| 6/E | CVaR p110 Penalty-X: 7/10 wins, median paired ratio 2.970995, MIXED_CVAR_RECOVERY | `results/cvar_robustness_v1/analysis_summary.json; FINAL_SCIENTIFIC_INTERPRETATION.md` |
| 6/E | CVaR p110 Global-Grover: 10/10 wins, median paired ratio 3.929006 | `results/cvar_robustness_v1/analysis_summary.json` |
| E | CVaR alphas 0.05, 0.10, 0.25, 0.50 plus expectation alpha=1; depths 50, 100, 110; 140 cells | `configs/cvar_robustness_v1.json; results/cvar_robustness_v1/frozen_manifest.json` |

## Generated figures and editable chart data

All depth figures were regenerated without optimization from `results/global_depth110/depth_by_depth.csv`. The PPTX uses native editable PowerPoint charts; the PNG assets are visually matched render references used for the PDF and slide audit.

| Asset | Data source | SHA-256 |
|---|---|---|
| `presentation/assets/cvar_summary.png` | `results/cvar_robustness_v1/analysis_summary.json` | `56acb43cb98d52a92ed2b5fe54add4e246c06678cc628f8a0c311d33706babf5` |
| `presentation/assets/pcond_depth.png` | `results/global_depth110/depth_by_depth.csv` | `be4826f4abc28f918741fb3d08a96a9c280aa9afaea5f810a9fa9ac30a36b9ac` |
| `presentation/assets/pfeas_depth.png` | `results/global_depth110/depth_by_depth.csv` | `a6b233b51a23bb7339560c33f49d3cb40045a994c7d014d918d05bc38173853a` |
| `presentation/assets/popt_depth.png` | `results/global_depth110/depth_by_depth.csv` | `5f0cdcd85bd15325db36a8918684a0d34258c12bf3276da872ff74534e9dd64e` |
| `presentation/assets/popt_near100.png` | `results/global_depth110/depth_by_depth.csv` | `b10c1ec22447605984423fd0eb7821e11aefd718f21bfdf7dadb5bab9280aad9` |

The routing graph on slide 2 is not a raster copy: it is drawn as editable PowerPoint nodes, edges, weights, and labels directly from `data/graph.json`. The PDF reference uses the same build coordinates.

## Exact validation commands run

```bash
.venv/bin/python -m pytest -q
# 167 passed in 49.06s

.venv/bin/python src/main.py
# reproduced 14 edges, 16,384 states, route 0->1->2->4->5->6, cost 10

.venv/bin/python scripts/run_penalty_qaoa.py --budget 8
# smoke completed; intentionally budget-limited at 8 evaluations

.venv/bin/python scripts/run_course_final.py --seed 2601 --depth 1
# smoke completed; 20-route feasible basis, p_feas=1, 81 evaluations

.venv/bin/python -m pytest -q tests/test_qaoa_visualization.py tests/test_qaoa_dynamics_visualization.py
# 23 passed in 5.22s
```

No expensive optimizer sweep was rerun, and no file under `results/` was modified.

## Deliberately excluded or corrected material

- `figures/course/02`–`05`, `configs/course_final.json`, and the Q2-F/Q2-Final artifacts describe separate feasible-subspace, path-exchange, or threshold-Grover studies. They are not the full-space Penalty-X vs Global-Grover p=1…110 experiment and were excluded from the main deck.
- `results/qaoa_dynamics_deep_dive/v1` is a useful p=1/p=2 mechanism pilot, but its older scaled-X convention and shallow results were not used as the final depth evidence.
- `results/global_depth110_recovery_bfo_v1` is exploratory/pilot evidence and was excluded from headline claims.
- Q2 revision/extension material, benchmark-portfolio material, and larger research framing were excluded to keep this a DTU teaching-scale course story.
- The prompt's tentative wording “fixed evaluation budget” was corrected to the repository's depth-dependent but mixer-matched rule `max(120,4p+64)`.
- The prompt's tentative wording “each depth independently optimized” was corrected: each method-depth has an optimization, but p>1 is initialized by deterministic layerwise continuation from p−1.
- The old presentation-story claim of 103 tests was excluded; the current suite has 167 passing tests.
- No standard-Grover quadratic speedup, Grover-scale transition, hardware performance, depth scaling law, or quantum advantage is claimed.

## Working-tree status at build time (including the generated presentation directory)

```text
M  .gitattributes
M  .gitignore
M  README.md
M  archive/development/legacy_core_pipeline/experiment.py
M  archive/development/q2r/q2_revision.py
M  archive/development/q2r/test_q2_revision.py
M  archive/development/warm_start_tuning/test_tuning.py
M  archive/development/warm_start_tuning/tune_warm_start.py
M  archive/development/warm_start_tuning/tuning.py
M  pyproject.toml
A  reports/QAOA_ROUTING_ALGORITHM_LAYER_CN.md
A  reports/QAOA_ROUTING_ALGORITHM_LAYER_CN.pdf
A  reports/QAOA_ROUTING_FINAL_REPORT_CN.md
A  reports/QAOA_ROUTING_FINAL_REPORT_CN.pdf
A  reports/QAOA_ROUTING_FINAL_REPORT_CN_EVIDENCE_INDEX.md
A  scripts/make_algorithm_report_cn.py
M  scripts/make_course_figures.py
AM scripts/make_final_report_cn.py
M  scripts/run_course_final.py
M  scripts/run_penalty_qaoa.py
M  scripts/run_qaoa_visualizer.py
M  src/experiments/course_final.py
M  src/experiments/dynamics_study.py
M  src/experiments/dynamics_trace.py
D  src/experiments/penalty_demo.py
M  src/feasible_experiments.py
M  src/feasible_qaoa.py
M  src/graph.py
MM src/main.py
A  src/metrics.py
M  src/qaoa.py
M  src/support/__init__.py
M  src/support/dynamics_figures.py
D  src/support/exact_reference.py
D  src/support/metrics.py
M  src/support/qaoa_dynamics_visualization.py
M  src/support/qaoa_visualization.py
D  src/support/sealed_results.py
D  src/support/warm_start.py
D  src/utils.py
M  tests/conftest.py
M  tests/test_course_commands.py
M  tests/test_feasible_qaoa.py
M  tests/test_graph_exact.py
M  tests/test_metrics.py
M  tests/test_objectives.py
M  tests/test_q2f_final_improvement.py
M  tests/test_qaoa.py
M  tests/test_qaoa_dynamics.py
D  tests/test_warm_start.py
D  todo.txt
?? presentation/
?? reports/FINAL_EXPERIMENT_RESULTS_CN.md
?? reports/FINAL_EXPERIMENT_RESULTS_CN.pdf
```
