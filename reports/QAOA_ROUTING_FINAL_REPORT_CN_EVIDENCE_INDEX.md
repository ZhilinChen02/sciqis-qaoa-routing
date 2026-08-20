# QAOA 路由项目最终报告：来源与证据索引

**用途：** 本文件是 `QAOA_ROUTING_FINAL_REPORT_CN.md` 的逐主题证据入口。它只索引仓库中现存的源代码、冻结配置、canonical 数据、验证记录和图；不把叙述性总结本身当作原始数据，也不合并相互独立的 seed 集。

**审阅日期：** 2026-08-19
**仓库：** `ZhilinChen02/sciqis-qaoa-routing`
**证据范围：** 理想 statevector 模拟、单一冻结有向加权路由实例；不含 QPU、噪声模型或量子优势证据。

证据状态使用以下词汇：`CONFIRMATORY / STRONGLY_SUPPORTED`、`DESCRIPTIVE`、`MECHANISTIC`、`EXPLORATORY`、`HISTORICAL/PILOT`、`UNSUPPORTED / PROHIBITED`。其中 `CONFIRMATORY` 只用于冻结协议明确指定的 primary endpoint；`STRONGLY_SUPPORTED` 表示多块相互独立的新证据方向一致，但不代表跨实例普适性。

| Scientific topic | Artifact path | Evidence status | Key quantities |
| --- | --- | --- | --- |
| 项目定义、运行入口与范围 | `README.md`; `pyproject.toml` | DOCUMENTATION / IMPLEMENTATION | DTU SCIQIS course project；Python 3.11–3.13；NumPy/SciPy/Qiskit；明确不主张 quantum advantage |
| 冻结图与 edge ordering | `data/graph.json`; `src/graph.py` | STRONGLY_SUPPORTED（定义） | 7 nodes，14 directed edges，source=0，target=6；q0…q13 顺序由 JSON 固定 |
| 经典最短路与独立参考 | `src/graph.py`; `src/qubo.py`; `results/global_depth110/reference_solution.json` | STRONGLY_SUPPORTED | weighted shortest path、全部 simple-route 枚举与 edge-selection decoder 一致；unique optimum `0→1→2→4→5→6`，cost=10，state index=10377，bitstring=`10010001000101`；20 条 feasible simple routes |
| QUBO、flow penalty、decoder | `src/qubo.py`; `docs/QAOA_DYNAMICS_DEEP_DIVE.md` | STRONGLY_SUPPORTED（实现与枚举核验） | `A=6`；routing cost + squared flow residual；16,384 states；DAG 上 penalty-feasible 与 decoder-valid 一致；最低 infeasible QUBO energy=12 |
| QUBO→Ising 映射 | `src/qubo.py`; `results/qaoa_dynamics_deep_dive/v1/scientific_validations.json` | STRONGLY_SUPPORTED（数值恒等核验） | `x=(I-Z)/2`；exhaustive max error=0；共享 diagonal SHA-256=`66e208…542b` |
| QAOA statevector 与 Penalty-X | `src/qaoa.py`; `results/qaoa_dynamics_deep_dive/v1/experiment_config.json` | IMPLEMENTATION / HISTORICAL | full `2^14` basis；uniform `|+⟩^14`；penalized cost；X mixer；早期 course 代码含 `β/q` 缩放，深度研究明确不缩放 |
| repository-specific Global-Grover | `src/qaoa.py`; `results/global_depth110/experiment_config.json`; `results/global_depth110/validation_summary.json` | STRONGLY_SUPPORTED（定义） | full-space `H_M=|s⟩⟨s|`，`U=I+(e^{-iβ}-1)|s⟩⟨s|`；无 feasible projection、无 threshold oracle；16,384-state rank-one update |
| 浅层 p=1/p=2 dynamics | `results/qaoa_dynamics_deep_dive/v1/final_summary.json`; `postrun_scientific_validation.json`; `dynamics_trace.csv` | HISTORICAL/PILOT + MECHANISTIC | 6 cells；Penalty-X p1/p2 均 `p_opt=6.1035e-5`；Global p2 `p_opt=2.13372e-4`；cost layer 保持概率，mixer 改变概率 |
| 原图与 QUBO/QAOA 流程图 | `figures/qaoa_dynamics_deep_dive/v1/01_fixed_weighted_routing_graph.png`; `02_qubo_ising_qaoa_pipeline.png` | DOCUMENTATION（由冻结定义生成） | 图结构、权重、source/target；graph→QUBO→Ising→QAOA→decode 流程 |
| Warm-start Q2-R 正式执行 | `results/q2_revision_formal/q2r-9c98b90049545d0508fb20bb488019608a4356d55fe91ff809bce8dc5c5e0c69/summary/formal_results.json`; `execution_receipt.json`; `archive/development/q2r/Q2_LITERATURE_GUIDED_REVISION.md` | HISTORICAL/PILOT | 8/8 一次完成，seed=2601；aligned incumbent-product state/mixer；p3 expectation A0 `p_opt=0.00228946`，CVaR-0.25 A1 `0.00308458`；单 seed，不作 robustness 推断 |
| Q2-R 验证事件 | `results/q2_revision_formal/q2r-9c98b90049545d0508fb20bb488019608a4356d55fe91ff809bce8dc5c5e0c69/validation/postrun_validation.json`; `postrun_validation_reaudit.json`; `execution_receipt.json` | INTEGRITY EVIDENCE | 初始 validator 因 JSON list/tuple 表示差异失败；未改 raw、未 rerun；append-only reaudit 108/108 通过；事件保留而未隐藏 |
| Feasible-subspace/path-exchange Q2-F | `docs/methods/Q2F_FEASIBLE_WARM_START.md`; `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/summary/q2f_results.csv`; `aggregate_by_objective_depth.csv`; `validation/postrun_validation.json` | DESCRIPTIVE / MECHANISTIC | logical 20-route basis；106 path exchanges；27 cells，3 seeds，p1–3，cap=100；p3 median `p_opt`: E=0.264901，CVaR-.25=0.258176，ascending=0.263191 |
| Feasible Global/threshold extension | `docs/methods/Q2F_FINAL_IMPROVEMENT.md`; `results/q2f_final_improvement/q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1/summary/Q2F_FINAL_IMPROVEMENT_RESULTS.md`; `aggregate_by_method_depth.csv`; `validation/postrun_validation.json` | DESCRIPTIVE / EXPLORATORY | 36/36，3 seeds，p1–4，cap=150；GM-Th p3 median `p_opt=0.998712`；只在预枚举 20-route logical space，非 scalable hardware implementation |
| 缺失的早期提案数值 | 旧 proposal/已删除旧 final report（仅在历史 Git 状态中出现） | DOCUMENTED BUT NOT AVAILABLE AS FINAL SCIENTIFIC EVIDENCE | 不从 proposal 或 Git 删除清单恢复任何数值；现存冻结结果优先 |
| p=1…110 协议与 canonical 数据 | `results/global_depth110/experiment_config.json`; `canonical_rows.csv`; `canonical_rows.xlsx`; `depth_by_depth.csv` | FROZEN PRIMARY / DESCRIPTIVE | 220 rows=2 mixers×110 depths；同一 full-space、同一 penalized Hamiltonian；deterministic layerwise continuation；p110 budget=504 evaluations |
| p=1…110 结果与比较 | `results/global_depth110/analysis_summary.json`; `REPORT.md`; `near_p100.csv` | DESCRIPTIVE | Penalty-X best p=21, `p_opt=0.00283315`；Global best p=110, `0.00102978`；crossover p=22；Global conditional superiority 89/110；两者均未达 0.10 |
| p=1…110 验证与 runtime | `results/global_depth110/validation_summary.json`; `runtime_profile.csv`; `PROVENANCE.json` | INTEGRITY EVIDENCE | 43/43 checks；220/220 rows budget-limited；recorded wall time 2823.653 s；理想 complex128 statevector |
| 深度图 | `results/global_depth110/figures/01_optimal_probability_vs_depth.png`; `02_feasible_probability_vs_depth.png`; `03_conditional_optimal_probability.png` | DESCRIPTIVE VISUAL EVIDENCE | oscillatory/nonmonotonic depth trajectory；Penalty-X feasible mass 高，Global conditional optimal fraction 常更高 |
| 初次 budget/Fourier/CVaR recovery pilot | `results/global_depth110_recovery_bfo_v1/REPORT.md`; `canonical_rows.csv`; `analysis_summary.json`; `validation_summary.json` | HISTORICAL/PILOT / EXPLORATORY | 74 cells，40/40 validation；budget alone 0/10 material；direct CVaR comparisons 15 material；single-cell extremes only motivated replication，不可作 robustness 结论 |
| CVaR robustness 冻结协议与 manifest | `results/cvar_robustness_v1/OPTIMIZER_BUDGET_ROBUSTNESS_V1_PROTOCOL.md`; `frozen_manifest.json`; `canonical_rows.csv`; `canonical_rows.xlsx` | FROZEN CONFIRMATORY/DESCRIPTIVE | 140/140 cells；primary seeds 与 secondary seeds 分离；p50/p100/p110 budgets=2500/5000/5500 |
| Penalty-X CVaR robustness | `results/cvar_robustness_v1/paired_seed_comparison.csv`; `analysis_summary.json`; `FINAL_SCIENTIFIC_INTERPRETATION.md` | CONFIRMATORY | p110 E median `5.509e-5`，CVaR-.10 `2.727e-4`；7/10；median ratio=2.971，GM=2.766，CI=[0.613,9.097]；`MIXED_CVAR_RECOVERY` |
| Global-Grover CVaR robustness | 同上；`depth_replication_summary.csv`; `figures/02_globalgrover_p110_seed_pairs_popt.png` | DESCRIPTIVE SECONDARY，后被 fresh study 强化 | p110 10/10；E=`5.464e-4`，C=`2.264e-3`；median ratio=3.929，CI=[3.115,4.910]；p50/p100 均 5/5 |
| alpha response | `results/cvar_robustness_v1/alpha_response_summary.csv`; `figures/03_penaltyx_alpha_response.png`; `04_globalgrover_alpha_response.png` | EXPLORATORY | Global observed best α=.10；Penalty observed best α=.05；非单调；不支持 universal alpha optimum |
| 第一阶段 distribution mechanism | `results/cvar_robustness_v1/distribution_concentration_summary.csv`; `FINAL_SCIENTIFIC_INTERPRETATION.md`; `figures/08_penalized_energy_cumulative_mass.png` | MECHANISTIC / DESCRIPTIVE | Global Δ`p_feas`=+0.031455，ΔLowE100=+0.092148，Δ`p_opt_given_feas`=-0.002311；`FEASIBLE_REGION_AMPLIFICATION` |
| robustness 完整性 | `results/cvar_robustness_v1/validation_summary.json`; `run_history.json`; `REPORT.md` | INTEGRITY EVIDENCE | 43/43 checks；140/140，0 failures；全部耗尽相应 budget；历史 `global_depth110` 和 BFO root 未变 |
| Search-control prospective protocol | `results/cvar_search_control_mechanism_v1/PROTOCOL.md`; `frozen_manifest.json`; `execution_plan.md`; `experiment_config.json` | FROZEN PROSPECTIVE | exactly 60 cells；A=20、B=10、C=20、D=10；planned max 495,000 evaluations；seeds 8801–8810，B/C/D 用 8801–8805 |
| Search-control 完成与 canonical | `results/cvar_search_control_mechanism_v1/canonical_rows.csv`; `canonical_rows.xlsx`; `run_history.json` | STRONGLY_SUPPORTED（完整性） | 60/60；0 failed/censored/duplicate/missing；495,000/495,000 actual evaluations；所有 cells budget exhausted |
| Experiment A budget response | `trajectory_checkpoints.csv`; `budget_response_summary.csv`; `analysis_summary.json`; `figures/01_global_grover_budget_response_popt.png` | CONFIRMATORY endpoint + MECHANISTIC checkpoints | B=1000…11000 全部 10/10；B11000 E=`0.000542156`，C=`0.002164731`，median ratio=2.889，GM=3.744，CI=[2.618,5.766]；`CVAR_RECOVERY_PERSISTS_WITH_BUDGET` |
| Experiment B optimizer sensitivity | `canonical_rows.csv`; `analysis_summary.json`; `figures/05_optimizer_sensitivity_popt.png` | DESCRIPTIVE / SENSITIVITY | COBYLA 5/5 ratio=3.219；Nelder-Mead 5/5 ratio=1.768；`OBJECTIVE_OPTIMIZER_INTERACTION` |
| Experiment C cross-evaluation 与 basin transfer | `basin_transfer_rows.csv`; `canonical_rows.csv`; `analysis_summary.json`; `figures/06_basin_transfer_popt.png` | MECHANISTIC（5 paired seeds） | `C→E`: `0.002209→0.001132`, 5/5 decline；`E→C`: `0.000616→0.002626`, 5/5 rise；`OBJECTIVE_PRESSURE_REQUIRED` |
| Experiment D mixer transport | `trajectory_checkpoints.csv`; `analysis_summary.json`; `figures/07_mixer_transport_pfeas.png`; `08_mixer_transport_low_energy_top100.png` | MECHANISTIC（matched 5 seeds） | B11000 Global Δ`p_feas`=+0.028270、ΔLowE100=+0.079438；Penalty +0.004718、+0.011686；`MIXER_SPECIFIC_TRANSPORT_SUPPORTED` |
| Global mechanism trajectory | `trajectory_checkpoints.csv`; `analysis_summary.json`; `figures/03_global_grover_low_energy_top100_trajectory.png`; `04_global_grover_popt_given_feas_trajectory.png`; `09_entropy_trajectories.png` | MECHANISTIC | LowE100、`p_feas`、`p_opt` 在 B=1000 已过门槛；conditional sharpening 从未过门槛；`SIMULTANEOUS_WITHIN_CHECKPOINT_RESOLUTION`；FRA strengthened |
| Search-control verification 与 preservation | `verification/preflight_review.json`; `stage_a_validation.json`; `final_validation.json`; `historical_evidence_inventory.json`; `REPORT.md` | INTEGRITY EVIDENCE | final 836/836；312-file `cvar_robustness_v1` byte-for-byte unchanged；manifest仍exact 60 unique；冻结执行报告记录当时 full suite 176 tests |
| 当前工作区软件复核 | `tests/`; `src/main.py` | CURRENT IMPLEMENTATION CHECK | 2026-08-19 实跑 `.venv/bin/python -m pytest -q`：167 passed in 56.03 s；`src/main.py` 成功复现 14 edges、16,384 states、唯一 cost-10 route 与 p=1 baseline |
| Absolute performance | `trajectory_checkpoints.csv`; `canonical_rows.csv`; `FINAL_SCIENTIFIC_INTERPRETATION.md` | DESCRIPTIVE | B11000 medians: Global E .0005422/C .0021647；Penalty E .00007417/C .00043382；best retained Global-CVaR .00453160；均低于1% |
| 跨实例、hardware、noise、advantage、depth law | 各 frozen protocol/report 的 prohibited-claims sections | UNSUPPORTED / PROHIBITED | 单实例、ideal simulator、budget exhausted；不支持 quantum advantage、QPU/noise generality、universal CVaR/α/mixer/optimizer、fully converged optimum、depth scaling law |

## 证据分层与不可合并项

1. `global_depth110` 是每个 mixer 一条 deterministic layerwise-continuation trajectory，不是多 seed 推断样本。
2. `global_depth110_recovery_bfo_v1` 是探索性 pilot；其中单 cell 的大倍率不能替代 paired fresh-seed robustness。
3. `cvar_robustness_v1` 与 `cvar_search_control_mechanism_v1` 的 fresh seeds 统计上保持分离，只作描述性前后对照。
4. Search-control 的 Experiment A 有 10 pairs；B/C/D 只有 5 pairs，因而后者是 sensitivity/mechanistic evidence。
5. Q2-F 与 GM-Th 使用预枚举 20-route logical basis，不能与 full `2^14` Penalty-X/Global-Grover 的绝对概率作公平算法排名。
6. 冻结 execution receipt/report 的历史 test count 保持原样；本次 final-writing review 另行执行当前工作区测试并得到 167 passed，但没有重跑任何 scientific cell。

## 本次索引生成边界

本索引的生成仅包含读取、CSV/JSON 字段核对和文档编排。未启动、恢复、延长、修复或重跑任何科学实验；未修改任何 `results/`、`checkpoints/`、distribution、manifest、canonical row 或历史证据。
