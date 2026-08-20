# Presentation Data Audit

The final deck is course-first and reads only tracked course figures. It does not read or present depth-110, recovery, robustness, or search-control campaigns as course results.

| Claim | Frozen course source |
| --- | --- |
| 7 nodes, 14 edge variables, exact route cost 10 | `data/graph.json`; `src/graph.py` |
| QUBO/Ising equality over 16,384 states | `tests/test_encoding.py` |
| 20 feasible routes and structural p_feas=1 | `results/q2f_final_improvement/`; `tests/test_feasible_qaoa.py` |
| GM-Th depth-3 median p_opt=0.998712 | `results/q2f_final_improvement/*/summary/aggregate_by_method_depth.csv` |
| CVaR did not clearly improve depth-3 expectation | `results/q2f_course_extension/*/summary/aggregate_by_objective_depth.csv` |

## Figure inputs

- `figures/course/01_routing_graph_and_routes.png`
- `figures/course/02_popt_versus_depth.png`
- `figures/course/04_final_probability_distribution.png`
- `figures/course/05_grover_amplification.png`
- `figures/course/simple_ising_circuit.png`
- `figures/qaoa_dynamics_deep_dive/v1/02_qubo_ising_qaoa_pipeline.png`
- `figures/qaoa_dynamics_deep_dive/v1/03_three_search_spaces_and_mixers.png`
- `figures/qaoa_dynamics_deep_dive/v1/06_probability_mass_decomposition.png`
- `figures/qaoa_dynamics_deep_dive/v1/07_layer_by_layer_expected_hc.png`
- `presentation/assets/eq_final_qubo.png`
- `results/q2f_course_extension/q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7/figures/03_popt_versus_depth.png`

Generated outputs: PPTX, PDF, Notes, and this audit. Local rendered slide PNGs are ignored by Git.
