<div align="center">

# QAOA Routing

**From a weighted routing problem to QUBO, Ising, and 110-layer QAOA dynamics**

![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-full_reproduction-F37626?logo=jupyter&logoColor=white)
![Validation](https://img.shields.io/badge/validation-ALL_PASS-2EA44F)
![State space](https://img.shields.io/badge/state_space-16,384-16324F)

[Full notebook](./QAOA_Routing_Project_Full_Reproduction.ipynb) ·
[Results](#results) ·
[Reproduce](#reproduce-the-notebook) ·
[Repository layout](#repository-layout)

</div>

## Overview

This project studies a small directed routing problem as a complete QAOA
workflow:

```text
weighted graph → binary edge variables → QUBO → Ising Hamiltonian
               → cost phase → mixer → interference → route probabilities
```

Every directed edge is represented by one qubit. The objective combines route
cost with a squared flow-conservation penalty,

$$
Q(x)=\sum_e w_e x_e+A\sum_v
\left(\sum_{e\in\mathrm{out}(v)}x_e-
\sum_{e\in\mathrm{in}(v)}x_e-b_v\right)^2,
$$

with penalty coefficient $A=6$. The notebook compares two full-space mixers:

- **Penalty-X:** independent single-qubit $X$ rotations;
- **Global-Grover:** a rank-one phase operation around the uniform state.

The simulation is an ideal statevector study. It is intended to expose the
mechanism and trade-offs of the algorithms, not to claim quantum advantage.

## Routing instance

<p align="center">
  <img src="./assets/readme/routing_problem.png" alt="Directed seven-node routing graph with the unique optimal path highlighted" width="920">
</p>

| Property | Value |
|---|---:|
| Nodes / directed edges | 7 / 14 |
| Binary states | $2^{14}=16{,}384$ |
| Decoder-valid routes | 20 |
| Unique optimal route | `0 → 1 → 2 → 4 → 5 → 6` |
| Optimal route cost | **10** |
| QAOA depths studied | $p=1,\ldots,110$ |

## Results

The plots below use the frozen depth sweep in
[`results/global_depth110/depth_by_depth.csv`](./results/global_depth110/depth_by_depth.csv).
Both methods use the same 14-qubit Hamiltonian, initialization policy,
continuation scheme, and depth-dependent COBYLA evaluation budget.

<p align="center">
  <img src="./assets/readme/depth_results.png" alt="Optimal and feasible route probabilities versus QAOA depth for Penalty-X and Global-Grover" width="1100">
</p>

### Depth-110 comparison

| Method | $P(\mathrm{feasible})$ | $P(\mathrm{optimal})$ | $P(\mathrm{optimal}\mid\mathrm{feasible})$ | $\langle H_C\rangle$ | Evaluations |
|---|---:|---:|---:|---:|---:|
| Penalty-X | **14.6970%** | 0.035787% | 0.24350% | **18.3693** | 504 |
| Global-Grover | 2.19461% | **0.102978%** | **4.69233%** | 33.2746 | 504 |

### Main observations

- **Penalty-X favors feasibility.** It places more probability on valid routes
  at every one of the 110 recorded depths.
- **Global-Grover favors concentration within the feasible subspace.** It has
  the higher conditional optimal probability at 89 of 110 depths.
- The first raw-success crossover occurs at **$p=22$**. Global-Grover has the
  higher $P(\mathrm{optimal})$ for every recorded depth from 22 through 110.
- The largest success probability in the complete sweep is nevertheless the
  Penalty-X result at **$p=21$**, where
  $P(\mathrm{optimal})=0.283315\%$. Global-Grover reaches its maximum at
  **$p=110$**, with $P(\mathrm{optimal})=0.102978\%$.

These results reveal two different bottlenecks: Penalty-X is better at moving
mass into the valid routing subspace, while Global-Grover is better at selecting
the optimum after feasibility is reached.

> [!NOTE]
> The frozen optimizer runs are evaluation-budget limited. The reported points
> are reproducible outcomes of this protocol, not certificates of globally
> optimal QAOA parameters.

## Reproduce the notebook

Python 3.11–3.13 is supported.

```bash
git clone https://github.com/ZhilinChen02/sciqis-qaoa-routing.git
cd sciqis-qaoa-routing
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
python -m jupyter lab QAOA_Routing_Project_Full_Reproduction.ipynb
```

For a non-interactive end-to-end run:

```bash
MPLBACKEND=Agg python -m jupyter nbconvert \
  --to notebook --execute QAOA_Routing_Project_Full_Reproduction.ipynb \
  --output /tmp/QAOA_Routing_Project_Full_Reproduction.executed.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=300
```

A successful run executes all 51 cells and finishes with:

```text
ALL VALIDATIONS PASS
```

The default execution replays the saved depth-110 parameters and normally
finishes in under two minutes. The optional full depth-1-to-110 re-optimization
is disabled by default because it is substantially more expensive.

## What is validated

The notebook checks the scientific-computing chain directly:

- QUBO, Ising, and penalized routing energies agree on all 16,384 basis states;
- exactly 20 states decode as valid routes and exactly one is optimal;
- cost evolution preserves computational-basis probabilities;
- both mixer implementations preserve state normalization;
- saved Penalty-X and Global-Grover depth-110 metrics are reproduced within
  numerical tolerance;
- the final validation table passes every assertion.

## Repository layout

```text
.
├── QAOA_Routing_Project_Full_Reproduction.ipynb
├── assets/readme/                  # two curated README figures
├── data/graph.json                 # fixed routing instance
├── results/global_depth110/
│   ├── depth_by_depth.csv          # p = 1,…,110 summary
│   └── checkpoints/                # two p = 110 parameter sets
├── src/
│   ├── graph.py                    # graph and route encoding
│   ├── qubo.py                     # QUBO/Ising construction
│   ├── qaoa.py                     # exact statevector evolution
│   └── metrics.py                  # route-distribution metrics
└── pyproject.toml
```

Figures and tables generated during a local notebook run are written to
`notebook_output/qaoa_project_reproduction/` and ignored by Git. Only the two
curated figures used on this page are versioned.
