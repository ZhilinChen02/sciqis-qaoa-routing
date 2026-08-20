# QAOA Routing — Full Reproduction Notebook

This repository is intentionally minimal. It contains only the code and frozen
numerical inputs required to run
`QAOA_Routing_Project_Full_Reproduction.ipynb` from top to bottom.

The notebook builds a directed-routing QUBO, converts it to an Ising
Hamiltonian, compares Penalty-X and Global-Grover QAOA, replays the saved
depth-110 circuits, and checks the numerical results. It uses ideal statevector
simulation and makes no claim of quantum advantage.

## Quick start

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
python -m jupyter lab QAOA_Routing_Project_Full_Reproduction.ipynb
```

Run the complete notebook non-interactively:

```bash
MPLBACKEND=Agg python -m jupyter nbconvert \
  --to notebook --execute QAOA_Routing_Project_Full_Reproduction.ipynb \
  --output /tmp/QAOA_Routing_Project_Full_Reproduction.executed.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=300
```

The default run takes about two minutes on a typical laptop. It reads the
frozen sweep and replays the two depth-110 states; the much more expensive full
depth-1-to-110 re-optimization remains disabled by default in the notebook.

Generated figures and tables are written to
`notebook_output/qaoa_project_reproduction/` and are intentionally ignored by
Git.

## Repository contents

- `QAOA_Routing_Project_Full_Reproduction.ipynb` — the complete reproduction.
- `src/` — graph, QUBO/Ising, QAOA, and metric helpers used by the notebook.
- `data/graph.json` — the fixed 7-node, 14-edge routing instance.
- `results/global_depth110/depth_by_depth.csv` — the frozen depth summary.
- `results/global_depth110/checkpoints/*_p110.json` — the two saved depth-110
  parameter sets used for exact replay.

The notebook finishes with `ALL VALIDATIONS PASS` when the reproduction is
successful.
