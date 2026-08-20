# Execution Plan

## Frozen order

1. Run and validate Experiment A.
2. Run Experiments B and D only after A checkpoint artifacts are valid.
3. Validate B and D.
4. Independently verify the ten required A B=5,500 parameter checkpoints.
5. Run Experiment C.
6. Rebuild canonical outputs, analyze, generate figures, and independently verify.
7. Recompute the complete `cvar_robustness_v1` inventory and compare it with the frozen preservation manifest.

## Planned commands

All execution commands use:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The restart-safe runner stages are `A`, `B`, `D`, and `C`, with eight independent worker processes unless preflight finds that unsafe. Checkpoints are reconstructed within long A/D trajectories and are never submitted as separate cells.

## Compute accounting

| Experiment | Cells | Evaluations per cell | Maximum evaluations |
|---|---:|---:|---:|
| A | 20 | 11,000 | 220,000 |
| B | 10 | 5,500 | 55,000 |
| C | 20 | 5,500 | 110,000 |
| D | 10 | 11,000 | 110,000 |
| Total | 60 | — | 495,000 |

Historical p=110 medians imply approximately 360 seconds per 5,500-evaluation Global-Grover cell and 2,098 seconds per 5,500-evaluation Penalty-X cell. Linear evaluation-count projection gives roughly 20.8 serial CPU-hours for the full plan. With eight one-thread workers and stage dependencies, expected wall-clock is approximately 3–4 hours, with Experiment D dominating. This is a planning estimate, not a completion guarantee.

## Resume contract

The canonical CSV is rebuilt from atomically valid per-cell checkpoints after every completed cell. A process interruption can lose only an in-progress cell. A valid retained failure is terminal and is not retried. Invalid artifacts fail closed for review rather than being promoted or silently replaced.
