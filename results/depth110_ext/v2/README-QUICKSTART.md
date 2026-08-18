# Depth-110 Main Track v2

This directory contains the corrected 220-row main track:

- 110 verified Penalty-X rows from the saved source trajectory.
- 110 newly optimized Grover-Mixer rows using the route-cost phase.

The Grover cost layer is `exp(-i * gamma * C(P_i))`.  The incumbent-threshold
mask is used only for the auxiliary BSP metric.  The earlier threshold-phase
Grover trajectory is not part of this directory or the v2 package.

`evaluation_budget_exhausted` is reported as `BUDGET_LIMITED` in
`failure_taxonomy.json`; it does not mean the process crashed or exceeded its
wall-clock guard.

Run the figure, report, and validation scripts to rebuild the analysis from the
saved files without rerunning either optimizer.

The repository also contains every module used by the two sweep runners and by
the v2 builder. See `REPORT.md` for the commands that reproduce a new 220-row
main track in separate result directories.
