# Dynamic QAOA Visualization

## Launch

The unified, read-only QAOA Dynamics Demo uses only the Python standard-library
web server and browser-native JavaScript, Canvas, and SVG. From the repository
root run:

```bash
python scripts/run_qaoa_dynamics_visualizer.py
```

It opens `http://127.0.0.1:8766/`. On a remote machine, suppress the automatic
browser launch:

```bash
python scripts/run_qaoa_dynamics_visualizer.py --no-browser
```

Use another port with `--port 9000`. The server binds only to `127.0.0.1` by
default. No Node installation, cloud service, CDN, or internet connection is
needed.

To validate the data without starting the interface:

```bash
python scripts/run_qaoa_dynamics_visualizer.py --check
```

## What is being visualized

The selectors cover the six frozen primary runs:

- Penalty-X QAOA at `p=1` and `p=2`;
- Global Grover-Mixer QAOA at `p=1` and `p=2`;
- Feasible Grover-Mixer QAOA at `p=1` and `p=2`.

The checkpoint slider adapts to the selected depth. A `p=1` run has Initial,
Cost-1, and Mixer-1. A `p=2` run additionally has Cost-2 and Mixer-2. Changing
algorithm, depth, or checkpoint updates every panel from one shared run frame;
there are no independent panel clocks.

The central teaching mechanism is:

> The cost Hamiltonian encodes objective information into relative phase. The
> mixer converts that phase structure into interference and probability
> redistribution.

A cost unitary commutes with its own Hamiltonian. Consequently it changes
complex phases but preserves every computational-basis probability and
preserves the expected cost energy. A subsequent mixer may change amplitudes,
probabilities, feasibility, optimality, and expected energy. Neither the
intermediate nor final energy is required to decrease monotonically.

## Panels

### A. QAOA circuit timeline

Shows Initial → cost → mixer → cost → mixer → Measurement, highlights the
current checkpoint, and displays the actual frozen `gamma` or `beta` for the
selected layer. Parameters use the project convention: all gammas followed by
all betas.

### B. Expected energy

Shows exact layer-by-layer `⟨H_C⟩`. For Penalty-X and Global-Grover it also
shows the saved decomposition into the raw routing contribution and
`A⟨H_penalty⟩`. Cost-step deltas are shown numerically as approximately zero.
The plot deliberately does not imply monotonic descent.

### C. Feasibility, optimality, and invalid mass

Shows exact statevector probability masses and trajectories. Feasible-Grover is
labelled as structurally feasible: its logical representation contains only
valid routes, so `p_feas≈1` is not an optimization achievement.

### D. Probability flow

For each full-space method the display contains the exact optimum, up to seven
other highest-probability feasible states, up to seven highest-probability
infeasible states, and exact aggregate remainder masses. The rows always sum to
one. For Feasible-Grover all 20 logical paths are displayed. Click a state to
connect it to the routing graph.

### E. Complex amplitude and phase

Plots representative amplitudes as vectors `a_x=|a_x|exp(i phi_x)`. Feasible,
infeasible, and exact-optimal states use distinct colors. The phase, magnitude,
and probability toggles change radial normalization but never alter the saved
state data. Between a pre-cost and post-cost frame vector lengths stay fixed
while angles change; after mixing, lengths may change.

### F. Phase versus energy

Plots wrapped phase against state energy. All feasible states and the optimum
are retained. Full-space views use a labelled fixed-seed random sample of roughly
1,400 infeasible states plus every state required by the probability and energy
inspectors; the display never describes this sample as all 16,384 points.
Feasible-Grover shows all 20 logical states.

### G. Energy landscape

Uses exact all-16,384-state counts from `energy_landscape.csv`, not a sample.
The logarithmic histogram distinguishes feasible and infeasible energies and
marks the confirmed optimum 10, second feasible energy 11, and lowest
infeasible energy 12. Select a bin and then a representative state to inspect
its edge selection and flow residuals.

### H. Search-space geometry

Uses a labelled conceptual `q=3` diagram. Penalty-X shows hypercube edges for
one-bit flips. Global-Grover uses a projector hub over all illustrative states.
Feasible-Grover highlights only an illustrative feasible subset. The hub is a
notation for rank-one projector action, not a claim that the hardware contains
a complete graph, and the diagram is not the real 16,384-node state graph.

### I. Route and edge-selection view

Uses the original seven-node, fourteen-edge graph. A valid selection highlights
its route and reports cost, current probability, and global-optimum status. An
invalid selection highlights all selected edges, nodes with nonzero flow
residuals, raw selected-edge cost, flow penalty, and `A`-weighted penalty
contribution.

### J. Optimizer landscape

For `p=1`, the panel displays the existing `25×25` gamma/beta scan with toggles
for expected energy, `p_feas`, and `p_opt`. It marks the frozen optimizer result
and the best grid point. The exact evaluation trajectory was not retained, so
the interface says that it is unavailable instead of inventing it. For `p=2`,
no local two-dimensional slice was saved, and the interface refuses to show a
misleading projection of the four-dimensional landscape.

## Playback and presentation mode

The normal controls provide Previous, Play/Pause, Next, and four speeds. The
default interval is 1.4 seconds, and all panels advance together.

The **Present** button enables a clean seven-scene sequence intended for the
15-minute DTU presentation:

1. initial full-space uniform state;
2. cost phase without probability or energy change;
3. mixer interference and non-monotonic energy;
4. Penalty-X versus Global-Grover—same `H_C`, different mixer;
5. full-space versus 20-route feasible-space restriction;
6. final `p=2` feasibility and optimality;
7. the classical optimizer around the quantum ansatz.

Use the on-screen arrows or keyboard left/right arrows. Escape exits
presentation mode.

## Data provenance and numerical validation

The primary inputs are read from
`results/qaoa_dynamics_deep_dive/v1/`: `final_summary.json`,
`dynamics_trace.csv`, `energy_landscape.csv`, the saved parameter landscapes,
final probability NPZ files, p=2 amplitude NPZ files, and phase-analysis CSVs.
The server never writes to this result namespace and never calls an optimizer.

The p=2 statevectors are loaded and compared with deterministic evolution from
the frozen parameters. The p=1 intermediate vectors, which were not persisted
as full amplitude arrays, are reconstructed from the saved parameters and
checked against every stored checkpoint metric and final probability vector.
The server refuses to start when any scientific consistency check fails.

Checks cover:

- exact checkpoint order;
- norms and probability sums;
- cost-step probability and `⟨H_C⟩` invariance;
- saved phase, magnitude, and probability values;
- `p_opt≤p_feas`;
- feasible-route decoding and Feasible-Grover validity;
- 16,384-state full-space dimensions;
- probability-flow aggregate normalization;
- displayed checkpoint metrics versus `dynamics_trace.csv`;
- displayed final metrics versus `final_summary.json`.

The validation tolerance is `1e-10`. On the retained artifacts the largest
cost-step probability change is about `1.39e-17`, the largest cost-step energy
change is about `1.42e-14`, and displayed saved metrics match exactly.

## Presentation-safe exports

The live HTML interface is primary. A small optional exporter creates a GIF and
three static PNG frames without rerunning optimization:

```bash
python scripts/export_qaoa_dynamics_animation.py
```

Outputs are written under `figures/qaoa_dynamics_visualizer/v1/`:

- `global_grover_p2_cost_mixer_dynamics.gif`;
- `fallback_initial.png`;
- `fallback_cost_1.png`;
- `fallback_mixer_2.png`.

The exporter refuses a non-empty output directory unless `--force` is supplied.
Use `--output /tmp/qaoa-dynamics-export` for a non-destructive smoke run.

## Implementation and performance

The server caches six immutable browser payloads after one strict startup
validation. The frontend uses ordinary Canvas and SVG and requests one run at a
time. It never creates or transmits a dense `16,384×16,384` object. Full-space
amplitudes remain vectors; probability bars are aggregated; phase rendering is
sampled with all feasible states retained; and the exact energy histogram is
pre-binned server-side.

The implementation is isolated in:

- `src/qaoa_dynamics_visualization.py`;
- `scripts/run_qaoa_dynamics_visualizer.py`;
- `web/qaoa_dynamics_visualizer/`;
- `tests/test_qaoa_dynamics_visualization.py`.

It does not modify the separate historical feasible-route optimizer visualizer
in `web/qaoa_visualizer/`.

## Scientific limitations

- This is ideal statevector simulation of one seven-node graph.
- Feasible-Grover explicitly enumerates 20 routes; taking the classical argmin
  already exposes the optimum.
- Structural `p_feas=1` is not quantum advantage or a scalability result.
- Global projector mixing in the logical simulator is not automatically a
  hardware-efficient mixer.
- The fixed-seed full-space phase sample is for rendering only; all exact
  metrics and energy-histogram counts use complete probability vectors/data.
- The absent optimizer evaluation trajectory and absent `p=2` parameter slice
  are labelled unavailable rather than reconstructed or fabricated.
- The interface supports teaching and mechanism comparison; it does not claim
  superiority over classical shortest-path algorithms.
