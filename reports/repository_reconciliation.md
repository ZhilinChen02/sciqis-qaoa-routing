# Repository reconciliation for the v3.0 public release

This report records the repository state found at baseline commit
`398d76024b6f0427cd418af2adaacf2b347a29e8`. Repository cleanup is explicitly
non-scientific: no frozen contract, result, or conclusion may change.

## Scientific freeze gate

Before cleanup, every artifact declared in `data/scientific_freeze_v3.json`
matched its recorded SHA-256. The scientific status was
`COURSE_PROJECT_SCIENCE_COMPLETE`; the exact route, cost, penalty grid, depth
grid, 24 optimization runs, and eight selected cells all reconciled.

## Classification rules

- `RELEASE_REQUIRED`: part of the final v3.0 implementation or release record.
- `LEGACY_SUPERSEDED`: empty scaffolding or older code replaced by v3.0.
- `COURSE_DOCUMENTATION`: a course proposal/report document.
- `ENVIRONMENT_REQUIRED`: the active dependency/build contract.
- `UNRELATED_OR_AMBIGUOUS`: ownership cannot be established safely.

## Dirty-path inventory and decision

| Path | Baseline status | Classification | Reason | Modern replacement, if any | Proposed action |
|---|---|---|---|---|---|
| `.python-version` | unstaged deletion | LEGACY_SUPERSEDED | Tracked value was Python 3.13, while the frozen work ran on 3.12.4; retaining two version contracts would be confusing. | `pyproject.toml` Python range and README tested-version note | Keep deletion. |
| `DTU-QAOA-RCSP-Mixers_Project_Proposal_中文版.docx` | staged add, worktree deleted (`AD`) | COURSE_DOCUMENTATION | Version 1.0 describes the superseded RCSP/path-exchange project, not frozen v3.0. Index blob SHA-256: `fa43ff5c9b8a8a1bdf5027477983ca1674480229049311558280a920f6475da6`. | v3.0 proposal under `docs/` | Remove the staged addition; keep absent. |
| `DTU_SCIQIS_QAOA_Routing_Project_Proposal_v2.0.docx` | staged addition | COURSE_DOCUMENTATION | Version 2.0 still includes warm-start/path-exchange scope removed from the final science. SHA-256: `7ce141a642b0b6de91e6f0dc7cfecd729dec140236bab09f9585695939f969a6`. | v3.0 proposal under `docs/` | Remove from index and working tree. |
| `LICENSE` | unstaged deletion | LEGACY_SUPERSEDED | The tracked file is zero bytes and grants no license. User intent is required before choosing a real license. | None pending owner decision | Keep deletion; document `LICENSE_DECISION_REQUIRED`. |
| `data/instances/.gitkeep` | unstaged deletion | LEGACY_SUPERSEDED | Empty placeholder predates the canonical v3 contracts in `data/`. | `data/graph.json` and frozen contracts | Keep deletion. |
| `docs/project_plan.md` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte placeholder. | Final report, presentation outline, and v3.0 proposal | Keep deletion. |
| `notebooks/project_demo.ipynb` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte placeholder, not a notebook. | `notebooks/01_...` through `04_...` | Keep deletion. |
| `pyproject.toml` | unstaged modification | ENVIRONMENT_REQUIRED | The dirty draft removes obsolete Qutip, but still packages the four obsolete untracked modules and omits Qiskit/Pillow/notebook dependencies. | Reconciled release `pyproject.toml` | Replace with one tested pip/setuptools contract for the modern flat modules. |
| `scripts/make_figures.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte placeholder. | Day-specific build scripts | Keep deletion. |
| `scripts/run_demo.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte placeholder. | Day-specific build and notebook workflow | Keep deletion. |
| `scripts/run_experiments.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte placeholder. | `scripts/run_day4_core_experiment.py` | Keep deletion. |
| `slides/.gitkeep` | unstaged deletion | LEGACY_SUPERSEDED | Empty placeholder; the release supplies a presentation outline instead of a slide deck. | `reports/presentation_outline_15min.md` | Keep deletion. |
| `figures/.gitkeep` | clean at baseline; found during tree audit | LEGACY_SUPERSEDED | The release tracks complete PNG/SVG evidence, so the placeholder is redundant. | Tracked Figures 1–13 | Remove. |
| `results/.gitkeep` | clean at baseline; found during tree audit | LEGACY_SUPERSEDED | The release tracks complete JSON/CSV evidence, so the placeholder is redundant. | Tracked result artifacts | Remove. |
| `src/sciqis_qaoa_routing/__init__.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold never implemented. | Modern flat `src/*.py` modules | Keep deletion. |
| `src/sciqis_qaoa_routing/cli.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `scripts/*.py` entry scripts | Keep deletion. |
| `src/sciqis_qaoa_routing/decoder.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `src/graph.py` bit/edge conversion and validation | Keep deletion. |
| `src/sciqis_qaoa_routing/encoding.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `src/qubo.py` and `src/ising.py` | Keep deletion. |
| `src/sciqis_qaoa_routing/hamiltonian.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `src/qubo.py` and `src/ising.py` | Keep deletion. |
| `src/sciqis_qaoa_routing/instances.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `data/graph.json` and `src/graph.py` | Keep deletion. |
| `src/sciqis_qaoa_routing/metrics.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `src/optimization.py` metric functions | Keep deletion. |
| `src/sciqis_qaoa_routing/mixers.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | Explicit RX mixer in `src/circuit.py` and `src/statevector_reference.py` | Keep deletion. |
| `src/sciqis_qaoa_routing/optimizer.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `src/optimization.py` | Keep deletion. |
| `src/sciqis_qaoa_routing/simulator.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte package scaffold. | `src/statevector_reference.py` and Qiskit Statevector checks | Keep deletion. |
| `tests/__init__.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte marker unnecessary for pytest collection. | `tests/conftest.py` plus focused v3 tests | Keep deletion. |
| `tests/test_decoder.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte test placeholder. | `tests/test_graph.py`, `test_bit_order.py`, and `test_core_metrics.py` | Keep deletion. |
| `tests/test_encoding.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte test placeholder. | `tests/test_qubo.py` and `test_ising.py` | Keep deletion. |
| `tests/test_hamiltonian.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte test placeholder. | `tests/test_qubo.py` and `test_ising.py` | Keep deletion. |
| `tests/test_instances.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte test placeholder. | `tests/test_graph.py` and `test_exact_reference.py` | Keep deletion. |
| `tests/test_mixers.py` | unstaged deletion | LEGACY_SUPERSEDED | Zero-byte test placeholder. | `tests/test_day3_circuit.py`, `test_p2_circuit.py`, and `test_statevector_reference.py` | Keep deletion. |
| `tests/conftest.py` | clean at baseline; found during package audit | LEGACY_SUPERSEDED | It inserted `src/` into `sys.path`, which could mask an incomplete package definition. | Editable installation defined by `pyproject.toml` | Remove after validating installation and imports. |
| `uv.lock` | unstaged deletion | LEGACY_SUPERSEDED | Lock targets Python >=3.13 and the original Qutip/Jupyter scaffold, not the environment used for frozen v3.0. | Release `pyproject.toml`; pip is the documented environment tool | Keep deletion; do not regenerate an unused uv lock. |
| `src/qaoa.py` | untracked | LEGACY_SUPERSEDED | Older NumPy-only prototype uses q0 as the most-significant bit, conflicting with frozen q0-as-LSB. No release artifact imports it. SHA-256: `199e272813c339acce54b20158ff6c502bf5c0ca1800d97d74a7ee5e5f88a30c`. | `src/circuit.py` and `src/statevector_reference.py` | Remove after recording this hash. |
| `src/hamiltonians.py` | untracked | LEGACY_SUPERSEDED | Older 13-qubit/API prototype assumes `graph.EDGES`, `graph.N_QUBITS`, and an unfrozen default penalty. No release artifact imports it. SHA-256: `74d5394f7a029d58fee014d4792413b85d3034d1f112a6936222981595a9aa9b`. | `src/qubo.py` and `src/ising.py` | Remove after recording this hash. |
| `src/metrics.py` | untracked | LEGACY_SUPERSEDED | Imports the obsolete prototypes and old graph API; its metric functionality is already independently tested in v3. SHA-256: `5f8abafdeb3711db8b312f172e635317db131a8648976c18ebfa0325f9845691`. | `src/optimization.py::compute_core_metrics` | Remove after recording this hash. |
| `tests/test_circuit.py` | untracked | LEGACY_SUPERSEDED | Tests a different graph (`C*=8`, effectively 13 qubits) and causes six obsolete-API failures. SHA-256: `1fe52219d4ec0b1c73dd1eb2af2cddca1d1c9e7cd4049b4e5470820777d27e79`. | Focused tracked Day 1–5 test suite | Remove after recording this hash. |

No baseline dirty path remained `UNRELATED_OR_AMBIGUOUS` after content, history,
import, and replacement inspection.

## Proposal decision

The v3.0 proposal supplied with this project is the only proposal aligned with
the frozen implementation. Its source SHA-256 is
`7b3bb824b7f174f31f0c4bddef29c9121ea40f35b5116b94178aa406f60b8bf3`.
It will be retained without content modification as
`docs/DTU_SCIQIS_QAOA_Routing_Project_Proposal_v3.0.docx`.

## Environment decision

The release uses standard `venv` + `pip` installation from `pyproject.toml`.
The stale uv lock is not part of the release contract. Runtime dependencies are
limited to packages imported by the implementation; test/notebook tools are
optional dependency groups. The frozen runtime versions remain recorded in the
profiling artifact, while release bounds permit compatible installations.

## License decision

`LICENSE_DECISION_REQUIRED`: the historical file is empty. No license is added
or guessed during reconciliation; the repository owner must choose one before
granting open-source reuse rights.

## Resolution outcome

- The four untracked legacy files were removed only after their hashes and
  replacements were recorded above. None contained functionality absent from
  the frozen v3.0 modules.
- Both older proposals were removed from the release index/tree. The unmodified
  v3.0 proposal was copied to `docs/` and retained its source SHA-256.
- All zero-byte legacy package, test, script, notebook, data, slide, and plan
  placeholders were removed. Redundant `figures/.gitkeep` and
  `results/.gitkeep` placeholders were also removed after their directories
  became populated.
- The environment was consolidated into `pyproject.toml`; the obsolete Python
  3.13 marker and uv lock were removed. Editable installation and wheel build
  both succeeded, and every `src/*.py` module is declared in the package.
- The test-only `sys.path` injection was removed. Imports work from outside the
  checkout after installation.
- Repository-local pytest caches, bytecode caches, editable-install metadata,
  and IDE metadata were removed. The ignored local `.venv` was left alone as a
  user environment, not release content.

## Validation record

| Gate | Result |
|---|---|
| Scientific hashes before versus after all cheap rebuilds | Byte-identical |
| `python -m pip install -e ".[dev]"` | Passed |
| Wheel construction | Passed |
| Unfiltered `pytest -q` after obsolete-test removal | 96 passed |
| Four numbered notebooks executed with saved-results default | Passed |
| Day 1, Day 2, Day 3, Day 4-figure, and final-report builds | Passed |
| Day-4 expensive optimization rerun | Not run, by design |
| Requested personal-home prefix in README/scripts/notebooks/reports | No occurrences |
| Runnable instructions requiring an absolute personal path | None |
