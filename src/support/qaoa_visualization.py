"""Load saved final-study results and prepare them for the browser animation.

This module only reads files.  The repository class checks the input, rebuilds
the 20-amplitude state after each step, and returns ordinary dictionaries that
the web server can send as JSON.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from support.sealed_results import Q2F_FINAL


METHOD_ORDER = (
    "bsp_path_exchange",
    "gm_qaoa_expectation",
    "gm_th_qaoa",
)


@dataclass(frozen=True)
class MethodDisplay:
    label: str
    short_label: str
    explanation: str


METHOD_DISPLAYS = {
    "bsp_path_exchange": MethodDisplay(
        label="BSP Path-Exchange QAOA",
        short_label="BSP + Path Exchange",
        explanation="Cost phase, path-exchange mixer, optimize probability below the incumbent cost.",
    ),
    "gm_qaoa_expectation": MethodDisplay(
        label="Grover-Mixer QAOA (Expectation)",
        short_label="GM-QAOA Expectation",
        explanation="Cost phase, Grover feasible mixer, optimize normalized expected route cost.",
    ),
    "gm_th_qaoa": MethodDisplay(
        label="Grover-Mixer Threshold QAOA",
        short_label="GM-Th-QAOA",
        explanation="Incumbent-threshold phase, Grover feasible mixer, optimize better-solution probability.",
    ),
}


class VisualizationDataError(RuntimeError):
    """Raised when retained visualization inputs are malformed."""


class QAOAVisualizationRepository:
    """Load, check and reconstruct saved QAOA runs."""

    def __init__(self, result_root: str | Path = Q2F_FINAL.root):
        # Load the shared route basis first, then discover individual runs.
        self.result_root = Path(result_root).resolve()
        self.raw_root = self.result_root / "raw"
        self._basis_payload = self._load_json(self.result_root / "basis.json")
        self._validate_basis()
        self._run_paths = self._discover_runs()

        basis = self._basis_payload["basis"]
        costs = self._basis_payload["cost_hamiltonian"]
        self._routes = tuple(basis["routes"])
        self._raw_costs = np.asarray(costs["raw_energies"], dtype=np.float64)
        self._normalized_costs = np.asarray(
            costs["normalized_energies"], dtype=np.float64
        )
        self._normalization_shift = float(costs["normalization_shift"])
        self._normalization_scale = float(costs["normalization_scale"])
        self._better_mask = np.asarray(
            self._basis_payload["incumbent_threshold"]["better_mask"], dtype=bool
        )
        self._optimal_mask = np.asarray(
            [bool(route["exact_optimal"]) for route in self._routes], dtype=bool
        )

        path_hamiltonian = np.asarray(
            self._basis_payload["path_exchange_mixer"]["hamiltonian"],
            dtype=np.complex128,
        )
        self._path_eigenvalues, self._path_eigenvectors = np.linalg.eigh(
            path_hamiltonian
        )
        self._uniform_state = np.asarray(
            self._basis_payload["grover_mixer"]["uniform_state_real"],
            dtype=np.complex128,
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VisualizationDataError(f"cannot_load_visualization_input:{path}") from exc
        if not isinstance(payload, dict):
            raise VisualizationDataError(f"visualization_input_is_not_an_object:{path}")
        return payload

    def _validate_basis(self) -> None:
        required = {
            "basis",
            "cost_hamiltonian",
            "grover_mixer",
            "incumbent_biased_initial_state",
            "incumbent_threshold",
            "path_exchange_mixer",
            "uniform_initial_state",
        }
        if not required <= self._basis_payload.keys():
            raise VisualizationDataError("visualization_basis_missing_sections")
        basis = self._basis_payload["basis"]
        route_count = int(basis.get("basis_size", 0))
        routes = basis.get("routes", [])
        costs = self._basis_payload["cost_hamiltonian"]
        if route_count != 20 or len(routes) != route_count:
            raise VisualizationDataError("visualization_basis_size_mismatch")
        if len(costs.get("raw_energies", [])) != route_count:
            raise VisualizationDataError("visualization_cost_size_mismatch")

    def _discover_runs(self) -> dict[str, Path]:
        if not self.raw_root.is_dir():
            raise VisualizationDataError(f"visualization_raw_root_missing:{self.raw_root}")
        paths: dict[str, Path] = {}
        for path in sorted(self.raw_root.glob("*.json")):
            payload = self._load_json(path)
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            run_id = result.get("run_id")
            method = result.get("method")
            if not isinstance(run_id, str) or method not in METHOD_DISPLAYS:
                continue
            if run_id != path.stem or run_id in paths:
                raise VisualizationDataError(f"visualization_run_identity_mismatch:{path}")
            paths[run_id] = path
        if not paths:
            raise VisualizationDataError("no_visualization_runs_found")
        return paths

    def catalog(self) -> dict[str, Any]:
        """Return the available method/depth/seed combinations."""

        runs: list[dict[str, Any]] = []
        for run_id, path in self._run_paths.items():
            result = self._load_json(path)["result"]
            method = str(result["method"])
            display = METHOD_DISPLAYS[method]
            runs.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "method_label": display.label,
                    "method_short_label": display.short_label,
                    "depth": int(result["depth"]),
                    "seed": int(result["seed"]),
                    "evaluations": int(result["optimizer"]["evaluations"]),
                    "final_expected_cost": float(
                        result["metrics"]["expected_route_cost"]
                    ),
                    "final_p_opt": float(result["metrics"]["p_opt"]),
                }
            )
        order = {method: index for index, method in enumerate(METHOD_ORDER)}
        runs.sort(key=lambda item: (order[item["method"]], item["depth"], item["seed"]))
        return {
            "schema": "dtu-sciqis-qaoa-visualization-catalog",
            "version": "1.0",
            "result_identity": Q2F_FINAL.identity,
            "read_only_replay": True,
            "method_count": len({item["method"] for item in runs}),
            "run_count": len(runs),
            "runs": runs,
        }
    def load_run(self, run_id: str) -> dict[str, Any]:
        """Return browser-ready optimizer and per-operator animation frames."""

        try:
            path = self._run_paths[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown_visualization_run:{run_id}") from exc
        result = self._load_json(path)["result"]
        method = str(result["method"])
        display = METHOD_DISPLAYS[method]
        depth = int(result["depth"])
        if depth not in (1, 2, 3, 4):
            raise VisualizationDataError(f"invalid_visualization_depth:{depth}")

        frames = self._build_frames(result)
        circuit = self._circuit_description(result)
        metrics = result["metrics"]
        return {
            "schema": "dtu-sciqis-qaoa-animation",
            "version": "1.0",
            "source": {
                "kind": "sealed_optimizer_trace",
                "result_identity": Q2F_FINAL.identity,
                "read_only": True,
            },
            "run_id": run_id,
            "method": method,
            "method_label": display.label,
            "method_short_label": display.short_label,
            "method_explanation": display.explanation,
            "depth": depth,
            "seed": int(result["seed"]),
            "circuit": circuit,
            "energy": {
                "label": "Expected route cost",
                "symbol": "<C>",
                "minimum_basis_cost": float(np.min(self._raw_costs)),
                "maximum_basis_cost": float(np.max(self._raw_costs)),
                "normalization_shift": self._normalization_shift,
                "normalization_scale": self._normalization_scale,
            },
            "routes": [
                {
                    "route_id": int(route["route_id"]),
                    "node_sequence": list(route["node_sequence"]),
                    "routing_cost": float(route["routing_cost"]),
                    "exact_optimal": bool(route["exact_optimal"]),
                }
                for route in self._routes
            ],
            "frames": frames,
            "final": {
                "parameters": list(map(float, result["optimizer"]["final_parameters"])),
                "expected_cost": float(metrics["expected_route_cost"]),
                "expected_normalized_cost": float(metrics["expected_normalized_cost"]),
                "p_opt": float(metrics["p_opt"]),
                "p_feas": float(metrics["p_feas"]),
                "bsp": float(metrics["bsp"]),
                "most_probable_route": list(metrics["most_probable_route"]),
                "most_probable_route_cost": float(metrics["most_probable_route_cost"]),
                "termination": str(result["optimizer"]["reason"]),
            },
        }

    def _circuit_description(self, result: dict[str, Any]) -> dict[str, Any]:
        phase_kind = str(result["phase_kind"])
        mixer_kind = str(result["mixer_kind"])
        initialization_mode = str(result["initialization_mode"])
        phase = (
            {
                "symbol": "U_T",
                "label": "Threshold phase",
                "formula": "exp(-i gamma h_T)",
            }
            if phase_kind == "strict_incumbent_threshold_phase"
            else {
                "symbol": "U_C",
                "label": "Cost phase",
                "formula": "exp(-i gamma H_C)",
            }
        )
        mixer = (
            {
                "symbol": "U_G",
                "label": "Grover feasible mixer",
                "formula": "exp(-i beta |F><F|)",
            }
            if mixer_kind == "logical_grover_feasible_mixer"
            else {
                "symbol": "U_PE",
                "label": "Path-exchange mixer",
                "formula": "exp(-i beta H_PE)",
            }
        )
        initial = (
            {"symbol": "|F>", "label": "Uniform feasible state"}
            if initialization_mode == "uniform_feasible"
            else {"symbol": "|psi_inc>", "label": "Incumbent-biased feasible state"}
        )
        return {
            "representation": "20-state logical feasible-route register",
            "hardware_gate_decomposition": False,
            "initial": initial,
            "phase": phase,
            "mixer": mixer,
            "measurement": {"symbol": "M", "label": "Route measurement"},
            "parameter_order": "all gammas, then all betas",
            "layer_order": "phase, then mixer",
        }

    def _build_frames(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        depth = int(result["depth"])
        trace = result.get("optimizer", {}).get("evaluation_trace", [])
        if not isinstance(trace, list) or not trace:
            raise VisualizationDataError("visualization_trace_missing")
        frames: list[dict[str, Any]] = []
        best_loss = float("inf")
        for record in trace:
            parameters = np.asarray(record.get("parameters", []), dtype=np.float64)
            if parameters.shape != (2 * depth,) or np.any(~np.isfinite(parameters)):
                raise VisualizationDataError("visualization_parameter_shape_mismatch")
            in_bounds = bool(record.get("in_bounds"))
            loss = float(record["loss"])
            stages = self._simulate_stages(result, parameters) if in_bounds else []
            expected_normalized = record.get("expected_normalized_cost")
            expected_cost: float | None = None
            if expected_normalized is not None:
                expected_cost = self._normalization_shift + self._normalization_scale * float(
                    expected_normalized
                )
            if stages and expected_cost is not None:
                reconstructed = float(stages[-1]["expected_cost"])
                if not np.isclose(reconstructed, expected_cost, atol=1e-9, rtol=0.0):
                    raise VisualizationDataError(
                        f"visualization_energy_reconstruction_mismatch:{reconstructed}:{expected_cost}"
                    )
            improved = bool(in_bounds and loss < best_loss)
            if improved:
                best_loss = loss
            frames.append(
                {
                    "evaluation": int(record["evaluation_index"]),
                    "in_bounds": in_bounds,
                    "gammas": list(map(float, parameters[:depth])),
                    "betas": list(map(float, parameters[depth:])),
                    "loss": loss,
                    "bsp": None if record.get("bsp") is None else float(record["bsp"]),
                    "p_feas": (
                        None
                        if record.get("p_feas") is None
                        else float(record["p_feas"])
                    ),
                    "expected_normalized_cost": (
                        None
                        if expected_normalized is None
                        else float(expected_normalized)
                    ),
                    "expected_cost": expected_cost,
                    "best_objective_so_far": improved,
                    "stages": stages,
                }
            )
        return frames

    def _simulate_stages(
        self, result: dict[str, Any], parameters: np.ndarray
    ) -> list[dict[str, Any]]:
        depth = int(result["depth"])
        initialization_mode = str(result["initialization_mode"])
        initial_key = (
            "uniform_initial_state"
            if initialization_mode == "uniform_feasible"
            else "incumbent_biased_initial_state"
        )
        initial_probabilities = np.asarray(
            self._basis_payload[initial_key]["probabilities"], dtype=np.float64
        )
        state = np.sqrt(initial_probabilities).astype(np.complex128)
        if str(result["phase_kind"]) == "strict_incumbent_threshold_phase":
            phase_values = self._better_mask.astype(np.float64)
        else:
            phase_values = self._normalized_costs

        gammas, betas = parameters[:depth], parameters[depth:]
        stages = [self._stage_payload(state, kind="initial", layer=0, angle=None)]
        for layer, (gamma, beta) in enumerate(zip(gammas, betas), start=1):
            state = state * np.exp(-1j * float(gamma) * phase_values)
            stages.append(
                self._stage_payload(
                    state, kind="phase", layer=layer, angle=float(gamma)
                )
            )
            if str(result["mixer_kind"]) == "logical_grover_feasible_mixer":
                overlap = np.vdot(self._uniform_state, state)
                state = state + (np.exp(-1j * float(beta)) - 1.0) * self._uniform_state * overlap
            else:
                coefficients = self._path_eigenvectors.conj().T @ state
                phases = np.exp(-1j * float(beta) * self._path_eigenvalues)
                state = self._path_eigenvectors @ (phases * coefficients)
            stages.append(
                self._stage_payload(
                    state, kind="mixer", layer=layer, angle=float(beta)
                )
            )
        return stages

    def _stage_payload(
        self,
        state: np.ndarray,
        *,
        kind: str,
        layer: int,
        angle: float | None,
    ) -> dict[str, Any]:
        probabilities = np.abs(state) ** 2
        probabilities /= float(np.sum(probabilities))
        most_probable = int(np.argmax(probabilities))
        return {
            "kind": kind,
            "layer": int(layer),
            "angle": angle,
            "expected_cost": float(probabilities @ self._raw_costs),
            "expected_normalized_cost": float(probabilities @ self._normalized_costs),
            "bsp": float(np.sum(probabilities[self._better_mask])),
            "p_opt": float(np.sum(probabilities[self._optimal_mask])),
            "most_probable_route_id": most_probable,
            "probabilities": list(map(float, probabilities)),
        }
