"""Prepare saved feasible-route QAOA runs for the browser animation."""

import json
from pathlib import Path
from typing import Any

import numpy as np

METHOD_ORDER = ("bsp_path_exchange", "gm_qaoa_expectation", "gm_th_qaoa")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "results/q2f_final_improvement"
    / "q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1"
)
METHOD_DISPLAYS = {
    "bsp_path_exchange": {
        "label": "BSP Path-Exchange QAOA",
        "short": "BSP + Path Exchange",
        "explanation": "Cost phase, path-exchange mixer, optimize probability below the incumbent cost.",
    },
    "gm_qaoa_expectation": {
        "label": "Grover-Mixer QAOA (Expectation)",
        "short": "GM-QAOA Expectation",
        "explanation": "Cost phase, Grover feasible mixer, optimize normalized expected route cost.",
    },
    "gm_th_qaoa": {
        "label": "Grover-Mixer Threshold QAOA",
        "short": "GM-Th-QAOA",
        "explanation": "Incumbent-threshold phase, Grover feasible mixer, optimize better-solution probability.",
    },
}


class VisualizationDataError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VisualizationDataError(f"cannot load {path}") from error
    if not isinstance(data, dict):
        raise VisualizationDataError(f"expected a JSON object in {path}")
    return data


class QAOAVisualizationRepository:
    """Small read-only adapter between saved JSON files and the web page."""

    def __init__(self, result_root: str | Path = DEFAULT_RESULT_ROOT):
        self.result_root = Path(result_root).resolve()
        self.basis_data = _read_json(self.result_root / "basis.json")
        basis = self.basis_data["basis"]
        costs = self.basis_data["cost_hamiltonian"]
        self.routes = tuple(basis["routes"])
        if len(self.routes) != 20:
            raise VisualizationDataError("expected 20 feasible routes")

        self.raw_costs = np.asarray(costs["raw_energies"], dtype=float)
        self.normalized_costs = np.asarray(costs["normalized_energies"], dtype=float)
        self.shift = float(costs["normalization_shift"])
        self.scale = float(costs["normalization_scale"])
        self.better_mask = np.asarray(
            self.basis_data["incumbent_threshold"]["better_mask"], dtype=bool
        )
        self.optimal_mask = np.asarray(
            [route["exact_optimal"] for route in self.routes], dtype=bool
        )

        path_hamiltonian = np.asarray(
            self.basis_data["path_exchange_mixer"]["hamiltonian"], dtype=complex
        )
        self.path_eigenvalues, self.path_eigenvectors = np.linalg.eigh(path_hamiltonian)
        self.uniform_state = np.asarray(
            self.basis_data["grover_mixer"]["uniform_state_real"], dtype=complex
        )

        self.runs = {}
        for path in sorted((self.result_root / "raw").glob("*.json")):
            result = _read_json(path).get("result")
            if not isinstance(result, dict) or result.get("method") not in METHOD_ORDER:
                continue
            run_id = result.get("run_id")
            if run_id != path.stem or run_id in self.runs:
                raise VisualizationDataError(f"invalid run identity: {path}")
            self.runs[run_id] = result
        if not self.runs:
            raise VisualizationDataError("no saved visualization runs found")

    def catalog(self) -> dict[str, Any]:
        """List the available method, depth and seed combinations."""

        rows = []
        for run_id, result in self.runs.items():
            display = METHOD_DISPLAYS[result["method"]]
            rows.append(
                {
                    "run_id": run_id,
                    "method": result["method"],
                    "method_label": display["label"],
                    "method_short_label": display["short"],
                    "depth": int(result["depth"]),
                    "seed": int(result["seed"]),
                    "evaluations": int(result["optimizer"]["evaluations"]),
                    "final_expected_cost": float(result["metrics"]["expected_route_cost"]),
                    "final_p_opt": float(result["metrics"]["p_opt"]),
                }
            )
        method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
        rows.sort(key=lambda row: (method_index[row["method"]], row["depth"], row["seed"]))
        return {
            "schema": "dtu-sciqis-qaoa-visualization-catalog",
            "version": "1.0",
            "result_identity": self.result_root.name,
            "read_only_replay": True,
            "method_count": len({row["method"] for row in rows}),
            "run_count": len(rows),
            "runs": rows,
        }

    def load_run(self, run_id: str) -> dict[str, Any]:
        """Build optimizer frames and circuit labels for one saved run."""

        if run_id not in self.runs:
            raise KeyError(f"unknown_visualization_run:{run_id}")
        result = self.runs[run_id]
        method = result["method"]
        display = METHOD_DISPLAYS[method]
        metrics = result["metrics"]
        return {
            "schema": "dtu-sciqis-qaoa-animation",
            "version": "1.0",
            "source": {
                "kind": "sealed_optimizer_trace",
                "result_identity": self.result_root.name,
                "read_only": True,
            },
            "run_id": run_id,
            "method": method,
            "method_label": display["label"],
            "method_short_label": display["short"],
            "method_explanation": display["explanation"],
            "depth": int(result["depth"]),
            "seed": int(result["seed"]),
            "circuit": self._circuit(result),
            "energy": {
                "label": "Expected route cost",
                "symbol": "<C>",
                "minimum_basis_cost": float(self.raw_costs.min()),
                "maximum_basis_cost": float(self.raw_costs.max()),
                "normalization_shift": self.shift,
                "normalization_scale": self.scale,
            },
            "routes": [
                {
                    "route_id": int(route["route_id"]),
                    "node_sequence": list(route["node_sequence"]),
                    "routing_cost": float(route["routing_cost"]),
                    "exact_optimal": bool(route["exact_optimal"]),
                }
                for route in self.routes
            ],
            "frames": self._frames(result),
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

    @staticmethod
    def _circuit(result):
        depth = int(result["depth"])
        threshold_phase = result["phase_kind"] == "strict_incumbent_threshold_phase"
        grover_mixer = result["mixer_kind"] == "logical_grover_feasible_mixer"
        uniform = result["initialization_mode"] == "uniform_feasible"
        return {
            "representation": "20-state logical feasible-route register",
            "hardware_gate_decomposition": False,
            "resources": {
                "logical_states": 20,
                "qaoa_layers": depth,
                "variational_parameters": 2 * depth,
                "operator_blocks": 2 * depth,
                "measurement_outcomes": 20,
            },
            "initial": (
                {"symbol": "|F>", "label": "Uniform feasible state"}
                if uniform
                else {"symbol": "|psi_inc>", "label": "Incumbent-biased feasible state"}
            ),
            "phase": (
                {"symbol": "U_T", "label": "Threshold phase", "formula": "exp(-i gamma h_T)"}
                if threshold_phase
                else {"symbol": "U_C", "label": "Cost phase", "formula": "exp(-i gamma H_C)"}
            ),
            "mixer": (
                {"symbol": "U_G", "label": "Grover feasible mixer", "formula": "exp(-i beta |F><F|)"}
                if grover_mixer
                else {"symbol": "U_PE", "label": "Path-exchange mixer", "formula": "exp(-i beta H_PE)"}
            ),
            "measurement": {"symbol": "M", "label": "Route measurement"},
            "parameter_order": "all gammas, then all betas",
            "layer_order": "phase, then mixer",
        }

    def _frames(self, result):
        depth = int(result["depth"])
        frames = []
        best_loss = float("inf")
        for record in result["optimizer"]["evaluation_trace"]:
            parameters = np.asarray(record["parameters"], dtype=float)
            in_bounds = bool(record["in_bounds"])
            loss = float(record["loss"])
            stages = self._simulate(result, parameters) if in_bounds else []
            normalized = record.get("expected_normalized_cost")
            expected_cost = None if normalized is None else self.shift + self.scale * float(normalized)
            improved = in_bounds and loss < best_loss
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
                    "p_feas": None if record.get("p_feas") is None else float(record["p_feas"]),
                    "expected_normalized_cost": None if normalized is None else float(normalized),
                    "expected_cost": expected_cost,
                    "best_objective_so_far": bool(improved),
                    "stages": stages,
                }
            )
        return frames

    def _simulate(self, result, parameters):
        depth = int(result["depth"])
        initial_key = (
            "uniform_initial_state"
            if result["initialization_mode"] == "uniform_feasible"
            else "incumbent_biased_initial_state"
        )
        probabilities = np.asarray(self.basis_data[initial_key]["probabilities"])
        state = np.sqrt(probabilities).astype(complex)
        phase = (
            self.better_mask.astype(float)
            if result["phase_kind"] == "strict_incumbent_threshold_phase"
            else self.normalized_costs
        )
        stages = [self._stage(state, "initial", 0, None)]
        for layer, (gamma, beta) in enumerate(
            zip(parameters[:depth], parameters[depth:]), start=1
        ):
            state *= np.exp(-1j * gamma * phase)
            stages.append(self._stage(state, "phase", layer, float(gamma)))
            if result["mixer_kind"] == "logical_grover_feasible_mixer":
                overlap = np.vdot(self.uniform_state, state)
                state += (np.exp(-1j * beta) - 1) * self.uniform_state * overlap
            else:
                coefficients = self.path_eigenvectors.conj().T @ state
                phases = np.exp(-1j * beta * self.path_eigenvalues)
                state = self.path_eigenvectors @ (phases * coefficients)
            stages.append(self._stage(state, "mixer", layer, float(beta)))
        return stages

    def _stage(self, state, kind, layer, angle):
        probabilities = np.abs(state) ** 2
        probabilities /= probabilities.sum()
        return {
            "kind": kind,
            "layer": layer,
            "angle": angle,
            "expected_cost": float(probabilities @ self.raw_costs),
            "expected_normalized_cost": float(probabilities @ self.normalized_costs),
            "bsp": float(probabilities[self.better_mask].sum()),
            "p_opt": float(probabilities[self.optimal_mask].sum()),
            "most_probable_route_id": int(np.argmax(probabilities)),
            "probabilities": list(map(float, probabilities)),
        }
