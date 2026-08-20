"""Build a compact, self-contained QAOA demo for the browser visualizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from feasible_experiments import build_grover_feasible_mixer
from feasible_qaoa import (
    INCUMBENT_BIASED_FEASIBLE,
    UNIFORM_FEASIBLE,
    build_feasible_initial_state,
    build_feasible_route_basis,
    build_logical_cost_hamiltonian,
    build_logical_path_exchange_mixer,
)
from graph import load_graph


METHOD_ORDER = ("bsp_path_exchange", "gm_qaoa_expectation", "gm_th_qaoa")
METHOD_DISPLAYS = {
    "bsp_path_exchange": {
        "label": "BSP Path-Exchange QAOA",
        "short": "BSP + Path Exchange",
        "explanation": "Cost phase, path-exchange mixer, and better-solution probability.",
    },
    "gm_qaoa_expectation": {
        "label": "Grover-Mixer QAOA (Expectation)",
        "short": "GM-QAOA Expectation",
        "explanation": "Cost phase, Grover feasible mixer, and normalized expected route cost.",
    },
    "gm_th_qaoa": {
        "label": "Grover-Mixer Threshold QAOA",
        "short": "GM-Th-QAOA",
        "explanation": "Incumbent-threshold phase, Grover feasible mixer, and better-solution probability.",
    },
}

# These deterministic angle sequences are enough to demonstrate all three
# circuits. Every state and metric is rebuilt from the tracked core code.
DEMO_RUNS = (
    {
        "run_id": "bsp_path_exchange_p1_demo",
        "method": "bsp_path_exchange",
        "depth": 1,
        "seed": 2601,
        "parameters": (0.5904949711373307, 1.814207376939347),
    },
    {
        "run_id": "gm_qaoa_expectation_p2_demo",
        "method": "gm_qaoa_expectation",
        "depth": 2,
        "seed": 2601,
        "parameters": (
            3.0950079456075264e-09,
            5.673086608771164,
            2.461793769694505,
            0.2509334405358042,
        ),
    },
    {
        "run_id": "gm_th_qaoa_p3_demo",
        "method": "gm_th_qaoa",
        "depth": 3,
        "seed": 2601,
        "parameters": (
            2.9806517181260643,
            2.8789040050030557,
            2.940387676195214,
            2.8489898303512913,
            3.1031165823822953,
            2.6411237545732305,
        ),
    },
)


class VisualizationDataError(RuntimeError):
    """Raised when the built-in demonstration cannot be constructed."""


class QAOAVisualizationRepository:
    """Build browser-ready runs directly from the graph and QAOA primitives."""

    def __init__(self) -> None:
        try:
            self.basis = build_feasible_route_basis(load_graph())
            self.costs = build_logical_cost_hamiltonian(self.basis)
            self.path_mixer = build_logical_path_exchange_mixer(self.basis)
            self.grover_mixer = build_grover_feasible_mixer(self.basis.size)
            self.uniform_initial = build_feasible_initial_state(
                self.basis, self.costs, mode=UNIFORM_FEASIBLE
            )
            self.biased_initial = build_feasible_initial_state(
                self.basis,
                self.costs,
                mode=INCUMBENT_BIASED_FEASIBLE,
                bias_lambda=1.0,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise VisualizationDataError("cannot build the visualizer demo") from error

        self.raw_costs = np.asarray(self.costs.raw_energies, dtype=float)
        self.normalized_costs = np.asarray(
            self.costs.normalized_energies, dtype=float
        )
        incumbent_cost = self.raw_costs[self.basis.incumbent_route_id]
        self.better_mask = self.raw_costs < incumbent_cost
        self.optimal_mask = np.asarray(
            [route.exact_optimal for route in self.basis.routes], dtype=bool
        )
        self.runs = {str(run["run_id"]): run for run in DEMO_RUNS}
        self.payloads = {
            run_id: self._build_run(spec) for run_id, spec in self.runs.items()
        }

    def catalog(self) -> dict[str, Any]:
        """List the three small built-in method demonstrations."""

        rows = []
        for run_id, result in self.payloads.items():
            display = METHOD_DISPLAYS[str(result["method"])]
            rows.append(
                {
                    "run_id": run_id,
                    "method": result["method"],
                    "method_label": display["label"],
                    "method_short_label": display["short"],
                    "depth": result["depth"],
                    "seed": result["seed"],
                    "evaluations": len(result["frames"]),
                    "final_expected_cost": result["final"]["expected_cost"],
                    "final_p_opt": result["final"]["p_opt"],
                }
            )
        method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
        rows.sort(key=lambda row: method_index[str(row["method"])])
        return {
            "schema": "dtu-sciqis-qaoa-visualization-catalog",
            "version": "1.0",
            "result_identity": "built-in-core-demo",
            "read_only_replay": True,
            "method_count": len(rows),
            "run_count": len(rows),
            "runs": rows,
        }

    def load_run(self, run_id: str) -> dict[str, Any]:
        """Return one fully reconstructed demonstration run."""

        if run_id not in self.payloads:
            raise KeyError(f"unknown_visualization_run:{run_id}")
        return self.payloads[run_id]

    def _build_run(self, spec: dict[str, object]) -> dict[str, Any]:
        method = str(spec["method"])
        depth = int(spec["depth"])
        final_parameters = np.asarray(spec["parameters"], dtype=float)
        parameter_rows = (
            np.zeros(2 * depth, dtype=float),
            0.5 * final_parameters,
            final_parameters,
        )
        frames = []
        best_loss = float("inf")
        for evaluation, parameters in enumerate(parameter_rows, start=1):
            stages = self._simulate(method, depth, parameters)
            final_stage = stages[-1]
            loss = (
                final_stage["expected_normalized_cost"]
                if method == "gm_qaoa_expectation"
                else -final_stage["bsp"]
            )
            improved = float(loss) < best_loss
            if improved:
                best_loss = float(loss)
            frames.append(
                {
                    "evaluation": evaluation,
                    "in_bounds": True,
                    "gammas": list(map(float, parameters[:depth])),
                    "betas": list(map(float, parameters[depth:])),
                    "loss": float(loss),
                    "bsp": final_stage["bsp"],
                    "p_feas": 1.0,
                    "expected_normalized_cost": final_stage[
                        "expected_normalized_cost"
                    ],
                    "expected_cost": final_stage["expected_cost"],
                    "best_objective_so_far": improved,
                    "stages": stages,
                }
            )

        final_stage = frames[-1]["stages"][-1]
        probabilities = np.asarray(final_stage["probabilities"])
        most_probable_id = int(np.argmax(probabilities))
        most_probable = self.basis.routes[most_probable_id]
        display = METHOD_DISPLAYS[method]
        return {
            "schema": "dtu-sciqis-qaoa-animation",
            "version": "1.0",
            "source": {
                "kind": "built_in_example_trajectory",
                "result_identity": "built-in-core-demo",
                "read_only": True,
            },
            "run_id": spec["run_id"],
            "method": method,
            "method_label": display["label"],
            "method_short_label": display["short"],
            "method_explanation": display["explanation"],
            "depth": depth,
            "seed": int(spec["seed"]),
            "circuit": self._circuit(method, depth),
            "energy": {
                "label": "Expected route cost",
                "symbol": "<C>",
                "minimum_basis_cost": float(self.raw_costs.min()),
                "maximum_basis_cost": float(self.raw_costs.max()),
                "normalization_shift": self.costs.normalization_shift,
                "normalization_scale": self.costs.normalization_scale,
            },
            "routes": [
                {
                    "route_id": route.route_id,
                    "node_sequence": list(route.node_sequence),
                    "routing_cost": float(route.routing_cost),
                    "exact_optimal": route.exact_optimal,
                }
                for route in self.basis.routes
            ],
            "frames": frames,
            "final": {
                "parameters": list(map(float, final_parameters)),
                "expected_cost": final_stage["expected_cost"],
                "expected_normalized_cost": final_stage[
                    "expected_normalized_cost"
                ],
                "p_opt": final_stage["p_opt"],
                "p_feas": 1.0,
                "bsp": final_stage["bsp"],
                "most_probable_route": list(most_probable.node_sequence),
                "most_probable_route_cost": float(most_probable.routing_cost),
                "termination": "built_in_demo_parameters",
            },
        }

    @staticmethod
    def _circuit(method: str, depth: int) -> dict[str, Any]:
        threshold_phase = method == "gm_th_qaoa"
        grover_mixer = method != "bsp_path_exchange"
        uniform = method != "bsp_path_exchange"
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
                else {
                    "symbol": "|psi_inc>",
                    "label": "Incumbent-biased feasible state",
                }
            ),
            "phase": (
                {
                    "symbol": "U_T",
                    "label": "Threshold phase",
                    "formula": "exp(-i gamma h_T)",
                }
                if threshold_phase
                else {
                    "symbol": "U_C",
                    "label": "Cost phase",
                    "formula": "exp(-i gamma H_C)",
                }
            ),
            "mixer": (
                {
                    "symbol": "U_G",
                    "label": "Grover feasible mixer",
                    "formula": "exp(-i beta |F><F|)",
                }
                if grover_mixer
                else {
                    "symbol": "U_PE",
                    "label": "Path-exchange mixer",
                    "formula": "exp(-i beta H_PE)",
                }
            ),
            "measurement": {"symbol": "M", "label": "Route measurement"},
            "parameter_order": "all gammas, then all betas",
            "layer_order": "phase, then mixer",
        }

    def _simulate(
        self, method: str, depth: int, parameters: np.ndarray
    ) -> list[dict[str, Any]]:
        initial = (
            self.biased_initial
            if method == "bsp_path_exchange"
            else self.uniform_initial
        )
        state = np.asarray(initial.amplitudes, dtype=complex).copy()
        phase = (
            self.better_mask.astype(float)
            if method == "gm_th_qaoa"
            else self.normalized_costs
        )
        stages = [self._stage(state, "initial", 0, None)]
        for layer, (gamma, beta) in enumerate(
            zip(parameters[:depth], parameters[depth:]), start=1
        ):
            state *= np.exp(-1j * gamma * phase)
            stages.append(self._stage(state, "phase", layer, float(gamma)))
            mixer = (
                self.path_mixer
                if method == "bsp_path_exchange"
                else self.grover_mixer
            )
            state = mixer.evolve(state, float(beta))
            stages.append(self._stage(state, "mixer", layer, float(beta)))
        return stages

    def _stage(
        self, state: np.ndarray, kind: str, layer: int, angle: float | None
    ) -> dict[str, Any]:
        probabilities = np.abs(state) ** 2
        probabilities /= probabilities.sum()
        return {
            "kind": kind,
            "layer": layer,
            "angle": angle,
            "expected_cost": float(probabilities @ self.raw_costs),
            "expected_normalized_cost": float(
                probabilities @ self.normalized_costs
            ),
            "bsp": float(probabilities[self.better_mask].sum()),
            "p_opt": float(probabilities[self.optimal_mask].sum()),
            "most_probable_route_id": int(np.argmax(probabilities)),
            "probabilities": list(map(float, probabilities)),
        }
