"""Read-only, layer-resolved replay of the frozen global depth experiment.

The module never calls an optimizer.  It reads the optimized angles stored in
``results/global_depth110/checkpoints`` and deterministically replays the exact
cost and mixer operations used by that experiment.  Browser payloads are kept
small: summaries contain scalar trajectories, while one checkpoint endpoint
returns only the requested state rows plus compact aggregate views.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from experiments.dynamics_trace import BasisMetadata, EvolutionTrace, trace_qaoa_evolution
from graph import DEFAULT_GRAPH_PATH, EXPECTED_EDGE_COUNT, get_edge_order, load_graph
from metrics import distribution_metrics, probability_mass
from qaoa import (
    X_MIXER,
    apply_grover_mixer,
    apply_x_mixer,
    initial_state,
    qaoa_state,
    simulate_global_grover_state,
)
from qubo import build_qubo, enumerate_state_space, qubo_to_ising


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results/global_depth110"
ALGORITHMS = ("penalty_x", "global_grover")
DISPLAY_NAMES = {
    "penalty_x": "Penalty-X",
    "global_grover": "Global-Grover",
}
STAGES = ("before_cost", "after_cost", "after_mixer")
TOLERANCE = 1e-10
MAX_TOP_STATES = 200
DEFAULT_TOP_STATES = 20
REPLAY_CACHE_SIZE = 2


class EvolutionMicroscopeDataError(RuntimeError):
    """Raised when a frozen artifact or a scientific invariant is invalid."""


@dataclass(frozen=True)
class EvolutionKey:
    algorithm: str
    depth: int

    @property
    def artifact_id(self) -> str:
        return f"{self.algorithm}_p{self.depth:03d}"


@dataclass(frozen=True)
class ReplayBundle:
    key: EvolutionKey
    trace: EvolutionTrace
    artifact: dict[str, Any]
    validation: dict[str, Any]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvolutionMicroscopeDataError(f"cannot_load_json:{path}") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise EvolutionMicroscopeDataError(f"cannot_load_csv:{path}") from error


def _float(value: Fraction | float | int) -> float:
    return float(value.numerator / value.denominator) if isinstance(value, Fraction) else float(value)


def _wrapped_phase_delta(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.angle(after) - np.angle(before))))


def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


class QAOAEvolutionMicroscopeRepository:
    """Validated view over the frozen p=1..110 experiment and exact replays."""

    def __init__(self, result_root: str | Path = DEFAULT_RESULT_ROOT):
        self.result_root = Path(result_root).resolve()
        if not self.result_root.is_dir():
            raise EvolutionMicroscopeDataError(f"missing_result_root:{self.result_root}")

        self.config = _read_json(self.result_root / "experiment_config.json")
        self.saved_validation = _read_json(self.result_root / "validation_summary.json")
        self.reference = _read_json(self.result_root / "reference_solution.json")
        self.optimized_parameters = _read_json(self.result_root / "optimized_parameters.json")
        self.depth_rows = _read_csv(self.result_root / "depth_by_depth.csv")

        self.graph = load_graph(DEFAULT_GRAPH_PATH)
        self.edge_order = get_edge_order(self.graph)
        self.states = enumerate_state_space(self.graph)
        self.qubo = build_qubo(self.graph, self.config["penalty_coefficient"])
        self.ising = qubo_to_ising(self.qubo)

        with np.load(self.result_root / "reference_arrays.npz") as arrays:
            self.energies = np.asarray(arrays["energies"], dtype=np.float64).copy()
            self.route_costs = np.asarray(arrays["route_cost"], dtype=np.float64).copy()
            self.flow_penalties = np.asarray(arrays["penalty_cost"], dtype=np.float64).copy()
            self.feasible = np.asarray(arrays["is_feasible"], dtype=bool).copy()
            self.optimal = np.asarray(arrays["is_optimal"], dtype=bool).copy()

        self.bit_matrix = (
            (np.arange(2**EXPECTED_EDGE_COUNT, dtype=np.uint16)[:, None]
             >> np.arange(EXPECTED_EDGE_COUNT, dtype=np.uint16)[None, :])
            & 1
        ).astype(np.float64)
        self.energy_values, self.energy_inverse = np.unique(
            self.energies, return_inverse=True
        )
        self.optimal_index = int(np.flatnonzero(self.optimal)[0])
        self.low_energy_infeasible_index = int(
            np.flatnonzero((~self.feasible) & (self.energies == self.energies[~self.feasible].min()))[0]
        )
        feasible_nonoptimal = np.flatnonzero(self.feasible & ~self.optimal)
        self.best_nonoptimal_feasible_index = int(
            feasible_nonoptimal[np.lexsort((feasible_nonoptimal, self.energies[feasible_nonoptimal]))[0]]
        )

        self.metadata = BasisMetadata(
            basis_labels=tuple(state.canonical_bitstring for state in self.states),
            decoded_routes=tuple(state.decoded_route for state in self.states),
            # The trace API accepts sequences. Keeping the frozen numeric arrays
            # as arrays avoids rebuilding five 16,384-entry arrays at every one
            # of the 221 p=110 checkpoints.
            route_costs=self.route_costs,
            flow_penalties=self.flow_penalties,
            total_energies=self.energies,
            feasible_mask=self.feasible,
            optimal_mask=self.optimal,
            penalty_coefficient=float(self.config["penalty_coefficient"]),
            representation="edge_bits_all_bitstrings",
        )
        self.graph_payload = self._graph_payload()
        self.hamiltonian_payload = self._hamiltonian_payload()
        self.depth_trajectories = self._depth_trajectories()
        self.static_validation = self._validate_static_artifacts()
        self._cache: OrderedDict[EvolutionKey, ReplayBundle] = OrderedDict()
        self._cache_lock = RLock()

    @staticmethod
    def _validate_key(algorithm: str, depth: int) -> EvolutionKey:
        algorithm = str(algorithm)
        depth = int(depth)
        if algorithm not in ALGORITHMS or not 1 <= depth <= 110:
            raise KeyError(f"unknown_evolution:{algorithm}:p{depth}")
        return EvolutionKey(algorithm, depth)

    def _artifact(self, key: EvolutionKey) -> dict[str, Any]:
        path = self.result_root / "checkpoints" / f"{key.artifact_id}.json"
        artifact = _read_json(path)
        if artifact.get("algorithm") != key.algorithm or int(artifact.get("depth", -1)) != key.depth:
            raise EvolutionMicroscopeDataError(f"checkpoint_identity_mismatch:{key.artifact_id}")
        return artifact

    def _graph_payload(self) -> dict[str, Any]:
        positions = {
            0: (0.06, 0.51),
            1: (0.23, 0.20),
            2: (0.41, 0.12),
            3: (0.41, 0.72),
            4: (0.62, 0.24),
            5: (0.79, 0.69),
            6: (0.95, 0.45),
        }
        optimal_pairs = set(zip(self.reference["exact_route"][:-1], self.reference["exact_route"][1:]))
        return {
            "source": int(self.config["source"]),
            "target": int(self.config["target"]),
            "nodes": [
                {"id": int(node), "x": positions[node][0], "y": positions[node][1]}
                for node in sorted(self.graph.nodes)
            ],
            "edges": [
                {
                    "qubit": index,
                    "u": int(u),
                    "v": int(v),
                    "weight": int(self.graph.edges[u, v]["weight"]),
                    "optimal": (u, v) in optimal_pairs,
                }
                for index, (u, v) in enumerate(self.edge_order)
            ],
            "optimal_route": list(map(int, self.reference["exact_route"])),
            "optimal_cost": float(self.reference["optimal_feasible_cost"]),
        }

    def _hamiltonian_payload(self) -> dict[str, Any]:
        linear = [
            {"kind": "Z", "qubits": [index], "label": f"Z{index}", "coefficient": _float(value)}
            for index, value in enumerate(self.ising.h)
            if value
        ]
        couplings = [
            {
                "kind": "ZZ",
                "qubits": [i, j],
                "label": f"Z{i} Z{j}",
                "coefficient": _float(value),
            }
            for (i, j), value in sorted(self.ising.coupling.items())
            if value
        ]
        return {
            "fixed_structure": True,
            "cost": {
                "formula": r"H_C=c_0I+\sum_jh_jZ_j+\sum_{j<k}J_{jk}Z_jZ_k",
                "constant": _float(self.ising.constant),
                "linear": linear,
                "couplings": couplings,
                "counts": {"constant": 1, "linear": len(linear), "zz": len(couplings)},
                "source": "src/qubo.py build_qubo(A=6) → qubo_to_ising",
            },
            "mixers": {
                "penalty_x": {
                    "formula": r"H_M^{(X)}=\sum_{j=1}^nX_j,\quad U_M^{(X)}(\beta_\ell)=e^{-i\beta_\ell H_M^{(X)}}",
                    "fixed_structure": True,
                    "geometry": "local single-bit-flip transport on the 14-dimensional hypercube",
                    "implementation": "src/qaoa.py apply_x_mixer; unscaled β in the frozen depth sweep",
                    "feasibility_preserving": False,
                },
                "global_grover": {
                    "formula": r"H_M^{(G)}=\Pi_s=\lvert s\rangle\langle s\rvert,\quad U_M^{(G)}(\beta_\ell)=e^{-i\beta_\ell\Pi_s}=I+(e^{-i\beta_\ell}-1)\Pi_s",
                    "reference_state": r"\lvert s\rangle=2^{-n/2}\sum_{x\in\{0,1\}^n}\lvert x\rangle=\lvert+\rangle^{\otimes n},\quad n=14",
                    "fixed_structure": True,
                    "geometry": "rank-one global amplitude redistribution relative to the uniform full-space reference state",
                    "implementation": "src/qaoa.py apply_grover_mixer",
                    "feasibility_preserving": False,
                    "standard_grover_search": False,
                },
            },
        }

    def _depth_trajectories(self) -> dict[str, list[dict[str, Any]]]:
        result = {algorithm: [] for algorithm in ALGORITHMS}
        for row in self.depth_rows:
            depth = int(row["p"])
            result["penalty_x"].append(
                {
                    "depth": depth,
                    "p_opt": float(row["penalty_x_p_opt"]),
                    "p_feas": float(row["penalty_x_p_feas"]),
                    "p_opt_given_feasible": float(row["penalty_x_p_opt_given_feasible"]),
                    "expected_hc": float(row["penalty_x_expected_penalized_cost"]),
                }
            )
            result["global_grover"].append(
                {
                    "depth": depth,
                    "p_opt": float(row["global_grover_p_opt"]),
                    "p_feas": float(row["global_grover_p_feas"]),
                    "p_opt_given_feasible": float(row["global_grover_p_opt_given_feasible"]),
                    "expected_hc": float(row["global_grover_expected_penalized_cost"]),
                }
            )
        return result

    def _validate_static_artifacts(self) -> dict[str, Any]:
        linear = np.asarray(list(map(float, self.qubo.linear)), dtype=np.float64)
        computed = np.full(
            len(self.states), float(self.qubo.constant), dtype=np.float64
        )
        computed += np.sum(self.bit_matrix * linear[None, :], axis=1)
        for (i, j), coefficient in self.qubo.pair.items():
            computed += float(coefficient) * self.bit_matrix[:, i] * self.bit_matrix[:, j]
        parameter_lengths = all(
            len(self.optimized_parameters[f"{algorithm}_p{depth:03d}"]) == 2 * depth
            for algorithm in ALGORITHMS
            for depth in range(1, 111)
        )
        checks = {
            "saved_global_depth_validation_43_of_43": bool(self.saved_validation["all_passed"]),
            "algorithm_names_exact": tuple(self.config["algorithms"]) == ALGORITHMS,
            "all_depths_1_to_110_present": len(self.depth_rows) == 110
            and [int(row["p"]) for row in self.depth_rows] == list(range(1, 111)),
            "all_220_parameter_vectors_have_length_2p": parameter_lengths,
            "reference_dimension_is_2_to_14": self.energies.shape == (2**14,),
            "reference_energy_matches_current_qubo": np.array_equal(computed, self.energies),
            "reference_feasible_mask_matches_decoder": np.array_equal(
                np.asarray([state.is_decoder_valid for state in self.states]), self.feasible
            ),
            "reference_optimal_mask_matches_decoder": np.array_equal(
                np.asarray(
                    [state.is_decoder_valid and state.routing_cost == 10 for state in self.states]
                ),
                self.optimal,
            ),
            "mixer_contract_is_full_space": not bool(self.config.get("feasible_projection", False))
            and self.config["initial_state"] == "uniform_full_space_plus_state",
            "penalty_x_contract_is_unscaled": "without beta/n scaling" in self.config["x_mixer"],
        }
        return {
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "tolerance": TOLERANCE,
            "optimization_rerun": False,
            "frozen_artifacts_modified": False,
        }

    def catalog(self) -> dict[str, Any]:
        best = {
            algorithm: max(rows, key=lambda row: row["p_opt"])
            for algorithm, rows in self.depth_trajectories.items()
        }
        return {
            "schema": "dtu-sciqis-qaoa-evolution-microscope-catalog",
            "version": "2.0",
            "read_only": True,
            "optimization_rerun": False,
            "source": "frozen results/global_depth110 plus deterministic replay",
            "algorithms": [
                {"id": algorithm, "label": DISPLAY_NAMES[algorithm]}
                for algorithm in ALGORITHMS
            ],
            "depths": list(range(1, 111)),
            "default": {"algorithm": "penalty_x", "depth": 110, "layer": 21, "stage": "before_cost"},
            "stages": list(STAGES),
            "graph": self.graph_payload,
            "hamiltonian": self.hamiltonian_payload,
            "depth_trajectories": self.depth_trajectories,
            "best_outcomes": best,
            "reference_states": {
                "optimal": self._state_identity(self.optimal_index),
                "best_nonoptimal_feasible": self._state_identity(self.best_nonoptimal_feasible_index),
                "low_energy_infeasible": self._state_identity(self.low_energy_infeasible_index),
            },
            "validation": self.static_validation,
            "pedagogy": {
                "cost": "The cost layer encodes basis-state energy into relative phase; computational-basis probabilities stay fixed.",
                "mixer": "The mixer converts phase structure into constructive and destructive interference, redistributing amplitudes.",
                "repetition": "QAOA repeatedly alternates phase encoding and interference.",
                "non_monotonic": "QAOA is coherent, interference-driven, and generally non-monotonic; more layers are not automatically better.",
            },
        }

    def run_config(self, algorithm: str, depth: int) -> dict[str, Any]:
        key = self._validate_key(algorithm, depth)
        artifact = self._artifact(key)
        parameters = list(map(float, artifact["optimized_parameters"]))
        return {
            "schema": "dtu-sciqis-qaoa-evolution-run-config",
            "version": "2.0",
            "algorithm": key.algorithm,
            "algorithm_label": DISPLAY_NAMES[key.algorithm],
            "optimized_depth": key.depth,
            "parameter_count": 2 * key.depth,
            "parameters": [
                {"layer": layer, "gamma": parameters[layer - 1], "beta": parameters[key.depth + layer - 1]}
                for layer in range(1, key.depth + 1)
            ],
            "initial_state": r"\lvert\psi_0\rangle=\lvert+\rangle^{\otimes14}",
            "cost_hamiltonian": "fixed raw routing cost + 6 × squared flow residual penalty",
            "mixer": self.hamiltonian_payload["mixers"][key.algorithm],
            "hamiltonian_structure_fixed": True,
            "layer_angles_change": True,
            "generated_unitaries_layer_dependent": True,
            "optimizer": {
                "name": artifact["optimizer"],
                "objective": self.config["optimizer_objective"],
                "evaluation_budget": int(artifact["evaluation_budget"]),
                "evaluations": int(artifact["nfev"]),
                "status": artifact["status"],
                "termination": artifact["termination_reason"],
                "runtime_seconds": float(artifact["optimizer_wall_time_s"]),
                "trace_available": False,
                "trace_reason": "Optimizer evaluation history was not saved in the frozen artifact; only final optimized parameters are available.",
            },
            "frozen_final": {
                "expected_hc": float(artifact["expected_penalized_cost"]),
                "p_feas": float(artifact["p_feas"]),
                "p_opt": float(artifact["p_opt"]),
                "p_opt_given_feasible": float(artifact["p_opt_given_feasible"]),
            },
            "depth_outcome": self.depth_trajectories[key.algorithm][key.depth - 1],
            "best_outcome": max(self.depth_trajectories[key.algorithm], key=lambda row: row["p_opt"]),
            "source": {
                "parameters": str(self.result_root / "checkpoints" / f"{key.artifact_id}.json"),
                "final_distribution": str(self.result_root / "distributions" / f"{key.artifact_id}.npz"),
                "optimization_rerun": False,
                "intermediate_states": "deterministic replay from frozen optimized parameters",
            },
        }

    def _mixer(self, algorithm: str):
        if algorithm == "penalty_x":
            return lambda state, beta: apply_x_mixer(state, beta, EXPECTED_EDGE_COUNT)
        return apply_grover_mixer

    def _replay(self, key: EvolutionKey) -> ReplayBundle:
        with self._cache_lock:
            if key in self._cache:
                bundle = self._cache.pop(key)
                self._cache[key] = bundle
                return bundle

            artifact = self._artifact(key)
            parameters = tuple(map(float, artifact["optimized_parameters"]))
            trace = trace_qaoa_evolution(
                initial_state(EXPECTED_EDGE_COUNT),
                self.energies,
                self._mixer(key.algorithm),
                parameters,
                depth=key.depth,
                metadata=self.metadata,
            )
            validation = self._validate_replay(key, trace, artifact)
            if not validation["all_checks_passed"]:
                raise EvolutionMicroscopeDataError(f"replay_validation_failed:{key.artifact_id}")
            bundle = ReplayBundle(key, trace, artifact, validation)
            self._cache[key] = bundle
            while len(self._cache) > REPLAY_CACHE_SIZE:
                self._cache.popitem(last=False)
            return bundle

    def _validate_replay(
        self, key: EvolutionKey, trace: EvolutionTrace, artifact: dict[str, Any]
    ) -> dict[str, Any]:
        parameters = np.asarray(artifact["optimized_parameters"], dtype=np.float64)
        if key.algorithm == "penalty_x":
            formal_final = qaoa_state(
                self.energies,
                parameters,
                depth=key.depth,
                mixer=X_MIXER,
                scale_x=False,
            )
        else:
            formal_final = simulate_global_grover_state(
                self.energies, parameters, depth=key.depth
            )
        final = trace.statevectors[-1]
        final_probabilities = np.abs(final) ** 2
        with np.load(self.result_root / "distributions" / f"{key.artifact_id}.npz") as saved:
            saved_probabilities = np.asarray(saved["probabilities"], dtype=np.float64)

        metric_object = distribution_metrics(
            final_probabilities, self.states, optimal_cost=10
        )
        independent_metric_error = max(
            abs(metric_object.p_feas - float(artifact["p_feas"])),
            abs(metric_object.p_opt - float(artifact["p_opt"])),
            abs((metric_object.p_opt_given_feas or 0.0) - float(artifact["p_opt_given_feasible"])),
            abs(
                float(np.sum(final_probabilities * self.energies))
                - float(artifact["expected_penalized_cost"])
            ),
        )
        trace_metric_error = 0.0
        representative_indices = sorted(
            {0, 1, 2, len(trace.checkpoints) // 2, len(trace.checkpoints) - 1}
        )
        for checkpoint_index in representative_indices:
            state = trace.statevectors[checkpoint_index]
            checkpoint = trace.checkpoints[checkpoint_index]
            probabilities = np.abs(state) ** 2
            recomputed = (
                float(np.sum(probabilities * self.energies)),
                probability_mass(probabilities, self.feasible),
                probability_mass(probabilities, self.optimal),
            )
            reported = (checkpoint.expected_hc, checkpoint.p_feas, checkpoint.p_opt)
            trace_metric_error = max(
                trace_metric_error,
                *(abs(left - right) for left, right in zip(recomputed, reported)),
            )

        maxima = {
            "normalization_error": max(
                abs(float(np.sum(np.abs(state) ** 2)) - 1.0)
                for state in trace.statevectors
            ),
            "cost_probability_delta": max(
                record.cost_probability_max_delta for record in trace.physics
            ),
            "cost_energy_delta": max(
                abs(record.cost_energy_delta) for record in trace.physics
            ),
            "mixer_normalization_error": max(
                abs(float(np.sum(np.abs(trace.statevectors[2 * layer]) ** 2)) - 1.0)
                for layer in range(1, key.depth + 1)
            ),
            "final_formal_statevector_error": float(np.max(np.abs(final - formal_final))),
            "final_frozen_distribution_error": float(
                np.max(np.abs(final_probabilities - saved_probabilities))
            ),
            "final_frozen_metrics_error": max(
                abs(trace.checkpoints[-1].expected_hc - float(artifact["expected_penalized_cost"])),
                abs(trace.checkpoints[-1].p_feas - float(artifact["p_feas"])),
                abs(trace.checkpoints[-1].p_opt - float(artifact["p_opt"])),
            ),
            "metrics_implementation_error": independent_metric_error,
            "checkpoint_metric_recomputation_error": trace_metric_error,
        }
        checks = {name: value <= TOLERANCE for name, value in maxima.items()}
        return {
            "algorithm": key.algorithm,
            "depth": key.depth,
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "maxima": maxima,
            "tolerance": TOLERANCE,
            "optimization_rerun": False,
            "provenance": "deterministic replay from frozen optimized parameters",
        }

    @staticmethod
    def _checkpoint_index(layer: int, stage: str, depth: int) -> int:
        layer = int(layer)
        stage = str(stage)
        if layer == 0:
            return 0
        if not 1 <= layer <= depth or stage not in STAGES:
            raise KeyError(f"unknown_checkpoint:layer={layer}:stage={stage}")
        return {
            "before_cost": 2 * (layer - 1),
            "after_cost": 2 * layer - 1,
            "after_mixer": 2 * layer,
        }[stage]

    def _metrics(self, checkpoint) -> dict[str, Any]:
        p_feas = float(checkpoint.p_feas)
        p_opt = float(checkpoint.p_opt)
        return {
            "expected_hc": float(checkpoint.expected_hc),
            "expected_route_cost_unconditional": float(checkpoint.expected_routing_term),
            "expected_flow_penalty": _json_number(checkpoint.expected_flow_penalty),
            "expected_feasible_route_cost": None,
            "p_feas": p_feas,
            "p_opt": p_opt,
            "p_opt_given_feasible": None if p_feas == 0 else p_opt / p_feas,
            "invalid_mass": float(checkpoint.invalid_mass),
            "entropy": float(checkpoint.shannon_entropy),
            "top_state_probability": float(checkpoint.max_basis_probability),
            "top_state_index": int(checkpoint.top_basis_index),
        }

    def _metrics_for_state(self, state: np.ndarray, checkpoint) -> dict[str, Any]:
        values = self._metrics(checkpoint)
        probabilities = np.abs(state) ** 2
        values["expected_feasible_route_cost"] = (
            None
            if checkpoint.p_feas == 0
            else float(
                np.sum(probabilities[self.feasible] * self.route_costs[self.feasible])
                / checkpoint.p_feas
            )
        )
        return values

    def evolution_summary(self, algorithm: str, depth: int) -> dict[str, Any]:
        key = self._validate_key(algorithm, depth)
        bundle = self._replay(key)
        trace = bundle.trace
        layers = []
        for layer in range(1, key.depth + 1):
            layer_item = {"layer": layer}
            for stage in STAGES:
                index = self._checkpoint_index(layer, stage, key.depth)
                layer_item[stage] = self._metrics_for_state(
                    trace.statevectors[index], trace.checkpoints[index]
                )
            physics = trace.physics[layer - 1]
            layer_item["physics"] = physics.as_dict()
            layers.append(layer_item)

        tracked = {}
        for name, index in (
            ("optimal", self.optimal_index),
            ("best_nonoptimal_feasible", self.best_nonoptimal_feasible_index),
            ("low_energy_infeasible", self.low_energy_infeasible_index),
        ):
            tracked[name] = self._track_payload(trace, index)

        return {
            "schema": "dtu-sciqis-qaoa-evolution-summary",
            "version": "2.0",
            "algorithm": key.algorithm,
            "algorithm_label": DISPLAY_NAMES[key.algorithm],
            "optimized_depth": key.depth,
            "initial": self._metrics_for_state(trace.statevectors[0], trace.checkpoints[0]),
            "layers": layers,
            "tracked_reference_states": tracked,
            "validation": bundle.validation,
            "raw_unsmoothed": True,
            "checkpoint_count": 1 + 2 * key.depth,
        }

    def _state_identity(self, index: int) -> dict[str, Any]:
        state = self.states[int(index)]
        return {
            "basis_index": int(index),
            "bitstring": state.canonical_bitstring,
            "energy": float(self.energies[index]),
            "feasible": bool(self.feasible[index]),
            "optimal": bool(self.optimal[index]),
            "route": None if state.decoded_route is None else list(state.decoded_route),
            "route_text": "" if state.decoded_route is None else " → ".join(map(str, state.decoded_route)),
            "route_cost": float(state.routing_cost),
        }

    def _state_row(
        self,
        index: int,
        amplitude: complex,
        *,
        probability: float | None = None,
        rank: int | None = None,
        gamma: float | None = None,
    ) -> dict[str, Any]:
        identity = self._state_identity(index)
        value = complex(amplitude)
        identity.update(
            {
                "rank": rank,
                "real": float(value.real),
                "imag": float(value.imag),
                "magnitude": float(abs(value)),
                "probability": float(abs(value) ** 2 if probability is None else probability),
                "phase": float(np.angle(value)),
                "cost_phase_rotation": None if gamma is None else float(np.angle(np.exp(-1j * gamma * self.energies[index]))),
                "selected_edge_indices": list(map(int, np.flatnonzero(self.bit_matrix[index]))),
            }
        )
        return identity

    def _candidate_indices(self, state_filter: str, query: str | None) -> np.ndarray:
        state_filter = str(state_filter or "all")
        if state_filter == "feasible":
            indices = np.flatnonzero(self.feasible)
        elif state_filter == "infeasible":
            indices = np.flatnonzero(~self.feasible)
        elif state_filter == "optimal":
            indices = np.flatnonzero(self.optimal)
        elif state_filter == "low_energy":
            indices = np.flatnonzero(self.energies <= 14)
        elif state_filter == "all":
            indices = np.arange(len(self.states))
        else:
            raise KeyError(f"unknown_state_filter:{state_filter}")
        if query:
            cleaned = str(query).replace(" ", "").strip()
            matches = [
                index
                for index in indices
                if cleaned in self.states[int(index)].canonical_bitstring
                or cleaned == str(int(index))
                or (
                    self.states[int(index)].decoded_route is not None
                    and cleaned in "->".join(map(str, self.states[int(index)].decoded_route))
                )
            ]
            return np.asarray(matches, dtype=int)
        return np.asarray(indices, dtype=int)

    def _transition_payload(
        self, trace: EvolutionTrace, index: int, selected_index: int
    ) -> dict[str, Any]:
        if index == 0:
            return {
                "from": None,
                "to": "initial",
                "probability_max_abs_delta": 0.0,
                "probability_l1_delta": 0.0,
                "phase_max_abs_delta": 0.0,
                "metrics_delta": {name: 0.0 for name in ("expected_hc", "p_feas", "p_opt", "p_opt_given_feasible")},
                "top_probability_gains": [],
                "top_probability_losses": [],
                "largest_phase_rotations": [],
            }
        before = trace.statevectors[index - 1]
        after = trace.statevectors[index]
        before_probability = np.abs(before) ** 2
        after_probability = np.abs(after) ** 2
        probability_delta = after_probability - before_probability
        support = (np.abs(before) > 1e-14) & (np.abs(after) > 1e-14)
        phase_delta = np.zeros(len(before), dtype=float)
        phase_delta[support] = _wrapped_phase_delta(after[support], before[support])
        gain_order = np.argsort(-probability_delta, kind="stable")[:5]
        loss_order = np.argsort(probability_delta, kind="stable")[:5]
        phase_order = np.argsort(-np.abs(phase_delta), kind="stable")[:5]
        before_metrics = self._metrics_for_state(before, trace.checkpoints[index - 1])
        after_metrics = self._metrics_for_state(after, trace.checkpoints[index])
        metric_names = ("expected_hc", "p_feas", "p_opt", "p_opt_given_feasible")

        def change_row(state_index: int, delta: float, field: str) -> dict[str, Any]:
            item = self._state_identity(int(state_index))
            item[field] = float(delta)
            return item

        return {
            "from": trace.checkpoints[index - 1].checkpoint,
            "to": trace.checkpoints[index].checkpoint,
            "operation": trace.checkpoints[index].operation,
            "probability_max_abs_delta": float(np.max(np.abs(probability_delta))),
            "probability_l1_delta": float(np.sum(np.abs(probability_delta))),
            "phase_max_abs_delta": float(np.max(np.abs(phase_delta))),
            "selected_probability_delta": float(probability_delta[selected_index]),
            "selected_phase_delta": float(phase_delta[selected_index]),
            "metrics_delta": {
                name: None
                if before_metrics[name] is None or after_metrics[name] is None
                else float(after_metrics[name] - before_metrics[name])
                for name in metric_names
            },
            "top_probability_gains": [
                change_row(int(i), probability_delta[i], "probability_delta") for i in gain_order
            ],
            "top_probability_losses": [
                change_row(int(i), probability_delta[i], "probability_delta") for i in loss_order
            ],
            "largest_phase_rotations": [
                change_row(int(i), phase_delta[i], "phase_delta") for i in phase_order
            ],
        }

    @staticmethod
    def _input_transition(layer: int) -> dict[str, Any]:
        return {
            "from": "Initial" if int(layer) == 0 else f"Mixer-{int(layer) - 1}" if int(layer) > 1 else "Initial",
            "to": "Initial" if int(layer) == 0 else f"Before-Cost-{int(layer)}",
            "operation": "input",
            "probability_max_abs_delta": 0.0,
            "probability_l1_delta": 0.0,
            "phase_max_abs_delta": 0.0,
            "selected_probability_delta": 0.0,
            "selected_phase_delta": 0.0,
            "metrics_delta": {
                name: 0.0
                for name in ("expected_hc", "p_feas", "p_opt", "p_opt_given_feasible")
            },
            "top_probability_gains": [],
            "top_probability_losses": [],
            "largest_phase_rotations": [],
        }

    def _track_payload(self, trace: EvolutionTrace, index: int) -> dict[str, Any]:
        values = []
        for checkpoint, state in zip(trace.checkpoints, trace.statevectors):
            amplitude = state[index]
            values.append(
                {
                    "checkpoint_index": int(checkpoint.checkpoint_index),
                    "layer": int(checkpoint.layer),
                    "operation": checkpoint.operation,
                    "probability": float(abs(amplitude) ** 2),
                    "magnitude": float(abs(amplitude)),
                    "phase": float(np.angle(amplitude)),
                    "real": float(amplitude.real),
                    "imag": float(amplitude.imag),
                }
            )
        return {"state": self._state_identity(index), "values": values}

    def track_state(self, algorithm: str, depth: int, state_index: int) -> dict[str, Any]:
        key = self._validate_key(algorithm, depth)
        state_index = int(state_index)
        if not 0 <= state_index < len(self.states):
            raise KeyError(f"unknown_basis_state:{state_index}")
        return self._track_payload(self._replay(key).trace, state_index)

    def checkpoint(
        self,
        algorithm: str,
        depth: int,
        layer: int,
        stage: str,
        *,
        top_n: int = DEFAULT_TOP_STATES,
        state_filter: str = "all",
        query: str | None = None,
        selected_state: int | None = None,
    ) -> dict[str, Any]:
        key = self._validate_key(algorithm, depth)
        index = self._checkpoint_index(layer, stage, key.depth)
        bundle = self._replay(key)
        trace = bundle.trace
        state = trace.statevectors[index]
        checkpoint = trace.checkpoints[index]
        probabilities = np.abs(state) ** 2
        selected_index = self.optimal_index if selected_state is None else int(selected_state)
        if not 0 <= selected_index < len(self.states):
            raise KeyError(f"unknown_basis_state:{selected_index}")

        candidates = self._candidate_indices(state_filter, query)
        order = candidates[np.lexsort((candidates, -np.round(probabilities[candidates], 15)))]
        requested = max(1, min(int(top_n), MAX_TOP_STATES))
        shown = list(map(int, order[:requested]))
        if selected_index not in shown:
            shown.append(selected_index)
        gamma = None if int(layer) == 0 else float(bundle.artifact["optimized_parameters"][int(layer) - 1])
        beta = None if int(layer) == 0 else float(bundle.artifact["optimized_parameters"][key.depth + int(layer) - 1])
        rows = [
            self._state_row(
                state_index,
                state[state_index],
                probability=probabilities[state_index],
                rank=position + 1,
                gamma=gamma if stage == "after_cost" else None,
            )
            for position, state_index in enumerate(shown)
        ]

        marginals = probabilities @ self.bit_matrix
        top_state_index = int(np.argmax(probabilities))
        top_route = self.states[top_state_index].decoded_route
        top_route_edges = set()
        if top_route is not None:
            top_route_edges = set(zip(top_route[:-1], top_route[1:]))
        selected_edges = set(map(int, np.flatnonzero(self.bit_matrix[selected_index])))
        graph_edges = []
        for edge, marginal in zip(self.graph_payload["edges"], marginals):
            item = dict(edge)
            item["selection_probability"] = float(marginal)
            item["top_route"] = (item["u"], item["v"]) in top_route_edges
            item["selected_state"] = item["qubit"] in selected_edges
            graph_edges.append(item)

        energy_mass = np.bincount(
            self.energy_inverse, weights=probabilities, minlength=len(self.energy_values)
        )
        feasible_energy_mass = np.bincount(
            self.energy_inverse,
            weights=probabilities * self.feasible,
            minlength=len(self.energy_values),
        )

        before_index = 0 if int(layer) == 0 else self._checkpoint_index(int(layer), "before_cost", key.depth)
        cost_index = 0 if int(layer) == 0 else self._checkpoint_index(int(layer), "after_cost", key.depth)
        mixer_index = 0 if int(layer) == 0 else self._checkpoint_index(int(layer), "after_mixer", key.depth)
        selected_triplet = {
            name: self._state_row(selected_index, trace.statevectors[position][selected_index])
            for name, position in (
                ("before_cost", before_index),
                ("after_cost", cost_index),
                ("after_mixer", mixer_index),
            )
        }
        physics = None if int(layer) == 0 else trace.physics[int(layer) - 1].as_dict()
        return {
            "schema": "dtu-sciqis-qaoa-evolution-checkpoint",
            "version": "2.0",
            "algorithm": key.algorithm,
            "algorithm_label": DISPLAY_NAMES[key.algorithm],
            "optimized_depth": key.depth,
            "layer": int(layer),
            "stage": "initial" if int(layer) == 0 else stage,
            "checkpoint_index": index,
            "checkpoint": checkpoint.checkpoint,
            "current_operation": {
                "kind": "initial" if int(layer) == 0 else ("cost" if stage == "after_cost" else "mixer" if stage == "after_mixer" else "input"),
                "gamma": gamma,
                "beta": beta,
                "cost_unitary": None if gamma is None else rf"U_C(\gamma_{{{layer}}})=e^{{-i\gamma_{{{layer}}}H_C}}",
                "mixer_unitary": None if beta is None else self.hamiltonian_payload["mixers"][key.algorithm]["formula"],
            },
            "metrics": self._metrics_for_state(state, checkpoint),
            "physics": physics,
            "transition": (
                self._input_transition(int(layer))
                if int(layer) == 0 or stage == "before_cost"
                else self._transition_payload(trace, index, selected_index)
            ),
            "states": {
                "rows": rows,
                "requested_top_n": requested,
                "filter": state_filter,
                "query": query or "",
                "matching_count": int(len(candidates)),
                "returned_count": len(rows),
                "full_dimension": len(self.states),
                "browser_receives_full_statevector": False,
            },
            "selected_state": self._state_row(selected_index, state[selected_index]),
            "selected_stage_triplet": selected_triplet,
            "graph": {**self.graph_payload, "edges": graph_edges},
            "energy_histogram": [
                {
                    "energy": float(energy),
                    "probability_mass": float(mass),
                    "feasible_probability_mass": float(feasible_mass),
                }
                for energy, mass, feasible_mass in zip(
                    self.energy_values, energy_mass, feasible_energy_mass
                )
                if mass > 1e-16
            ],
            "validation": bundle.validation,
        }

    def full_validation(self, depth: int = 110) -> dict[str, Any]:
        depth = int(depth)
        runs = [self._replay(EvolutionKey(algorithm, depth)).validation for algorithm in ALGORITHMS]
        maxima = {
            name: max(float(run["maxima"][name]) for run in runs)
            for name in runs[0]["maxima"]
        }
        return {
            "schema": "dtu-sciqis-qaoa-evolution-microscope-validation",
            "version": "2.0",
            "depth": depth,
            "all_checks_passed": self.static_validation["all_checks_passed"]
            and all(run["all_checks_passed"] for run in runs),
            "static": self.static_validation,
            "runs": runs,
            "maxima": maxima,
            "tolerance": TOLERANCE,
            "optimization_rerun": False,
            "frozen_artifacts_modified": False,
        }
