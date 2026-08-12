"""Read-only, browser-ready visualization data for the QAOA dynamics study.

The six optimized runs are never re-optimized here.  Exact statevectors are
loaded from the saved p=2 amplitude artifacts or deterministically reconstructed
from the saved final parameters when a p=1 intermediate amplitude was not
persisted.  Every reconstruction is checked against the saved scientific trace
before it is exposed to the frontend.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import json
from math import pi
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from feasible_qaoa import (
    build_feasible_route_basis,
)
from global_grover import build_global_grover_mixer
from graph import DEFAULT_GRAPH_PATH, EXPECTED_EDGE_COUNT, get_edge_order, load_graph
from qaoa import apply_x_mixer, standard_plus_state
from qaoa_dynamics import BasisMetadata, EvolutionTrace, trace_qaoa_evolution
from q2f_final_improvement import build_grover_feasible_mixer
from qubo import edge_vector_to_state_index, enumerate_state_space


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "qaoa_dynamics_deep_dive" / "v1"
ALGORITHM_ORDER = ("penalty_x", "grover_global", "grover_feasible")
DEPTHS = (1, 2)
CHECKPOINTS = {
    1: ("Initial", "Cost-1", "Mixer-1"),
    2: ("Initial", "Cost-1", "Mixer-1", "Cost-2", "Mixer-2"),
}
DISPLAY_NAMES = {
    "penalty_x": "Penalty-X QAOA",
    "grover_global": "Global Grover-Mixer QAOA",
    "grover_feasible": "Feasible Grover-Mixer QAOA",
}
SHORT_NAMES = {
    "penalty_x": "Penalty-X",
    "grover_global": "Global-Grover",
    "grover_feasible": "Feasible-Grover",
}
TOP_K = 7
PHASE_SAMPLE_TARGET = 1400
VALIDATION_TOLERANCE = 1e-10


class DynamicsVisualizationDataError(RuntimeError):
    """Raised when saved and reconstructed visualization data disagree."""


@dataclass(frozen=True)
class RunKey:
    algorithm: str
    depth: int

    @property
    def run_id(self) -> str:
        return f"{self.algorithm}_p{self.depth}_seed2601"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicsVisualizationDataError(f"cannot_load_visualization_json:{path}") from exc


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise DynamicsVisualizationDataError(f"cannot_load_visualization_csv:{path}") from exc


def _float_or_none(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def _wrapped_phase_error(left: float, right: float) -> float:
    return float(abs(np.angle(np.exp(1j * (float(left) - float(right))))))


class QAOADynamicsVisualizationRepository:
    """Validate, reconstruct, aggregate, and cache the six dynamics runs."""

    def __init__(self, result_root: str | Path = DEFAULT_RESULT_ROOT):
        self.result_root = Path(result_root).resolve()
        if not self.result_root.is_dir():
            raise DynamicsVisualizationDataError(
                f"dynamics_result_root_missing:{self.result_root}"
            )
        self.graph = load_graph(DEFAULT_GRAPH_PATH)
        self.edge_order = get_edge_order(self.graph)
        self.states = enumerate_state_space(self.graph)
        self.basis = build_feasible_route_basis(self.graph)
        self.feasible_mixer = build_grover_feasible_mixer(self.basis.size)
        self.global_mixer = build_global_grover_mixer(EXPECTED_EDGE_COUNT)

        summaries = _load_json(self.result_root / "final_summary.json")
        if not isinstance(summaries, list) or len(summaries) != 6:
            raise DynamicsVisualizationDataError("visualization_summary_matrix_mismatch")
        self.summaries = {
            RunKey(str(row["algorithm"]), int(row["p"])): row for row in summaries
        }
        expected_keys = {
            RunKey(algorithm, depth)
            for algorithm in ALGORITHM_ORDER
            for depth in DEPTHS
        }
        if set(self.summaries) != expected_keys:
            raise DynamicsVisualizationDataError("visualization_summary_keys_mismatch")

        self.trace_rows = self._load_trace_rows()
        self.energy_rows = self._load_and_validate_energy_rows()
        self.energy_summary = _load_json(
            self.result_root / "energy_landscape_summary.json"
        )
        self.scientific_validation = _load_json(
            self.result_root / "scientific_validations.json"
        )
        self.environment = _load_json(self.result_root / "environment.json")
        self._full_metadata = self._build_full_metadata()
        self._feasible_metadata = self._build_feasible_metadata()
        self._energy_landscape = self._build_energy_landscape_payload()
        self._graph_payload = self._build_graph_payload()

        self._payloads: dict[RunKey, dict[str, Any]] = {}
        validation_rows: list[dict[str, Any]] = []
        for key in sorted(expected_keys, key=lambda item: (ALGORITHM_ORDER.index(item.algorithm), item.depth)):
            payload, validation = self._build_run_payload(key)
            self._payloads[key] = payload
            validation_rows.append(validation)
        self.validation_report = self._aggregate_validation(validation_rows)

    def _load_trace_rows(self) -> dict[RunKey, list[dict[str, Any]]]:
        grouped: dict[RunKey, list[dict[str, Any]]] = {}
        integer_fields = {
            "p",
            "checkpoint_index",
            "layer",
            "top_basis_index",
        }
        boolean_fields = {"top_is_feasible", "top_is_optimal"}
        text_fields = {
            "run_id",
            "algorithm",
            "checkpoint",
            "operation",
            "top_basis_label",
        }
        for raw in _load_csv(self.result_root / "dynamics_trace.csv"):
            key = RunKey(raw["algorithm"], int(raw["p"]))
            row: dict[str, Any] = {}
            for name, value in raw.items():
                if name in text_fields:
                    row[name] = value
                elif name in integer_fields:
                    row[name] = int(value)
                elif name in boolean_fields:
                    row[name] = _bool(value)
                elif name == "top_decoded_route":
                    row[name] = None if not value else list(ast.literal_eval(value))
                else:
                    row[name] = _float_or_none(value)
            grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            rows.sort(key=lambda item: item["checkpoint_index"])
            if tuple(row["checkpoint"] for row in rows) != CHECKPOINTS[key.depth]:
                raise DynamicsVisualizationDataError(
                    f"visualization_checkpoint_order_mismatch:{key.run_id}"
                )
        return grouped

    def _load_and_validate_energy_rows(self) -> list[dict[str, Any]]:
        raw_rows = _load_csv(self.result_root / "energy_landscape.csv")
        if len(raw_rows) != 2**EXPECTED_EDGE_COUNT:
            raise DynamicsVisualizationDataError("visualization_energy_state_count_mismatch")
        rows: list[dict[str, Any]] = []
        for expected_index, (raw, state) in enumerate(zip(raw_rows, self.states)):
            row = {
                "state_index": int(raw["state_index"]),
                "canonical_bitstring": raw["canonical_bitstring"],
                "routing_term": float(raw["routing_term"]),
                "flow_penalty": float(raw["flow_penalty"]),
                "penalty_contribution": float(raw["penalty_contribution"]),
                "total_qubo_energy": float(raw["total_qubo_energy"]),
                "feasible": _bool(raw["feasible"]),
                "exact_optimal": _bool(raw["exact_optimal"]),
                "decoded_route": raw["decoded_route"],
            }
            if (
                row["state_index"] != expected_index
                or row["canonical_bitstring"] != state.canonical_bitstring
                or row["routing_term"] != state.routing_cost
                or row["flow_penalty"] != state.flow_penalty
                or row["feasible"] != state.is_decoder_valid
                or row["total_qubo_energy"]
                != state.routing_cost + 6.0 * state.flow_penalty
            ):
                raise DynamicsVisualizationDataError(
                    f"visualization_energy_row_mismatch:{expected_index}"
                )
            rows.append(row)
        return rows

    def _build_full_metadata(self) -> BasisMetadata:
        return BasisMetadata(
            basis_labels=tuple(state.canonical_bitstring for state in self.states),
            decoded_routes=tuple(state.decoded_route for state in self.states),
            route_costs=tuple(float(state.routing_cost) for state in self.states),
            flow_penalties=tuple(float(state.flow_penalty) for state in self.states),
            total_energies=tuple(
                float(state.routing_cost + 6.0 * state.flow_penalty)
                for state in self.states
            ),
            feasible_mask=tuple(state.is_decoder_valid for state in self.states),
            optimal_mask=tuple(
                state.is_decoder_valid and state.routing_cost == 10
                for state in self.states
            ),
            penalty_coefficient=6.0,
            representation="edge_bits_all_bitstrings",
        )

    def _build_feasible_metadata(self) -> BasisMetadata:
        return BasisMetadata(
            basis_labels=tuple(
                f"route_{route.route_id}:{route.bitstring_text}"
                for route in self.basis.routes
            ),
            decoded_routes=tuple(route.node_sequence for route in self.basis.routes),
            route_costs=tuple(float(route.routing_cost) for route in self.basis.routes),
            flow_penalties=tuple(0.0 for _route in self.basis.routes),
            total_energies=tuple(float(route.routing_cost) for route in self.basis.routes),
            feasible_mask=tuple(True for _route in self.basis.routes),
            optimal_mask=tuple(route.exact_optimal for route in self.basis.routes),
            penalty_coefficient=None,
            representation="logical_feasible_routes",
        )

    def _build_graph_payload(self) -> dict[str, Any]:
        positions = {
            0: (0.05, 0.50),
            1: (0.22, 0.20),
            2: (0.39, 0.13),
            3: (0.39, 0.70),
            4: (0.59, 0.24),
            5: (0.76, 0.69),
            6: (0.94, 0.45),
        }
        return {
            "source": int(self.graph.graph["source"]),
            "target": int(self.graph.graph["target"]),
            "nodes": [
                {"id": node, "x": positions[node][0], "y": positions[node][1]}
                for node in sorted(self.graph.nodes)
            ],
            "edges": [
                {
                    "qubit_index": index,
                    "u": edge[0],
                    "v": edge[1],
                    "weight": int(self.graph.edges[edge]["weight"]),
                }
                for index, edge in enumerate(self.edge_order)
            ],
            "optimal_route": [0, 1, 2, 4, 5, 6],
            "optimal_cost": 10,
        }

    def _full_state_payload(self, state_index: int) -> dict[str, Any]:
        state = self.states[int(state_index)]
        violations = [
            {"node": node, "residual": int(value)}
            for node, value in zip(sorted(self.graph.nodes), state.flow_residuals)
            if value != 0
        ]
        if state.is_decoder_valid:
            explanation = "Valid source-to-target route; every flow residual is zero."
        else:
            nodes = ", ".join(
                f"v{item['node']}={item['residual']:+d}" for item in violations
            )
            explanation = (
                f"Invalid edge selection: nonzero flow residuals ({nodes})."
                if violations
                else "Invalid edge selection: the independent route decoder rejected it."
            )
        return {
            "basis_index": int(state.state_index),
            "full_state_index": int(state.state_index),
            "identity": state.canonical_bitstring,
            "bitstring": state.canonical_bitstring,
            "route_id": None,
            "route": None if state.decoded_route is None else list(state.decoded_route),
            "route_text": ""
            if state.decoded_route is None
            else " → ".join(map(str, state.decoded_route)),
            "route_cost": float(state.routing_cost),
            "energy": float(state.routing_cost + 6.0 * state.flow_penalty),
            "flow_penalty": float(state.flow_penalty),
            "penalty_contribution": float(6.0 * state.flow_penalty),
            "feasible": bool(state.is_decoder_valid),
            "optimal": bool(state.is_decoder_valid and state.routing_cost == 10),
            "selected_edge_indices": [
                index for index, bit in enumerate(state.edge_vector) if bit
            ],
            "flow_violations": violations,
            "validity_explanation": explanation,
        }

    def _feasible_state_payload(self, route_id: int) -> dict[str, Any]:
        route = self.basis.routes[int(route_id)]
        full_index = edge_vector_to_state_index(route.edge_bitstring)
        return {
            "basis_index": int(route.route_id),
            "full_state_index": int(full_index),
            "identity": f"P{route.route_id}",
            "bitstring": route.bitstring_text,
            "route_id": int(route.route_id),
            "route": list(route.node_sequence),
            "route_text": " → ".join(map(str, route.node_sequence)),
            "route_cost": float(route.routing_cost),
            "energy": float(route.routing_cost),
            "flow_penalty": 0.0,
            "penalty_contribution": 0.0,
            "feasible": True,
            "optimal": bool(route.exact_optimal),
            "selected_edge_indices": [
                index for index, bit in enumerate(route.edge_bitstring) if bit
            ],
            "flow_violations": [],
            "validity_explanation": (
                "Exact globally optimal route in the logical feasible basis."
                if route.exact_optimal
                else "Valid route in the logical feasible basis; flow residuals are zero."
            ),
        }

    def _build_energy_landscape_payload(self) -> dict[str, Any]:
        energies = np.asarray(
            [row["total_qubo_energy"] for row in self.energy_rows], dtype=float
        )
        feasible = np.asarray([row["feasible"] for row in self.energy_rows], dtype=bool)
        edges = np.linspace(float(np.min(energies)) - 0.5, float(np.max(energies)) + 0.5, 41)
        feasible_counts, _ = np.histogram(energies[feasible], bins=edges)
        infeasible_counts, _ = np.histogram(energies[~feasible], bins=edges)
        logical_by_full = {
            edge_vector_to_state_index(route.edge_bitstring): route.route_id
            for route in self.basis.routes
        }
        bins = []
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            in_bin = (energies >= left) & (
                (energies <= right) if index == len(edges) - 2 else (energies < right)
            )
            candidates = list(map(int, np.flatnonzero(in_bin)))
            candidates.sort(
                key=lambda state_index: (
                    not self.energy_rows[state_index]["exact_optimal"],
                    not self.energy_rows[state_index]["feasible"],
                    energies[state_index],
                    state_index,
                )
            )
            examples = []
            for state_index in candidates[:6]:
                item = self._full_state_payload(state_index)
                item["logical_route_id"] = logical_by_full.get(state_index)
                examples.append(item)
            bins.append(
                {
                    "index": index,
                    "left": float(left),
                    "right": float(right),
                    "feasible_count": int(feasible_counts[index]),
                    "infeasible_count": int(infeasible_counts[index]),
                    "examples": examples,
                }
            )
        return {
            "total_state_count": int(self.energy_summary["state_count"]),
            "feasible_state_count": int(self.energy_summary["feasible_state_count"]),
            "infeasible_state_count": int(self.energy_summary["infeasible_state_count"]),
            "feasible_fraction": float(self.energy_summary["feasible_fraction"]),
            "minimum_energy": float(np.min(energies)),
            "maximum_energy": float(np.max(energies)),
            "markers": [
                {"label": "exact optimum", "energy": 10.0, "category": "optimal"},
                {"label": "second feasible", "energy": 11.0, "category": "feasible"},
                {"label": "lowest infeasible", "energy": 12.0, "category": "infeasible"},
            ],
            "bins": bins,
            "source": "energy_landscape.csv; exact all-16,384-state counts",
        }

    def _algorithm_objects(
        self, key: RunKey
    ) -> tuple[np.ndarray, np.ndarray, Any, BasisMetadata]:
        if key.algorithm in ("penalty_x", "grover_global"):
            total = np.asarray(self._full_metadata.total_energies, dtype=float)
            phase = (total - float(np.min(total))) / float(np.max(total) - np.min(total))
            initial = standard_plus_state(EXPECTED_EDGE_COUNT)
            if key.algorithm == "penalty_x":
                def mixer(state: Sequence[complex], beta: float) -> np.ndarray:
                    return apply_x_mixer(
                        np.asarray(state, dtype=np.complex128),
                        float(beta) / EXPECTED_EDGE_COUNT,
                        EXPECTED_EDGE_COUNT,
                    )
            else:
                mixer = self.global_mixer.evolve
            return initial, phase, mixer, self._full_metadata
        raw = np.asarray(self._feasible_metadata.total_energies, dtype=float)
        phase = (raw - float(np.min(raw))) / float(np.max(raw) - np.min(raw))
        initial = np.full(self.basis.size, 1.0 / np.sqrt(self.basis.size), dtype=np.complex128)
        return initial, phase, self.feasible_mixer.evolve, self._feasible_metadata

    def _probability_flow(
        self, probabilities: np.ndarray, metadata: BasisMetadata, algorithm: str
    ) -> tuple[list[dict[str, Any]], set[int]]:
        feasible = np.asarray(metadata.feasible_mask, dtype=bool)
        optimal = np.asarray(metadata.optimal_mask, dtype=bool)
        selected: list[dict[str, Any]] = []
        indices: set[int] = set()
        if algorithm == "grover_feasible":
            for index in range(metadata.dimension):
                selected.append(
                    {
                        "kind": "state",
                        "basis_index": index,
                        "probability": float(probabilities[index]),
                        "category": "optimal" if optimal[index] else "feasible",
                    }
                )
                indices.add(index)
            return selected, indices

        optimal_indices = list(map(int, np.flatnonzero(optimal)))
        feasible_indices = list(map(int, np.flatnonzero(feasible & ~optimal)))
        infeasible_indices = list(map(int, np.flatnonzero(~feasible)))
        feasible_indices.sort(key=lambda index: (-probabilities[index], index))
        infeasible_indices.sort(key=lambda index: (-probabilities[index], index))
        shown_feasible = optimal_indices + feasible_indices[:TOP_K]
        shown_infeasible = infeasible_indices[:TOP_K]
        for index in shown_feasible:
            selected.append(
                {
                    "kind": "state",
                    "basis_index": index,
                    "probability": float(probabilities[index]),
                    "category": "optimal" if optimal[index] else "feasible",
                }
            )
            indices.add(index)
        remaining_feasible = float(
            np.sum(probabilities[feasible]) - np.sum(probabilities[shown_feasible])
        )
        selected.append(
            {
                "kind": "aggregate",
                "identity": "remaining feasible",
                "probability": max(0.0, remaining_feasible),
                "category": "feasible_aggregate",
                "state_count": int(np.sum(feasible)) - len(shown_feasible),
            }
        )
        for index in shown_infeasible:
            selected.append(
                {
                    "kind": "state",
                    "basis_index": index,
                    "probability": float(probabilities[index]),
                    "category": "infeasible",
                }
            )
            indices.add(index)
        remaining_infeasible = float(
            np.sum(probabilities[~feasible]) - np.sum(probabilities[shown_infeasible])
        )
        selected.append(
            {
                "kind": "aggregate",
                "identity": "remaining infeasible",
                "probability": max(0.0, remaining_infeasible),
                "category": "infeasible_aggregate",
                "state_count": int(np.sum(~feasible)) - len(shown_infeasible),
            }
        )
        return selected, indices

    def _representative_indices(
        self, trace: EvolutionTrace, metadata: BasisMetadata, algorithm: str
    ) -> list[int]:
        if algorithm == "grover_feasible":
            return list(range(metadata.dimension))
        energies = np.asarray(metadata.total_energies, dtype=float)
        feasible = np.asarray(metadata.feasible_mask, dtype=bool)
        optimal = np.asarray(metadata.optimal_mask, dtype=bool)
        indices = list(map(int, np.flatnonzero(optimal)))
        feasible_order = np.flatnonzero(feasible & ~optimal)
        feasible_order = feasible_order[np.argsort(energies[feasible_order], kind="stable")]
        indices.extend(map(int, feasible_order[:5]))
        infeasible_order = np.flatnonzero(~feasible)
        infeasible_order = infeasible_order[
            np.lexsort((infeasible_order, energies[infeasible_order]))
        ]
        indices.extend(map(int, infeasible_order[:5]))
        final_probability = np.abs(trace.statevectors[-1]) ** 2
        top_infeasible = np.flatnonzero(~feasible)[
            np.argsort(-final_probability[~feasible], kind="stable")[:3]
        ]
        indices.extend(map(int, top_infeasible))
        return list(dict.fromkeys(indices))

    def _phase_sample_indices(
        self, metadata: BasisMetadata, required: Iterable[int], algorithm: str
    ) -> list[int]:
        if algorithm == "grover_feasible":
            return list(range(metadata.dimension))
        feasible_mask = np.asarray(metadata.feasible_mask, dtype=bool)
        feasible = np.flatnonzero(feasible_mask)
        infeasible = np.flatnonzero(~feasible_mask)
        random = np.random.default_rng(2601)
        sample = random.choice(
            infeasible,
            size=min(PHASE_SAMPLE_TARGET, len(infeasible)),
            replace=False,
        )
        values = set(map(int, sample))
        values.update(map(int, feasible))
        values.update(map(int, required))
        return sorted(values)

    def _state_payload(self, algorithm: str, basis_index: int) -> dict[str, Any]:
        return (
            self._feasible_state_payload(basis_index)
            if algorithm == "grover_feasible"
            else self._full_state_payload(basis_index)
        )

    def _landscape_payload(self, key: RunKey, parameters: Sequence[float]) -> dict[str, Any]:
        if key.depth != 1:
            return {
                "available": False,
                "reason": (
                    "No saved p=2 local 2D slice exists. A projection of the "
                    "four-dimensional landscape would be misleading."
                ),
                "final_parameters": list(map(float, parameters)),
                "optimizer_trajectory": {
                    "available": False,
                    "reason": "Exact evaluation history was not saved for the primary study.",
                },
            }
        rows = _load_csv(
            self.result_root / "parameter_landscapes" / f"{key.algorithm}_p1.csv"
        )
        gammas = sorted({float(row["gamma"]) for row in rows})
        betas = sorted({float(row["beta"]) for row in rows})
        if len(gammas) != 25 or len(betas) != 25 or len(rows) != 625:
            raise DynamicsVisualizationDataError(
                f"visualization_parameter_grid_mismatch:{key.run_id}"
            )
        metrics = {
            "expected_hc": [float(row["expected_hc"]) for row in rows],
            "p_feas": [float(row["p_feas"]) for row in rows],
            "p_opt": [float(row["p_opt"]) for row in rows],
        }
        best_points: dict[str, dict[str, float]] = {}
        for metric, values in metrics.items():
            index = int(np.argmin(values) if metric == "expected_hc" else np.argmax(values))
            best_points[metric] = {
                "gamma": float(rows[index]["gamma"]),
                "beta": float(rows[index]["beta"]),
                "value": float(values[index]),
            }
        return {
            "available": True,
            "grid_shape": [len(gammas), len(betas)],
            "gammas": gammas,
            "betas": betas,
            "metrics": metrics,
            "final_point": {
                "gamma": float(parameters[0]),
                "beta": float(parameters[1]),
            },
            "best_grid_points": best_points,
            "optimizer_trajectory": {
                "available": False,
                "reason": (
                    "The primary optimizer evaluation history was not saved; "
                    "only the exact final point is shown."
                ),
            },
            "teaching_note": (
                "The ansatz landscape can contain structure that a fixed local "
                "optimizer start does not reach. The grid was not used for retuning."
            ),
        }

    def _circuit_payload(self, key: RunKey, parameters: Sequence[float]) -> dict[str, Any]:
        gates: list[dict[str, Any]] = [
            {
                "checkpoint_index": 0,
                "kind": "initial",
                "label": "Initial",
                "symbol": "|+⟩⊗q" if key.algorithm != "grover_feasible" else "|s_F⟩",
                "detail": (
                    "uniform over all bitstrings"
                    if key.algorithm != "grover_feasible"
                    else "uniform over 20 feasible routes"
                ),
            }
        ]
        for layer in range(1, key.depth + 1):
            gates.extend(
                (
                    {
                        "checkpoint_index": 2 * layer - 1,
                        "kind": "cost",
                        "label": f"U_C(γ{layer})",
                        "symbol": "U_C",
                        "angle_name": f"γ{layer}",
                        "angle": float(parameters[layer - 1]),
                        "detail": "objective-dependent phase evolution",
                    },
                    {
                        "checkpoint_index": 2 * layer,
                        "kind": "mixer",
                        "label": f"U_M(β{layer})",
                        "symbol": "U_M",
                        "angle_name": f"β{layer}",
                        "angle": float(parameters[key.depth + layer - 1]),
                        "detail": "amplitude mixing / interference",
                    },
                )
            )
        gates.append(
            {
                "checkpoint_index": None,
                "kind": "measurement",
                "label": "Measurement",
                "symbol": "M",
                "detail": "probability distribution and route decoding",
            }
        )
        return {
            "parameter_order": "all gammas, then all betas",
            "layer_order": "cost, then mixer",
            "gates": gates,
        }

    def _validate_phase_csv(self, key: RunKey, trace: EvolutionTrace) -> tuple[float, float, float]:
        if key.depth != 2:
            return 0.0, 0.0, 0.0
        path = self.result_root / "phase_analysis" / f"{key.run_id}.csv"
        phase_error = magnitude_error = probability_error = 0.0
        checkpoint_by_name = {
            checkpoint.checkpoint: index
            for index, checkpoint in enumerate(trace.checkpoints)
        }
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    checkpoint_index = checkpoint_by_name[raw["checkpoint"]]
                    basis_index = int(raw["basis_index"])
                    amplitude = trace.statevectors[checkpoint_index][basis_index]
                    phase_error = max(
                        phase_error,
                        _wrapped_phase_error(float(raw["phase"]), float(np.angle(amplitude))),
                    )
                    magnitude_error = max(
                        magnitude_error,
                        abs(float(raw["magnitude"]) - float(abs(amplitude))),
                    )
                    probability_error = max(
                        probability_error,
                        abs(float(raw["probability"]) - float(abs(amplitude) ** 2)),
                    )
        except OSError as exc:
            raise DynamicsVisualizationDataError(
                f"visualization_phase_file_missing:{path}"
            ) from exc
        return phase_error, magnitude_error, probability_error

    def _build_run_payload(self, key: RunKey) -> tuple[dict[str, Any], dict[str, Any]]:
        summary = self.summaries[key]
        parameters = tuple(map(float, summary["optimized_parameters"]))
        initial, phase, mixer, metadata = self._algorithm_objects(key)
        trace = trace_qaoa_evolution(
            initial,
            phase,
            mixer,
            parameters,
            depth=key.depth,
            metadata=metadata,
        )
        stored = self.trace_rows[key]
        if len(stored) != len(trace.checkpoints):
            raise DynamicsVisualizationDataError(
                f"visualization_trace_length_mismatch:{key.run_id}"
            )

        metric_fields = (
            "norm",
            "probability_sum",
            "expected_hc",
            "expected_routing_term",
            "expected_flow_penalty",
            "expected_penalty_contribution",
            "p_feas",
            "p_opt",
            "invalid_mass",
            "shannon_entropy",
            "max_basis_probability",
        )
        stored_metric_error = 0.0
        for reconstructed, saved in zip(trace.checkpoints, stored):
            reconstructed_dict = reconstructed.as_dict()
            for field in metric_fields:
                saved_value = saved[field]
                reconstructed_value = reconstructed_dict[field]
                if saved_value is None and reconstructed_value is None:
                    continue
                if saved_value is None or reconstructed_value is None:
                    raise DynamicsVisualizationDataError(
                        f"visualization_metric_null_mismatch:{key.run_id}:{field}"
                    )
                stored_metric_error = max(
                    stored_metric_error,
                    abs(float(saved_value) - float(reconstructed_value)),
                )

        saved_amplitude_error = 0.0
        if key.depth == 2:
            amplitude_path = self.result_root / "amplitudes" / f"{key.run_id}.npz"
            with np.load(amplitude_path) as saved_amplitudes:
                saved_states = saved_amplitudes["real"] + 1j * saved_amplitudes["imag"]
                checkpoint_names = tuple(map(str, saved_amplitudes["checkpoint"]))
            if checkpoint_names != CHECKPOINTS[key.depth] or saved_states.shape != (
                len(trace.statevectors),
                metadata.dimension,
            ):
                raise DynamicsVisualizationDataError(
                    f"visualization_saved_amplitude_shape_mismatch:{key.run_id}"
                )
            saved_amplitude_error = float(
                np.max(np.abs(saved_states - np.stack(trace.statevectors)))
            )

        final_path = self.result_root / "distributions" / f"{key.run_id}.npz"
        with np.load(final_path) as saved_distribution:
            saved_probability = np.asarray(saved_distribution["probability"], dtype=float)
        reconstructed_final = np.abs(trace.statevectors[-1]) ** 2
        final_distribution_error = float(
            np.max(np.abs(saved_probability - reconstructed_final))
        )
        summary_error = max(
            abs(float(summary["p_feas"]) - trace.checkpoints[-1].p_feas),
            abs(float(summary["p_opt"]) - trace.checkpoints[-1].p_opt),
            abs(float(summary["invalid_mass"]) - trace.checkpoints[-1].invalid_mass),
            abs(float(summary["expected_hc"]) - trace.checkpoints[-1].expected_hc),
        )
        phase_error, phase_magnitude_error, phase_probability_error = (
            self._validate_phase_csv(key, trace)
        )

        flows: list[list[dict[str, Any]]] = []
        flow_indices: set[int] = set()
        for statevector in trace.statevectors:
            items, indices = self._probability_flow(
                np.abs(statevector) ** 2, metadata, key.algorithm
            )
            flows.append(items)
            flow_indices.update(indices)
        representatives = self._representative_indices(trace, metadata, key.algorithm)

        energy_example_indices = {
            int(example["full_state_index"])
            for bin_payload in self._energy_landscape["bins"]
            for example in bin_payload["examples"]
        }
        required = set(flow_indices) | set(representatives)
        if key.algorithm != "grover_feasible":
            required.update(energy_example_indices)
        phase_indices = self._phase_sample_indices(metadata, required, key.algorithm)
        display_indices = sorted(set(phase_indices) | required)
        display_position = {
            basis_index: position for position, basis_index in enumerate(display_indices)
        }
        display_states = [
            self._state_payload(key.algorithm, basis_index)
            for basis_index in display_indices
        ]
        phase_positions = [display_position[index] for index in phase_indices]
        representative_positions = [
            display_position[index] for index in representatives
        ]

        frames: list[dict[str, Any]] = []
        for checkpoint_index, (statevector, saved, flow) in enumerate(
            zip(trace.statevectors, stored, flows)
        ):
            values = []
            for basis_index in display_indices:
                amplitude = statevector[basis_index]
                values.append(
                    [
                        float(amplitude.real),
                        float(amplitude.imag),
                        float(abs(amplitude)),
                        float(abs(amplitude) ** 2),
                        float(np.angle(amplitude)),
                    ]
                )
            angle = None
            angle_name = None
            if saved["operation"] == "cost":
                angle_name = f"γ{saved['layer']}"
                angle = parameters[saved["layer"] - 1]
            elif saved["operation"] == "mixer":
                angle_name = f"β{saved['layer']}"
                angle = parameters[key.depth + saved["layer"] - 1]
            teaching = {
                "initial": (
                    "Uniform amplitudes establish the representation-dependent baseline."
                ),
                "cost": (
                    "Objective-dependent phases change; basis probabilities and ⟨H_C⟩ stay fixed."
                ),
                "mixer": (
                    "Interference can redistribute probability and may raise or lower ⟨H_C⟩."
                ),
            }[saved["operation"]]
            frames.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "checkpoint": saved["checkpoint"],
                    "operation": saved["operation"],
                    "layer": saved["layer"],
                    "angle_name": angle_name,
                    "angle": None if angle is None else float(angle),
                    "teaching": teaching,
                    "metrics": {
                        field: saved[field]
                        for field in stored[checkpoint_index]
                        if field
                        not in {
                            "run_id",
                            "algorithm",
                            "p",
                            "checkpoint_index",
                            "checkpoint",
                            "operation",
                            "layer",
                        }
                    },
                    "probability_flow": flow,
                    "state_values": values,
                }
            )

        for frame in frames:
            for item in frame["probability_flow"]:
                if item["kind"] == "state":
                    item["display_position"] = display_position[item["basis_index"]]

        payload = {
            "schema": "dtu-sciqis-qaoa-dynamics-visualization-run",
            "version": "1.0",
            "source": {
                "kind": "saved_dynamics_artifacts_with_verified_reconstruction",
                "result_root": str(self.result_root),
                "run_id": key.run_id,
                "optimization_rerun": False,
                "amplitude_source": (
                    "saved_npz_verified_against_reconstruction"
                    if key.depth == 2
                    else "deterministic_reconstruction_from_saved_parameters"
                ),
            },
            "run_id": key.run_id,
            "algorithm": key.algorithm,
            "algorithm_label": DISPLAY_NAMES[key.algorithm],
            "algorithm_short_label": SHORT_NAMES[key.algorithm],
            "depth": key.depth,
            "seed": int(summary["seed"]),
            "representation": summary["representation"],
            "search_dimension": int(summary["search_dimension"]),
            "mixer_type": summary["mixer_type"],
            "structural_feasibility": key.algorithm == "grover_feasible",
            "parameters": {
                "ordered": list(parameters),
                "gammas": list(parameters[: key.depth]),
                "betas": list(parameters[key.depth :]),
            },
            "optimizer": {
                "evaluations": int(summary["optimizer_evaluations"]),
                "termination": summary["optimizer_reason"],
                "runtime_seconds": float(summary["optimizer_runtime_seconds"]),
                "trajectory_available": False,
                "trajectory_reason": (
                    "The primary dynamics artifacts retain final parameters but "
                    "not the per-evaluation optimizer path."
                ),
            },
            "circuit": self._circuit_payload(key, parameters),
            "checkpoints": list(CHECKPOINTS[key.depth]),
            "display_states": display_states,
            "phase_scatter": {
                "display_positions": phase_positions,
                "downsampled": key.algorithm != "grover_feasible",
                "displayed_count": len(phase_positions),
                "total_count": metadata.dimension,
                "sampling": (
                    "fixed seed-2601 sample of infeasible indices plus all 20 "
                    "feasible states, all plotted top-k states, and energy inspectors"
                    if key.algorithm != "grover_feasible"
                    else "all 20 logical feasible states"
                ),
                "all_feasible_preserved": True,
            },
            "complex_amplitudes": {
                "display_positions": representative_positions,
                "all_feasible_routes": key.algorithm == "grover_feasible",
                "default_mode": "magnitude",
            },
            "frames": frames,
            "energy_landscape": self._energy_landscape,
            "graph": self._graph_payload,
            "parameter_landscape": self._landscape_payload(key, parameters),
            "summary": summary,
        }
        validation = {
            "run_id": key.run_id,
            "checkpoint_order_correct": tuple(payload["checkpoints"])
            == CHECKPOINTS[key.depth],
            "dimension_correct": metadata.dimension
            == (20 if key.algorithm == "grover_feasible" else 2**14),
            "stored_metric_max_error": stored_metric_error,
            "saved_amplitude_max_error": saved_amplitude_error,
            "final_distribution_max_error": final_distribution_error,
            "summary_metric_max_error": summary_error,
            "saved_phase_max_wrapped_error": phase_error,
            "saved_phase_magnitude_max_error": phase_magnitude_error,
            "saved_phase_probability_max_error": phase_probability_error,
            "maximum_norm_error": max(
                abs(float(np.linalg.norm(statevector)) - 1.0)
                for statevector in trace.statevectors
            ),
            "maximum_probability_sum_error": max(
                abs(float(np.sum(np.abs(statevector) ** 2)) - 1.0)
                for statevector in trace.statevectors
            ),
            "maximum_cost_probability_delta": max(
                record.cost_probability_max_delta for record in trace.physics
            ),
            "maximum_cost_energy_delta": max(
                abs(record.cost_energy_delta) for record in trace.physics
            ),
            "p_opt_not_above_p_feas": all(
                checkpoint.p_opt <= checkpoint.p_feas + VALIDATION_TOLERANCE
                for checkpoint in trace.checkpoints
            ),
            "probability_flow_normalized": all(
                abs(sum(float(item["probability"]) for item in frame["probability_flow"]) - 1.0)
                <= VALIDATION_TOLERANCE
                for frame in frames
            ),
            "all_feasible_states_in_phase_view": all(
                index in phase_indices
                for index in np.flatnonzero(np.asarray(metadata.feasible_mask, dtype=bool))
            ),
        }
        return payload, validation

    def _aggregate_validation(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        feasible_routes_decode = all(
            self.states[edge_vector_to_state_index(route.edge_bitstring)].decoded_route
            == route.node_sequence
            for route in self.basis.routes
        )
        checks = {
            "checkpoint_order_correct": all(row["checkpoint_order_correct"] for row in rows),
            "trace_norms_approximately_one": max(row["maximum_norm_error"] for row in rows)
            <= VALIDATION_TOLERANCE,
            "probabilities_sum_approximately_one": max(
                row["maximum_probability_sum_error"] for row in rows
            )
            <= VALIDATION_TOLERANCE,
            "cost_step_probability_invariance": max(
                row["maximum_cost_probability_delta"] for row in rows
            )
            <= VALIDATION_TOLERANCE,
            "cost_step_expected_hc_invariance": max(
                row["maximum_cost_energy_delta"] for row in rows
            )
            <= VALIDATION_TOLERANCE,
            "selected_phase_values_match_saved_data": max(
                row["saved_phase_max_wrapped_error"] for row in rows
            )
            <= VALIDATION_TOLERANCE,
            "p_opt_not_above_p_feas": all(row["p_opt_not_above_p_feas"] for row in rows),
            "route_identities_decode_consistently": feasible_routes_decode,
            "feasible_grover_states_all_validate": all(
                route.node_sequence
                == self.states[edge_vector_to_state_index(route.edge_bitstring)].decoded_route
                for route in self.basis.routes
            ),
            "full_space_dimensions_are_2_to_14": all(
                row["dimension_correct"] for row in rows
            ),
            "displayed_metrics_match_saved_trace": max(
                row["stored_metric_max_error"] for row in rows
            )
            <= VALIDATION_TOLERANCE,
            "displayed_final_metrics_match_summary": max(
                row["summary_metric_max_error"] for row in rows
            )
            <= VALIDATION_TOLERANCE,
            "saved_distributions_match_reconstruction": max(
                row["final_distribution_max_error"] for row in rows
            )
            <= VALIDATION_TOLERANCE,
            "probability_flow_aggregates_normalize": all(
                row["probability_flow_normalized"] for row in rows
            ),
            "phase_view_preserves_all_feasible_states": all(
                row["all_feasible_states_in_phase_view"] for row in rows
            ),
            "no_dense_full_space_object": True,
        }
        return {
            "schema": "dtu-sciqis-qaoa-dynamics-visualization-validation",
            "version": "1.0",
            "result_root": str(self.result_root),
            "optimization_rerun": False,
            "tolerance": VALIDATION_TOLERANCE,
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "maxima": {
                "stored_metric_error": max(row["stored_metric_max_error"] for row in rows),
                "saved_amplitude_error": max(row["saved_amplitude_max_error"] for row in rows),
                "saved_phase_wrapped_error": max(row["saved_phase_max_wrapped_error"] for row in rows),
                "saved_phase_magnitude_error": max(row["saved_phase_magnitude_max_error"] for row in rows),
                "saved_phase_probability_error": max(row["saved_phase_probability_max_error"] for row in rows),
                "final_distribution_error": max(row["final_distribution_max_error"] for row in rows),
                "summary_metric_error": max(row["summary_metric_max_error"] for row in rows),
                "norm_error": max(row["maximum_norm_error"] for row in rows),
                "probability_sum_error": max(row["maximum_probability_sum_error"] for row in rows),
                "cost_probability_delta": max(row["maximum_cost_probability_delta"] for row in rows),
                "cost_energy_delta": max(row["maximum_cost_energy_delta"] for row in rows),
            },
            "runs": rows,
        }

    def catalog(self) -> dict[str, Any]:
        """Return selectors, summaries, teaching scenes, and validation status."""

        summaries = []
        for algorithm in ALGORITHM_ORDER:
            for depth in DEPTHS:
                key = RunKey(algorithm, depth)
                row = self.summaries[key]
                summaries.append(
                    {
                        "algorithm": algorithm,
                        "algorithm_label": DISPLAY_NAMES[algorithm],
                        "algorithm_short_label": SHORT_NAMES[algorithm],
                        "depth": depth,
                        "search_dimension": int(row["search_dimension"]),
                        "p_feas": float(row["p_feas"]),
                        "p_opt": float(row["p_opt"]),
                        "invalid_mass": float(row["invalid_mass"]),
                        "expected_hc": float(row["expected_hc"]),
                    }
                )
        scenes = [
            {
                "number": 1,
                "title": "Initial uniform state",
                "algorithm": "grover_global",
                "depth": 2,
                "checkpoint": 0,
                "panels": ["circuit", "metrics", "probability", "route"],
                "message": "The full-space run begins uniformly: every bitstring has probability 1/16,384.",
            },
            {
                "number": 2,
                "title": "Cost encodes phase—not probability",
                "algorithm": "grover_global",
                "depth": 2,
                "checkpoint": 1,
                "panels": ["circuit", "energy", "complex", "phase"],
                "message": "U_C changes relative phase while probability and ⟨H_C⟩ remain invariant.",
            },
            {
                "number": 3,
                "title": "Mixer interference",
                "algorithm": "grover_global",
                "depth": 2,
                "checkpoint": 2,
                "panels": ["circuit", "energy", "metrics", "probability", "complex"],
                "message": "The mixer recombines phase-tagged amplitudes; this step raises energy, demonstrating non-monotonic dynamics.",
            },
            {
                "number": 4,
                "title": "Same H_C, different mixer",
                "algorithm": "grover_global",
                "depth": 2,
                "checkpoint": 4,
                "panels": ["circuit", "energy", "metrics", "geometry"],
                "message": "Penalty-X and Global-Grover share 16,384 states, the initial state, and byte-identical H_C; only mixer geometry differs.",
            },
            {
                "number": 5,
                "title": "Restrict to feasible routes",
                "algorithm": "grover_feasible",
                "depth": 2,
                "checkpoint": 0,
                "panels": ["metrics", "probability", "geometry", "route"],
                "message": "Feasible-Grover starts in a 20-route logical basis, so p_feas≈1 is structural.",
            },
            {
                "number": 6,
                "title": "p=2 final comparison",
                "algorithm": "grover_feasible",
                "depth": 2,
                "checkpoint": 4,
                "panels": ["energy", "metrics", "probability", "route"],
                "message": "Compare final probability mass against each representation's very different initial baseline.",
            },
            {
                "number": 7,
                "title": "Classical optimizer around the circuit",
                "algorithm": "grover_global",
                "depth": 1,
                "checkpoint": 2,
                "panels": ["circuit", "optimizer"],
                "message": "The ansatz defines a landscape; COBYLA chooses angles from objective evaluations, and a local run need not find the best grid point.",
            },
        ]
        return {
            "schema": "dtu-sciqis-qaoa-dynamics-visualization-catalog",
            "version": "1.0",
            "read_only": True,
            "optimization_rerun": False,
            "algorithms": [
                {
                    "id": algorithm,
                    "label": DISPLAY_NAMES[algorithm],
                    "short_label": SHORT_NAMES[algorithm],
                }
                for algorithm in ALGORITHM_ORDER
            ],
            "depths": list(DEPTHS),
            "default": {"algorithm": "grover_global", "depth": 2},
            "summaries": summaries,
            "presentation_scenes": scenes,
            "pedagogy": [
                "H_C tells the circuit what is good by encoding objective values into relative phase.",
                "H_M determines how amplitudes can mix.",
                "Interference converts phase structure into probability redistribution.",
                "The classical optimizer selects gamma and beta using measurements of the quantum objective.",
                "A cost layer does not itself lower <H_C>.",
                "QAOA intermediate energy need not decrease monotonically.",
                "Feasible-Grover p_feas=1 follows from representation restriction, not quantum advantage.",
            ],
            "energy_landscape_summary": {
                key: self._energy_landscape[key]
                for key in (
                    "total_state_count",
                    "feasible_state_count",
                    "infeasible_state_count",
                    "feasible_fraction",
                )
            },
            "environment": self.environment,
            "validation": {
                "all_checks_passed": self.validation_report["all_checks_passed"],
                "tolerance": self.validation_report["tolerance"],
            },
        }

    def load_run(self, algorithm: str, depth: int) -> dict[str, Any]:
        """Return one cached browser payload after strict selector validation."""

        key = RunKey(str(algorithm), int(depth))
        try:
            return self._payloads[key]
        except KeyError as exc:
            raise KeyError(f"unknown_dynamics_visualization_run:{algorithm}:p{depth}") from exc
