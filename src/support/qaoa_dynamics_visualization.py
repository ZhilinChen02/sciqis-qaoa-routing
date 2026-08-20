"""Turn six saved QAOA dynamics runs into data for the browser dashboard.

The optimizer is never rerun here.  We read its saved final angles, rebuild the
state after each cost and mixer operation, and prepare ordinary JSON data.
"""

import ast
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.dynamics_trace import BasisMetadata, trace_qaoa_evolution
from feasible_experiments import build_grover_feasible_mixer
from feasible_qaoa import build_feasible_route_basis
from graph import DEFAULT_GRAPH_PATH, EXPECTED_EDGE_COUNT, get_edge_order, load_graph
from qaoa import (
    apply_x_mixer,
    build_global_grover_mixer,
    initial_state,
)
from qubo import edge_vector_to_state_index, enumerate_state_space


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results/qaoa_dynamics_deep_dive/v1"
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
    pass


@dataclass(frozen=True)
class RunKey:
    algorithm: str
    depth: int

    @property
    def run_id(self) -> str:
        return f"{self.algorithm}_p{self.depth}_seed2601"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DynamicsVisualizationDataError(f"cannot load {path}") from error


def _read_csv(path: Path):
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise DynamicsVisualizationDataError(f"cannot load {path}") from error


def _as_bool(value):
    return value is True or str(value).lower() == "true"


def _phase_error(left, right):
    return float(abs(np.angle(np.exp(1j * (float(left) - float(right))))))


class QAOADynamicsVisualizationRepository:
    """Read saved results once and cache six browser-ready run dictionaries."""

    def __init__(self, result_root: str | Path = DEFAULT_RESULT_ROOT):
        self.result_root = Path(result_root).resolve()
        if not self.result_root.is_dir():
            raise DynamicsVisualizationDataError(f"missing result folder: {self.result_root}")

        self.graph = load_graph(DEFAULT_GRAPH_PATH)
        self.edge_order = get_edge_order(self.graph)
        self.states = enumerate_state_space(self.graph)
        self.basis = build_feasible_route_basis(self.graph)
        self.feasible_mixer = build_grover_feasible_mixer(self.basis.size)
        self.global_mixer = build_global_grover_mixer(EXPECTED_EDGE_COUNT)

        summary_rows = _read_json(self.result_root / "final_summary.json")
        self.summaries = {
            RunKey(row["algorithm"], int(row["p"])): row for row in summary_rows
        }
        self.keys = [
            RunKey(algorithm, depth)
            for algorithm in ALGORITHM_ORDER
            for depth in DEPTHS
        ]
        if set(self.summaries) != set(self.keys):
            raise DynamicsVisualizationDataError("the six summary rows are incomplete")

        self.trace_rows = self._load_trace_rows()
        self.energy_summary = _read_json(self.result_root / "energy_landscape_summary.json")
        self.environment = _read_json(self.result_root / "environment.json")
        self.full_metadata = self._metadata(full_space=True)
        self.feasible_metadata = self._metadata(full_space=False)
        self.graph_payload = self._graph_payload()
        self.energy_landscape = self._energy_landscape()

        self.payloads = {}
        validation_rows = []
        for key in self.keys:
            payload, validation = self._build_run(key)
            self.payloads[key] = payload
            validation_rows.append(validation)
        self.validation_report = self._validation_report(validation_rows)

    def _load_trace_rows(self):
        integer_fields = {"p", "checkpoint_index", "layer", "top_basis_index"}
        boolean_fields = {"top_is_feasible", "top_is_optimal"}
        text_fields = {"run_id", "algorithm", "checkpoint", "operation", "top_basis_label"}
        grouped = {}
        for raw in _read_csv(self.result_root / "dynamics_trace.csv"):
            key = RunKey(raw["algorithm"], int(raw["p"]))
            row = {}
            for name, value in raw.items():
                if name in text_fields:
                    row[name] = value
                elif name in integer_fields:
                    row[name] = int(value)
                elif name in boolean_fields:
                    row[name] = _as_bool(value)
                elif name == "top_decoded_route":
                    row[name] = None if not value else list(ast.literal_eval(value))
                else:
                    row[name] = None if value in (None, "") else float(value)
            grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            rows.sort(key=lambda row: row["checkpoint_index"])
            if tuple(row["checkpoint"] for row in rows) != CHECKPOINTS[key.depth]:
                raise DynamicsVisualizationDataError(f"wrong checkpoints: {key.run_id}")
        return grouped

    def _metadata(self, *, full_space):
        if full_space:
            return BasisMetadata(
                basis_labels=tuple(state.canonical_bitstring for state in self.states),
                decoded_routes=tuple(state.decoded_route for state in self.states),
                route_costs=tuple(float(state.routing_cost) for state in self.states),
                flow_penalties=tuple(float(state.flow_penalty) for state in self.states),
                total_energies=tuple(
                    float(state.routing_cost + 6 * state.flow_penalty)
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
        return BasisMetadata(
            basis_labels=tuple(
                f"route_{route.route_id}:{route.bitstring_text}" for route in self.basis.routes
            ),
            decoded_routes=tuple(route.node_sequence for route in self.basis.routes),
            route_costs=tuple(float(route.routing_cost) for route in self.basis.routes),
            flow_penalties=(0.0,) * self.basis.size,
            total_energies=tuple(float(route.routing_cost) for route in self.basis.routes),
            feasible_mask=(True,) * self.basis.size,
            optimal_mask=tuple(route.exact_optimal for route in self.basis.routes),
            penalty_coefficient=None,
            representation="logical_feasible_routes",
        )

    def _graph_payload(self):
        positions = {
            0: (0.05, 0.50), 1: (0.22, 0.20), 2: (0.39, 0.13),
            3: (0.39, 0.70), 4: (0.59, 0.24), 5: (0.76, 0.69), 6: (0.94, 0.45),
        }
        return {
            "source": self.graph.graph["source"],
            "target": self.graph.graph["target"],
            "nodes": [
                {"id": node, "x": positions[node][0], "y": positions[node][1]}
                for node in sorted(self.graph.nodes)
            ],
            "edges": [
                {
                    "qubit_index": index,
                    "u": u,
                    "v": v,
                    "weight": self.graph.edges[u, v]["weight"],
                }
                for index, (u, v) in enumerate(self.edge_order)
            ],
            "optimal_route": [0, 1, 2, 4, 5, 6],
            "optimal_cost": 10,
        }

    def _full_state(self, index):
        state = self.states[int(index)]
        violations = [
            {"node": node, "residual": int(value)}
            for node, value in zip(sorted(self.graph.nodes), state.flow_residuals)
            if value
        ]
        if state.is_decoder_valid:
            explanation = "Valid source-to-target route; every flow residual is zero."
        elif violations:
            values = ", ".join(
                f"v{item['node']}={item['residual']:+d}" for item in violations
            )
            explanation = f"Invalid edge selection: nonzero flow residuals ({values})."
        else:
            explanation = "Invalid edge selection: the independent route decoder rejected it."
        route = state.decoded_route
        return {
            "basis_index": state.state_index,
            "full_state_index": state.state_index,
            "identity": state.canonical_bitstring,
            "bitstring": state.canonical_bitstring,
            "route_id": None,
            "route": None if route is None else list(route),
            "route_text": "" if route is None else " → ".join(map(str, route)),
            "route_cost": float(state.routing_cost),
            "energy": float(state.routing_cost + 6 * state.flow_penalty),
            "flow_penalty": float(state.flow_penalty),
            "penalty_contribution": float(6 * state.flow_penalty),
            "feasible": bool(state.is_decoder_valid),
            "optimal": bool(state.is_decoder_valid and state.routing_cost == 10),
            "selected_edge_indices": [i for i, bit in enumerate(state.edge_vector) if bit],
            "flow_violations": violations,
            "validity_explanation": explanation,
        }

    def _feasible_state(self, index):
        route = self.basis.routes[int(index)]
        return {
            "basis_index": route.route_id,
            "full_state_index": edge_vector_to_state_index(route.edge_bitstring),
            "identity": f"P{route.route_id}",
            "bitstring": route.bitstring_text,
            "route_id": route.route_id,
            "route": list(route.node_sequence),
            "route_text": " → ".join(map(str, route.node_sequence)),
            "route_cost": float(route.routing_cost),
            "energy": float(route.routing_cost),
            "flow_penalty": 0.0,
            "penalty_contribution": 0.0,
            "feasible": True,
            "optimal": bool(route.exact_optimal),
            "selected_edge_indices": [i for i, bit in enumerate(route.edge_bitstring) if bit],
            "flow_violations": [],
            "validity_explanation": (
                "Exact globally optimal route in the logical feasible basis."
                if route.exact_optimal
                else "Valid route in the logical feasible basis; flow residuals are zero."
            ),
        }

    def _state(self, algorithm, index):
        return self._feasible_state(index) if algorithm == "grover_feasible" else self._full_state(index)

    def _energy_landscape(self):
        energies = np.asarray(self.full_metadata.total_energies)
        feasible = np.asarray(self.full_metadata.feasible_mask)
        edges = np.linspace(float(energies.min()) - 0.5, float(energies.max()) + 0.5, 41)
        feasible_counts, _ = np.histogram(energies[feasible], bins=edges)
        infeasible_counts, _ = np.histogram(energies[~feasible], bins=edges)
        logical_ids = {
            edge_vector_to_state_index(route.edge_bitstring): route.route_id
            for route in self.basis.routes
        }
        bins = []
        for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            inclusive = bin_index == len(edges) - 2
            inside = (energies >= left) & ((energies <= right) if inclusive else (energies < right))
            candidates = list(map(int, np.flatnonzero(inside)))
            candidates.sort(
                key=lambda index: (
                    not self.full_metadata.optimal_mask[index],
                    not self.full_metadata.feasible_mask[index],
                    energies[index],
                    index,
                )
            )
            examples = []
            for index in candidates[:6]:
                item = self._full_state(index)
                item["logical_route_id"] = logical_ids.get(index)
                examples.append(item)
            bins.append(
                {
                    "index": bin_index,
                    "left": float(left),
                    "right": float(right),
                    "feasible_count": int(feasible_counts[bin_index]),
                    "infeasible_count": int(infeasible_counts[bin_index]),
                    "examples": examples,
                }
            )
        return {
            "total_state_count": int(self.energy_summary["state_count"]),
            "feasible_state_count": int(self.energy_summary["feasible_state_count"]),
            "infeasible_state_count": int(self.energy_summary["infeasible_state_count"]),
            "feasible_fraction": float(self.energy_summary["feasible_fraction"]),
            "minimum_energy": float(energies.min()),
            "maximum_energy": float(energies.max()),
            "markers": [
                {"label": "exact optimum", "energy": 10.0, "category": "optimal"},
                {"label": "second feasible", "energy": 11.0, "category": "feasible"},
                {"label": "lowest infeasible", "energy": 12.0, "category": "infeasible"},
            ],
            "bins": bins,
            "source": "energy_landscape.csv; exact all-16,384-state counts",
        }

    def _algorithm(self, key):
        if key.algorithm != "grover_feasible":
            metadata = self.full_metadata
            energy = np.asarray(metadata.total_energies)
            phase = (energy - energy.min()) / (energy.max() - energy.min())
            initial = initial_state(EXPECTED_EDGE_COUNT)
            if key.algorithm == "penalty_x":
                def mixer(state: Sequence[complex], beta: float):
                    return apply_x_mixer(
                        np.asarray(state), beta / EXPECTED_EDGE_COUNT, EXPECTED_EDGE_COUNT
                    )
            else:
                mixer = self.global_mixer.evolve
            return initial, phase, mixer, metadata
        metadata = self.feasible_metadata
        energy = np.asarray(metadata.total_energies)
        phase = (energy - energy.min()) / (energy.max() - energy.min())
        initial = np.full(self.basis.size, 1 / np.sqrt(self.basis.size), dtype=complex)
        return initial, phase, self.feasible_mixer.evolve, metadata

    def _probability_flow(self, probabilities, metadata, algorithm):
        feasible = np.asarray(metadata.feasible_mask)
        optimal = np.asarray(metadata.optimal_mask)
        if algorithm == "grover_feasible":
            items = [
                {
                    "kind": "state",
                    "basis_index": index,
                    "probability": float(probabilities[index]),
                    "category": "optimal" if optimal[index] else "feasible",
                }
                for index in range(metadata.dimension)
            ]
            return items, set(range(metadata.dimension))

        optimal_indices = list(map(int, np.flatnonzero(optimal)))
        feasible_indices = list(map(int, np.flatnonzero(feasible & ~optimal)))
        infeasible_indices = list(map(int, np.flatnonzero(~feasible)))
        feasible_indices.sort(key=lambda i: (-probabilities[i], i))
        infeasible_indices.sort(key=lambda i: (-probabilities[i], i))
        shown_feasible = optimal_indices + feasible_indices[:TOP_K]
        shown_infeasible = infeasible_indices[:TOP_K]
        items = [
            {
                "kind": "state",
                "basis_index": index,
                "probability": float(probabilities[index]),
                "category": "optimal" if optimal[index] else "feasible",
            }
            for index in shown_feasible
        ]
        items.append(
            {
                "kind": "aggregate",
                "identity": "remaining feasible",
                "probability": max(0.0, float(probabilities[feasible].sum() - probabilities[shown_feasible].sum())),
                "category": "feasible_aggregate",
                "state_count": int(feasible.sum()) - len(shown_feasible),
            }
        )
        items.extend(
            {
                "kind": "state",
                "basis_index": index,
                "probability": float(probabilities[index]),
                "category": "infeasible",
            }
            for index in shown_infeasible
        )
        items.append(
            {
                "kind": "aggregate",
                "identity": "remaining infeasible",
                "probability": max(0.0, float(probabilities[~feasible].sum() - probabilities[shown_infeasible].sum())),
                "category": "infeasible_aggregate",
                "state_count": int((~feasible).sum()) - len(shown_infeasible),
            }
        )
        return items, set(shown_feasible + shown_infeasible)

    def _representatives(self, trace, metadata, algorithm):
        if algorithm == "grover_feasible":
            return list(range(metadata.dimension))
        energy = np.asarray(metadata.total_energies)
        feasible = np.asarray(metadata.feasible_mask)
        optimal = np.asarray(metadata.optimal_mask)
        indices = list(map(int, np.flatnonzero(optimal)))
        candidates = np.flatnonzero(feasible & ~optimal)
        indices.extend(map(int, candidates[np.argsort(energy[candidates], kind="stable")[:5]]))
        candidates = np.flatnonzero(~feasible)
        indices.extend(map(int, candidates[np.lexsort((candidates, energy[candidates]))[:5]]))
        final_probability = np.abs(trace.statevectors[-1]) ** 2
        indices.extend(map(int, candidates[np.argsort(-final_probability[candidates], kind="stable")[:3]]))
        return list(dict.fromkeys(indices))

    def _phase_indices(self, metadata, required, algorithm):
        if algorithm == "grover_feasible":
            return list(range(metadata.dimension))
        feasible = np.asarray(metadata.feasible_mask)
        infeasible_indices = np.flatnonzero(~feasible)
        sample = np.random.default_rng(2601).choice(
            infeasible_indices,
            min(PHASE_SAMPLE_TARGET, len(infeasible_indices)),
            replace=False,
        )
        return sorted(set(map(int, sample)) | set(map(int, np.flatnonzero(feasible))) | set(required))

    def _landscape(self, key, parameters):
        if key.depth != 1:
            return {
                "available": False,
                "reason": "No saved p=2 local 2D slice exists. A projection of the four-dimensional landscape would be misleading.",
                "final_parameters": list(map(float, parameters)),
                "optimizer_trajectory": {
                    "available": False,
                    "reason": "Exact evaluation history was not saved for the primary study.",
                },
            }
        rows = _read_csv(
            self.result_root / f"parameter_landscapes/{key.algorithm}_p1.csv"
        )
        gammas = sorted({float(row["gamma"]) for row in rows})
        betas = sorted({float(row["beta"]) for row in rows})
        metrics = {
            name: [float(row[name]) for row in rows]
            for name in ("expected_hc", "p_feas", "p_opt")
        }
        best_points = {}
        for name, values in metrics.items():
            index = int(np.argmin(values) if name == "expected_hc" else np.argmax(values))
            best_points[name] = {
                "gamma": float(rows[index]["gamma"]),
                "beta": float(rows[index]["beta"]),
                "value": values[index],
            }
        return {
            "available": True,
            "grid_shape": [len(gammas), len(betas)],
            "gammas": gammas,
            "betas": betas,
            "metrics": metrics,
            "final_point": {"gamma": float(parameters[0]), "beta": float(parameters[1])},
            "best_grid_points": best_points,
            "optimizer_trajectory": {
                "available": False,
                "reason": "The primary optimizer evaluation history was not saved; only the exact final point is shown.",
            },
            "teaching_note": "The grid shows the ansatz landscape; it was not used for retuning.",
        }

    @staticmethod
    def _circuit(key, parameters):
        gates = [
            {
                "checkpoint_index": 0,
                "kind": "initial",
                "label": "Initial",
                "symbol": "|s_F⟩" if key.algorithm == "grover_feasible" else "|+⟩⊗q",
                "detail": "uniform over 20 feasible routes" if key.algorithm == "grover_feasible" else "uniform over all bitstrings",
            }
        ]
        for layer in range(1, key.depth + 1):
            gates.extend(
                [
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
                ]
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

    def _phase_csv_error(self, key, trace):
        if key.depth != 2:
            return 0.0, 0.0, 0.0
        checkpoint_index = {
            checkpoint.checkpoint: index for index, checkpoint in enumerate(trace.checkpoints)
        }
        phase_error = magnitude_error = probability_error = 0.0
        for row in _read_csv(self.result_root / f"phase_analysis/{key.run_id}.csv"):
            amplitude = trace.statevectors[checkpoint_index[row["checkpoint"]]][int(row["basis_index"])]
            phase_error = max(phase_error, _phase_error(row["phase"], np.angle(amplitude)))
            magnitude_error = max(magnitude_error, abs(float(row["magnitude"]) - abs(amplitude)))
            probability_error = max(probability_error, abs(float(row["probability"]) - abs(amplitude) ** 2))
        return phase_error, magnitude_error, probability_error

    def _saved_errors(self, key, trace, metadata):
        stored = self.trace_rows[key]
        metric_names = (
            "norm", "probability_sum", "expected_hc", "expected_routing_term",
            "expected_flow_penalty", "expected_penalty_contribution", "p_feas",
            "p_opt", "invalid_mass", "shannon_entropy", "max_basis_probability",
        )
        metric_error = 0.0
        for checkpoint, saved in zip(trace.checkpoints, stored):
            rebuilt = checkpoint.as_dict()
            for name in metric_names:
                if saved[name] is None and rebuilt[name] is None:
                    continue
                metric_error = max(metric_error, abs(float(saved[name]) - float(rebuilt[name])))

        amplitude_error = 0.0
        if key.depth == 2:
            with np.load(self.result_root / f"amplitudes/{key.run_id}.npz") as saved:
                states = saved["real"] + 1j * saved["imag"]
            amplitude_error = float(np.max(np.abs(states - np.stack(trace.statevectors))))

        with np.load(self.result_root / f"distributions/{key.run_id}.npz") as saved:
            final_probability = np.asarray(saved["probability"])
        distribution_error = float(
            np.max(np.abs(final_probability - np.abs(trace.statevectors[-1]) ** 2))
        )
        summary = self.summaries[key]
        final = trace.checkpoints[-1]
        summary_error = max(
            abs(float(summary["p_feas"]) - final.p_feas),
            abs(float(summary["p_opt"]) - final.p_opt),
            abs(float(summary["invalid_mass"]) - final.invalid_mass),
            abs(float(summary["expected_hc"]) - final.expected_hc),
        )
        phase, magnitude, probability = self._phase_csv_error(key, trace)
        return metric_error, amplitude_error, distribution_error, summary_error, phase, magnitude, probability

    def _build_run(self, key):
        summary = self.summaries[key]
        parameters = tuple(map(float, summary["optimized_parameters"]))
        initial, phase, mixer, metadata = self._algorithm(key)
        trace = trace_qaoa_evolution(
            initial, phase, mixer, parameters, depth=key.depth, metadata=metadata
        )
        errors = self._saved_errors(key, trace, metadata)

        flows = []
        flow_indices = set()
        for state in trace.statevectors:
            flow, indices = self._probability_flow(np.abs(state) ** 2, metadata, key.algorithm)
            flows.append(flow)
            flow_indices.update(indices)

        representatives = self._representatives(trace, metadata, key.algorithm)
        energy_indices = {
            example["full_state_index"]
            for item in self.energy_landscape["bins"]
            for example in item["examples"]
        }
        required = flow_indices | set(representatives)
        if key.algorithm != "grover_feasible":
            required |= energy_indices
        phase_indices = self._phase_indices(metadata, required, key.algorithm)
        display_indices = sorted(set(phase_indices) | required)
        positions = {index: position for position, index in enumerate(display_indices)}

        teaching = {
            "initial": "Uniform amplitudes establish the representation-dependent baseline.",
            "cost": "Objective-dependent phases change; basis probabilities and ⟨H_C⟩ stay fixed.",
            "mixer": "Interference can redistribute probability and may raise or lower ⟨H_C⟩.",
        }
        frames = []
        for index, (state, checkpoint, flow) in enumerate(
            zip(trace.statevectors, trace.checkpoints, flows)
        ):
            values = [
                [float(state[i].real), float(state[i].imag), float(abs(state[i])),
                 float(abs(state[i]) ** 2), float(np.angle(state[i]))]
                for i in display_indices
            ]
            angle_name = angle = None
            if checkpoint.operation == "cost":
                angle_name = f"γ{checkpoint.layer}"
                angle = parameters[checkpoint.layer - 1]
            elif checkpoint.operation == "mixer":
                angle_name = f"β{checkpoint.layer}"
                angle = parameters[key.depth + checkpoint.layer - 1]
            metrics = checkpoint.as_dict()
            for name in ("checkpoint_index", "checkpoint", "operation", "layer"):
                metrics.pop(name)
            for item in flow:
                if item["kind"] == "state":
                    item["display_position"] = positions[item["basis_index"]]
            frames.append(
                {
                    "checkpoint_index": index,
                    "checkpoint": checkpoint.checkpoint,
                    "operation": checkpoint.operation,
                    "layer": checkpoint.layer,
                    "angle_name": angle_name,
                    "angle": None if angle is None else float(angle),
                    "teaching": teaching[checkpoint.operation],
                    "metrics": metrics,
                    "probability_flow": flow,
                    "state_values": values,
                }
            )

        payload = {
            "schema": "dtu-sciqis-qaoa-dynamics-visualization-run",
            "version": "1.0",
            "source": {
                "kind": "saved_dynamics_artifacts_with_verified_reconstruction",
                "result_root": str(self.result_root),
                "run_id": key.run_id,
                "optimization_rerun": False,
                "amplitude_source": "saved_npz_verified_against_reconstruction" if key.depth == 2 else "deterministic_reconstruction_from_saved_parameters",
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
                "gammas": list(parameters[:key.depth]),
                "betas": list(parameters[key.depth:]),
            },
            "optimizer": {
                "evaluations": int(summary["optimizer_evaluations"]),
                "termination": summary["optimizer_reason"],
                "runtime_seconds": float(summary["optimizer_runtime_seconds"]),
                "trajectory_available": False,
                "trajectory_reason": "The primary artifacts retain final parameters but not the optimizer path.",
            },
            "circuit": self._circuit(key, parameters),
            "checkpoints": list(CHECKPOINTS[key.depth]),
            "display_states": [self._state(key.algorithm, i) for i in display_indices],
            "phase_scatter": {
                "display_positions": [positions[i] for i in phase_indices],
                "downsampled": key.algorithm != "grover_feasible",
                "displayed_count": len(phase_indices),
                "total_count": metadata.dimension,
                "sampling": "all 20 logical feasible states" if key.algorithm == "grover_feasible" else "fixed sample plus all feasible and inspected states",
                "all_feasible_preserved": True,
            },
            "complex_amplitudes": {
                "display_positions": [positions[i] for i in representatives],
                "all_feasible_routes": key.algorithm == "grover_feasible",
                "default_mode": "magnitude",
            },
            "frames": frames,
            "energy_landscape": self.energy_landscape,
            "graph": self.graph_payload,
            "parameter_landscape": self._landscape(key, parameters),
            "summary": summary,
        }

        validation = {
            "run_id": key.run_id,
            "checkpoint_order_correct": tuple(payload["checkpoints"]) == CHECKPOINTS[key.depth],
            "dimension_correct": metadata.dimension == (20 if key.algorithm == "grover_feasible" else 2**14),
            "stored_metric_max_error": errors[0],
            "saved_amplitude_max_error": errors[1],
            "final_distribution_max_error": errors[2],
            "summary_metric_max_error": errors[3],
            "saved_phase_max_wrapped_error": errors[4],
            "saved_phase_magnitude_max_error": errors[5],
            "saved_phase_probability_max_error": errors[6],
            "maximum_norm_error": max(abs(np.linalg.norm(state) - 1) for state in trace.statevectors),
            "maximum_probability_sum_error": max(abs(np.sum(np.abs(state) ** 2) - 1) for state in trace.statevectors),
            "maximum_cost_probability_delta": max(item.cost_probability_max_delta for item in trace.physics),
            "maximum_cost_energy_delta": max(abs(item.cost_energy_delta) for item in trace.physics),
            "p_opt_not_above_p_feas": all(item.p_opt <= item.p_feas + VALIDATION_TOLERANCE for item in trace.checkpoints),
            "probability_flow_normalized": all(abs(sum(item["probability"] for item in frame["probability_flow"]) - 1) <= VALIDATION_TOLERANCE for frame in frames),
            "all_feasible_states_in_phase_view": all(i in phase_indices for i in np.flatnonzero(metadata.feasible_mask)),
        }
        return payload, validation

    def _validation_report(self, rows):
        maximum = lambda name: max(float(row[name]) for row in rows)
        routes_decode = all(
            self.states[edge_vector_to_state_index(route.edge_bitstring)].decoded_route == route.node_sequence
            for route in self.basis.routes
        )
        checks = {
            "checkpoint_order_correct": all(row["checkpoint_order_correct"] for row in rows),
            "trace_norms_approximately_one": maximum("maximum_norm_error") <= VALIDATION_TOLERANCE,
            "probabilities_sum_approximately_one": maximum("maximum_probability_sum_error") <= VALIDATION_TOLERANCE,
            "cost_step_probability_invariance": maximum("maximum_cost_probability_delta") <= VALIDATION_TOLERANCE,
            "cost_step_expected_hc_invariance": maximum("maximum_cost_energy_delta") <= VALIDATION_TOLERANCE,
            "selected_phase_values_match_saved_data": maximum("saved_phase_max_wrapped_error") <= VALIDATION_TOLERANCE,
            "p_opt_not_above_p_feas": all(row["p_opt_not_above_p_feas"] for row in rows),
            "route_identities_decode_consistently": routes_decode,
            "feasible_grover_states_all_validate": routes_decode,
            "full_space_dimensions_are_2_to_14": all(row["dimension_correct"] for row in rows),
            "displayed_metrics_match_saved_trace": maximum("stored_metric_max_error") <= VALIDATION_TOLERANCE,
            "displayed_final_metrics_match_summary": maximum("summary_metric_max_error") <= VALIDATION_TOLERANCE,
            "saved_distributions_match_reconstruction": maximum("final_distribution_max_error") <= VALIDATION_TOLERANCE,
            "probability_flow_aggregates_normalize": all(row["probability_flow_normalized"] for row in rows),
            "phase_view_preserves_all_feasible_states": all(row["all_feasible_states_in_phase_view"] for row in rows),
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
                "stored_metric_error": maximum("stored_metric_max_error"),
                "saved_amplitude_error": maximum("saved_amplitude_max_error"),
                "saved_phase_wrapped_error": maximum("saved_phase_max_wrapped_error"),
                "saved_phase_magnitude_error": maximum("saved_phase_magnitude_max_error"),
                "saved_phase_probability_error": maximum("saved_phase_probability_max_error"),
                "final_distribution_error": maximum("final_distribution_max_error"),
                "summary_metric_error": maximum("summary_metric_max_error"),
                "norm_error": maximum("maximum_norm_error"),
                "probability_sum_error": maximum("maximum_probability_sum_error"),
                "cost_probability_delta": maximum("maximum_cost_probability_delta"),
                "cost_energy_delta": maximum("maximum_cost_energy_delta"),
            },
            "runs": rows,
        }

    def catalog(self) -> dict[str, Any]:
        summaries = []
        for key in self.keys:
            row = self.summaries[key]
            summaries.append(
                {
                    "algorithm": key.algorithm,
                    "algorithm_label": DISPLAY_NAMES[key.algorithm],
                    "algorithm_short_label": SHORT_NAMES[key.algorithm],
                    "depth": key.depth,
                    "search_dimension": int(row["search_dimension"]),
                    "p_feas": float(row["p_feas"]),
                    "p_opt": float(row["p_opt"]),
                    "invalid_mass": float(row["invalid_mass"]),
                    "expected_hc": float(row["expected_hc"]),
                }
            )
        scenes = [
            {"number": 1, "title": "Initial uniform state", "algorithm": "grover_global", "depth": 2, "checkpoint": 0, "panels": ["circuit", "metrics", "probability", "route"], "message": "The full-space run begins uniformly: every bitstring has probability 1/16,384."},
            {"number": 2, "title": "Cost encodes phase—not probability", "algorithm": "grover_global", "depth": 2, "checkpoint": 1, "panels": ["circuit", "energy", "complex", "phase"], "message": "U_C changes relative phase while probability and ⟨H_C⟩ remain invariant."},
            {"number": 3, "title": "Mixer interference", "algorithm": "grover_global", "depth": 2, "checkpoint": 2, "panels": ["circuit", "energy", "metrics", "probability", "complex"], "message": "The mixer recombines phase-tagged amplitudes; this step can raise or lower energy."},
            {"number": 4, "title": "Same H_C, different mixer", "algorithm": "grover_global", "depth": 2, "checkpoint": 4, "panels": ["circuit", "energy", "metrics", "geometry"], "message": "Penalty-X and Global-Grover share the full space and cost Hamiltonian; only mixer geometry differs."},
            {"number": 5, "title": "Restrict to feasible routes", "algorithm": "grover_feasible", "depth": 2, "checkpoint": 0, "panels": ["metrics", "probability", "geometry", "route"], "message": "Feasible-Grover starts in a 20-route basis, so p_feas≈1 is structural."},
            {"number": 6, "title": "p=2 final comparison", "algorithm": "grover_feasible", "depth": 2, "checkpoint": 4, "panels": ["energy", "metrics", "probability", "route"], "message": "Compare final probability mass against each representation's initial baseline."},
            {"number": 7, "title": "Classical optimizer around the circuit", "algorithm": "grover_global", "depth": 1, "checkpoint": 2, "panels": ["circuit", "optimizer"], "message": "COBYLA chooses angles from objective evaluations; the exact path was not saved."},
        ]
        return {
            "schema": "dtu-sciqis-qaoa-dynamics-visualization-catalog",
            "version": "1.0",
            "read_only": True,
            "optimization_rerun": False,
            "algorithms": [
                {"id": algorithm, "label": DISPLAY_NAMES[algorithm], "short_label": SHORT_NAMES[algorithm]}
                for algorithm in ALGORITHM_ORDER
            ],
            "depths": list(DEPTHS),
            "default": {"algorithm": "grover_global", "depth": 2},
            "summaries": summaries,
            "presentation_scenes": scenes,
            "pedagogy": [
                "H_C encodes objective values into relative phase.",
                "H_M determines how amplitudes can mix.",
                "Interference converts phase structure into probability redistribution.",
                "The classical optimizer selects gamma and beta.",
                "A cost layer does not itself lower <H_C>.",
                "Intermediate QAOA energy need not decrease monotonically.",
                "Feasible-Grover p_feas=1 follows from representation restriction.",
            ],
            "energy_landscape_summary": {
                name: self.energy_landscape[name]
                for name in ("total_state_count", "feasible_state_count", "infeasible_state_count", "feasible_fraction")
            },
            "environment": self.environment,
            "validation": {
                "all_checks_passed": self.validation_report["all_checks_passed"],
                "tolerance": self.validation_report["tolerance"],
            },
        }

    def load_run(self, algorithm: str, depth: int) -> dict[str, Any]:
        key = RunKey(str(algorithm), int(depth))
        if key not in self.payloads:
            raise KeyError(f"unknown_dynamics_visualization_run:{algorithm}:p{depth}")
        return self.payloads[key]
