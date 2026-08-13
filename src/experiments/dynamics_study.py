"""Run the course comparison of Penalty-X and two Grover mixers.

The file first prepares shared data, then runs each algorithm, and finally
saves CSV/JSON files for the figures.
"""

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
import json
import platform
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from support.exact_reference import compute_exact_reference
from feasible_qaoa import (
    UNIFORM_FEASIBLE,
    FeasibleRouteBasis,
    build_feasible_initial_state,
    build_feasible_route_basis,
    build_logical_cost_hamiltonian,
)
from qaoa import (
    GROVER_GLOBAL,
    GlobalGroverMixer,
    build_global_grover_mixer,
    simulate_global_grover_state,
)
from graph import DEFAULT_GRAPH_PATH, EXPECTED_EDGE_COUNT, load_graph
from qubo import max_qubo_ising_error, qubo_to_ising
from feasible_experiments import (
    EXPECTATION_LOSS,
    GM_QAOA_EXPECTATION,
    FinalOptimizationResult,
    GroverFeasibleMixer,
    build_grover_feasible_mixer,
    build_incumbent_threshold,
    optimize_final_variant,
    simulate_final_improvement,
)
from qaoa import (
    Q1_PENALTY_X,
    apply_x_mixer,
    normalized_diagonal,
    simulate_qaoa_state,
    standard_plus_state,
    state_probabilities,
)
from experiments.dynamics_trace import BasisMetadata, EvolutionTrace, trace_qaoa_evolution
from qubo import StateRecord, build_qubo, enumerate_state_space
from utils import OptimizationResult, optimize_cobyla


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "qaoa_dynamics_deep_dive.json"
PENALTY_X = "penalty_x"
GROVER_FEASIBLE = "grover_feasible"
ALGORITHMS = (PENALTY_X, GROVER_GLOBAL, GROVER_FEASIBLE)
DISPLAY_NAMES = {
    PENALTY_X: "Penalty-X QAOA",
    GROVER_GLOBAL: "Global Grover-Mixer QAOA",
    GROVER_FEASIBLE: "Feasible Grover-Mixer QAOA",
}


# Data shared by all runs, followed by the result from one run.


@dataclass(frozen=True)
class StudyContext:
    graph: Any
    states: tuple[StateRecord, ...]
    basis: FeasibleRouteBasis
    exact_route: tuple[int, ...]
    exact_cost: int
    penalty: float
    raw_full_diagonal: np.ndarray
    normalized_full_diagonal: np.ndarray
    full_normalization_shift: float
    full_normalization_scale: float
    normalized_feasible_diagonal: np.ndarray
    full_metadata: BasisMetadata
    feasible_metadata: BasisMetadata
    global_mixer: GlobalGroverMixer
    feasible_mixer: GroverFeasibleMixer
    full_initial_state: np.ndarray
    feasible_initial_state: np.ndarray
    setup_timings: dict[str, float]
    validations: dict[str, object]


@dataclass(frozen=True)
class StudyRun:
    run_id: str
    algorithm: str
    display_name: str
    depth: int
    seed: int
    evaluation_budget: int
    representation: str
    search_dimension: int
    mixer_type: str
    optimized_parameters: tuple[float, ...]
    optimizer_evaluations: int
    optimizer_success: bool
    optimizer_reason: str
    optimizer_runtime_seconds: float
    analysis_runtime_seconds: float
    conditional_feasible_cost: float | None
    trace: EvolutionTrace

    @property
    def final(self):
        return self.trace.checkpoints[-1]

    @property
    def initial(self):
        return self.trace.checkpoints[0]

    def summary_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "display_name": self.display_name,
            "representation": self.representation,
            "search_dimension": self.search_dimension,
            "mixer_type": self.mixer_type,
            "p": self.depth,
            "seed": self.seed,
            "evaluation_budget": self.evaluation_budget,
            "p_feas": self.final.p_feas,
            "p_opt": self.final.p_opt,
            "invalid_mass": self.final.invalid_mass,
            "expected_hc": self.final.expected_hc,
            "expected_route_cost_conditional_feasible": self.conditional_feasible_cost,
            "expected_routing_term": self.final.expected_routing_term,
            "expected_flow_penalty": self.final.expected_flow_penalty,
            "expected_penalty_contribution": self.final.expected_penalty_contribution,
            "optimized_parameters": list(self.optimized_parameters),
            "optimizer_evaluations": self.optimizer_evaluations,
            "optimizer_success": self.optimizer_success,
            "optimizer_reason": self.optimizer_reason,
            "optimizer_runtime_seconds": self.optimizer_runtime_seconds,
            "analysis_runtime_seconds": self.analysis_runtime_seconds,
            "initial_p_feas": self.initial.p_feas,
            "initial_p_opt": self.initial.p_opt,
            "initial_expected_hc": self.initial.expected_hc,
        }


def load_study_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "dtu-sciqis-qaoa-dynamics-deep-dive":
        raise ValueError("unsupported_dynamics_config_schema")
    if tuple(payload["algorithms"]) != ALGORITHMS or tuple(payload["depths"]) != (1, 2):
        raise ValueError("dynamics_primary_matrix_must_be_three_algorithms_p1_p2")
    if int(payload["seed"]) != 2601 or payload["optimizer"] != "COBYLA":
        raise ValueError("dynamics_frozen_optimizer_policy_mismatch")
    return payload


# Build all graph, QUBO, basis and mixer objects once before optimization.


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return sha256(contiguous.tobytes()).hexdigest()


def _full_metadata(
    states: Sequence[StateRecord], penalty: float, optimal_cost: int
) -> BasisMetadata:
    return BasisMetadata(
        basis_labels=tuple(state.canonical_bitstring for state in states),
        decoded_routes=tuple(state.decoded_route for state in states),
        route_costs=tuple(float(state.routing_cost) for state in states),
        flow_penalties=tuple(float(state.flow_penalty) for state in states),
        total_energies=tuple(
            float(state.routing_cost + penalty * state.flow_penalty) for state in states
        ),
        feasible_mask=tuple(state.is_decoder_valid for state in states),
        optimal_mask=tuple(
            state.is_decoder_valid and state.routing_cost == int(optimal_cost)
            for state in states
        ),
        penalty_coefficient=float(penalty),
        representation="edge_bits_all_bitstrings",
    )


def _feasible_metadata(basis: FeasibleRouteBasis) -> BasisMetadata:
    return BasisMetadata(
        basis_labels=tuple(
            f"route_{route.route_id}:{route.bitstring_text}" for route in basis.routes
        ),
        decoded_routes=tuple(route.node_sequence for route in basis.routes),
        route_costs=tuple(float(route.routing_cost) for route in basis.routes),
        flow_penalties=tuple(0.0 for _route in basis.routes),
        total_energies=tuple(float(route.routing_cost) for route in basis.routes),
        feasible_mask=tuple(True for _route in basis.routes),
        optimal_mask=tuple(route.exact_optimal for route in basis.routes),
        penalty_coefficient=None,
        representation="logical_feasible_routes",
    )


def prepare_study_context(*, penalty: float = 6.0) -> StudyContext:
    """Build and time the frozen scientific objects without optimizing."""

    timings: dict[str, float] = {}
    started = perf_counter()
    graph = load_graph(DEFAULT_GRAPH_PATH)
    exact_payload, _ = compute_exact_reference(graph, graph_path=DEFAULT_GRAPH_PATH)
    states = enumerate_state_space(graph)
    qubo = build_qubo(graph, float(penalty))
    ising = qubo_to_ising(qubo)
    raw = np.asarray(
        [float(qubo.evaluate(state.edge_vector)) for state in states], dtype=np.float64
    )
    normalized, shift, scale = normalized_diagonal(raw)
    qubo_ising_error = float(max_qubo_ising_error(states, [qubo]))
    timings["full_model_qubo_ising_build_seconds"] = perf_counter() - started

    exact_route = tuple(exact_payload["exact_reference"]["node_path"])
    exact_cost = int(exact_payload["exact_reference"]["cost"])
    full_metadata = _full_metadata(states, float(penalty), exact_cost)

    started = perf_counter()
    full_initial = standard_plus_state(EXPECTED_EDGE_COUNT)
    timings["penalty_x_state_mixer_build_seconds"] = perf_counter() - started
    started = perf_counter()
    global_mixer = build_global_grover_mixer(EXPECTED_EDGE_COUNT)
    global_initial = global_mixer.initial_state()
    timings["global_grover_state_mixer_build_seconds"] = perf_counter() - started

    started = perf_counter()
    basis = build_feasible_route_basis(graph)
    timings["feasible_path_enumeration_seconds"] = perf_counter() - started
    started = perf_counter()
    feasible_costs = build_logical_cost_hamiltonian(basis)
    timings["feasible_cost_vector_build_seconds"] = perf_counter() - started
    started = perf_counter()
    feasible_mixer = build_grover_feasible_mixer(basis.size)
    feasible_initial_record = build_feasible_initial_state(
        basis, feasible_costs, mode=UNIFORM_FEASIBLE
    )
    feasible_initial = np.asarray(feasible_initial_record.amplitudes, dtype=np.complex128)
    timings["feasible_grover_state_mixer_build_seconds"] = perf_counter() - started
    feasible_metadata = _feasible_metadata(basis)

    full_sha = _array_sha256(raw)
    validations: dict[str, object] = {
        "qubo_ising_max_error": qubo_ising_error,
        "penalty_x_hc_sha256": full_sha,
        "grover_global_hc_sha256": _array_sha256(raw.copy()),
        "penalty_x_global_hc_array_equal": bool(np.array_equal(raw, raw.copy())),
        "penalty_x_global_hc_byte_identical": bool(raw.tobytes() == raw.copy().tobytes()),
        "global_dimension": global_mixer.dimension,
        "expected_global_dimension": 2**EXPECTED_EDGE_COUNT,
        "full_initial_uniform_probability": float(abs(global_initial[0]) ** 2),
        "expected_full_initial_uniform_probability": 1.0 / (2**EXPECTED_EDGE_COUNT),
        "full_initial_states_identical": bool(np.array_equal(full_initial, global_initial)),
        "feasible_basis_size": basis.size,
        "feasible_initial_probability_sum": float(np.sum(np.abs(feasible_initial) ** 2)),
        "grover_convention_analogous": (
            "H_G,all=|s_all><s_all| and H_G,F=|s_F><s_F|; both use "
            "I+(exp(-i beta)-1)P"
        ),
        "dense_global_mixer_constructed": False,
    }
    if (
        qubo_ising_error != 0.0
        or not validations["penalty_x_global_hc_byte_identical"]
        or global_mixer.dimension != len(states)
        or basis.size != 20
    ):
        raise RuntimeError("dynamics_context_scientific_validation_failed")
    return StudyContext(
        graph=graph,
        states=tuple(states),
        basis=basis,
        exact_route=exact_route,
        exact_cost=exact_cost,
        penalty=float(penalty),
        raw_full_diagonal=raw,
        normalized_full_diagonal=normalized,
        full_normalization_shift=float(shift),
        full_normalization_scale=float(scale),
        normalized_feasible_diagonal=np.asarray(
            feasible_costs.normalized_energies, dtype=np.float64
        ),
        full_metadata=full_metadata,
        feasible_metadata=feasible_metadata,
        global_mixer=global_mixer,
        feasible_mixer=feasible_mixer,
        full_initial_state=full_initial,
        feasible_initial_state=feasible_initial,
        setup_timings=timings,
        validations=validations,
    )


def _x_mixer_evolve(state: Sequence[complex], beta: float) -> np.ndarray:
    # Preserve the existing course parameter convention exactly.
    return apply_x_mixer(
        np.asarray(state, dtype=np.complex128),
        float(beta) / EXPECTED_EDGE_COUNT,
        EXPECTED_EDGE_COUNT,
    )


def _optimize_full(
    algorithm: str,
    context: StudyContext,
    *,
    depth: int,
    seed: int,
    evaluation_budget: int,
    rhobeg: float,
    tolerance: float,
) -> OptimizationResult:
    if algorithm == PENALTY_X:
        def simulator(diagonal, parameters, *, depth):
            return simulate_qaoa_state(
                diagonal, parameters, depth=depth, solver=Q1_PENALTY_X
            )
    elif algorithm == GROVER_GLOBAL:
        simulator = simulate_global_grover_state
    else:
        raise ValueError(f"not_a_full_space_algorithm:{algorithm}")

    def objective(parameters: np.ndarray) -> float:
        state = simulator(
            context.normalized_full_diagonal, parameters, depth=int(depth)
        )
        probabilities = state_probabilities(state)
        return float(probabilities @ context.normalized_full_diagonal)

    return optimize_cobyla(
        objective,
        depth=int(depth),
        seed=int(seed),
        evaluation_budget=int(evaluation_budget),
        rhobeg=float(rhobeg),
        tolerance=float(tolerance),
    )


def _optimize_feasible(
    context: StudyContext,
    *,
    depth: int,
    seed: int,
    evaluation_budget: int,
    rhobeg: float,
    tolerance: float,
) -> FinalOptimizationResult:
    raw_costs = np.asarray(context.feasible_metadata.route_costs)
    incumbent_cost = raw_costs[context.basis.incumbent_route_id]
    threshold = build_incumbent_threshold(raw_costs, float(incumbent_cost))
    optimized, _ = optimize_final_variant(
        context.feasible_initial_state,
        context.normalized_feasible_diagonal,
        context.feasible_mixer,
        context.normalized_feasible_diagonal,
        threshold.better_mask,
        loss_kind=EXPECTATION_LOSS,
        depth=int(depth),
        seed=int(seed),
        evaluation_budget=int(evaluation_budget),
        rhobeg=float(rhobeg),
        tolerance=float(tolerance),
    )
    return optimized


def _conditional_feasible_cost(
    probabilities: np.ndarray, metadata: BasisMetadata
) -> float | None:
    feasible = np.asarray(metadata.feasible_mask, dtype=bool)
    mass = float(np.sum(probabilities[feasible]))
    if mass <= 1e-15:
        return None
    costs = np.asarray(metadata.route_costs, dtype=np.float64)
    return float(probabilities[feasible] @ costs[feasible] / mass)


def run_algorithm(
    context: StudyContext,
    algorithm: str,
    *,
    depth: int,
    seed: int = 2601,
    evaluation_budget: int = 100,
    rhobeg: float = 0.5,
    tolerance: float = 1e-8,
) -> StudyRun:
    """Optimize one primary cell and trace its retained final parameters."""

    if algorithm not in ALGORITHMS or int(depth) not in (1, 2):
        raise ValueError("invalid_primary_dynamics_cell")
    if algorithm in (PENALTY_X, GROVER_GLOBAL):
        optimized = _optimize_full(
            algorithm,
            context,
            depth=depth,
            seed=seed,
            evaluation_budget=evaluation_budget,
            rhobeg=rhobeg,
            tolerance=tolerance,
        )
        parameters = optimized.parameters
        metadata = context.full_metadata
        initial = context.full_initial_state
        phase = context.normalized_full_diagonal
        mixer = _x_mixer_evolve if algorithm == PENALTY_X else context.global_mixer.evolve
        representation = "14 edge bits; all 2^14 bitstrings"
        mixer_type = (
            "H_X=sum_j X_j (course beta/q scaling)"
            if algorithm == PENALTY_X
            else "H_G,all=|s_all><s_all| (rank-one update)"
        )
        evaluations = optimized.evaluations
        success = optimized.success
        reason = optimized.reason
        optimizer_runtime = optimized.wall_time
    else:
        feasible_optimized = _optimize_feasible(
            context,
            depth=depth,
            seed=seed,
            evaluation_budget=evaluation_budget,
            rhobeg=rhobeg,
            tolerance=tolerance,
        )
        parameters = feasible_optimized.final_parameters
        metadata = context.feasible_metadata
        initial = context.feasible_initial_state
        phase = context.normalized_feasible_diagonal
        mixer = context.feasible_mixer.evolve
        representation = "logical basis of enumerated feasible routes"
        mixer_type = "H_G,F=|s_F><s_F|"
        evaluations = feasible_optimized.evaluations
        success = feasible_optimized.success
        reason = feasible_optimized.reason
        optimizer_runtime = feasible_optimized.runtime_seconds

    started = perf_counter()
    trace = trace_qaoa_evolution(
        initial,
        phase,
        mixer,
        parameters,
        depth=int(depth),
        metadata=metadata,
    )
    analysis_runtime = perf_counter() - started
    run = StudyRun(
        run_id=f"{algorithm}_p{int(depth)}_seed{int(seed)}",
        algorithm=algorithm,
        display_name=DISPLAY_NAMES[algorithm],
        depth=int(depth),
        seed=int(seed),
        evaluation_budget=int(evaluation_budget),
        representation=representation,
        search_dimension=metadata.dimension,
        mixer_type=mixer_type,
        optimized_parameters=tuple(map(float, parameters)),
        optimizer_evaluations=int(evaluations),
        optimizer_success=bool(success),
        optimizer_reason=str(reason),
        optimizer_runtime_seconds=float(optimizer_runtime),
        analysis_runtime_seconds=float(analysis_runtime),
        conditional_feasible_cost=_conditional_feasible_cost(
            np.abs(trace.statevectors[-1]) ** 2, metadata
        ),
        trace=trace,
    )
    final_probabilities = np.abs(trace.statevectors[-1]) ** 2
    if run.final.p_opt > run.final.p_feas + 1e-10:
        raise RuntimeError("p_opt_exceeds_p_feas")
    if not np.isclose(np.sum(final_probabilities), 1.0, atol=1e-10):
        raise RuntimeError("saved_distribution_not_normalized")
    if algorithm == GROVER_FEASIBLE and abs(run.final.p_feas - 1.0) > 1e-10:
        raise RuntimeError("feasible_grover_leaked_outside_logical_basis")
    return run


def run_primary_matrix(
    context: StudyContext,
    config: dict[str, Any],
) -> tuple[StudyRun, ...]:
    runs = []
    for algorithm in config["algorithms"]:
        for depth in config["depths"]:
            runs.append(
                run_algorithm(
                    context,
                    algorithm,
                    depth=int(depth),
                    seed=int(config["seed"]),
                    evaluation_budget=int(config["objective_evaluation_cap"]),
                    rhobeg=float(config["optimizer_rhobeg"]),
                    tolerance=float(config["optimizer_tolerance"]),
                )
            )
    return tuple(runs)


def _algorithm_objects(
    context: StudyContext, algorithm: str
) -> tuple[np.ndarray, np.ndarray, Callable[[Sequence[complex], float], np.ndarray], BasisMetadata]:
    if algorithm == PENALTY_X:
        return (
            context.full_initial_state,
            context.normalized_full_diagonal,
            _x_mixer_evolve,
            context.full_metadata,
        )
    if algorithm == GROVER_GLOBAL:
        return (
            context.full_initial_state,
            context.normalized_full_diagonal,
            context.global_mixer.evolve,
            context.full_metadata,
        )
    if algorithm == GROVER_FEASIBLE:
        return (
            context.feasible_initial_state,
            context.normalized_feasible_diagonal,
            context.feasible_mixer.evolve,
            context.feasible_metadata,
        )
    raise ValueError(f"unsupported_algorithm:{algorithm}")


def parameter_landscape_rows(
    context: StudyContext,
    algorithm: str,
    *,
    grid_points: int,
) -> list[dict[str, object]]:
    """Evaluate a frozen p=1 grid; this never changes optimized parameters."""

    count = int(grid_points)
    if count < 3:
        raise ValueError("parameter_landscape_grid_too_small")
    initial, phase, mixer, metadata = _algorithm_objects(context, algorithm)
    feasible = np.asarray(metadata.feasible_mask, dtype=bool)
    optimal = np.asarray(metadata.optimal_mask, dtype=bool)
    total = np.asarray(metadata.total_energies, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for gamma in np.linspace(0.0, 2.0 * pi, count):
        after_cost = initial * np.exp(-1j * float(gamma) * phase)
        for beta in np.linspace(0.0, pi, count):
            state = np.asarray(mixer(after_cost, float(beta)), dtype=np.complex128)
            probabilities = np.abs(state) ** 2
            rows.append(
                {
                    "algorithm": algorithm,
                    "gamma": float(gamma),
                    "beta": float(beta),
                    "expected_normalized_objective": float(probabilities @ phase),
                    "expected_hc": float(probabilities @ total),
                    "p_feas": float(np.sum(probabilities[feasible])),
                    "p_opt": float(np.sum(probabilities[optimal])),
                    "probability_sum": float(np.sum(probabilities)),
                }
            )
    return rows


def energy_landscape_rows(context: StudyContext) -> list[dict[str, object]]:
    optimal = np.asarray(context.full_metadata.optimal_mask, dtype=bool)
    rows = []
    for state, is_optimal in zip(context.states, optimal):
        rows.append(
            {
                "state_index": state.state_index,
                "canonical_bitstring": state.canonical_bitstring,
                "routing_term": state.routing_cost,
                "flow_penalty": state.flow_penalty,
                "penalty_contribution": context.penalty * state.flow_penalty,
                "total_qubo_energy": state.routing_cost
                + context.penalty * state.flow_penalty,
                "feasible": state.is_decoder_valid,
                "exact_optimal": bool(is_optimal),
                "decoded_route": ""
                if state.decoded_route is None
                else "->".join(map(str, state.decoded_route)),
            }
        )
    return rows


def energy_landscape_summary(context: StudyContext) -> dict[str, object]:
    energies = np.asarray(context.full_metadata.total_energies)
    route_costs = np.asarray(context.full_metadata.route_costs)
    feasible = np.asarray(context.full_metadata.feasible_mask, dtype=bool)
    optimal = np.asarray(context.full_metadata.optimal_mask, dtype=bool)
    best = float(np.min(energies[optimal]))
    feasible_competitors = energies[feasible & ~optimal]
    infeasible_energies = energies[~feasible]
    return {
        "state_count": len(energies),
        "feasible_state_count": int(np.sum(feasible)),
        "feasible_fraction": float(np.mean(feasible)),
        "infeasible_state_count": int(np.sum(~feasible)),
        "optimal_state_count": int(np.sum(optimal)),
        "optimal_energy": best,
        "second_best_feasible_energy": float(np.min(feasible_competitors)),
        "feasible_optimal_gap": float(np.min(feasible_competitors) - best),
        "lowest_infeasible_energy": float(np.min(infeasible_energies)),
        "infeasible_gap_above_optimum": float(np.min(infeasible_energies) - best),
        "infeasible_state_below_optimum_count": int(np.sum(infeasible_energies < best)),
        "minimum_routing_term_infeasible": float(np.min(route_costs[~feasible])),
        "penalty": context.penalty,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot_write_empty_csv:{path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not_json_serializable:{type(value).__name__}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _phase_rows(run: StudyRun, metadata: BasisMetadata) -> list[dict[str, object]]:
    selected_checkpoints = {0, 1, 2, len(run.trace.checkpoints) - 1}
    rows: list[dict[str, object]] = []
    energies = np.asarray(metadata.total_energies)
    feasible = np.asarray(metadata.feasible_mask)
    optimal = np.asarray(metadata.optimal_mask)
    for checkpoint_index in sorted(selected_checkpoints):
        checkpoint = run.trace.checkpoints[checkpoint_index]
        vector = run.trace.statevectors[checkpoint_index]
        probabilities = np.abs(vector) ** 2
        for index, amplitude in enumerate(vector):
            rows.append(
                {
                    "run_id": run.run_id,
                    "checkpoint_index": checkpoint_index,
                    "checkpoint": checkpoint.checkpoint,
                    "basis_index": index,
                    "basis_label": metadata.basis_labels[index],
                    "energy": float(energies[index]),
                    "magnitude": float(abs(amplitude)),
                    "probability": float(probabilities[index]),
                    "phase": float(np.angle(amplitude)),
                    "feasible": bool(feasible[index]),
                    "exact_optimal": bool(optimal[index]),
                }
            )
    return rows


def _selected_state_indices(run: StudyRun, metadata: BasisMetadata) -> list[int]:
    energies = np.asarray(metadata.total_energies)
    feasible = np.asarray(metadata.feasible_mask, dtype=bool)
    optimal = np.asarray(metadata.optimal_mask, dtype=bool)
    indices = list(map(int, np.flatnonzero(optimal)))
    feasible_order = np.flatnonzero(feasible)[np.argsort(energies[feasible], kind="stable")]
    indices.extend(map(int, feasible_order[: min(4, len(feasible_order))]))
    if np.any(~feasible):
        infeasible_order = np.flatnonzero(~feasible)[
            np.argsort(energies[~feasible], kind="stable")
        ]
        indices.extend(map(int, infeasible_order[:3]))
        final_prob = np.abs(run.trace.statevectors[-1]) ** 2
        top_infeasible = np.flatnonzero(~feasible)[np.argmax(final_prob[~feasible])]
        indices.append(int(top_infeasible))
    return list(dict.fromkeys(indices))


def save_study_artifacts(
    context: StudyContext,
    runs: Sequence[StudyRun],
    config: dict[str, Any],
    *,
    result_root: Path,
) -> dict[str, object]:
    """Write all tables needed to regenerate analysis and figures."""

    result_root.mkdir(parents=True, exist_ok=True)
    write_json(result_root / "experiment_config.json", config)
    write_json(
        result_root / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
            "networkx": package_version("networkx"),
            "qiskit": package_version("qiskit"),
            "optimizer": "scipy.optimize.minimize(method=COBYLA)",
            "statevector_mode": "ideal_exact_numpy_complex128",
        },
    )
    write_json(result_root / "scientific_validations.json", context.validations)
    write_json(result_root / "energy_landscape_summary.json", energy_landscape_summary(context))
    _write_csv(result_root / "energy_landscape.csv", energy_landscape_rows(context))

    summaries = [run.summary_dict() for run in runs]
    _write_csv(result_root / "final_summary.csv", summaries)
    write_json(result_root / "final_summary.json", summaries)
    parameter_rows = []
    for run in runs:
        row = {
            "run_id": run.run_id,
            "algorithm": run.algorithm,
            "p": run.depth,
            "seed": run.seed,
            "gamma_1": run.optimized_parameters[0],
            "gamma_2": None,
            "beta_1": run.optimized_parameters[run.depth],
            "beta_2": None,
        }
        if run.depth >= 2:
            row["gamma_2"] = run.optimized_parameters[1]
            row["beta_2"] = run.optimized_parameters[run.depth + 1]
        parameter_rows.append(row)
    _write_csv(result_root / "optimized_parameters.csv", parameter_rows)

    trace_rows: list[dict[str, object]] = []
    physics_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for run in runs:
        metadata = (
            context.feasible_metadata
            if run.algorithm == GROVER_FEASIBLE
            else context.full_metadata
        )
        for checkpoint in run.trace.checkpoints:
            trace_rows.append(
                {
                    "run_id": run.run_id,
                    "algorithm": run.algorithm,
                    "p": run.depth,
                    **checkpoint.as_dict(),
                }
            )
        for record in run.trace.physics:
            physics_rows.append(
                {
                    "run_id": run.run_id,
                    "algorithm": run.algorithm,
                    "p": run.depth,
                    **record.as_dict(),
                }
            )
        final_probability = np.abs(run.trace.statevectors[-1]) ** 2
        distribution_dir = result_root / "distributions"
        distribution_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            distribution_dir / f"{run.run_id}.npz",
            probability=final_probability,
            parameters=np.asarray(run.optimized_parameters),
        )
        write_json(
            result_root / "runs" / f"{run.run_id}.json",
            {
                "summary": run.summary_dict(),
                "checkpoints": [item.as_dict() for item in run.trace.checkpoints],
                "physics": [item.as_dict() for item in run.trace.physics],
            },
        )
        if run.depth == 2:
            states_array = np.stack(run.trace.statevectors)
            amplitude_dir = result_root / "amplitudes"
            amplitude_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                amplitude_dir / f"{run.run_id}.npz",
                real=states_array.real,
                imag=states_array.imag,
                checkpoint=np.asarray(
                    [item.checkpoint for item in run.trace.checkpoints], dtype="U16"
                ),
            )
            _write_csv(
                result_root / "phase_analysis" / f"{run.run_id}.csv",
                _phase_rows(run, metadata),
            )
            selected = _selected_state_indices(run, metadata)
            for checkpoint, statevector in zip(run.trace.checkpoints, run.trace.statevectors):
                for index in selected:
                    amplitude = statevector[index]
                    top_rows.append(
                        {
                            "run_id": run.run_id,
                            "algorithm": run.algorithm,
                            "checkpoint_index": checkpoint.checkpoint_index,
                            "checkpoint": checkpoint.checkpoint,
                            "basis_index": index,
                            "basis_label": metadata.basis_labels[index],
                            "energy": metadata.total_energies[index],
                            "feasible": metadata.feasible_mask[index],
                            "exact_optimal": metadata.optimal_mask[index],
                            "decoded_route": ""
                            if metadata.decoded_routes[index] is None
                            else "->".join(map(str, metadata.decoded_routes[index])),
                            "magnitude": float(abs(amplitude)),
                            "probability": float(abs(amplitude) ** 2),
                            "phase": float(np.angle(amplitude)),
                            "real": float(amplitude.real),
                            "imag": float(amplitude.imag),
                        }
                    )
    _write_csv(result_root / "dynamics_trace.csv", trace_rows)
    _write_csv(result_root / "physics_validation.csv", physics_rows)
    _write_csv(result_root / "top_state_evolution.csv", top_rows)

    landscape_rows: list[dict[str, object]] = []
    count = int(config["parameter_landscape_grid_points_per_axis"])
    for algorithm in ALGORITHMS:
        rows = parameter_landscape_rows(context, algorithm, grid_points=count)
        _write_csv(result_root / "parameter_landscapes" / f"{algorithm}_p1.csv", rows)
        landscape_rows.extend(rows)

    runtime_rows = []
    for component, seconds in context.setup_timings.items():
        runtime_rows.append(
            {
                "scope": "setup",
                "run_id": "shared",
                "component": component,
                "seconds": seconds,
            }
        )
    for run in runs:
        runtime_rows.extend(
            (
                {
                    "scope": "run",
                    "run_id": run.run_id,
                    "component": "optimize_evaluate_seconds",
                    "seconds": run.optimizer_runtime_seconds,
                },
                {
                    "scope": "run",
                    "run_id": run.run_id,
                    "component": "decode_analysis_trace_seconds",
                    "seconds": run.analysis_runtime_seconds,
                },
            )
        )
    _write_csv(result_root / "runtime_profile.csv", runtime_rows)
    return {
        "result_root": str(result_root),
        "run_count": len(runs),
        "summary_rows": len(summaries),
        "trace_rows": len(trace_rows),
        "physics_rows": len(physics_rows),
        "parameter_landscape_rows": len(landscape_rows),
    }


def regression_comparison(context: StudyContext, run: StudyRun) -> dict[str, object]:
    """Compare traced final states with the untouched historical simulators."""

    if run.algorithm == PENALTY_X:
        reference = simulate_qaoa_state(
            context.normalized_full_diagonal,
            run.optimized_parameters,
            depth=run.depth,
            solver=Q1_PENALTY_X,
        )
    elif run.algorithm == GROVER_FEASIBLE:
        reference = np.asarray(
            simulate_final_improvement(
                context.feasible_initial_state,
                context.normalized_feasible_diagonal,
                context.feasible_mixer,
                run.optimized_parameters,
                depth=run.depth,
            ).state,
            dtype=np.complex128,
        )
    else:
        raise ValueError("regression_oracle_only_exists_for_preserved_algorithms")
    traced = run.trace.statevectors[-1]
    return {
        "run_id": run.run_id,
        "algorithm": run.algorithm,
        "max_amplitude_error": float(np.max(np.abs(reference - traced))),
        "max_probability_error": float(
            np.max(np.abs(np.abs(reference) ** 2 - np.abs(traced) ** 2))
        ),
        "unchanged_within_1e-12": bool(np.allclose(reference, traced, atol=1e-12, rtol=0.0)),
    }


def postrun_scientific_validation(runs: Sequence[StudyRun]) -> dict[str, object]:
    """Assert the core teaching identities across the real optimized runs."""

    checkpoints = [item for run in runs for item in run.trace.checkpoints]
    physics = [item for run in runs for item in run.trace.physics]
    payload = {
        "run_count": len(runs),
        "all_checkpoint_norms_within_1e-10": bool(
            all(abs(item.norm - 1.0) <= 1e-10 for item in checkpoints)
        ),
        "all_checkpoint_probability_sums_within_1e-10": bool(
            all(abs(item.probability_sum - 1.0) <= 1e-10 for item in checkpoints)
        ),
        "all_cost_layers_preserve_probabilities_within_1e-10": bool(
            all(item.cost_probability_max_delta <= 1e-10 for item in physics)
        ),
        "all_cost_layers_preserve_expected_hc_within_1e-10": bool(
            all(abs(item.cost_energy_delta) <= 1e-10 for item in physics)
        ),
        "at_least_one_real_cost_layer_changes_relative_phase": bool(
            any(item.cost_changed_phase for item in physics)
        ),
        "at_least_one_real_mixer_changes_probabilities": bool(
            any(item.mixer_changed_probability for item in physics)
        ),
        "at_least_one_real_mixer_changes_expected_hc": bool(
            any(abs(item.mixer_energy_delta) > 1e-10 for item in physics)
        ),
        "all_p_opt_not_above_p_feas": bool(
            all(item.p_opt <= item.p_feas + 1e-10 for item in checkpoints)
        ),
        "all_feasible_grover_checkpoints_feasible": bool(
            all(
                abs(item.p_feas - 1.0) <= 1e-10
                for run in runs
                if run.algorithm == GROVER_FEASIBLE
                for item in run.trace.checkpoints
            )
        ),
        "checkpoint_counts_correct": bool(
            all(len(run.trace.checkpoints) == 1 + 2 * run.depth for run in runs)
        ),
    }
    if not all(payload[key] for key in payload if key != "run_count"):
        raise RuntimeError(f"postrun_scientific_validation_failed:{payload}")
    return payload
