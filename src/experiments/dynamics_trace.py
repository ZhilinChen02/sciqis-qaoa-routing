"""Record what happens after every cost and mixer layer in QAOA."""

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from metrics import probability_mass, shannon_entropy


TRACE_TOLERANCE = 1e-10


# These data classes are containers for saved tables.  The calculations start
# at apply_cost_layer below.


@dataclass(frozen=True)
class BasisMetadata:
    """Classical labels and diagonal observables for one simulation basis."""

    basis_labels: tuple[str, ...]
    decoded_routes: tuple[tuple[int, ...] | None, ...]
    route_costs: tuple[float, ...]
    flow_penalties: tuple[float, ...]
    total_energies: tuple[float, ...]
    feasible_mask: tuple[bool, ...]
    optimal_mask: tuple[bool, ...]
    penalty_coefficient: float | None
    representation: str

    def __post_init__(self) -> None:
        size = len(self.basis_labels)
        fields = (
            self.decoded_routes,
            self.route_costs,
            self.flow_penalties,
            self.total_energies,
            self.feasible_mask,
            self.optimal_mask,
        )
        if size < 1 or any(len(values) != size for values in fields):
            raise ValueError("basis_metadata_dimension_mismatch")
        feasible = np.asarray(self.feasible_mask, dtype=bool)
        optimal = np.asarray(self.optimal_mask, dtype=bool)
        if not np.any(optimal) or np.any(optimal & ~feasible):
            raise ValueError("optimal_states_must_be_nonempty_and_feasible")

    @property
    def dimension(self) -> int:
        return len(self.basis_labels)


@dataclass(frozen=True)
class CheckpointMetrics:
    checkpoint_index: int
    checkpoint: str
    operation: str
    layer: int
    norm: float
    probability_sum: float
    expected_hc: float
    expected_routing_term: float
    expected_flow_penalty: float | None
    expected_penalty_contribution: float | None
    expected_total_qubo: float
    p_feas: float
    p_opt: float
    invalid_mass: float
    optimum_feasible_mass: float
    other_feasible_mass: float
    infeasible_mass: float
    shannon_entropy: float
    max_basis_probability: float
    top_basis_index: int
    top_basis_label: str
    top_is_feasible: bool
    top_is_optimal: bool
    top_decoded_route: tuple[int, ...] | None
    top_route_cost: float
    top_total_energy: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LayerPhysicsValidation:
    layer: int
    cost_probability_max_delta: float
    cost_energy_delta: float
    cost_norm_delta: float
    cost_state_max_delta: float
    cost_relative_phase_spread: float
    cost_changed_phase: bool
    mixer_probability_max_delta: float
    mixer_energy_delta: float
    mixer_changed_probability: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvolutionTrace:
    """Metrics and exact statevectors at ``initial/cost/mixer`` checkpoints."""

    depth: int
    parameters: tuple[float, ...]
    checkpoints: tuple[CheckpointMetrics, ...]
    statevectors: tuple[np.ndarray, ...]
    physics: tuple[LayerPhysicsValidation, ...]

    def __post_init__(self) -> None:
        expected = 1 + 2 * int(self.depth)
        if len(self.checkpoints) != expected or len(self.statevectors) != expected:
            raise ValueError("trace_checkpoint_count_mismatch")


def apply_cost_layer(
    state: Sequence[complex], phase_energies: Sequence[float], gamma: float
) -> np.ndarray:
    """Apply diagonal ``exp(-i gamma H_C)`` without changing magnitudes."""

    vector = np.asarray(state, dtype=np.complex128)
    diagonal = np.asarray(phase_energies, dtype=np.float64)
    if vector.ndim != 1 or vector.shape != diagonal.shape:
        raise ValueError("cost_layer_state_diagonal_dimension_mismatch")
    if np.any(~np.isfinite(diagonal)):
        raise ValueError("cost_layer_diagonal_nonfinite")
    return vector * np.exp(-1j * float(gamma) * diagonal)


def _checkpoint_metrics(
    state: np.ndarray,
    metadata: BasisMetadata,
    *,
    checkpoint_index: int,
    checkpoint: str,
    operation: str,
    layer: int,
) -> CheckpointMetrics:
    vector = np.asarray(state, dtype=np.complex128)
    if vector.shape != (metadata.dimension,):
        raise ValueError("checkpoint_state_metadata_dimension_mismatch")
    probabilities = np.abs(vector) ** 2
    probability_sum = float(np.sum(probabilities))
    if not np.isclose(probability_sum, 1.0, atol=TRACE_TOLERANCE, rtol=0.0):
        raise RuntimeError(f"trace_probability_not_normalized:{probability_sum}")
    feasible_mask = np.asarray(metadata.feasible_mask, dtype=bool)
    optimal_mask = np.asarray(metadata.optimal_mask, dtype=bool)
    route_costs = np.asarray(metadata.route_costs, dtype=np.float64)
    flow_penalties = np.asarray(metadata.flow_penalties, dtype=np.float64)
    total_energies = np.asarray(metadata.total_energies, dtype=np.float64)

    p_feas = probability_mass(probabilities, feasible_mask)
    p_opt = probability_mass(probabilities, optimal_mask)
    infeasible_mass = probability_mass(probabilities, ~feasible_mask)
    other_feasible = probability_mass(probabilities, feasible_mask & ~optimal_mask)
    order = np.lexsort((np.arange(metadata.dimension), -np.round(probabilities, 15)))
    top = int(order[0])
    expected_penalty = float(probabilities @ flow_penalties)
    coefficient = metadata.penalty_coefficient
    penalty_contribution = (
        None if coefficient is None else float(coefficient * expected_penalty)
    )
    return CheckpointMetrics(
        checkpoint_index=int(checkpoint_index),
        checkpoint=checkpoint,
        operation=operation,
        layer=int(layer),
        norm=float(np.linalg.norm(vector)),
        probability_sum=probability_sum,
        expected_hc=float(probabilities @ total_energies),
        expected_routing_term=float(probabilities @ route_costs),
        expected_flow_penalty=None if coefficient is None else expected_penalty,
        expected_penalty_contribution=penalty_contribution,
        expected_total_qubo=float(probabilities @ total_energies),
        p_feas=p_feas,
        p_opt=p_opt,
        invalid_mass=infeasible_mass,
        optimum_feasible_mass=p_opt,
        other_feasible_mass=other_feasible,
        infeasible_mass=infeasible_mass,
        shannon_entropy=shannon_entropy(probabilities),
        max_basis_probability=float(probabilities[top]),
        top_basis_index=top,
        top_basis_label=metadata.basis_labels[top],
        top_is_feasible=bool(feasible_mask[top]),
        top_is_optimal=bool(optimal_mask[top]),
        top_decoded_route=metadata.decoded_routes[top],
        top_route_cost=float(route_costs[top]),
        top_total_energy=float(total_energies[top]),
    )


def _relative_phase_spread(before: np.ndarray, after: np.ndarray) -> float:
    support = (np.abs(before) > 1e-14) & (np.abs(after) > 1e-14)
    if np.count_nonzero(support) < 2:
        return 0.0
    factors = after[support] / before[support]
    factors /= np.abs(factors)
    reference = factors[0]
    return float(np.max(np.abs(factors - reference)))


def _validate_physics(
    states: Sequence[np.ndarray], checkpoints: Sequence[CheckpointMetrics], depth: int
) -> tuple[LayerPhysicsValidation, ...]:
    records: list[LayerPhysicsValidation] = []
    for layer in range(int(depth)):
        before = np.asarray(states[2 * layer], dtype=np.complex128)
        after_cost = np.asarray(states[2 * layer + 1], dtype=np.complex128)
        after_mixer = np.asarray(states[2 * layer + 2], dtype=np.complex128)
        before_prob = np.abs(before) ** 2
        cost_prob = np.abs(after_cost) ** 2
        mixer_prob = np.abs(after_mixer) ** 2
        phase_spread = _relative_phase_spread(before, after_cost)
        record = LayerPhysicsValidation(
            layer=layer + 1,
            cost_probability_max_delta=float(np.max(np.abs(cost_prob - before_prob))),
            cost_energy_delta=float(
                checkpoints[2 * layer + 1].expected_hc
                - checkpoints[2 * layer].expected_hc
            ),
            cost_norm_delta=float(np.linalg.norm(after_cost) - np.linalg.norm(before)),
            cost_state_max_delta=float(np.max(np.abs(after_cost - before))),
            cost_relative_phase_spread=phase_spread,
            cost_changed_phase=bool(phase_spread > TRACE_TOLERANCE),
            mixer_probability_max_delta=float(np.max(np.abs(mixer_prob - cost_prob))),
            mixer_energy_delta=float(
                checkpoints[2 * layer + 2].expected_hc
                - checkpoints[2 * layer + 1].expected_hc
            ),
            mixer_changed_probability=bool(
                np.max(np.abs(mixer_prob - cost_prob)) > TRACE_TOLERANCE
            ),
        )
        if record.cost_probability_max_delta > TRACE_TOLERANCE:
            raise RuntimeError("cost_layer_changed_computational_probabilities")
        if abs(record.cost_energy_delta) > TRACE_TOLERANCE:
            raise RuntimeError("cost_layer_changed_expected_cost_energy")
        if abs(record.cost_norm_delta) > TRACE_TOLERANCE:
            raise RuntimeError("cost_layer_changed_state_norm")
        records.append(record)
    return tuple(records)


def trace_qaoa_evolution(
    initial_state: Sequence[complex],
    phase_energies: Sequence[float],
    mixer_evolve: Callable[[Sequence[complex], float], np.ndarray],
    parameters: Sequence[float],
    *,
    depth: int,
    metadata: BasisMetadata,
) -> EvolutionTrace:
    """Capture ``initial``, then every cost and mixer checkpoint exactly."""

    depth = int(depth)
    parameters = np.asarray(parameters, dtype=np.float64)
    phases = np.asarray(phase_energies, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.complex128).copy()
    if depth < 1 or parameters.shape != (2 * depth,) or np.any(~np.isfinite(parameters)):
        raise ValueError("trace_parameter_count_or_finiteness_error")
    if state.shape != (metadata.dimension,) or phases.shape != state.shape:
        raise ValueError("trace_state_phase_metadata_dimension_mismatch")
    if not np.isclose(np.linalg.norm(state), 1.0, atol=TRACE_TOLERANCE, rtol=0.0):
        raise ValueError("trace_initial_state_not_normalized")
    gammas = parameters[:depth]
    betas = parameters[depth:]
    states = [state.copy()]
    checkpoints = [
        _checkpoint_metrics(
            state,
            metadata,
            checkpoint_index=0,
            checkpoint="Initial",
            operation="initial",
            layer=0,
        )
    ]
    for layer, (gamma, beta) in enumerate(zip(gammas, betas), start=1):
        state = apply_cost_layer(state, phases, float(gamma))
        states.append(state.copy())
        checkpoints.append(
            _checkpoint_metrics(
                state,
                metadata,
                checkpoint_index=len(checkpoints),
                checkpoint=f"Cost-{layer}",
                operation="cost",
                layer=layer,
            )
        )
        state = np.asarray(mixer_evolve(state, float(beta)), dtype=np.complex128)
        if state.shape != (metadata.dimension,):
            raise ValueError("trace_mixer_returned_wrong_dimension")
        states.append(state.copy())
        checkpoints.append(
            _checkpoint_metrics(
                state,
                metadata,
                checkpoint_index=len(checkpoints),
                checkpoint=f"Mixer-{layer}",
                operation="mixer",
                layer=layer,
            )
        )
    physics = _validate_physics(states, checkpoints, depth)
    return EvolutionTrace(
        depth=depth,
        parameters=tuple(map(float, parameters)),
        checkpoints=tuple(checkpoints),
        statevectors=tuple(states),
        physics=physics,
    )
