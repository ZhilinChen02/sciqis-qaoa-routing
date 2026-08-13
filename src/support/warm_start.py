"""Build the classical route and quantum state used by warm-start QAOA."""

from dataclasses import dataclass
from math import asin, sqrt

import networkx as nx
import numpy as np

from graph import get_edge_order, path_to_edge_bitstring, validate_graph


@dataclass(frozen=True)
class WarmStartValue:
    variable: str
    incumbent_bit: int
    relaxed_value: float
    clipped_value: float
    preparation_angle: float
    mixer_x: float
    mixer_z: float


def _validate_epsilon(epsilon) -> float:
    epsilon = float(epsilon)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("warm_start_epsilon must satisfy 0 < epsilon < 0.5")
    return epsilon


def clip_relaxed_values(values, epsilon=0.1) -> np.ndarray:
    """Keep relaxed values away from the exact endpoints zero and one."""

    epsilon = _validate_epsilon(epsilon)
    values = np.asarray(tuple(values), dtype=np.float64)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("relaxed values must be a non-empty vector")
    if np.any(~np.isfinite(values)):
        raise ValueError("relaxed values must lie in [0,1]")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("relaxed values must lie in [0,1]")

    return np.clip(values, epsilon, 1.0 - epsilon)


def greedy_incumbent_route(graph: nx.DiGraph) -> tuple[int, ...]:
    """Choose the cheapest usable outgoing edge until the target is reached."""

    validate_graph(graph)
    source = graph.graph["source"]
    target = graph.graph["target"]
    route = [source]
    current_node = source

    while current_node != target:
        candidates = []

        for next_node in graph.successors(current_node):
            if next_node in route:
                continue
            if next_node != target and not nx.has_path(graph, next_node, target):
                continue

            weight = graph.edges[current_node, next_node]["weight"]
            candidates.append((weight, next_node))

        if not candidates:
            raise RuntimeError("greedy incumbent construction reached a dead end")

        _weight, current_node = min(candidates)
        route.append(current_node)

    return tuple(route)


def incumbent_bitstring(graph: nx.DiGraph) -> tuple[int, ...]:
    """Convert the greedy route into fourteen edge bits."""

    route = greedy_incumbent_route(graph)
    return path_to_edge_bitstring(graph, route)


def incumbent_relaxation(
    graph: nx.DiGraph,
    epsilon=0.1,
) -> tuple[WarmStartValue, ...]:
    """Calculate preparation and mixer values for all fourteen variables."""

    bits = incumbent_bitstring(graph)
    clipped_values = clip_relaxed_values(bits, epsilon)
    rows = []

    for index, edge in enumerate(get_edge_order(graph)):
        bit = bits[index]
        clipped_value = float(clipped_values[index])
        preparation_angle = 2.0 * asin(sqrt(clipped_value))
        mixer_x = 2.0 * sqrt(clipped_value * (1.0 - clipped_value))
        mixer_z = 1.0 - 2.0 * clipped_value

        rows.append(
            WarmStartValue(
                variable=f"x{index} ({edge[0]}->{edge[1]})",
                incumbent_bit=int(bit),
                relaxed_value=float(bit),
                clipped_value=clipped_value,
                preparation_angle=preparation_angle,
                mixer_x=mixer_x,
                mixer_z=mixer_z,
            )
        )

    return tuple(rows)


def product_state(clipped_values) -> np.ndarray:
    """Prepare a product state with P(q_i=1) equal to each clipped value."""

    values = np.asarray(clipped_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("clipped values must be a non-empty vector")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("clipped values must lie strictly inside (0,1)")

    dimension = 1 << len(values)
    basis_indices = np.arange(dimension, dtype=np.uint64)
    amplitudes = np.ones(dimension, dtype=np.float64)

    for qubit, probability_one in enumerate(values):
        bit_is_one = (
            (basis_indices >> np.uint64(qubit)) & np.uint64(1)
        ).astype(bool)
        zero_amplitude = sqrt(1.0 - float(probability_one))
        one_amplitude = sqrt(float(probability_one))
        amplitudes *= np.where(bit_is_one, one_amplitude, zero_amplitude)

    return amplitudes.astype(np.complex128)


def incumbent_product_warm_start_state(
    current_bitstring,
    epsilon=0.1,
) -> np.ndarray:
    """Prepare a product state directly from an incumbent bit string."""

    bits = tuple(int(bit) for bit in current_bitstring)
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError("invalid_current_bitstring")

    clipped_values = clip_relaxed_values(bits, epsilon)
    return product_state(clipped_values)


def mixer_hamiltonian(probability_one) -> np.ndarray:
    """Return the two-by-two warm-start mixer Hamiltonian."""

    probability_one = float(probability_one)
    if not 0.0 < probability_one < 1.0:
        raise ValueError("mixer probability must lie strictly inside (0,1)")

    x_value = 2.0 * sqrt(probability_one * (1.0 - probability_one))
    z_value = 1.0 - 2.0 * probability_one
    return np.asarray(
        [
            [-z_value, -x_value],
            [-x_value, z_value],
        ],
        dtype=np.complex128,
    )


def warm_start_single_qubit_mixer(current_bit, epsilon=0.1) -> np.ndarray:
    """Build a mixer Hamiltonian from one incumbent bit."""

    current_bit = int(current_bit)
    if current_bit not in (0, 1):
        raise ValueError("invalid_current_bit")

    epsilon = _validate_epsilon(epsilon)
    probability_one = 1.0 - epsilon if current_bit == 1 else epsilon
    return mixer_hamiltonian(probability_one)


def apply_single_qubit_hamiltonian_rotation(
    state,
    qubit,
    beta,
    hamiltonian,
) -> np.ndarray:
    """Apply exp(-i beta H) to one qubit of a statevector."""

    state = np.asarray(state, dtype=np.complex128).copy()
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.shape != (2, 2):
        raise ValueError("single_qubit_hamiltonian_shape")

    stride = 1 << int(qubit)
    if len(state) % (2 * stride) != 0:
        raise ValueError("statevector_dimension_mismatch")

    cosine = np.cos(float(beta))
    sine = np.sin(float(beta))
    blocks = state.reshape(-1, 2, stride)
    zero_part = blocks[:, 0, :].copy()
    one_part = blocks[:, 1, :].copy()

    h_zero = hamiltonian[0, 0] * zero_part + hamiltonian[0, 1] * one_part
    h_one = hamiltonian[1, 0] * zero_part + hamiltonian[1, 1] * one_part

    blocks[:, 0, :] = cosine * zero_part - 1j * sine * h_zero
    blocks[:, 1, :] = cosine * one_part - 1j * sine * h_one
    return state


def apply_warm_start_mixer(state, beta, clipped_values) -> np.ndarray:
    """Apply the warm-start mixer to every qubit."""

    clipped_values = tuple(float(value) for value in clipped_values)
    result = np.asarray(state, dtype=np.complex128).copy()
    if len(result) != 1 << len(clipped_values):
        raise ValueError("statevector_dimension_mismatch")

    for qubit, probability_one in enumerate(clipped_values):
        hamiltonian = mixer_hamiltonian(probability_one)
        result = apply_single_qubit_hamiltonian_rotation(
            result,
            qubit,
            beta,
            hamiltonian,
        )
    return result
