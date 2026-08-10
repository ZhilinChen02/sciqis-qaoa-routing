"""Incumbent-product warm start migrated from the validated Q2 baseline.

The source Q2 method uses a feasible incumbent bitstring, not an LP oracle.
For incumbent bit ``b_i`` it forms a relaxed/clipped value
``c_i = 1-epsilon`` when ``b_i=1`` and ``c_i=epsilon`` otherwise.  The
single-qubit state is ``sqrt(1-c_i)|0> + sqrt(c_i)|1>`` and its mixer is
``H_i = -sin(theta_i) X - cos(theta_i) Z``.  This state is the -1 ground
eigenstate of ``H_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, sqrt
from typing import Iterable, Sequence

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


def _validate_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not 0.0 < value < 0.5:
        raise ValueError("warm_start_epsilon must satisfy 0 < epsilon < 0.5")
    return value


def clip_relaxed_values(values: Iterable[float], epsilon: float = 0.1) -> np.ndarray:
    """Clip values in ``[0,1]`` away from pathological endpoints."""

    eps = _validate_epsilon(epsilon)
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("relaxed values must be a non-empty vector")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("relaxed values must lie in [0,1]")
    return np.clip(array, eps, 1.0 - eps)


def greedy_incumbent_route(graph: nx.DiGraph) -> tuple[int, ...]:
    """Build deterministic classical warm-start information without an oracle.

    At each node, choose the lightest outgoing edge that can still reach the
    target; ties are resolved by node id.  The exact optimum is never read.
    """

    validate_graph(graph)
    target = graph.graph["target"]
    current = graph.graph["source"]
    route = [current]
    while current != target:
        candidates = []
        for successor in graph.successors(current):
            if successor in route:
                continue
            if successor != target and not nx.has_path(graph, successor, target):
                continue
            candidates.append((graph.edges[current, successor]["weight"], successor))
        if not candidates:
            raise RuntimeError("greedy incumbent construction reached a dead end")
        _, current = min(candidates)
        route.append(current)
    return tuple(route)


def incumbent_bitstring(graph: nx.DiGraph) -> tuple[int, ...]:
    return path_to_edge_bitstring(graph, greedy_incumbent_route(graph))


def incumbent_relaxation(
    graph: nx.DiGraph,
    epsilon: float = 0.1,
) -> tuple[WarmStartValue, ...]:
    """Return the source-Q2 incumbent relaxation for every edge variable."""

    bits = incumbent_bitstring(graph)
    clipped = clip_relaxed_values(bits, epsilon)
    rows = []
    for index, (edge, bit, c_value) in enumerate(zip(get_edge_order(graph), bits, clipped)):
        theta = 2.0 * asin(sqrt(float(c_value)))
        rows.append(
            WarmStartValue(
                variable=f"x{index} ({edge[0]}->{edge[1]})",
                incumbent_bit=int(bit),
                relaxed_value=float(bit),
                clipped_value=float(c_value),
                preparation_angle=theta,
                mixer_x=2.0 * sqrt(float(c_value) * (1.0 - float(c_value))),
                mixer_z=1.0 - 2.0 * float(c_value),
            )
        )
    return tuple(rows)


def product_state(clipped_values: Sequence[float]) -> np.ndarray:
    """Prepare the little-endian product state with ``P(q_i=1)=c_i``."""

    values = np.asarray(clipped_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("clipped values must be a non-empty vector")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("clipped values must lie strictly inside (0,1)")
    dimension = 1 << len(values)
    indices = np.arange(dimension, dtype=np.uint64)
    amplitudes = np.ones(dimension, dtype=np.float64)
    for qubit, probability_one in enumerate(values):
        selected = ((indices >> np.uint64(qubit)) & np.uint64(1)).astype(bool)
        amplitudes *= np.where(
            selected,
            sqrt(float(probability_one)),
            sqrt(1.0 - float(probability_one)),
        )
    return amplitudes.astype(np.complex128)


def incumbent_product_warm_start_state(
    current_bitstring: Sequence[int],
    epsilon: float = 0.1,
) -> np.ndarray:
    """Compatibility form of the exact source Q2 state-preparation primitive."""

    bits = tuple(map(int, current_bitstring))
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError("invalid_current_bitstring")
    return product_state(clip_relaxed_values(bits, epsilon))


def mixer_hamiltonian(probability_one: float) -> np.ndarray:
    """Return ``-sin(theta)X-cos(theta)Z`` for one clipped value."""

    c_value = float(probability_one)
    if not 0.0 < c_value < 1.0:
        raise ValueError("mixer probability must lie strictly inside (0,1)")
    x_value = 2.0 * sqrt(c_value * (1.0 - c_value))
    z_value = 1.0 - 2.0 * c_value
    return np.asarray(
        [[-z_value, -x_value], [-x_value, z_value]], dtype=np.complex128
    )


def warm_start_single_qubit_mixer(current_bit: int, epsilon: float = 0.1) -> np.ndarray:
    """Compatibility form of the exact source Q2 mixer primitive."""

    bit = int(current_bit)
    if bit not in (0, 1):
        raise ValueError("invalid_current_bit")
    eps = _validate_epsilon(epsilon)
    return mixer_hamiltonian(1.0 - eps if bit else eps)


def apply_single_qubit_hamiltonian_rotation(
    state: np.ndarray,
    qubit: int,
    beta: float,
    hamiltonian: np.ndarray,
) -> np.ndarray:
    """Apply ``exp(-i beta H)`` using ``H^2=I`` and q0 as the LSB."""

    vector = np.asarray(state, dtype=np.complex128).copy()
    if hamiltonian.shape != (2, 2):
        raise ValueError("single_qubit_hamiltonian_shape")
    stride = 1 << int(qubit)
    if len(vector) % (2 * stride):
        raise ValueError("statevector_dimension_mismatch")
    cosine, sine = np.cos(float(beta)), np.sin(float(beta))
    blocks = vector.reshape(-1, 2, stride)
    left = blocks[:, 0, :].copy()
    right = blocks[:, 1, :].copy()
    h_left = hamiltonian[0, 0] * left + hamiltonian[0, 1] * right
    h_right = hamiltonian[1, 0] * left + hamiltonian[1, 1] * right
    blocks[:, 0, :] = cosine * left - 1j * sine * h_left
    blocks[:, 1, :] = cosine * right - 1j * sine * h_right
    return vector


def apply_warm_start_mixer(
    state: np.ndarray,
    beta: float,
    clipped_values: Sequence[float],
) -> np.ndarray:
    """Apply the separable source-Q2 warm-start mixer to all qubits."""

    values = tuple(map(float, clipped_values))
    result = np.asarray(state, dtype=np.complex128).copy()
    if len(result) != 1 << len(values):
        raise ValueError("statevector_dimension_mismatch")
    for qubit, probability_one in enumerate(values):
        result = apply_single_qubit_hamiltonian_rotation(
            result, qubit, beta, mixer_hamiltonian(probability_one)
        )
    return result
