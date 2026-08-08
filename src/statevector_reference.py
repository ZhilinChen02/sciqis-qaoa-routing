"""Independent NumPy reference evolution for the explicit p=1 circuit.

No Qiskit circuit object is imported here.  The cost layer is applied as a
diagonal energy phase, and the mixer uses transparent two-amplitude updates
for each qubit without constructing a dense 16384 x 16384 matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from graph import EXPECTED_EDGE_COUNT
from qubo import StateRecord


STATE_COUNT = 2**EXPECTED_EDGE_COUNT


def uniform_plus_state(num_qubits: int = EXPECTED_EDGE_COUNT) -> np.ndarray:
    """Return amplitudes of ``|+>^n`` in basis-index order."""

    if num_qubits != EXPECTED_EDGE_COUNT:
        raise ValueError("the frozen reference evolution requires 14 qubits")
    return np.full(2**num_qubits, 1.0 / np.sqrt(2**num_qubits), dtype=complex)


def penalty_energies(states: Sequence[StateRecord], penalty: int | float) -> np.ndarray:
    """Return direct ``C(x)+A*P_flow(x)`` energies in state-index order."""

    if len(states) != STATE_COUNT:
        raise ValueError("expected all 16384 frozen basis states")
    if any(state.state_index != index for index, state in enumerate(states)):
        raise ValueError("states must be ordered by basis-state index")
    return np.fromiter(
        (state.routing_cost + float(penalty) * state.flow_penalty for state in states),
        dtype=float,
        count=STATE_COUNT,
    )


def apply_cost_phase(
    statevector: Iterable[complex],
    energies: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply ``exp(-i*gamma*Q_A(x))`` independently to every amplitude."""

    vector = np.asarray(tuple(statevector), dtype=complex)
    if vector.shape != (STATE_COUNT,) or energies.shape != (STATE_COUNT,):
        raise ValueError("statevector and energies must both have length 16384")
    return vector * np.exp(-1j * float(gamma) * energies)


def apply_x_mixer(
    statevector: Iterable[complex],
    beta: float,
    *,
    num_qubits: int = EXPECTED_EDGE_COUNT,
) -> np.ndarray:
    """Apply ``tensor_i exp(-i*beta*X_i)`` via pairwise amplitude updates."""

    if num_qubits != EXPECTED_EDGE_COUNT:
        raise ValueError("the frozen reference evolution requires 14 qubits")
    vector = np.asarray(tuple(statevector), dtype=complex).copy()
    if vector.shape != (2**num_qubits,):
        raise ValueError("statevector must have length 16384")
    cosine = np.cos(float(beta))
    sine_factor = -1j * np.sin(float(beta))
    for qubit in range(num_qubits):
        stride = 1 << qubit
        block = stride << 1
        for start in range(0, vector.size, block):
            low_slice = slice(start, start + stride)
            high_slice = slice(start + stride, start + block)
            low = vector[low_slice].copy()
            high = vector[high_slice].copy()
            vector[low_slice] = cosine * low + sine_factor * high
            vector[high_slice] = sine_factor * low + cosine * high
    return vector


def reference_statevector_at_checkpoints(
    states: Sequence[StateRecord],
    *,
    penalty: int | float,
    gamma: float,
    beta: float,
) -> dict[str, np.ndarray]:
    """Return independent initial, post-cost, and post-mixer statevectors."""

    initial = uniform_plus_state()
    post_cost = apply_cost_phase(initial, penalty_energies(states, penalty), gamma)
    post_mixer = apply_x_mixer(post_cost, beta)
    return {
        "initial": initial,
        "post_cost": post_cost,
        "post_mixer": post_mixer,
    }


def align_global_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align ``candidate`` to ``reference`` by their overlap phase."""

    reference = np.asarray(reference, dtype=complex)
    candidate = np.asarray(candidate, dtype=complex)
    if reference.shape != candidate.shape:
        raise ValueError("statevectors must have matching shapes")
    overlap = np.vdot(reference, candidate)
    if abs(overlap) < 1e-15:
        raise ValueError("cannot align nearly orthogonal statevectors")
    return candidate * np.exp(-1j * np.angle(overlap))


@dataclass(frozen=True)
class StatevectorComparison:
    fidelity: float
    max_amplitude_absolute_error: float
    reference_norm_error: float
    candidate_norm_error: float
    probability_vector_max_error: float


def compare_statevectors(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> StatevectorComparison:
    """Compare normalized states up to one physically irrelevant global phase."""

    reference = np.asarray(reference, dtype=complex)
    candidate = np.asarray(candidate, dtype=complex)
    aligned = align_global_phase(reference, candidate)
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    overlap = np.vdot(reference, candidate)
    raw_fidelity = float(abs(overlap) ** 2 / (reference_norm**2 * candidate_norm**2))
    fidelity = min(1.0, max(0.0, raw_fidelity))
    return StatevectorComparison(
        fidelity=fidelity,
        max_amplitude_absolute_error=float(np.max(np.abs(reference - aligned))),
        reference_norm_error=abs(reference_norm - 1.0),
        candidate_norm_error=abs(candidate_norm - 1.0),
        probability_vector_max_error=float(
            np.max(np.abs(np.abs(reference) ** 2 - np.abs(candidate) ** 2))
        ),
    )


def probabilities(statevector: np.ndarray) -> np.ndarray:
    """Return computational-basis probabilities."""

    return np.abs(np.asarray(statevector, dtype=complex)) ** 2


def total_variation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return half the L1 distance between probability distributions."""

    return float(0.5 * np.sum(np.abs(np.asarray(first) - np.asarray(second))))
