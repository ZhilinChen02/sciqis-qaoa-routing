"""Exact-statevector Q1 and Q2 circuits for the 14-edge routing encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from ising import IsingHamiltonian
from warm_start import (
    apply_single_qubit_hamiltonian_rotation,
    apply_warm_start_mixer,
    mixer_hamiltonian,
    product_state,
)


Q1_PENALTY_X = "Q1 Penalty-X"
Q2_WARM_START = "Q2 Warm-Start"


@dataclass(frozen=True)
class CircuitStatistics:
    circuit_depth: int
    total_gate_count: int
    two_qubit_gate_count: int


def normalized_diagonal(raw_diagonal: Sequence[float]) -> tuple[np.ndarray, float, float]:
    """Return an affine normalization and its shift/scale."""

    raw = np.asarray(raw_diagonal, dtype=np.float64)
    if raw.ndim != 1 or len(raw) < 2 or np.any(~np.isfinite(raw)):
        raise ValueError("invalid cost diagonal")
    shift = float(np.min(raw))
    scale = float(np.max(raw) - shift)
    if scale <= 0.0:
        scale = 1.0
    return (raw - shift) / scale, shift, scale


def standard_plus_state(num_qubits: int) -> np.ndarray:
    if int(num_qubits) < 1:
        raise ValueError("num_qubits must be positive")
    dimension = 1 << int(num_qubits)
    return np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)


def apply_x_mixer(state: np.ndarray, beta: float, num_qubits: int) -> np.ndarray:
    """Apply the ordinary separable X mixer using source little-endian order."""

    x_matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    result = np.asarray(state, dtype=np.complex128).copy()
    if len(result) != 1 << int(num_qubits):
        raise ValueError("statevector_dimension_mismatch")
    for qubit in range(int(num_qubits)):
        result = apply_single_qubit_hamiltonian_rotation(
            result, qubit, beta, x_matrix
        )
    return result


def simulate_qaoa_state(
    normalized_cost_diagonal: Sequence[float],
    parameters: Sequence[float],
    *,
    depth: int,
    solver: str,
    warm_start_values: Sequence[float] | None = None,
) -> np.ndarray:
    """Evolve the exact state with source Q2 layer and angle conventions.

    Parameters are ``[gamma_1,...,gamma_p,beta_1,...,beta_p]``.  Each layer
    applies cost evolution followed by mixer evolution. The retained course
    convention scales the mixer angle by ``1/q``.
    """

    diagonal = np.asarray(normalized_cost_diagonal, dtype=np.float64)
    if diagonal.ndim != 1 or len(diagonal) == 0 or len(diagonal) & (len(diagonal) - 1):
        raise ValueError("cost diagonal dimension must be a positive power of two")
    num_qubits = len(diagonal).bit_length() - 1
    depth = int(depth)
    values = np.asarray(parameters, dtype=np.float64)
    if depth < 1 or values.shape != (2 * depth,):
        raise ValueError("qaoa_parameter_count_mismatch")
    gammas, betas = values[:depth], values[depth:]
    mixer_scale = 1.0 / float(num_qubits)
    if solver == Q1_PENALTY_X:
        state = standard_plus_state(num_qubits)
    elif solver == Q2_WARM_START:
        if warm_start_values is None or len(warm_start_values) != num_qubits:
            raise ValueError("Q2 requires one warm-start value per qubit")
        state = product_state(warm_start_values)
    else:
        raise ValueError(f"unsupported solver: {solver}")
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * float(gamma) * diagonal)
        scaled_beta = float(beta) * mixer_scale
        if solver == Q1_PENALTY_X:
            state = apply_x_mixer(state, scaled_beta, num_qubits)
        else:
            state = apply_warm_start_mixer(
                state, scaled_beta, warm_start_values  # type: ignore[arg-type]
            )
    return state


def state_probabilities(state: np.ndarray) -> np.ndarray:
    probabilities = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or not np.isclose(total, 1.0, atol=1e-10):
        raise RuntimeError(f"statevector is not normalized: probability sum={total}")
    return probabilities / total


def build_qaoa_circuit(
    ising: IsingHamiltonian,
    parameters: Sequence[float],
    *,
    depth: int,
    solver: str,
    normalization_scale: float,
    warm_start_values: Sequence[float] | None = None,
) -> QuantumCircuit:
    """Build the circuit equivalent used to compute reproducible resources."""

    num_qubits = len(ising.h)
    values = np.asarray(parameters, dtype=float)
    if values.shape != (2 * int(depth),):
        raise ValueError("qaoa_parameter_count_mismatch")
    if normalization_scale <= 0.0:
        raise ValueError("normalization scale must be positive")
    gammas, betas = values[:depth], values[depth:]
    circuit = QuantumCircuit(num_qubits, name=solver)
    if solver == Q1_PENALTY_X:
        circuit.h(range(num_qubits))
    elif solver == Q2_WARM_START:
        if warm_start_values is None or len(warm_start_values) != num_qubits:
            raise ValueError("Q2 requires one warm-start value per qubit")
        for qubit, probability_one in enumerate(warm_start_values):
            theta = 2.0 * np.arcsin(np.sqrt(float(probability_one)))
            circuit.ry(theta, qubit)
    else:
        raise ValueError(f"unsupported solver: {solver}")

    mixer_scale = 1.0 / float(num_qubits)
    for gamma, beta in zip(gammas, betas):
        for qubit, coefficient in enumerate(ising.h):
            angle = 2.0 * float(gamma) * float(coefficient) / normalization_scale
            if angle:
                circuit.rz(angle, qubit)
        for (left, right), coefficient in sorted(ising.coupling.items()):
            angle = 2.0 * float(gamma) * float(coefficient) / normalization_scale
            if angle:
                circuit.rzz(angle, left, right)
        scaled_beta = float(beta) * mixer_scale
        if solver == Q1_PENALTY_X:
            for qubit in range(num_qubits):
                circuit.rx(2.0 * scaled_beta, qubit)
        else:
            for qubit, probability_one in enumerate(warm_start_values):  # type: ignore[arg-type]
                theta = 2.0 * np.arcsin(np.sqrt(float(probability_one)))
                # exp(-i beta H), H=-RY(theta) Z RY(-theta)
                circuit.ry(-theta, qubit)
                circuit.rz(-2.0 * scaled_beta, qubit)
                circuit.ry(theta, qubit)
    return circuit


def circuit_statistics(circuit: QuantumCircuit) -> CircuitStatistics:
    """Count a backend-independent ``rz/sx/x/cx`` decomposition."""

    decomposed = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
        seed_transpiler=2601,
    )
    operations = decomposed.count_ops()
    two_qubit = sum(
        1 for instruction in decomposed.data if instruction.operation.num_qubits == 2
    )
    return CircuitStatistics(
        circuit_depth=int(decomposed.depth()),
        total_gate_count=int(sum(operations.values())),
        two_qubit_gate_count=int(two_qubit),
    )


def mixer_ground_state_error(probability_one: float) -> float:
    """Numerical check that the prepared qubit is the mixer's -1 eigenstate."""

    c_value = float(probability_one)
    state = np.asarray([np.sqrt(1.0 - c_value), np.sqrt(c_value)], dtype=complex)
    return float(np.max(np.abs(mixer_hamiltonian(c_value) @ state + state)))
