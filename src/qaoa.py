"""QAOA statevector simulation and equivalent Qiskit circuits."""

from collections import namedtuple

import numpy as np
from qiskit import QuantumCircuit, transpile

from qubo import IsingHamiltonian
from utils import (
    apply_single_qubit_hamiltonian_rotation,
    apply_warm_start_mixer,
    product_state,
)


Q1_PENALTY_X = "Q1 Penalty-X"
Q2_WARM_START = "Q2 Warm-Start"
GROVER_GLOBAL = "grover_global"
X_MATRIX = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
CircuitStatistics = namedtuple(
    "CircuitStatistics", "circuit_depth total_gate_count two_qubit_gate_count"
)


def _diagonal(values):
    values = np.asarray(values, dtype=np.float64)
    size = values.size
    if (
        values.ndim != 1
        or size < 2
        or size & (size - 1)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("cost diagonal must have a finite power-of-two length")
    return values, size.bit_length() - 1


def _parameters(values, depth):
    depth = int(depth)
    values = np.asarray(values, dtype=np.float64)
    if depth < 1 or values.shape != (2 * depth,) or not np.all(np.isfinite(values)):
        raise ValueError("QAOA needs p gamma values followed by p beta values")
    return values[:depth], values[depth:]


def _warm_values(solver, values, num_qubits):
    if solver == Q1_PENALTY_X:
        return None
    if solver != Q2_WARM_START:
        raise ValueError(f"unsupported solver: {solver}")
    if values is None or len(values) != num_qubits:
        raise ValueError("Q2 requires one warm-start value per qubit")
    return tuple(map(float, values))


def normalized_diagonal(values):
    """Scale QUBO energies to the interval 0..1."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("invalid cost diagonal")
    shift = float(np.min(values))
    scale = float(np.max(values) - shift) or 1.0
    return (values - shift) / scale, shift, scale


def standard_plus_state(num_qubits):
    """Return the uniform state |+>^q."""

    num_qubits = int(num_qubits)
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")
    dimension = 1 << num_qubits
    return np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)


def apply_x_mixer(state, beta, num_qubits):
    """Apply exp(-i beta X) to every qubit."""

    result = np.asarray(state, dtype=np.complex128).copy()
    if result.shape != (1 << int(num_qubits),):
        raise ValueError("statevector dimension does not match num_qubits")
    for qubit in range(int(num_qubits)):
        result = apply_single_qubit_hamiltonian_rotation(
            result, qubit, beta, X_MATRIX
        )
    return result


def simulate_qaoa_state(
    cost_diagonal,
    parameters,
    *,
    depth,
    solver,
    warm_start_values=None,
):
    """Alternate cost and mixer layers on an exact statevector."""

    cost_diagonal, num_qubits = _diagonal(cost_diagonal)
    gammas, betas = _parameters(parameters, depth)
    warm = _warm_values(solver, warm_start_values, num_qubits)
    state = standard_plus_state(num_qubits) if warm is None else product_state(warm)

    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * cost_diagonal)  # cost layer: change phases
        beta = beta / num_qubits
        if warm is None:
            state = apply_x_mixer(state, beta, num_qubits)
        else:
            state = apply_warm_start_mixer(state, beta, warm)
    return state


def state_probabilities(state):
    """Convert statevector amplitudes to probabilities."""

    probabilities = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or not np.isclose(total, 1.0, atol=1e-10):
        raise RuntimeError(f"statevector is not normalized: probability sum={total}")
    return probabilities / total


def build_qaoa_circuit(
    ising: IsingHamiltonian,
    parameters,
    *,
    depth,
    solver,
    normalization_scale,
    warm_start_values=None,
):
    """Build the Qiskit circuit matching ``simulate_qaoa_state``."""

    num_qubits = len(ising.h)
    gammas, betas = _parameters(parameters, depth)
    warm = _warm_values(solver, warm_start_values, num_qubits)
    scale = float(normalization_scale)
    if scale <= 0.0:
        raise ValueError("normalization scale must be positive")

    circuit = QuantumCircuit(num_qubits, name=solver)
    thetas = [] if warm is None else [2 * np.arcsin(np.sqrt(value)) for value in warm]
    if warm is None:
        circuit.h(range(num_qubits))
    else:
        for qubit, theta in enumerate(thetas):
            circuit.ry(theta, qubit)

    for gamma, beta in zip(gammas, betas):
        for qubit, coefficient in enumerate(ising.h):
            angle = 2 * gamma * float(coefficient) / scale
            if angle:
                circuit.rz(angle, qubit)
        for (left, right), coefficient in sorted(ising.coupling.items()):
            angle = 2 * gamma * float(coefficient) / scale
            if angle:
                circuit.rzz(angle, left, right)

        beta = beta / num_qubits
        if warm is None:
            for qubit in range(num_qubits):
                circuit.rx(2 * beta, qubit)
        else:
            for qubit, theta in enumerate(thetas):
                circuit.ry(-theta, qubit)
                circuit.rz(-2 * beta, qubit)
                circuit.ry(theta, qubit)
    return circuit


def circuit_statistics(circuit):
    """Transpile to common gates and count circuit resources."""

    circuit = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
        seed_transpiler=2601,
    )
    return CircuitStatistics(
        int(circuit.depth()),
        int(sum(circuit.count_ops().values())),
        sum(item.operation.num_qubits == 2 for item in circuit.data),
    )


class GlobalGroverMixer:
    """Full-space Grover mixer without a dense matrix."""

    def __init__(self, num_qubits):
        if isinstance(num_qubits, bool):
            raise ValueError("num_qubits must be positive")
        self.num_qubits = int(num_qubits)
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be positive")

    @property
    def dimension(self):
        return 1 << self.num_qubits

    def initial_state(self):
        return standard_plus_state(self.num_qubits)

    def evolve(self, state, beta):
        state = np.asarray(state, dtype=np.complex128)
        if state.shape != (self.dimension,):
            raise ValueError("state dimension does not match the Grover mixer")
        uniform = 1.0 / np.sqrt(self.dimension)
        overlap = np.sum(state) * uniform
        return state + (np.exp(-1j * beta) - 1.0) * overlap * uniform


def build_global_grover_mixer(num_qubits):
    return GlobalGroverMixer(num_qubits)


def simulate_global_grover_state(cost_diagonal, parameters, *, depth):
    """Run QAOA with the full-space Grover mixer."""

    cost_diagonal, num_qubits = _diagonal(cost_diagonal)
    gammas, betas = _parameters(parameters, depth)
    mixer = GlobalGroverMixer(num_qubits)
    state = mixer.initial_state()
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * cost_diagonal)
        state = mixer.evolve(state, beta)
    state_probabilities(state)
    return state
