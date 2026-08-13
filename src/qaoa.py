"""Statevector simulation and circuit construction for the two QAOA methods."""

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile

from qubo import IsingHamiltonian
from utils import (
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


def normalized_diagonal(raw_diagonal):
    """Scale a cost array to the interval from zero to one."""

    raw_diagonal = np.asarray(raw_diagonal, dtype=np.float64)
    if raw_diagonal.ndim != 1 or len(raw_diagonal) < 2:
        raise ValueError("invalid cost diagonal")
    if np.any(~np.isfinite(raw_diagonal)):
        raise ValueError("invalid cost diagonal")

    shift = float(np.min(raw_diagonal))
    scale = float(np.max(raw_diagonal) - shift)
    if scale <= 0.0:
        scale = 1.0

    normalized = (raw_diagonal - shift) / scale
    return normalized, shift, scale


def standard_plus_state(num_qubits: int) -> np.ndarray:
    """Return the uniform state |+>^q."""

    num_qubits = int(num_qubits)
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")

    dimension = 1 << num_qubits
    amplitude = 1.0 / np.sqrt(dimension)
    return np.full(dimension, amplitude, dtype=np.complex128)


def apply_x_mixer(state, beta, num_qubits) -> np.ndarray:
    """Apply the ordinary X mixer to every qubit."""

    num_qubits = int(num_qubits)
    result = np.asarray(state, dtype=np.complex128).copy()
    if len(result) != 1 << num_qubits:
        raise ValueError("statevector_dimension_mismatch")

    x_matrix = np.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    for qubit in range(num_qubits):
        result = apply_single_qubit_hamiltonian_rotation(
            result,
            qubit,
            beta,
            x_matrix,
        )
    return result


def simulate_qaoa_state(
    normalized_cost_diagonal,
    parameters,
    *,
    depth,
    solver,
    warm_start_values=None,
) -> np.ndarray:
    """Run exact QAOA evolution using gamma values followed by beta values."""

    cost_diagonal = np.asarray(normalized_cost_diagonal, dtype=np.float64)
    is_power_of_two = (
        len(cost_diagonal) > 0
        and len(cost_diagonal) & (len(cost_diagonal) - 1) == 0
    )
    if cost_diagonal.ndim != 1 or not is_power_of_two:
        raise ValueError("cost diagonal dimension must be a positive power of two")

    num_qubits = len(cost_diagonal).bit_length() - 1
    depth = int(depth)
    parameters = np.asarray(parameters, dtype=np.float64)
    if depth < 1 or parameters.shape != (2 * depth,):
        raise ValueError("qaoa_parameter_count_mismatch")

    gammas = parameters[:depth]
    betas = parameters[depth:]
    mixer_scale = 1.0 / num_qubits

    if solver == Q1_PENALTY_X:
        state = standard_plus_state(num_qubits)
    elif solver == Q2_WARM_START:
        if warm_start_values is None or len(warm_start_values) != num_qubits:
            raise ValueError("Q2 requires one warm-start value per qubit")
        state = product_state(warm_start_values)
    else:
        raise ValueError(f"unsupported solver: {solver}")

    for layer in range(depth):
        gamma = float(gammas[layer])
        beta = float(betas[layer]) / num_qubits

        # Cost layer: each basis amplitude receives an energy-dependent phase.
        state *= np.exp(-1j * gamma * cost_diagonal)

        # Mixer layer: Q1 and Q2 use different single-qubit Hamiltonians.
        if solver == Q1_PENALTY_X:
            state = apply_x_mixer(state, beta, num_qubits)
        else:
            state = apply_warm_start_mixer(state, beta, warm_start_values)

    return state


def state_probabilities(state) -> np.ndarray:
    """Convert complex amplitudes to a normalized probability vector."""

    probabilities = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    total_probability = float(np.sum(probabilities))
    if not np.isfinite(total_probability):
        raise RuntimeError(
            f"statevector is not normalized: probability sum={total_probability}"
        )
    if not np.isclose(total_probability, 1.0, atol=1e-10):
        raise RuntimeError(
            f"statevector is not normalized: probability sum={total_probability}"
        )
    return probabilities / total_probability


def build_qaoa_circuit(
    ising: IsingHamiltonian,
    parameters,
    *,
    depth,
    solver,
    normalization_scale,
    warm_start_values=None,
) -> QuantumCircuit:
    """Build the Qiskit circuit corresponding to the statevector simulation."""

    num_qubits = len(ising.h)
    depth = int(depth)
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (2 * depth,):
        raise ValueError("qaoa_parameter_count_mismatch")
    if normalization_scale <= 0.0:
        raise ValueError("normalization scale must be positive")

    gammas = parameters[:depth]
    betas = parameters[depth:]
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

    for layer in range(depth):
        gamma = float(gammas[layer])
        scaled_beta = float(betas[layer]) / num_qubits

        # Single-Z and ZZ gates implement the diagonal Ising cost layer.
        for qubit, coefficient in enumerate(ising.h):
            angle = 2.0 * gamma * float(coefficient) / normalization_scale
            if angle != 0.0:
                circuit.rz(angle, qubit)

        for (left, right), coefficient in sorted(ising.coupling.items()):
            angle = 2.0 * gamma * float(coefficient) / normalization_scale
            if angle != 0.0:
                circuit.rzz(angle, left, right)

        if solver == Q1_PENALTY_X:
            for qubit in range(num_qubits):
                circuit.rx(2.0 * scaled_beta, qubit)
        else:
            for qubit, probability_one in enumerate(warm_start_values):
                theta = 2.0 * np.arcsin(np.sqrt(float(probability_one)))
                circuit.ry(-theta, qubit)
                circuit.rz(-2.0 * scaled_beta, qubit)
                circuit.ry(theta, qubit)

    return circuit


def circuit_statistics(circuit: QuantumCircuit) -> CircuitStatistics:
    """Transpile to common gates and count circuit resources."""

    decomposed = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
        seed_transpiler=2601,
    )
    operations = decomposed.count_ops()

    two_qubit_gate_count = 0
    for instruction in decomposed.data:
        if instruction.operation.num_qubits == 2:
            two_qubit_gate_count += 1

    return CircuitStatistics(
        circuit_depth=int(decomposed.depth()),
        total_gate_count=int(sum(operations.values())),
        two_qubit_gate_count=two_qubit_gate_count,
    )


def mixer_ground_state_error(probability_one: float) -> float:
    """Check that a prepared warm-start qubit is a mixer ground state."""

    probability_one = float(probability_one)
    state = np.asarray(
        [np.sqrt(1.0 - probability_one), np.sqrt(probability_one)],
        dtype=complex,
    )
    error_vector = mixer_hamiltonian(probability_one) @ state + state
    return float(np.max(np.abs(error_vector)))


# ---------------------------------------------------------------------------
# Grover mixer over all edge bit strings
# ---------------------------------------------------------------------------

GROVER_GLOBAL = "grover_global"
GLOBAL_GROVER_CONVENTION = (
    "H_G,all=|s_all><s_all|; "
    "U_G(beta)=I+(exp(-i*beta)-1)|s_all><s_all|"
)


@dataclass(frozen=True)
class GlobalGroverMixer:
    """Memory-efficient Grover mixer over all 2^q states."""

    num_qubits: int
    mixer_name: str = GROVER_GLOBAL
    convention: str = GLOBAL_GROVER_CONVENTION

    def __post_init__(self):
        if isinstance(self.num_qubits, bool) or int(self.num_qubits) < 1:
            raise ValueError("global_grover_num_qubits_must_be_positive")

    @property
    def dimension(self):
        return 1 << int(self.num_qubits)

    @property
    def uniform_amplitude(self):
        return float(1.0 / np.sqrt(self.dimension))

    def initial_state(self):
        return standard_plus_state(self.num_qubits)

    def evolve(self, state, beta):
        vector = np.asarray(state, dtype=np.complex128)
        if vector.shape != (self.dimension,):
            raise ValueError("global_grover_mixer_state_dimension_mismatch")

        overlap = np.sum(vector) * self.uniform_amplitude
        coefficient = (np.exp(-1j * float(beta)) - 1.0) * overlap
        return vector + coefficient * self.uniform_amplitude

    def as_dict(self):
        return {
            "mixer_name": self.mixer_name,
            "num_qubits": int(self.num_qubits),
            "dimension": self.dimension,
            "uniform_amplitude": self.uniform_amplitude,
            "convention": self.convention,
            "implementation": "rank_one_statevector_update_no_dense_matrix",
        }


def build_global_grover_mixer(num_qubits):
    return GlobalGroverMixer(int(num_qubits))


def simulate_global_grover_state(
    normalized_cost_diagonal,
    parameters,
    *,
    depth,
):
    """Run QAOA with the full-space Grover mixer."""

    diagonal = np.asarray(normalized_cost_diagonal, dtype=np.float64)
    is_power_of_two = diagonal.size and not diagonal.size & (diagonal.size - 1)
    if diagonal.ndim != 1 or not is_power_of_two or np.any(~np.isfinite(diagonal)):
        raise ValueError("global_grover_cost_dimension_must_be_a_power_of_two")

    depth = int(depth)
    parameters = np.asarray(parameters, dtype=np.float64)
    if depth < 1 or parameters.shape != (2 * depth,):
        raise ValueError("global_grover_parameter_count_or_finiteness_error")
    if np.any(~np.isfinite(parameters)):
        raise ValueError("global_grover_parameter_count_or_finiteness_error")

    mixer = build_global_grover_mixer(diagonal.size.bit_length() - 1)
    state = mixer.initial_state()
    gammas = parameters[:depth]
    betas = parameters[depth:]

    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * float(gamma) * diagonal)
        state = mixer.evolve(state, float(beta))

    state_probabilities(state)
    return state
