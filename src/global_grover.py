"""Full-bitstring Grover-mixer QAOA for the frozen routing QUBO.

The computational basis is exactly the Penalty-X edge-bit basis: ``q`` edge
bits and dimension ``2**q``.  The mixer follows the frozen feasible-Grover
convention ``H_G = |s><s|`` but uses the uniform state over *all* bitstrings.
Its exponential is applied as a rank-one update, so no dense full-space
projector or unitary is ever constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from qaoa import standard_plus_state, state_probabilities


GROVER_GLOBAL = "grover_global"
GLOBAL_GROVER_CONVENTION = (
    "H_G,all=|s_all><s_all|; "
    "U_G(beta)=I+(exp(-i*beta)-1)|s_all><s_all|"
)


@dataclass(frozen=True)
class GlobalGroverMixer:
    """Rank-one projector mixer over all ``2**num_qubits`` bitstrings.

    Only the scalar uniform amplitude is stored.  :meth:`evolve` costs
    ``O(2**q)`` time and one output statevector; it never allocates an
    ``O(4**q)`` matrix.
    """

    num_qubits: int
    mixer_name: str = GROVER_GLOBAL
    convention: str = GLOBAL_GROVER_CONVENTION

    def __post_init__(self) -> None:
        if isinstance(self.num_qubits, bool) or int(self.num_qubits) < 1:
            raise ValueError("global_grover_num_qubits_must_be_positive")

    @property
    def dimension(self) -> int:
        return 1 << int(self.num_qubits)

    @property
    def uniform_amplitude(self) -> float:
        return float(1.0 / np.sqrt(self.dimension))

    def initial_state(self) -> np.ndarray:
        """Return ``|s_all> = |+>**q`` in little-endian basis order."""

        return standard_plus_state(int(self.num_qubits))

    def evolve(self, state: Sequence[complex], beta: float) -> np.ndarray:
        """Apply ``exp(-i beta |s_all><s_all|)`` by a rank-one update."""

        vector = np.asarray(state, dtype=np.complex128)
        if vector.shape != (self.dimension,):
            raise ValueError("global_grover_mixer_state_dimension_mismatch")
        # <s_all|psi> = sum_x psi_x / sqrt(N); the uniform vector need not be
        # materialized separately from the output-sized correction.
        overlap = np.sum(vector) * self.uniform_amplitude
        coefficient = (np.exp(-1j * float(beta)) - 1.0) * overlap
        return vector + coefficient * self.uniform_amplitude

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "mixer_name": self.mixer_name,
            "num_qubits": int(self.num_qubits),
            "dimension": self.dimension,
            "uniform_amplitude": self.uniform_amplitude,
            "convention": self.convention,
            "implementation": "rank_one_statevector_update_no_dense_matrix",
        }


def build_global_grover_mixer(num_qubits: int) -> GlobalGroverMixer:
    """Construct the matrix-free full-space mixer from ``q`` only."""

    return GlobalGroverMixer(int(num_qubits))


def simulate_global_grover_state(
    normalized_cost_diagonal: Sequence[float],
    parameters: Sequence[float],
    *,
    depth: int,
) -> np.ndarray:
    """Evolve full-space Grover QAOA using grouped ``gamma``/``beta`` order.

    Cost evolution is identical to Penalty-X.  In accordance with the frozen
    feasible-Grover convention, projector mixer angles are not divided by the
    qubit count.
    """

    diagonal = np.asarray(normalized_cost_diagonal, dtype=np.float64)
    if (
        diagonal.ndim != 1
        or diagonal.size == 0
        or diagonal.size & (diagonal.size - 1)
        or np.any(~np.isfinite(diagonal))
    ):
        raise ValueError("global_grover_cost_dimension_must_be_a_power_of_two")
    p = int(depth)
    values = np.asarray(parameters, dtype=np.float64)
    if p < 1 or values.shape != (2 * p,) or np.any(~np.isfinite(values)):
        raise ValueError("global_grover_parameter_count_or_finiteness_error")
    mixer = build_global_grover_mixer(diagonal.size.bit_length() - 1)
    state = mixer.initial_state()
    gammas, betas = values[:p], values[p:]
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * float(gamma) * diagonal)
        state = mixer.evolve(state, float(beta))
    # Reuse the course normalization assertion without perturbing amplitudes.
    state_probabilities(state)
    return state
