"""Exact canonical-QUBO to diagonal Ising-Hamiltonian conversion.

The mapping is fixed as ``x_i = (I - Z_i)/2``.  Consequently a computational
basis selection ``x_i`` has Z eigenvalue ``z_i = 1 - 2*x_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from graph import EXPECTED_EDGE_COUNT, validate_edge_vector
from qubo import Pair, QuboPolynomial, StateRecord


ISING_CONVENTION = "H = c0*I + sum_i h_i*Z_i + sum_{i<j} J_ij*Z_i*Z_j"
BIT_TO_Z_CONVENTION = "x_i = (1 - z_i)/2; equivalently z_i = 1 - 2*x_i"


@dataclass(frozen=True)
class IsingHamiltonian:
    """Diagonal Ising coefficients including the identity constant."""

    constant: Fraction
    h: tuple[Fraction, ...]
    coupling: Mapping[Pair, Fraction]

    def __post_init__(self) -> None:
        if len(self.h) != EXPECTED_EDGE_COUNT:
            raise ValueError("Ising Hamiltonian must have exactly 14 h coefficients")
        for (i, j), coefficient in self.coupling.items():
            if not 0 <= i < j < EXPECTED_EDGE_COUNT:
                raise ValueError(f"Ising pair key must satisfy 0 <= i < j < 14: {(i, j)}")
            if coefficient == 0:
                raise ValueError(f"zero Ising coupling should be omitted: {(i, j)}")

    def basis_energy(self, edge_vector: Iterable[int]) -> Fraction:
        """Evaluate ``<x|H|x>`` with ``z_i = 1 - 2*x_i``."""

        vector = validate_edge_vector(edge_vector)
        z = tuple(1 - 2 * bit for bit in vector)
        value = self.constant
        value += sum(coefficient * z[i] for i, coefficient in enumerate(self.h))
        value += sum(
            coefficient * z[i] * z[j]
            for (i, j), coefficient in self.coupling.items()
        )
        return value


def qubo_to_ising(qubo: QuboPolynomial) -> IsingHamiltonian:
    """Derive ``c0``, ``h_i``, and ``J_ij`` explicitly and exactly."""

    constant = qubo.constant + sum(qubo.linear, Fraction(0)) / 2
    h = [-coefficient / 2 for coefficient in qubo.linear]
    coupling: dict[Pair, Fraction] = {}

    for (i, j), coefficient in qubo.pair.items():
        quarter = coefficient / 4
        constant += quarter
        h[i] -= quarter
        h[j] -= quarter
        coupling[i, j] = quarter

    return IsingHamiltonian(constant, tuple(h), coupling)


def max_qubo_ising_error(
    states: Sequence[StateRecord],
    qubos: Sequence[QuboPolynomial],
) -> Fraction:
    """Exhaustively compare QUBO and Ising basis energies."""

    maximum = Fraction(0)
    for qubo in qubos:
        ising = qubo_to_ising(qubo)
        for state in states:
            maximum = max(
                maximum,
                abs(qubo.evaluate(state.edge_vector) - ising.basis_energy(state.edge_vector)),
            )
    return maximum


def z_eigenvalues(edge_vector: Iterable[int]) -> tuple[int, ...]:
    """Expose the frozen computational-basis bit-to-Z conversion."""

    return tuple(1 - 2 * bit for bit in validate_edge_vector(edge_vector))
