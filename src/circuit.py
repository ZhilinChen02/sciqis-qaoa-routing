"""Explicit shallow Penalty-X QAOA circuits from the frozen Ising model.

This module intentionally uses primitive gates rather than a QAOA ansatz or
optimizer.  Qubit ``q_i`` is the frozen Day-1 edge variable ``x_i`` and q0 is
the least-significant computational-basis index bit.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import Statevector

from graph import EXPECTED_EDGE_COUNT, edge_order_records, validate_graph
from ising import IsingHamiltonian


GAMMA_PARAMETER_NAME = "gamma_1"
BETA_PARAMETER_NAME = "beta_1"


def _coefficient_float(value: Fraction) -> float:
    return float(value)


def build_initial_state_circuit(graph) -> QuantumCircuit:
    """Prepare the uniform ``|+>^14`` state with one H gate per edge qubit."""

    validate_graph(graph)
    circuit = QuantumCircuit(EXPECTED_EDGE_COUNT, name="p1_penalty_x")
    circuit.h(range(EXPECTED_EDGE_COUNT))
    circuit.metadata = {
        "qubit_mapping": edge_order_records(graph),
        "initial_state": "|+>^14",
        "basis_index_convention": "q0 is least-significant",
    }
    return circuit


def append_cost_layer(
    circuit: QuantumCircuit,
    hamiltonian: IsingHamiltonian,
    gamma: float | ParameterExpression,
) -> QuantumCircuit:
    """Append commuting RZ/RZZ gates for one explicit Ising cost layer.

    ``RZ(2*gamma*h_i)`` and ``RZZ(2*gamma*J_ij)`` implement the non-identity
    terms.  The ``c0 I`` term is omitted physically and therefore changes the
    evolved state only by global phase ``exp(-i*gamma*c0)``.
    """

    if circuit.num_qubits != EXPECTED_EDGE_COUNT:
        raise ValueError("the frozen circuit must have exactly 14 qubits")
    for index, coefficient in enumerate(hamiltonian.h):
        if coefficient:
            circuit.rz(2.0 * _coefficient_float(coefficient) * gamma, index)
    for (i, j), coefficient in sorted(hamiltonian.coupling.items()):
        if coefficient:
            circuit.rzz(2.0 * _coefficient_float(coefficient) * gamma, i, j)
    return circuit


def append_mixer_layer(
    circuit: QuantumCircuit,
    beta: float | ParameterExpression,
) -> QuantumCircuit:
    """Append the standard transverse-field mixer as ``RX_i(2*beta)``."""

    if circuit.num_qubits != EXPECTED_EDGE_COUNT:
        raise ValueError("the frozen circuit must have exactly 14 qubits")
    for index in range(EXPECTED_EDGE_COUNT):
        circuit.rx(2.0 * beta, index)
    return circuit


def build_p1_qaoa_circuit(
    graph,
    hamiltonian: IsingHamiltonian,
    *,
    barriers: bool = False,
) -> QuantumCircuit:
    """Build parameterized ``H -> U_C(gamma_1) -> U_M(beta_1)`` explicitly."""

    return build_qaoa_circuit(graph, hamiltonian, p=1, barriers=barriers)


def build_qaoa_circuit(
    graph,
    hamiltonian: IsingHamiltonian,
    *,
    p: int,
    barriers: bool = False,
) -> QuantumCircuit:
    """Build explicit Penalty-X QAOA for frozen shallow depth ``p in {1,2}``."""

    if p not in (1, 2):
        raise ValueError("the frozen core experiment supports only p=1 or p=2")
    circuit = build_initial_state_circuit(graph)
    circuit.name = f"p{p}_penalty_x"
    if barriers:
        circuit.barrier(label="INITIAL")
    parameter_names = []
    for layer in range(1, p + 1):
        gamma_name = f"gamma_{layer}"
        beta_name = f"beta_{layer}"
        gamma = Parameter(gamma_name)
        beta = Parameter(beta_name)
        parameter_names.extend((gamma_name, beta_name))
        append_cost_layer(circuit, hamiltonian, gamma)
        if barriers:
            circuit.barrier(label=f"COST_{layer}")
        append_mixer_layer(circuit, beta)
        if barriers:
            circuit.barrier(label=f"MIXER_{layer}")
    circuit.metadata.update(
        {
            "p": p,
            "cost_layer_count": p,
            "mixer_layer_count": p,
            "parameter_names": parameter_names,
            "cost_identity_constant": float(hamiltonian.constant),
            "cost_identity_gate_omitted": True,
            "identity_effect": "global phase exp(-i*c0*sum_l gamma_l) only",
            "cost_gate_order": "RZ by q_i, then RZZ lexicographically by (i,j)",
            "mixer_gate_order": "RX by q0 through q13",
        }
    )
    return circuit


def bind_p1_parameters(
    circuit: QuantumCircuit,
    *,
    gamma: float,
    beta: float,
) -> QuantumCircuit:
    """Bind exactly ``gamma_1`` and ``beta_1`` by their explicit names."""

    return bind_qaoa_parameters(circuit, gammas=(gamma,), betas=(beta,))


def bind_qaoa_parameters(
    circuit: QuantumCircuit,
    *,
    gammas: tuple[float, ...] | list[float],
    betas: tuple[float, ...] | list[float],
) -> QuantumCircuit:
    """Bind interleaved ``gamma_l, beta_l`` parameters for p=1 or p=2."""

    if len(gammas) != len(betas) or len(gammas) not in (1, 2):
        raise ValueError("gammas and betas must have matching length p in {1,2}")

    parameters = {parameter.name: parameter for parameter in circuit.parameters}
    expected = {
        name
        for layer in range(1, len(gammas) + 1)
        for name in (f"gamma_{layer}", f"beta_{layer}")
    }
    if set(parameters) != expected:
        raise ValueError(
            f"expected free parameters {sorted(expected)}, got {sorted(parameters)}"
        )
    values = {}
    for layer, (gamma, beta) in enumerate(zip(gammas, betas), start=1):
        values[parameters[f"gamma_{layer}"]] = float(gamma)
        values[parameters[f"beta_{layer}"]] = float(beta)
    return circuit.assign_parameters(values, inplace=False)


def qaoa_statevector(
    graph,
    hamiltonian: IsingHamiltonian,
    *,
    gammas: tuple[float, ...] | list[float],
    betas: tuple[float, ...] | list[float],
) -> np.ndarray:
    """Evolve the explicit primitive-gate circuit for p=1 or p=2."""

    if len(gammas) != len(betas) or len(gammas) not in (1, 2):
        raise ValueError("gammas and betas must have matching length p in {1,2}")
    circuit = build_initial_state_circuit(graph)
    for gamma, beta in zip(gammas, betas):
        append_cost_layer(circuit, hamiltonian, float(gamma))
        append_mixer_layer(circuit, float(beta))
    return np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)


def statevector_at_checkpoints(
    graph,
    hamiltonian: IsingHamiltonian,
    *,
    gamma: float,
    beta: float,
) -> dict[str, np.ndarray]:
    """Return Qiskit statevectors after H, cost, and mixer checkpoints."""

    initial = build_initial_state_circuit(graph)
    post_cost = initial.copy()
    append_cost_layer(post_cost, hamiltonian, float(gamma))
    post_mixer = post_cost.copy()
    append_mixer_layer(post_mixer, float(beta))
    return {
        "initial": np.asarray(Statevector.from_instruction(initial).data, dtype=complex),
        "post_cost": np.asarray(Statevector.from_instruction(post_cost).data, dtype=complex),
        "post_mixer": np.asarray(Statevector.from_instruction(post_mixer).data, dtype=complex),
    }


def primitive_gate_counts(circuit: QuantumCircuit) -> dict[str, int]:
    """Return deterministic primitive gate counts without barriers."""

    counts = circuit.count_ops()
    return {
        "h": int(counts.get("h", 0)),
        "rz": int(counts.get("rz", 0)),
        "rzz": int(counts.get("rzz", 0)),
        "rx": int(counts.get("rx", 0)),
        "total": int(sum(counts.values())),
    }
