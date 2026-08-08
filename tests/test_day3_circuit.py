from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from circuit import (
    BETA_PARAMETER_NAME,
    GAMMA_PARAMETER_NAME,
    bind_p1_parameters,
    build_p1_qaoa_circuit,
    primitive_gate_counts,
)
from graph import edge_order_records, load_graph
from ising import qubo_to_ising
from qubo import build_qubo


GRAPH_SHA256 = "451a6e4cffd01552c1cb6e4290b5ee8d50fc97514aab8ed47a6b33a111f5e54e"
PENALTY_CONTRACT_SHA256 = "3e31dc31434dcd06c430e90e75884536cbef8cdfefec195afeb1acc1b77acd57"
PENALTIES = (2, 5, 6, 12)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.mark.parametrize(
    ("penalty", "expected_rz", "expected_total"),
    [(2, 13, 87), (5, 13, 87), (6, 14, 88), (12, 14, 88)],
)
def test_explicit_p1_circuit_identity_and_gate_counts(graph, penalty, expected_rz, expected_total):
    hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
    circuit = build_p1_qaoa_circuit(graph, hamiltonian)

    assert circuit.num_qubits == 14
    assert circuit.num_clbits == 0
    assert circuit.metadata["p"] == 1
    assert circuit.metadata["cost_layer_count"] == 1
    assert circuit.metadata["mixer_layer_count"] == 1
    assert circuit.metadata["qubit_mapping"] == edge_order_records(graph)
    assert circuit.metadata["cost_identity_gate_omitted"] is True
    assert primitive_gate_counts(circuit) == {
        "h": 14,
        "rz": expected_rz,
        "rzz": 46,
        "rx": 14,
        "total": expected_total,
    }

    names = [instruction.operation.name for instruction in circuit.data]
    assert names[:14] == ["h"] * 14
    assert names[-14:] == ["rx"] * 14
    assert names[14:-14] == ["rz"] * expected_rz + ["rzz"] * 46


def test_parameterized_circuit_has_exactly_gamma_1_and_beta_1(graph):
    hamiltonian = qubo_to_ising(build_qubo(graph, 6))
    circuit = build_p1_qaoa_circuit(graph, hamiltonian)
    assert {parameter.name for parameter in circuit.parameters} == {
        GAMMA_PARAMETER_NAME,
        BETA_PARAMETER_NAME,
    }
    bound = bind_p1_parameters(circuit, gamma=0.137, beta=0.223)
    assert not bound.parameters


@pytest.mark.parametrize("penalty", PENALTIES)
def test_factor_of_two_gate_angles_are_exactly_applied(graph, penalty):
    gamma = 0.137
    beta = 0.223
    hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
    parameterized = build_p1_qaoa_circuit(graph, hamiltonian)
    circuit = bind_p1_parameters(parameterized, gamma=gamma, beta=beta)

    rz_angles = {}
    rzz_angles = {}
    rx_angles = {}
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if name == "rz":
            rz_angles[qubits[0]] = float(instruction.operation.params[0])
        elif name == "rzz":
            rzz_angles[qubits] = float(instruction.operation.params[0])
        elif name == "rx":
            rx_angles[qubits[0]] = float(instruction.operation.params[0])

    expected_rz = {
        index: 2 * gamma * float(coefficient)
        for index, coefficient in enumerate(hamiltonian.h)
        if coefficient
    }
    expected_rzz = {
        pair: 2 * gamma * float(coefficient)
        for pair, coefficient in hamiltonian.coupling.items()
        if coefficient
    }
    assert rz_angles.keys() == expected_rz.keys()
    assert rzz_angles.keys() == expected_rzz.keys()
    assert rx_angles.keys() == set(range(14))
    assert all(np.isclose(rz_angles[index], angle) for index, angle in expected_rz.items())
    assert all(np.isclose(rzz_angles[pair], angle) for pair, angle in expected_rzz.items())
    assert all(np.isclose(angle, 2 * beta) for angle in rx_angles.values())


def test_committed_circuit_contract_reconciles_frozen_identity(graph):
    contract = json.loads(Path("data/circuit_contract.json").read_text(encoding="utf-8"))
    assert _sha256("data/graph.json") == GRAPH_SHA256
    assert _sha256("data/penalty_contract.json") == PENALTY_CONTRACT_SHA256
    assert contract["graph"]["sha256"] == GRAPH_SHA256
    assert contract["penalty_contract"]["sha256"] == PENALTY_CONTRACT_SHA256
    assert contract["qubit_count"] == 14
    assert contract["qubit_mapping"] == edge_order_records(graph)
    assert contract["parameters"] == [GAMMA_PARAMETER_NAME, BETA_PARAMETER_NAME]
    assert contract["optimization_performed"] is False
