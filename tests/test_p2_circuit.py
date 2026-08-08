from __future__ import annotations

import numpy as np
import pytest

from circuit import bind_qaoa_parameters, build_qaoa_circuit, primitive_gate_counts
from graph import load_graph
from ising import qubo_to_ising
from optimization import validate_p2_equivalence_gate
from qubo import build_qubo, enumerate_state_space


PENALTIES = (2, 5, 6, 12)


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.mark.parametrize(
    ("penalty", "rz_per_layer", "total"),
    ((2, 13, 160), (5, 13, 160), (6, 14, 162), (12, 14, 162)),
)
def test_p2_has_four_parameters_and_two_explicit_layers(
    graph, penalty, rz_per_layer, total
):
    hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
    circuit = build_qaoa_circuit(graph, hamiltonian, p=2)
    assert {parameter.name for parameter in circuit.parameters} == {
        "gamma_1", "beta_1", "gamma_2", "beta_2"
    }
    assert circuit.metadata["cost_layer_count"] == 2
    assert circuit.metadata["mixer_layer_count"] == 2
    assert primitive_gate_counts(circuit) == {
        "h": 14,
        "rz": 2 * rz_per_layer,
        "rzz": 92,
        "rx": 28,
        "total": total,
    }

    names = [instruction.operation.name for instruction in circuit.data]
    cost = ["rz"] * rz_per_layer + ["rzz"] * 46
    assert names == ["h"] * 14 + cost + ["rx"] * 14 + cost + ["rx"] * 14


def test_p2_factor_of_two_angles_hold_in_both_layers(graph):
    hamiltonian = qubo_to_ising(build_qubo(graph, 6))
    gammas = (0.137, 0.419)
    betas = (0.223, 0.317)
    circuit = bind_qaoa_parameters(
        build_qaoa_circuit(graph, hamiltonian, p=2),
        gammas=gammas,
        betas=betas,
    )

    cursor = 14
    for gamma, beta in zip(gammas, betas):
        for index, coefficient in enumerate(hamiltonian.h):
            if not coefficient:
                continue
            instruction = circuit.data[cursor]
            assert instruction.operation.name == "rz"
            assert circuit.find_bit(instruction.qubits[0]).index == index
            assert np.isclose(
                float(instruction.operation.params[0]), 2 * gamma * float(coefficient)
            )
            cursor += 1
        for pair, coefficient in sorted(hamiltonian.coupling.items()):
            if not coefficient:
                continue
            instruction = circuit.data[cursor]
            assert instruction.operation.name == "rzz"
            assert tuple(circuit.find_bit(q).index for q in instruction.qubits) == pair
            assert np.isclose(
                float(instruction.operation.params[0]), 2 * gamma * float(coefficient)
            )
            cursor += 1
        for index in range(14):
            instruction = circuit.data[cursor]
            assert instruction.operation.name == "rx"
            assert circuit.find_bit(instruction.qubits[0]).index == index
            assert np.isclose(float(instruction.operation.params[0]), 2 * beta)
            cursor += 1
    assert cursor == len(circuit.data)


def test_full_frozen_p2_qiskit_numpy_equivalence_gate(graph):
    report = validate_p2_equivalence_gate(graph, enumerate_state_space(graph))
    assert report["passed"] is True
    assert report["case_count"] == 8
    assert report["minimum_fidelity"] >= 1 - 1e-12
    assert report["maximum_amplitude_absolute_error"] < 1e-12
    assert report["maximum_probability_error"] < 1e-14
    assert report["maximum_norm_error"] < 1e-12
