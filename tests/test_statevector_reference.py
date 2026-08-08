from __future__ import annotations

import hashlib
import json
from math import pi
from pathlib import Path

import numpy as np
import pytest

from circuit import statevector_at_checkpoints
from exact_reference import networkx_shortest_reference
from graph import (
    edge_vector_to_canonical_bitstring,
    edge_vector_to_qiskit_display_bitstring,
    load_graph,
    path_to_edge_bitstring,
)
from ising import qubo_to_ising
from qubo import build_qubo, edge_vector_to_state_index, enumerate_state_space
from statevector_reference import (
    compare_statevectors,
    probabilities,
    reference_statevector_at_checkpoints,
    total_variation_distance,
)


GRAPH_SHA256 = "451a6e4cffd01552c1cb6e4290b5ee8d50fc97514aab8ed47a6b33a111f5e54e"
PENALTY_CONTRACT_SHA256 = "3e31dc31434dcd06c430e90e75884536cbef8cdfefec195afeb1acc1b77acd57"
PENALTIES = (2, 5, 6, 12)
PARAMETER_PAIRS = ((pi / 7, pi / 11), (0.173, 0.271), (-pi / 13, pi / 17))


@pytest.fixture(scope="module")
def frozen_model():
    graph = load_graph()
    states = enumerate_state_space(graph)
    return graph, states


@pytest.mark.parametrize("penalty", PENALTIES)
@pytest.mark.parametrize(("gamma", "beta"), PARAMETER_PAIRS)
def test_qiskit_and_independent_numpy_agree_at_every_checkpoint(
    frozen_model, penalty, gamma, beta
):
    graph, states = frozen_model
    hamiltonian = qubo_to_ising(build_qubo(graph, penalty))
    qiskit_checkpoints = statevector_at_checkpoints(
        graph, hamiltonian, gamma=gamma, beta=beta
    )
    numpy_checkpoints = reference_statevector_at_checkpoints(
        states, penalty=penalty, gamma=gamma, beta=beta
    )
    for checkpoint in ("initial", "post_cost", "post_mixer"):
        comparison = compare_statevectors(
            numpy_checkpoints[checkpoint], qiskit_checkpoints[checkpoint]
        )
        assert comparison.fidelity >= 1 - 1e-12
        assert comparison.max_amplitude_absolute_error < 1e-12
        assert comparison.reference_norm_error < 1e-12
        assert comparison.candidate_norm_error < 1e-12
        assert comparison.probability_vector_max_error < 1e-14


@pytest.mark.parametrize("penalty", PENALTIES)
def test_cost_layer_changes_phase_not_probability(frozen_model, penalty):
    _graph, states = frozen_model
    checkpoints = reference_statevector_at_checkpoints(
        states, penalty=penalty, gamma=pi / 7, beta=pi / 11
    )
    initial_probabilities = probabilities(checkpoints["initial"])
    cost_probabilities = probabilities(checkpoints["post_cost"])
    assert np.max(np.abs(cost_probabilities - initial_probabilities)) < 1e-15

    relative_phases = np.angle(checkpoints["post_cost"] / checkpoints["initial"])
    assert len(np.unique(np.round(relative_phases, decimals=12))) > 1


@pytest.mark.parametrize("penalty", PENALTIES)
def test_mixer_redistributes_probability_without_changing_norm(frozen_model, penalty):
    _graph, states = frozen_model
    checkpoints = reference_statevector_at_checkpoints(
        states, penalty=penalty, gamma=pi / 7, beta=pi / 11
    )
    for vector in checkpoints.values():
        assert abs(np.linalg.norm(vector) - 1.0) < 1e-12
    post_cost = probabilities(checkpoints["post_cost"])
    post_mixer = probabilities(checkpoints["post_mixer"])
    assert total_variation_distance(post_cost, post_mixer) > 1e-8
    assert np.max(np.abs(post_mixer - post_cost)) > 1e-8


def test_bit_order_and_exact_route_state_index_remain_frozen(frozen_model):
    graph, _states = frozen_model
    exact = networkx_shortest_reference(graph)
    vector = path_to_edge_bitstring(graph, exact.node_path)
    assert exact.node_path == (0, 1, 2, 4, 5, 6)
    assert exact.cost == 10
    assert edge_vector_to_canonical_bitstring(vector) == "10010001000101"
    assert edge_vector_to_qiskit_display_bitstring(vector) == "10100010001001"
    assert edge_vector_to_state_index(vector) == 10377


def test_graph_and_penalty_grid_identity_remain_frozen():
    assert hashlib.sha256(Path("data/graph.json").read_bytes()).hexdigest() == GRAPH_SHA256
    assert (
        hashlib.sha256(Path("data/penalty_contract.json").read_bytes()).hexdigest()
        == PENALTY_CONTRACT_SHA256
    )
    penalty = json.loads(Path("data/penalty_contract.json").read_text(encoding="utf-8"))
    assert penalty["A_crit"] == 5
    assert [(record["label"], record["A"]) for record in penalty["penalty_values"]] == [
        ("weak", 2),
        ("critical", 5),
        ("just-supercritical", 6),
        ("strong", 12),
    ]
