from __future__ import annotations

import numpy as np
import pytest

from graph import load_graph
from qubo import build_qubo, enumerate_state_space
from qaoa import normalized_diagonal


@pytest.fixture(scope="session")
def graph():
    return load_graph()


@pytest.fixture(scope="session")
def states(graph):
    return enumerate_state_space(graph)


@pytest.fixture(scope="session")
def qubo(graph):
    return build_qubo(graph, 6)


@pytest.fixture(scope="session")
def cost_diagonal(states, qubo):
    raw = np.asarray([float(qubo.evaluate(state.edge_vector)) for state in states])
    return normalized_diagonal(raw)[0]
