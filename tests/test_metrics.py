from __future__ import annotations

import numpy as np

from utils import distribution_metrics, top_state_rows


def test_feasibility_optimality_and_metrics_use_full_distribution(states):
    optimal_index = next(
        state.state_index for state in states
        if state.decoded_route == (0, 1, 2, 4, 5, 6)
    )
    other = next(state for state in states if state.is_decoder_valid and state.routing_cost == 11)
    invalid = next(state for state in states if not state.is_decoder_valid)
    probabilities = np.zeros(len(states))
    probabilities[optimal_index] = 0.6
    probabilities[other.state_index] = 0.3
    probabilities[invalid.state_index] = 0.1
    result = distribution_metrics(probabilities, states, optimal_cost=10)
    assert np.isclose(result.p_feas, 0.9)
    assert np.isclose(result.p_opt, 0.6)
    assert np.isclose(result.feasible_conditional_cost, (0.6 * 10 + 0.3 * 11) / 0.9)
    assert result.best_decoded_route == (0, 1, 2, 4, 5, 6)
    assert result.best_state_valid is True
    assert result.best_state_optimal is True
    assert result.optimal_state_rank == 1


def test_top_table_marks_invalid_and_optimal_states(states):
    optimal = next(state for state in states if state.decoded_route == (0, 1, 2, 4, 5, 6))
    invalid = next(state for state in states if not state.is_decoder_valid)
    probabilities = np.zeros(len(states))
    probabilities[invalid.state_index] = 0.55
    probabilities[optimal.state_index] = 0.45
    rows = top_state_rows(probabilities, states, optimal_cost=10, top_k=2)
    assert rows[0]["valid"] is False
    assert rows[0]["invalid_reason"]
    assert rows[1]["optimal"] is True
    assert rows[1]["decoded_cost"] == 10
