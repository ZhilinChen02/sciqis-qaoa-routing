"""Calculate useful numbers from a complete QAOA probability distribution."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistributionMetrics:
    p_feas: float
    p_opt: float
    feasible_conditional_cost: float | None
    best_decoded_route: tuple[int, ...] | None
    best_decoded_cost: int | None
    best_state_index: int
    best_state_probability: float
    best_state_valid: bool
    best_state_optimal: bool
    optimal_state_rank: int


def distribution_metrics(
    probabilities,
    states,
    *,
    optimal_cost,
    threshold=1e-15,
) -> DistributionMetrics:
    """Calculate feasibility, optimum probability and most likely states."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or len(probabilities) != len(states):
        raise ValueError("distribution and state-space sizes differ")
    if np.any(probabilities < -threshold):
        raise ValueError("invalid probability distribution")
    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-10):
        raise ValueError("invalid probability distribution")

    feasible_mask = np.asarray(
        [state.is_decoder_valid for state in states],
        dtype=bool,
    )
    optimal_mask = np.asarray(
        [
            state.is_decoder_valid and state.routing_cost == int(optimal_cost)
            for state in states
        ],
        dtype=bool,
    )

    p_feas = float(np.sum(probabilities[feasible_mask]))
    p_opt = float(np.sum(probabilities[optimal_mask]))

    conditional_cost = None
    if p_feas > threshold:
        costs = np.asarray([state.routing_cost for state in states], dtype=float)
        weighted_cost = probabilities[feasible_mask] @ costs[feasible_mask]
        conditional_cost = float(weighted_cost / p_feas)

    # The state index is used to break equal-probability ties reproducibly.
    state_indices = np.arange(len(probabilities))
    rounded_probabilities = np.round(probabilities, 15)
    probability_order = np.lexsort((state_indices, -rounded_probabilities))

    best_index = int(probability_order[0])
    best_state = states[best_index]

    best_decoded_state = None
    for index in probability_order:
        candidate = states[int(index)]
        if candidate.is_decoder_valid:
            best_decoded_state = candidate
            break

    optimal_probabilities = probabilities[optimal_mask]
    if len(optimal_probabilities) == 0:
        raise RuntimeError("no globally optimal feasible basis state found")

    best_optimal_probability = float(np.max(optimal_probabilities))
    better_state_count = int(
        np.sum(probabilities > best_optimal_probability + 1e-14)
    )

    if best_decoded_state is None:
        best_route = None
        best_route_cost = None
    else:
        best_route = best_decoded_state.decoded_route
        best_route_cost = best_decoded_state.routing_cost

    return DistributionMetrics(
        p_feas=p_feas,
        p_opt=p_opt,
        feasible_conditional_cost=conditional_cost,
        best_decoded_route=best_route,
        best_decoded_cost=best_route_cost,
        best_state_index=best_index,
        best_state_probability=float(probabilities[best_index]),
        best_state_valid=best_state.is_decoder_valid,
        best_state_optimal=bool(optimal_mask[best_index]),
        optimal_state_rank=1 + better_state_count,
    )


def top_state_rows(
    probabilities,
    states,
    *,
    optimal_cost,
    top_k=12,
) -> list[dict[str, object]]:
    """Prepare the most probable states for printing or saving as a table."""

    probabilities = np.asarray(probabilities, dtype=float)
    top_k = int(top_k)
    if len(probabilities) != len(states) or top_k < 1:
        raise ValueError("invalid top-state request")

    state_indices = np.arange(len(probabilities))
    rounded_probabilities = np.round(probabilities, 15)
    order = np.lexsort((state_indices, -rounded_probabilities))[:top_k]

    rows = []
    for rank, index in enumerate(order, start=1):
        index = int(index)
        state = states[index]
        valid = state.is_decoder_valid

        if valid:
            decoded_route = "->".join(str(node) for node in state.decoded_route)
            decoded_cost = state.routing_cost
            invalid_reason = ""
        else:
            decoded_route = ""
            decoded_cost = ""
            if state.flow_penalty > 0:
                invalid_reason = "flow_constraints_violated"
            else:
                invalid_reason = "route_decoder_rejected"

        rows.append(
            {
                "rank": rank,
                "bitstring": state.canonical_bitstring,
                "probability": float(probabilities[index]),
                "valid": valid,
                "optimal": bool(
                    valid and state.routing_cost == int(optimal_cost)
                ),
                "decoded_route": decoded_route,
                "decoded_cost": decoded_cost,
                "invalid_reason": invalid_reason,
            }
        )

    return rows
