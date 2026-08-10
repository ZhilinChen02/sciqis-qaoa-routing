"""Metrics computed from the complete exact probability distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from qubo import StateRecord


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
    probabilities: Sequence[float],
    states: Sequence[StateRecord],
    *,
    optimal_cost: int,
    threshold: float = 1e-15,
) -> DistributionMetrics:
    """Calculate route metrics from all ``2^q`` probabilities, never top-k."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or len(probs) != len(states):
        raise ValueError("distribution and state-space sizes differ")
    if np.any(probs < -threshold) or not np.isclose(np.sum(probs), 1.0, atol=1e-10):
        raise ValueError("invalid probability distribution")
    feasible = np.asarray([state.is_decoder_valid for state in states], dtype=bool)
    optimal = np.asarray(
        [state.is_decoder_valid and state.routing_cost == int(optimal_cost) for state in states],
        dtype=bool,
    )
    p_feas = float(np.sum(probs[feasible]))
    p_opt = float(np.sum(probs[optimal]))
    conditional = None
    if p_feas > threshold:
        costs = np.asarray([state.routing_cost for state in states], dtype=float)
        conditional = float(probs[feasible] @ costs[feasible] / p_feas)

    # Round only for deterministic tie handling; stored probabilities remain raw.
    order = np.lexsort((np.arange(len(probs)), -np.round(probs, 15)))
    best_index = int(order[0])
    best_state = states[best_index]
    feasible_order = [int(index) for index in order if states[int(index)].is_decoder_valid]
    best_decoded = states[feasible_order[0]] if feasible_order else None
    optimal_probabilities = probs[optimal]
    if len(optimal_probabilities) == 0:
        raise RuntimeError("no globally optimal feasible basis state found")
    best_optimal_probability = float(np.max(optimal_probabilities))
    optimal_rank = 1 + int(np.sum(probs > best_optimal_probability + 1e-14))
    return DistributionMetrics(
        p_feas=p_feas,
        p_opt=p_opt,
        feasible_conditional_cost=conditional,
        best_decoded_route=None if best_decoded is None else best_decoded.decoded_route,
        best_decoded_cost=None if best_decoded is None else best_decoded.routing_cost,
        best_state_index=best_index,
        best_state_probability=float(probs[best_index]),
        best_state_valid=best_state.is_decoder_valid,
        best_state_optimal=bool(optimal[best_index]),
        optimal_state_rank=optimal_rank,
    )


def top_state_rows(
    probabilities: Sequence[float],
    states: Sequence[StateRecord],
    *,
    optimal_cost: int,
    top_k: int = 12,
) -> list[dict[str, object]]:
    """Return presentation rows only; scientific metrics use the full vector."""

    probs = np.asarray(probabilities, dtype=float)
    if len(probs) != len(states) or int(top_k) < 1:
        raise ValueError("invalid top-state request")
    order = np.lexsort((np.arange(len(probs)), -np.round(probs, 15)))[: int(top_k)]
    rows = []
    for rank, index in enumerate(order, start=1):
        state = states[int(index)]
        valid = state.is_decoder_valid
        rows.append(
            {
                "rank": rank,
                "bitstring": state.canonical_bitstring,
                "probability": float(probs[int(index)]),
                "valid": valid,
                "optimal": bool(valid and state.routing_cost == int(optimal_cost)),
                "decoded_route": "" if not valid else "->".join(map(str, state.decoded_route)),
                "decoded_cost": "" if not valid else state.routing_cost,
                "invalid_reason": "" if valid else (
                    "flow_constraints_violated"
                    if state.flow_penalty > 0
                    else "route_decoder_rejected"
                ),
            }
        )
    return rows
