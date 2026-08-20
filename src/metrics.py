"""Metrics calculated from a complete QAOA probability distribution."""

from dataclasses import dataclass

import numpy as np


def probability_mass(probabilities, mask=None) -> float:
    """Add total probability, optionally restricted by a Boolean mask."""

    probabilities = np.asarray(probabilities)
    if mask is None:
        return float(np.sum(probabilities))
    return float(np.sum(probabilities[np.asarray(mask, dtype=bool)]))


def indexed_probability_mass(probabilities, indices) -> float:
    """Add probability on an explicit collection of state indices."""

    return float(np.sum(np.asarray(probabilities)[np.asarray(indices, dtype=int)]))


def conditional_probability(part: float, whole: float) -> float | None:
    """Return ``part / whole``, or ``None`` when the conditioning mass is zero."""

    return float(part / whole) if whole > 0.0 else None


def shannon_entropy(probabilities) -> float:
    """Return Shannon entropy in nats, ignoring zero-probability states."""

    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def conditional_entropy(probabilities, mask) -> float | None:
    """Return entropy after conditioning on the selected states."""

    probabilities = np.asarray(probabilities, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    mass = probability_mass(probabilities, mask)
    return None if mass == 0.0 else shannon_entropy(probabilities[mask] / mass)


def lowest_energy_mass(probabilities, energies, count) -> float:
    """Add probability on the ``count`` lowest-energy basis states."""

    probabilities = np.asarray(probabilities)
    energies = np.asarray(energies)
    order = np.lexsort((np.arange(len(energies)), energies))
    return indexed_probability_mass(probabilities, order[: int(count)])


def expected_feasible_cost(probabilities, feasible, costs) -> float | None:
    """Return expected route cost conditioned on the decoded feasible states."""

    probabilities = np.asarray(probabilities, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    p_feas = probability_mass(probabilities, feasible)
    if p_feas == 0.0:
        return None
    costs = np.asarray(costs, dtype=float)
    return float(probabilities[feasible] @ costs[feasible] / p_feas)


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

    @property
    def p_opt_given_feas(self) -> float | None:
        """Probability of an optimal route conditioned on feasibility."""

        return conditional_probability(self.p_opt, self.p_feas)


def _probability_order(probabilities):
    """Sort high probability first and use the state index for ties."""

    indices = np.arange(len(probabilities))
    return np.lexsort((indices, -np.round(probabilities, 15)))


def distribution_metrics(
    probabilities,
    states,
    *,
    optimal_cost,
    threshold=1e-15,
) -> DistributionMetrics:
    """Calculate feasibility, optimality and the most likely decoded route."""

    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(states) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("invalid probability distribution")
    if np.any(probabilities < -threshold):
        raise ValueError("invalid probability distribution")

    feasible = np.array([state.is_decoder_valid for state in states])
    optimal = np.array(
        [state.is_decoder_valid and state.routing_cost == optimal_cost for state in states]
    )
    p_feas = probability_mass(probabilities, feasible)
    p_opt = probability_mass(probabilities, optimal)
    costs = np.array([state.routing_cost for state in states], dtype=float)

    order = _probability_order(probabilities)
    best_index = int(order[0])
    best_state = states[best_index]
    best_decoded = next(
        (states[int(i)] for i in order if states[int(i)].is_decoder_valid), None
    )

    optimal_probabilities = probabilities[optimal]
    if len(optimal_probabilities) == 0:
        raise RuntimeError("no optimal feasible state found")
    best_optimal_probability = float(optimal_probabilities.max())
    optimal_rank = 1 + int(np.sum(probabilities > best_optimal_probability + 1e-14))

    return DistributionMetrics(
        p_feas=p_feas,
        p_opt=p_opt,
        feasible_conditional_cost=expected_feasible_cost(
            probabilities, feasible, costs
        ),
        best_decoded_route=None if best_decoded is None else best_decoded.decoded_route,
        best_decoded_cost=None if best_decoded is None else best_decoded.routing_cost,
        best_state_index=best_index,
        best_state_probability=float(probabilities[best_index]),
        best_state_valid=best_state.is_decoder_valid,
        best_state_optimal=bool(optimal[best_index]),
        optimal_state_rank=optimal_rank,
    )


def top_state_rows(
    probabilities,
    states,
    *,
    optimal_cost,
    top_k=12,
) -> list[dict[str, object]]:
    """Return a small printable table of the most probable states."""

    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(states) or int(top_k) < 1:
        raise ValueError("invalid top-state request")

    rows = []
    for rank, index in enumerate(
        _probability_order(probabilities)[: int(top_k)], start=1
    ):
        state = states[int(index)]
        valid = state.is_decoder_valid
        rows.append(
            {
                "rank": rank,
                "bitstring": state.canonical_bitstring,
                "probability": float(probabilities[index]),
                "valid": valid,
                "optimal": bool(valid and state.routing_cost == optimal_cost),
                "decoded_route": (
                    "->".join(map(str, state.decoded_route)) if valid else ""
                ),
                "decoded_cost": state.routing_cost if valid else "",
                "invalid_reason": "" if valid else (
                    "flow_constraints_violated"
                    if state.flow_penalty > 0
                    else "route_decoder_rejected"
                ),
            }
        )
    return rows
