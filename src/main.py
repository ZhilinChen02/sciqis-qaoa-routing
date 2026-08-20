"""Small graph -> QUBO -> Ising -> QAOA -> route-metrics example."""

import numpy as np

from graph import load_graph
from metrics import distribution_metrics
from qaoa import EXPECTATION, GROVER_MIXER, X_MIXER, normalized_diagonal, optimize
from qubo import (
    build_qubo,
    enumerate_state_space,
    max_qubo_ising_error,
    minimum_energy_states,
)


PENALTY = 6.0
DEPTH = 1
SEED = 2601
EVALUATION_BUDGET = 100
OBJECTIVE = EXPECTATION
MIXERS = (X_MIXER, GROVER_MIXER)


def run(*, penalty=PENALTY, depth=DEPTH, seed=SEED, evaluation_budget=EVALUATION_BUDGET):


    graph = load_graph()
    states = enumerate_state_space(graph)
    qubo = build_qubo(graph, penalty)
    if max_qubo_ising_error(states, [qubo]):
        raise RuntimeError("QUBO and Ising energies disagree")

    exact_cost, ground = minimum_energy_states(states, penalty)
    if len(ground) != 1 or ground[0].decoded_route is None:
        raise RuntimeError("ground state does not match the exact route")

    raw_energies = np.asarray([float(qubo.evaluate(state.edge_vector)) for state in states])
    energies, _, _ = normalized_diagonal(raw_energies)
    results = {}
    for mixer in MIXERS:
        optimizer, final_probabilities = optimize(
            energies,
            depth=depth,
            mixer=mixer,
            objective=OBJECTIVE,
            seed=seed,
            evaluation_budget=evaluation_budget,
        )
        metrics = distribution_metrics(
            final_probabilities, states, optimal_cost=float(exact_cost)
        )
        results[mixer] = {
            "optimizer": optimizer,
            "p_feas": metrics.p_feas,
            "p_opt": metrics.p_opt,
            "p_opt_given_feas": metrics.p_opt_given_feas,
        }

    return {
        "edges": graph.number_of_edges(),
        "states": len(states),
        "exact_route": ground[0].decoded_route,
        "exact_cost": int(exact_cost),
        "depth": int(depth),
        "seed": int(seed),
        "results": results,
    }


def print_summary(result):
    route = "->".join(map(str, result["exact_route"]))
    print(f"routing: {result['edges']} edges, {result['states']} bit strings")
    print(f"exact route: {route}, cost={result['exact_cost']}")
    for mixer, values in result["results"].items():
        optimizer = values["optimizer"]
        print(
            f"{mixer}: evals={optimizer.evaluations}, {optimizer.reason}; "
            f"p_feas={values['p_feas']:.8g}, p_opt={values['p_opt']:.8g}, "
            f"p_opt|feas={values['p_opt_given_feas']:.8g}"
        )


if __name__ == "__main__":
    print_summary(run())
