"""Start here: the routing problem from graph to QUBO in one short file."""

import networkx as nx

from graph import load_graph, path_cost, path_to_edge_bitstring
from qubo import build_qubo, flow_penalty, routing_cost


def run(penalty=6.0):
    """Return the main intermediate values of the routing problem."""

    # 1. Read the seven nodes and fourteen directed edges.
    graph = load_graph()
    source = graph.graph["source"]
    target = graph.graph["target"]

    # 2. Find the classical shortest route.  This is our reference answer.
    shortest_route = nx.shortest_path(
        graph,
        source=source,
        target=target,
        weight="weight",
    )
    shortest_route = tuple(shortest_route)
    shortest_cost = path_cost(graph, shortest_route)

    # 3. Convert the route into one bit for every edge.
    edge_bits = path_to_edge_bitstring(graph, shortest_route)

    # 4. Build the QUBO and evaluate the reference route.
    qubo = build_qubo(graph, penalty)
    direct_cost = routing_cost(graph, edge_bits)
    constraint_penalty = flow_penalty(graph, edge_bits)
    qubo_energy = qubo.evaluate(edge_bits)

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "source": source,
        "target": target,
        "shortest_route": shortest_route,
        "shortest_cost": shortest_cost,
        "edge_bits": edge_bits,
        "direct_cost": direct_cost,
        "flow_penalty": constraint_penalty,
        "qubo_energy": float(qubo_energy),
        "penalty_coefficient": float(penalty),
    }


def print_summary(result):
    """Print the pipeline result without any report-specific formatting."""

    route_text = " -> ".join(str(node) for node in result["shortest_route"])
    bit_text = "".join(str(bit) for bit in result["edge_bits"])

    print(f"graph: {result['node_count']} nodes, {result['edge_count']} edges")
    print(f"source and target: {result['source']} -> {result['target']}")
    print(f"shortest route: {route_text}")
    print(f"route cost: {result['shortest_cost']}")
    print(f"edge bits: {bit_text}")
    print(f"flow penalty: {result['flow_penalty']}")
    print(f"QUBO energy: {result['qubo_energy']:g}")


if __name__ == "__main__":
    print_summary(run())
