"""Build and draw one small QAOA circuit with an Ising cost layer."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


def ising_gate_angles(
    gamma: float,
    beta: float,
    h0: float,
    h1: float,
    coupling: float,
) -> dict[str, float]:
    """Convert Ising and QAOA parameters into rotation angles."""

    return {
        "rz_q0": 2.0 * gamma * h0,
        "rz_q1": 2.0 * gamma * h1,
        "rz_zz": 2.0 * gamma * coupling,
        "rx": 2.0 * beta,
    }


def build_simple_ising_circuit(
    gamma: float = 0.4,
    beta: float = 0.3,
    h0: float = 1.0,
    h1: float = -0.5,
    coupling: float = 0.8,
) -> QuantumCircuit:
    """Build a two-qubit example with initial, cost, mixer, and measurement steps."""

    angles = ising_gate_angles(gamma, beta, h0, h1, coupling)
    circuit = QuantumCircuit(2, 2)

    # Initial state: H creates an equal superposition on each qubit.
    circuit.h(0)
    circuit.h(1)
    circuit.barrier(label="initial")

    # Local Ising terms h_0 Z_0 and h_1 Z_1 become single-qubit RZ gates.
    circuit.rz(angles["rz_q0"], 0)
    circuit.rz(angles["rz_q1"], 1)

    # The interaction J Z_0 Z_1 becomes the sequence CX-RZ-CX.
    circuit.cx(0, 1)
    circuit.rz(angles["rz_zz"], 1)
    circuit.cx(0, 1)
    circuit.barrier(label="cost")

    # The standard X mixer applies RX(2 beta) to both qubits.
    circuit.rx(angles["rx"], 0)
    circuit.rx(angles["rx"], 1)
    circuit.barrier(label="mixer")

    # Measure both qubits in the computational basis.
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    return circuit


def save_circuit_image(circuit: QuantumCircuit, output_path: Path) -> None:
    """Save the Qiskit text drawing as a simple PNG image."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        drawing = circuit.draw(output="text", fold=-1).single_string()
    lines = drawing.splitlines()
    longest_line = max(len(line) for line in lines)

    width = max(10.0, longest_line * 0.11)
    height = max(3.0, len(lines) * 0.32)
    figure, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    axis.text(
        0.01,
        0.5,
        drawing,
        family="DejaVu Sans Mono",
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="center",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    gamma = 0.4
    beta = 0.3
    h0 = 1.0
    h1 = -0.5
    coupling = 0.8

    circuit = build_simple_ising_circuit(gamma, beta, h0, h1, coupling)
    angles = ising_gate_angles(gamma, beta, h0, h1, coupling)
    repository_root = Path(__file__).resolve().parents[1]
    output_path = repository_root / "figures" / "course" / "simple_ising_circuit.png"
    save_circuit_image(circuit, output_path)

    print("Simple two-qubit Ising QAOA circuit")
    print(f"gamma = {gamma}, beta = {beta}")
    print(f"h0 = {h0}, h1 = {h1}, coupling = {coupling}")
    print("rotation angles:")
    for name, value in angles.items():
        print(f"  {name} = {value:.3f}")
    print(f"circuit depth = {circuit.depth()}")
    print(f"operation counts = {dict(circuit.count_ops())}")
    print(f"image saved to {output_path}")


if __name__ == "__main__":
    main()
