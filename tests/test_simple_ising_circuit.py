from pathlib import Path

from scripts.show_simple_ising_circuit import (
    build_simple_ising_circuit,
    ising_gate_angles,
    save_circuit_image,
)


def test_ising_gate_angles() -> None:
    angles = ising_gate_angles(
        gamma=0.4,
        beta=0.3,
        h0=1.0,
        h1=-0.5,
        coupling=0.8,
    )
    assert abs(angles["rz_q0"] - 0.8) < 1e-12
    assert abs(angles["rz_q1"] + 0.4) < 1e-12
    assert abs(angles["rz_zz"] - 0.64) < 1e-12
    assert abs(angles["rx"] - 0.6) < 1e-12


def test_circuit_contains_the_expected_basic_steps() -> None:
    circuit = build_simple_ising_circuit()
    counts = circuit.count_ops()
    assert counts["h"] == 2
    assert counts["rz"] == 3
    assert counts["cx"] == 2
    assert counts["rx"] == 2
    assert counts["measure"] == 2


def test_circuit_image_is_created(tmp_path: Path) -> None:
    output_path = tmp_path / "simple_ising_circuit.png"
    save_circuit_image(build_simple_ising_circuit(), output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
