from __future__ import annotations

from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import numpy as np
import pytest

from support.qaoa_dynamics_visualization import (
    ALGORITHM_ORDER,
    CHECKPOINTS,
    QAOADynamicsVisualizationRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dynamics_visualization_repository() -> QAOADynamicsVisualizationRepository:
    return QAOADynamicsVisualizationRepository()


def test_visualization_catalog_is_complete_read_only_and_validated(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
):
    catalog = dynamics_visualization_repository.catalog()
    assert catalog["read_only"] is True
    assert catalog["optimization_rerun"] is False
    assert catalog["validation"]["all_checks_passed"] is True
    assert {item["id"] for item in catalog["algorithms"]} == set(ALGORITHM_ORDER)
    assert catalog["depths"] == [1, 2]
    assert len(catalog["summaries"]) == 6
    assert len(catalog["presentation_scenes"]) == 7


@pytest.mark.parametrize("algorithm", ALGORITHM_ORDER)
@pytest.mark.parametrize("depth", (1, 2))
def test_visualization_run_has_synchronous_valid_checkpoints_and_metrics(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
    algorithm: str,
    depth: int,
):
    run = dynamics_visualization_repository.load_run(algorithm, depth)
    assert run["source"]["optimization_rerun"] is False
    assert tuple(run["checkpoints"]) == CHECKPOINTS[depth]
    assert len(run["frames"]) == 1 + 2 * depth
    assert run["search_dimension"] == (20 if algorithm == "grover_feasible" else 2**14)
    for frame in run["frames"]:
        metrics = frame["metrics"]
        assert metrics["norm"] == pytest.approx(1.0, abs=1e-12)
        assert metrics["probability_sum"] == pytest.approx(1.0, abs=1e-12)
        assert metrics["p_opt"] <= metrics["p_feas"] + 1e-12
        assert sum(item["probability"] for item in frame["probability_flow"]) == pytest.approx(
            1.0, abs=1e-12
        )


@pytest.mark.parametrize("algorithm", ALGORITHM_ORDER)
@pytest.mark.parametrize("depth", (1, 2))
def test_cost_frames_preserve_displayed_probabilities_and_energy(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
    algorithm: str,
    depth: int,
):
    run = dynamics_visualization_repository.load_run(algorithm, depth)
    for checkpoint_index in range(1, len(run["frames"]), 2):
        before = run["frames"][checkpoint_index - 1]
        after = run["frames"][checkpoint_index]
        assert after["operation"] == "cost"
        assert after["metrics"]["expected_hc"] == pytest.approx(
            before["metrics"]["expected_hc"], abs=1e-12
        )
        assert np.asarray(after["state_values"])[:, 3] == pytest.approx(
            np.asarray(before["state_values"])[:, 3], abs=1e-12
        )


def test_feasible_visualization_displays_and_validates_all_twenty_routes(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
):
    run = dynamics_visualization_repository.load_run("grover_feasible", 2)
    assert run["structural_feasibility"] is True
    assert len(run["display_states"]) == 20
    assert all(state["feasible"] for state in run["display_states"])
    assert all(not state["flow_violations"] for state in run["display_states"])
    assert all(len(frame["probability_flow"]) == 20 for frame in run["frames"])
    assert all(frame["metrics"]["p_feas"] == pytest.approx(1.0, abs=1e-12) for frame in run["frames"])


def test_full_space_visualization_aggregates_without_losing_probability_mass(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
):
    for algorithm in ("penalty_x", "grover_global"):
        run = dynamics_visualization_repository.load_run(algorithm, 2)
        assert run["phase_scatter"]["total_count"] == 2**14
        assert run["phase_scatter"]["downsampled"] is True
        assert run["phase_scatter"]["all_feasible_preserved"] is True
        assert len(run["phase_scatter"]["display_positions"]) < 2**14
        assert run["energy_landscape"]["total_state_count"] == 2**14
        assert run["energy_landscape"]["feasible_state_count"] == 20
        for frame in run["frames"]:
            assert {item["kind"] for item in frame["probability_flow"]} == {
                "state",
                "aggregate",
            }


def test_parameter_landscape_does_not_invent_optimizer_history_or_p2_projection(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
):
    p1 = dynamics_visualization_repository.load_run("grover_global", 1)[
        "parameter_landscape"
    ]
    assert p1["available"] is True
    assert p1["grid_shape"] == [25, 25]
    assert len(p1["metrics"]["expected_hc"]) == 625
    assert p1["optimizer_trajectory"]["available"] is False
    p2 = dynamics_visualization_repository.load_run("grover_global", 2)[
        "parameter_landscape"
    ]
    assert p2["available"] is False
    assert "four-dimensional" in p2["reason"]


def test_visualization_validation_report_has_no_failed_scientific_check(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
):
    report = dynamics_visualization_repository.validation_report
    assert report["all_checks_passed"] is True
    assert all(report["checks"].values())
    assert report["maxima"]["cost_probability_delta"] < 1e-12
    assert report["maxima"]["cost_energy_delta"] < 1e-12
    assert report["maxima"]["stored_metric_error"] < 1e-12


def test_standard_library_visualization_server_serves_assets_and_validated_api(
    dynamics_visualization_repository: QAOADynamicsVisualizationRepository,
):
    script_path = PROJECT_ROOT / "scripts" / "run_qaoa_dynamics_visualizer.py"
    spec = importlib.util.spec_from_file_location("run_qaoa_dynamics_visualizer", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), module.build_handler(dynamics_visualization_repository)
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/api/health", timeout=5) as response:
            health = json.load(response)
        assert health == {"status": "ok", "scientific_validation": True}
        with urlopen(f"{base}/api/run/grover_global/2", timeout=5) as response:
            run = json.load(response)
        assert run["run_id"] == "grover_global_p2_seed2601"
        with urlopen(f"{base}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        with urlopen(f"{base}/app.js", timeout=5) as response:
            javascript = response.read().decode("utf-8")
        assert "PANEL J" in html
        assert "presentation_scenes" in javascript
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
