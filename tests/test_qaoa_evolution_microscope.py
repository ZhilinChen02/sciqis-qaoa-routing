from __future__ import annotations

from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import pytest

from support.qaoa_evolution_microscope import (
    ALGORITHMS,
    QAOAEvolutionMicroscopeRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def microscope() -> QAOAEvolutionMicroscopeRepository:
    return QAOAEvolutionMicroscopeRepository()


def test_catalog_exposes_two_full_space_algorithms_and_all_110_depths(microscope):
    catalog = microscope.catalog()
    assert catalog["read_only"] is True
    assert catalog["optimization_rerun"] is False
    assert [item["id"] for item in catalog["algorithms"]] == list(ALGORITHMS)
    assert catalog["depths"] == list(range(1, 111))
    assert catalog["default"] == {
        "algorithm": "penalty_x",
        "depth": 110,
        "layer": 21,
        "stage": "before_cost",
    }
    assert catalog["hamiltonian"]["fixed_structure"] is True
    assert catalog["hamiltonian"]["cost"]["counts"] == {
        "constant": 1,
        "linear": 14,
        "zz": 46,
    }
    assert catalog["validation"]["all_checks_passed"] is True


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_p110_config_has_all_220_frozen_angles_and_no_optimizer_trace(
    microscope, algorithm
):
    config = microscope.run_config(algorithm, 110)
    assert config["optimized_depth"] == 110
    assert config["parameter_count"] == 220
    assert len(config["parameters"]) == 110
    assert all(set(row) == {"layer", "gamma", "beta"} for row in config["parameters"])
    assert config["optimizer"]["trace_available"] is False
    assert config["source"]["optimization_rerun"] is False
    assert config["hamiltonian_structure_fixed"] is True
    assert config["layer_angles_change"] is True


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_p110_replay_has_every_layer_and_matches_frozen_final_metrics(
    microscope, algorithm
):
    summary = microscope.evolution_summary(algorithm, 110)
    artifact = microscope.run_config(algorithm, 110)["frozen_final"]
    assert summary["checkpoint_count"] == 221
    assert len(summary["layers"]) == 110
    assert summary["validation"]["all_checks_passed"] is True
    final = summary["layers"][-1]["after_mixer"]
    assert final["expected_hc"] == pytest.approx(artifact["expected_hc"], abs=1e-11)
    assert final["p_feas"] == pytest.approx(artifact["p_feas"], abs=1e-12)
    assert final["p_opt"] == pytest.approx(artifact["p_opt"], abs=1e-12)
    assert final["p_opt_given_feasible"] == pytest.approx(
        artifact["p_opt_given_feasible"], abs=1e-12
    )


def test_layer_21_cost_encodes_phase_without_probability_or_energy_change(microscope):
    before = microscope.checkpoint("penalty_x", 110, 21, "before_cost")
    after = microscope.checkpoint("penalty_x", 110, 21, "after_cost")
    assert after["transition"]["probability_max_abs_delta"] < 1e-12
    assert abs(after["transition"]["metrics_delta"]["expected_hc"]) < 1e-12
    assert after["transition"]["phase_max_abs_delta"] > 0.1
    assert after["physics"]["cost_changed_phase"] is True
    assert after["metrics"]["p_opt"] == pytest.approx(before["metrics"]["p_opt"])


def test_layer_21_mixer_redistributes_probability_and_updates_graph_marginals(microscope):
    after = microscope.checkpoint("penalty_x", 110, 21, "after_mixer", top_n=50)
    assert after["transition"]["probability_max_abs_delta"] > 1e-6
    assert after["physics"]["mixer_changed_probability"] is True
    assert len(after["states"]["rows"]) in (50, 51)
    assert len(after["graph"]["edges"]) == 14
    assert all(0 <= edge["selection_probability"] <= 1 for edge in after["graph"]["edges"])
    assert after["states"]["browser_receives_full_statevector"] is False


def test_full_p110_validation_covers_both_mixers(microscope):
    report = microscope.full_validation(110)
    assert report["all_checks_passed"] is True
    assert len(report["runs"]) == 2
    assert report["maxima"]["normalization_error"] < 1e-10
    assert report["maxima"]["cost_probability_delta"] < 1e-10
    assert report["maxima"]["cost_energy_delta"] < 1e-10
    assert report["maxima"]["final_formal_statevector_error"] < 1e-10
    assert report["maxima"]["final_frozen_distribution_error"] < 1e-10
    assert report["maxima"]["metrics_implementation_error"] < 1e-10


def test_server_exposes_compact_summary_checkpoint_and_track_apis(microscope):
    script_path = PROJECT_ROOT / "scripts" / "run_qaoa_dynamics_visualizer.py"
    spec = importlib.util.spec_from_file_location("run_qaoa_dynamics_visualizer_v2", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.build_handler(microscope))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for asset, content_type in (
            ("/", "text/html"),
            ("/math.js", "text/javascript"),
            ("/vendor/katex/katex.min.css", "text/css"),
            ("/vendor/katex/fonts/KaTeX_Main-Regular.woff2", "font/woff2"),
        ):
            with urlopen(f"{base}{asset}", timeout=10) as response:
                assert response.headers.get_content_type() == content_type
                assert response.read()
        with urlopen(f"{base}/api/evolution/global_grover/3/summary", timeout=10) as response:
            summary = json.load(response)
        assert summary["checkpoint_count"] == 7
        with urlopen(
            f"{base}/api/evolution/global_grover/3/checkpoint?layer=2&stage=after_cost&top=10",
            timeout=10,
        ) as response:
            checkpoint = json.load(response)
        assert checkpoint["stage"] == "after_cost"
        assert checkpoint["states"]["returned_count"] in (10, 11)
        with urlopen(f"{base}/api/evolution/global_grover/3/track/10377", timeout=10) as response:
            track = json.load(response)
        assert len(track["values"]) == 7
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
