from __future__ import annotations

import numpy as np
import pytest

from support.qaoa_visualization import METHOD_ORDER, QAOAVisualizationRepository


@pytest.fixture(scope="module")
def visualization_repository() -> QAOAVisualizationRepository:
    return QAOAVisualizationRepository()


def test_catalog_contains_one_self_contained_demo_for_every_method(
    visualization_repository: QAOAVisualizationRepository,
):
    catalog = visualization_repository.catalog()
    assert catalog["read_only_replay"] is True
    assert catalog["method_count"] == 3
    assert catalog["run_count"] == 3
    assert {run["method"] for run in catalog["runs"]} == set(METHOD_ORDER)
    assert {run["depth"] for run in catalog["runs"]} == {1, 2, 3}
    assert {run["seed"] for run in catalog["runs"]} == {2601}


def test_gm_threshold_animation_reconstructs_operator_stages_and_energy(
    visualization_repository: QAOAVisualizationRepository,
):
    run = visualization_repository.load_run("gm_th_qaoa_p3_demo")
    assert run["source"]["read_only"] is True
    assert run["depth"] == 3
    assert run["circuit"]["phase"]["symbol"] == "U_T"
    assert run["circuit"]["mixer"]["symbol"] == "U_G"
    assert run["circuit"]["hardware_gate_decomposition"] is False
    assert run["circuit"]["resources"] == {
        "logical_states": 20,
        "qaoa_layers": 3,
        "variational_parameters": 6,
        "operator_blocks": 6,
        "measurement_outcomes": 20,
    }

    frame = next(frame for frame in run["frames"] if frame["in_bounds"])
    assert len(frame["gammas"]) == len(frame["betas"]) == 3
    assert [stage["kind"] for stage in frame["stages"]] == [
        "initial",
        "phase",
        "mixer",
        "phase",
        "mixer",
        "phase",
        "mixer",
    ]
    assert frame["stages"][-1]["expected_cost"] == pytest.approx(
        frame["expected_cost"], abs=1e-9
    )
    assert frame["stages"][-1]["bsp"] == pytest.approx(frame["bsp"], abs=1e-12)
    assert sum(frame["stages"][-1]["probabilities"]) == pytest.approx(1.0, abs=1e-12)


def test_cost_phase_changes_phase_before_mixer_changes_probabilities(
    visualization_repository: QAOAVisualizationRepository,
):
    run = visualization_repository.load_run("bsp_path_exchange_p1_demo")
    frame = run["frames"][1]
    initial, phase, mixer = frame["stages"]
    assert run["circuit"]["phase"]["symbol"] == "U_C"
    assert run["circuit"]["mixer"]["symbol"] == "U_PE"
    assert phase["expected_cost"] == pytest.approx(initial["expected_cost"], abs=1e-12)
    assert phase["probabilities"] == pytest.approx(initial["probabilities"], abs=1e-12)
    assert not np.allclose(mixer["probabilities"], phase["probabilities"], atol=1e-12)


def test_expectation_demo_uses_expected_normalized_cost_as_loss(
    visualization_repository: QAOAVisualizationRepository,
):
    run = visualization_repository.load_run("gm_qaoa_expectation_p2_demo")
    assert run["source"]["kind"] == "built_in_example_trajectory"
    for frame in run["frames"]:
        assert frame["loss"] == pytest.approx(frame["expected_normalized_cost"])
        assert frame["p_feas"] == pytest.approx(1.0)


def test_unknown_run_id_cannot_escape_the_retained_catalog(
    visualization_repository: QAOAVisualizationRepository,
):
    with pytest.raises(KeyError, match="unknown_visualization_run"):
        visualization_repository.load_run("../../basis")
