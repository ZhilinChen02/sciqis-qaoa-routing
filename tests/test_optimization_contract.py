from __future__ import annotations

from math import pi

import numpy as np
import scipy

from optimization import (
    EXPECTED_OPTIMIZATION_CONTRACT_SHA256,
    file_sha256,
    load_optimization_contract,
    parameter_bounds,
)


def test_optimization_contract_is_frozen_and_complete():
    contract, digest = load_optimization_contract()
    assert digest == EXPECTED_OPTIMIZATION_CONTRACT_SHA256
    assert file_sha256("data/optimization_contract.json") == digest
    assert contract["optimizer"]["scipy_version"] == scipy.__version__ == "1.18.0"
    assert contract["optimizer"]["method"] == "COBYLA"
    assert contract["optimizer"]["maximum_objective_evaluations_per_start"] == 240
    assert contract["optimizer"]["options"] == {
        "maxiter": 240,
        "rhobeg": 0.5,
        "tol": 1e-6,
        "catol": 1e-8,
        "disp": False,
    }
    assert contract["experimental_matrix"]["starts_per_cell"] == 3
    assert contract["experimental_matrix"]["total_optimization_runs"] == 24
    assert contract["cell_selection_rule"]["selection_key"] == [
        "final_expected_qubo_energy", "start_id"
    ]
    assert "p_feas" in contract["cell_selection_rule"]["forbidden_selection_metrics"]


def test_starts_regenerate_once_from_seed_and_are_reused_by_penalty():
    contract, _digest = load_optimization_contract()
    generation = contract["start_generation"]
    assert generation["seed"] == 10387
    assert generation["generated_once"] is True
    assert generation["same_depth_specific_starts_reused_for_every_A"] is True

    generator = np.random.default_rng(10387)
    expected = {}
    for p in (1, 2):
        expected[str(p)] = []
        for start_id in range(3):
            values = []
            for _layer in range(p):
                values.extend((generator.uniform(0, 2 * pi), generator.uniform(0, pi)))
            expected[str(p)].append(
                {"start_id": start_id, "parameters": values}
            )
    for p in ("1", "2"):
        actual = generation["starts_by_depth"][p]
        assert [row["start_id"] for row in actual] == [0, 1, 2]
        assert np.allclose(
            [row["parameters"] for row in actual],
            [row["parameters"] for row in expected[p]],
            rtol=0,
            atol=0,
        )


def test_parameter_bounds_are_identical_by_layer_and_depth():
    for p in (1, 2):
        bounds = parameter_bounds(p)
        assert np.array_equal(bounds.lb, np.asarray([0.0, 0.0] * p))
        assert np.array_equal(bounds.ub, np.asarray([2 * pi, pi] * p))
