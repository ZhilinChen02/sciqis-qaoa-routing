from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from day5_analysis import (
    EXPECTED_DAY5_CONTRACT_SHA256,
    load_day5_contract,
    probability_vector_sha256,
    representative_distribution,
)


def _rows():
    with Path("results/finite_shot_replicates.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def test_day5_contract_is_frozen_before_descriptive_extensions():
    contract, digest = load_day5_contract()
    assert digest == EXPECTED_DAY5_CONTRACT_SHA256
    assert contract["classification"].keys() == {
        "CORE_CONFIRMATORY_FOR_COURSE_STORY",
        "POST_CORE_DESCRIPTIVE_EXTENSION",
    }
    assert contract["finite_shot_sampling"]["seed"] == 1038705
    assert contract["finite_shot_sampling"]["shot_counts"] == [
        256, 1024, 4096, 16384
    ]
    assert contract["finite_shot_sampling"]["independent_replicates_per_shot_count"] == 200
    assert contract["profiling"]["warmup_repeats_per_stage"] == 3
    assert contract["profiling"]["timed_repeats_per_stage"] == 31
    assert not any(contract["immutability"].values())


def test_finite_shot_source_is_exact_committed_A2_p2_distribution():
    states, cell, distribution = representative_distribution()
    summary = json.loads(Path("results/finite_shot_summary.json").read_text())
    assert len(states) == 16384
    assert (cell["identity"]["A"], cell["identity"]["p"]) == (2, 2)
    assert cell["optimization"]["selected_start_id"] == 2
    assert summary["source_state"]["optimized_parameters"] == cell["optimization"]["final_parameters"]
    assert summary["source_state"]["probability_vector_float64_sha256"] == probability_vector_sha256(distribution)
    feasible = np.asarray([state.flow_penalty == 0 for state in states], dtype=bool)
    assert np.isclose(
        distribution[feasible].sum(), cell["quality"]["p_feas"], atol=1e-15, rtol=0
    )
    assert np.isclose(
        distribution[10377], cell["quality"]["p_opt"], atol=1e-15, rtol=0
    )


def test_sampling_matrix_is_complete_and_estimates_are_probabilities():
    rows = _rows()
    assert len(rows) == 800
    for shots in (256, 1024, 4096, 16384):
        group = [row for row in rows if int(row["shots"]) == shots]
        assert len(group) == 200
        assert [int(row["replicate_id"]) for row in group] == list(range(200))
    for row in rows:
        shots = int(row["shots"])
        p_feas = float(row["p_feas_hat"])
        p_opt = float(row["p_opt_hat"])
        invalid = float(row["invalid_mass_hat"])
        assert 0 <= p_opt <= p_feas <= 1
        assert 0 <= invalid <= 1
        assert np.isclose(invalid, 1 - p_feas, atol=1e-15, rtol=0)
        assert int(row["exact_optimum_observation_count"]) == round(p_opt * shots)
        assert int(row["feasible_observation_count"]) == round(p_feas * shots)


def test_exact_optimum_counts_are_reproducibly_taken_from_state_10377():
    contract, _digest = load_day5_contract()
    states, _cell, distribution = representative_distribution()
    feasible = np.asarray(
        [state.state_index for state in states if state.flow_penalty == 0], dtype=int
    )
    rng = np.random.default_rng(contract["finite_shot_sampling"]["seed"])
    rows = _rows()
    for row in rows:
        shots = int(row["shots"])
        counts = rng.multinomial(shots, distribution)
        assert int(row["exact_optimum_observation_count"]) == int(counts[10377])
        assert int(row["feasible_observation_count"]) == int(counts[feasible].sum())
        assert int(row["most_frequent_state_index"]) == int(np.argmax(counts))


def test_sampling_summary_reconciles_with_all_raw_replicates():
    summary = json.loads(Path("results/finite_shot_summary.json").read_text())
    rows = _rows()
    for record in summary["shot_summaries"]:
        group = [row for row in rows if int(row["shots"]) == record["shots"]]
        p_feas = np.asarray([float(row["p_feas_hat"]) for row in group])
        p_opt = np.asarray([float(row["p_opt_hat"]) for row in group])
        counts = np.asarray(
            [int(row["exact_optimum_observation_count"]) for row in group]
        )
        for values, key in ((p_feas, "p_feas_sampling"), (p_opt, "p_opt_sampling")):
            observed = record[key]
            assert observed["mean"] == float(np.mean(values))
            assert observed["empirical_standard_deviation"] == float(np.std(values, ddof=1))
            assert observed["quantile_05"] == float(np.quantile(values, 0.05))
            assert observed["quantile_50"] == float(np.quantile(values, 0.50))
            assert observed["quantile_95"] == float(np.quantile(values, 0.95))
        assert record["zero_exact_optimum_observation_fraction"] == float(
            np.mean(counts == 0)
        )
