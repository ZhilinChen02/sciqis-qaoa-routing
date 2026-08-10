"""Archived smoke test for the superseded comparison pipeline."""

from __future__ import annotations

import csv
import json

from experiment import DEFAULT_CONFIG_PATH, run_experiment


def test_q1_p1_and_q2_p1_p2_smoke(tmp_path):
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["optimizer_budget"] = 6
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    results = tmp_path / "results"
    summary = run_experiment(config_path=config_path, results_dir=results)
    assert len(summary["results"]) == 3
    assert {(row["solver"], row["p"]) for row in summary["results"]} == {
        ("Q1 Penalty-X", 1),
        ("Q2 Warm-Start", 1),
        ("Q2 Warm-Start", 2),
    }
    for filename in (
        "metrics.csv", "top_bitstrings.csv", "runtime_profile.csv",
        "warm_start_values.csv", "exact_reference.json", "experiment_summary.json",
    ):
        assert (results / filename).is_file()
    with (results / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(abs(float(row["probability_sum"]) - 1.0) < 1e-10 for row in rows)
    assert all(int(row["optimizer_evaluations"]) <= 6 for row in rows)
