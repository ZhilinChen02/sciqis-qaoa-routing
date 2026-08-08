from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from day5_analysis import (
    EXPECTED_CORE_RESULTS_SHA256,
    EXPECTED_CORE_SUMMARY_SHA256,
    EXPECTED_GRAPH_SHA256,
    EXPECTED_OPTIMIZATION_RUNS_SHA256,
    EXPECTED_OPTIMIZATION_SHA256,
    EXPECTED_PENALTY_SHA256,
    file_sha256,
    reconcile_frozen_day4,
)


def test_all_frozen_day1_through_day4_identities_remain_exact():
    core, reconciliation = reconcile_frozen_day4()
    hashes = reconciliation["required_sha256"]
    assert hashes["data/graph.json"] == EXPECTED_GRAPH_SHA256
    assert hashes["data/penalty_contract.json"] == EXPECTED_PENALTY_SHA256
    assert hashes["data/optimization_contract.json"] == EXPECTED_OPTIMIZATION_SHA256
    assert hashes["results/core_experiment_results.json"] == EXPECTED_CORE_RESULTS_SHA256
    assert hashes["results/core_experiment_summary.csv"] == EXPECTED_CORE_SUMMARY_SHA256
    assert hashes["results/optimization_runs.csv"] == EXPECTED_OPTIMIZATION_RUNS_SHA256
    assert len(core["optimization_runs"]) == 24
    assert len(core["selected_cells"]) == 8


def test_final_summary_reconciles_with_original_artifacts():
    final = json.loads(Path("results/final_project_summary.json").read_text())
    exact = json.loads(Path("results/exact_reference.json").read_text())
    threshold = json.loads(Path("results/penalty_threshold_analysis.json").read_text())
    core = json.loads(Path("results/core_experiment_results.json").read_text())
    sampling = json.loads(Path("results/finite_shot_summary.json").read_text())
    runtime = json.loads(Path("results/runtime_profile_summary.json").read_text())
    assert final["problem"]["node_count"] == exact["graph"]["node_count"] == 7
    assert final["problem"]["directed_edge_count"] == exact["graph"]["edge_count"] == 14
    assert final["problem"]["valid_route_count"] == exact["simple_path_count"] == 20
    assert final["problem"]["exact_cost"] == exact["exact_reference"]["cost"] == 10
    assert final["modeling"]["A_crit"] == threshold["A_crit"]["value"] == 5
    assert final["modeling"]["flow_penalty_zero_iff_valid_route"] is True
    assert final["core_results"]["run_count"] == len(core["optimization_runs"]) == 24
    assert final["core_results"]["selected_cell_count"] == len(core["selected_cells"]) == 8
    for compact, source in zip(
        final["core_results"]["selected_cells"],
        sorted(core["selected_cells"], key=lambda cell: (cell["identity"]["A"], cell["identity"]["p"])),
    ):
        assert (compact["A"], compact["p"]) == (source["identity"]["A"], source["identity"]["p"])
        assert compact["final_parameters"] == source["optimization"]["final_parameters"]
        assert compact["p_feas"] == source["quality"]["p_feas"]
        assert compact["p_opt"] == source["quality"]["p_opt"]
    assert final["sampling"] == sampling
    assert final["computing"] == runtime


def test_runtime_profile_has_every_required_positive_stage():
    contract = json.loads(Path("data/day5_analysis_contract.json").read_text())
    summary = json.loads(Path("results/runtime_profile_summary.json").read_text())
    with Path("results/runtime_profile.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["stage"] for row in summary["stage_summaries"]] == contract["profiling"]["stages"]
    assert len(rows) == 11 * 31
    assert all(int(row["duration_ns"]) > 0 for row in rows)
    assert all(record["median_ns"] > 0 for record in summary["stage_summaries"])
    assert summary["day4_observed_optimization_runtimes"]["overall"]["run_count"] == 24
    assert summary["day4_observed_optimization_runtimes"]["total_objective_evaluations"] == 5630


def test_figure_manifest_references_renderable_png_and_svg_files():
    manifest = json.loads(Path("results/final_figure_manifest.json").read_text())
    assert manifest["figure_count"] == 13
    assert [row["figure"] for row in manifest["figures"]] == list(range(1, 14))
    assert 6 <= len(manifest["core_story_figure_numbers"]) <= 8
    for row in manifest["figures"]:
        png = Path(row["png"])
        svg = Path(row["svg"])
        assert png.exists() and svg.exists()
        with Image.open(png) as image:
            assert image.format == "PNG"
            assert list(image.size) == row["audit"]["png_dimensions"]
        ElementTree.parse(svg)
        assert file_sha256(png) == row["audit"]["png_sha256"]
        assert file_sha256(svg) == row["audit"]["svg_sha256"]


def test_final_report_uses_only_explicitly_negated_evidence_boundaries():
    report = Path("reports/final_course_project_report.md").read_text(encoding="utf-8")
    lowered = report.lower()
    prohibited_positive_claims = (
        "demonstrates quantum advantage",
        "achieves quantum advantage",
        "qaoa outperforms classical",
        "faster than classical",
    )
    assert not any(claim in lowered for claim in prohibited_positive_claims)
    assert "no quantum advantage is claimed" in lowered
    assert "not a hardware experiment" in lowered
    assert "not a contradiction" in lowered


def test_scientific_freeze_hashes_every_declared_artifact():
    freeze = json.loads(Path("data/scientific_freeze_v3.json").read_text())
    assert freeze["scientific_status"] == "COURSE_PROJECT_SCIENCE_COMPLETE"
    assert freeze["scope"] == "ONE_GRAPH_IDEAL_SIMULATION_SHALLOW_QAOA"
    assert freeze["prohibited_post_freeze_changes"] == [
        "graph", "A grid", "optimizer protocol", "selected core results"
    ]
    for record in freeze["artifact_hashes"].values():
        assert file_sha256(record["path"]) == record["sha256"]


def test_presentation_outline_has_exactly_ten_concise_slides():
    outline = Path("reports/presentation_outline_15min.md").read_text(encoding="utf-8")
    assert outline.count("## Slide ") == 10
    for number in range(1, 11):
        assert f"## Slide {number} —" in outline
