import importlib.util
from pathlib import Path

import pytest

from csubst.structural_validation import PredictionRecord


def load_tool(name):
    path = Path(__file__).resolve().parents[2] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_summary_balances_families_and_compares_same_sequence_structures():
    data = dict(schema_version=1, records=[
        dict(id="one", amino_acids="MKK", structure_3di="AAA", family="big", group="group",
             split="test", truth_source="experimental", source="fixture"),
        dict(id="two", amino_acids="L", structure_3di="C", family="small", group="group",
             split="test", truth_source="experimental", source="fixture"),
        dict(id="three", amino_acids="L", structure_3di="A", family="small", group="group",
             split="test", truth_source="experimental", source="fixture"),
    ])
    predictions = {name: {row["id"]: PredictionRecord(row["amino_acids"], character * len(row["amino_acids"]), name, "fixture")
                          for row in data["records"]} for name, character in [("first", "A"), ("second", "C")]}
    report = load_tool("compare_3di_validation").compare(data, predictions)
    assert report["summaries"][0]["q20"] == .8
    assert report["summaries"][0]["macro_family_q20"] == .75
    assert report["same_sequence_structure_pairs"][0]["reference_disagreement"] == 1
    assert report["predictor_common_errors"][0]["both_wrong"] == 0
    assert report["predictor_common_errors"][0]["marginal_product"] == pytest.approx(.16)
    assert report["structural_omega_calibrated"] is False


def test_fixed_gtr_report_keeps_scope_and_zero_error_parity():
    result = load_tool("validate_3di_fixed_gtr").run(sites=100, seed=4)
    assert result["fitted_parameters"] is False
    assert result["structural_omega_calibrated"] is False
    assert len(result["results"]) == 8
    a, b = result["results"][:2]
    for key in ("root_brier", "root_log_loss", "edge_joint_log_loss", "mean_edge_change_probability"):
        assert a[key] == b[key]
