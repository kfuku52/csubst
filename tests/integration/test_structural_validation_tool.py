import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from csubst import structural_validation as sv


def test_validation_tool_reuses_artifact_without_models(tmp_path):
    manifest = {"schema_version": 1, "records": [
        {"id": "one", "amino_acids": "MK", "structure_3di": "AC", "family": "fixture",
         "split": "test", "group": "fixture", "truth_source": "simulated", "source": "unit fixture"},
    ]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    predictions = {"one": sv.PredictionRecord.from_logits(
        "MK", np.zeros((2, 20)), sv.STATE_ORDER, "esm3di-35m", "fixture")}
    archive = tmp_path / "input.npz"
    sv.save_predictions(archive, predictions, {"purpose": "test"})
    out = tmp_path / "results"
    tool = Path(__file__).resolve().parents[2] / "tools" / "validate_3di_predictions.py"
    command = [sys.executable, str(tool), "--manifest", str(manifest_path),
               "--predictions", str(archive), "--output-dir", str(out)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads((out / "metrics.json").read_text())
    assert report["summaries"][0]["q20"] == .5
    assert report["structural_omega_calibrated"] is False
    saved, provenance = sv.load_predictions(out / "predictions.npz")
    assert provenance == {"purpose": "test"}
    np.testing.assert_array_equal(saved["one"].logits, predictions["one"].logits)
    repeat = subprocess.run(command, capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "must not already exist" in repeat.stderr


def test_validation_tool_rejects_mismatched_inputs_before_output(tmp_path):
    data = {"schema_version": 1, "records": [
        {"id": "one", "amino_acids": "MK", "structure_3di": "A", "family": "fixture",
         "split": "test", "group": "fixture", "truth_source": "simulated", "source": "unit fixture"},
    ]}
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps(data))
    out = tmp_path / "results"
    tool = Path(__file__).resolve().parents[2] / "tools" / "validate_3di_predictions.py"
    run = subprocess.run([sys.executable, str(tool), "--manifest", str(manifest),
                          "--backend", "esm3di-35m", "--output-dir", str(out)], capture_output=True, text=True)
    assert run.returncode != 0
    assert "residue-for-residue" in run.stderr
    assert not out.exists()
