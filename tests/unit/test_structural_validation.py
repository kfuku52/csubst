import contextlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from csubst import structural_prediction as sp
from csubst import structural_validation as sv


def record(aa="MK", logits=None):
    if logits is None:
        logits = np.zeros((len(aa), 20))
    return sv.PredictionRecord.from_logits(aa, logits, sv.STATE_ORDER, "esm3di-35m", "fixture-v1")


def row(identifier="one", aa="MK", truth="AC", family="family1", **changes):
    result = dict(id=identifier, amino_acids=aa, structure_3di=truth,
                  family=family, split="test", group="soluble", source="synthetic fixture",
                  truth_source="simulated")
    result.update(changes)
    return result


def manifest(*rows):
    return dict(schema_version=1, records=list(rows))


def test_logits_canonicalization_preserves_model_ties_and_not_one_hot():
    labels = tuple(reversed(sv.STATE_ORDER))
    values = np.zeros((2, 20))
    values[1, 2] = 2
    result = sv.PredictionRecord.from_logits("MK", values, labels, "esm3di-35m", "fixture")
    assert result.prediction == labels[0] + labels[2]
    assert result.probability_kind == "uncalibrated_softmax"
    np.testing.assert_allclose(result.probabilities[0], .05)
    expected = np.ones(20)
    expected[sv.STATE_ORDER.index(labels[2])] = np.exp(2)
    np.testing.assert_allclose(result.probabilities[1], expected / expected.sum())
    values[:] = 100
    assert result.logits[0, 0] == 0
    with pytest.raises(ValueError):
        result.logits[0, 0] = 10


def test_hard_only_has_no_probabilities():
    result = sv.PredictionRecord("MK", "AC", "prostt5", "fixture")
    assert result.probabilities is None
    assert result.log_probabilities is None
    assert result.probability_kind == "unavailable"


@pytest.mark.parametrize("values", [np.zeros((2, 19)), np.zeros((1, 20)),
                                  np.full((2, 20), np.nan), np.full((2, 20), np.inf)])
def test_invalid_logits_rejected(values):
    with pytest.raises(ValueError, match="finite residue-by-20"):
        record(logits=values)


def test_duplicate_labels_and_inconsistent_hard_state_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        sv.PredictionRecord.from_logits("M", np.zeros((1, 20)), ("A",) * 20, "esm3di-35m", "v1")
    values = np.zeros((1, 20))
    values[0, 0] = 2
    with pytest.raises(ValueError, match="disagrees"):
        sv.PredictionRecord("M", "C", "esm3di-35m", "v1", values)


def test_pickle_free_artifact_roundtrip_and_identity(tmp_path):
    predictions = {"one": record(), "empty": record("")}
    path = tmp_path / "predictions.npz"
    sv.save_predictions(path, predictions, {"source": "fixture"})
    restored, provenance = sv.load_predictions(path, expected_model_key="fixture-v1")
    assert provenance == {"source": "fixture"}
    assert list(restored) == ["one", "empty"]
    np.testing.assert_array_equal(restored["one"].logits, predictions["one"].logits)
    assert restored["empty"].probabilities.shape == (0, 20)
    with pytest.raises(ValueError, match="identity mismatch"):
        sv.load_predictions(path, expected_model_key="changed-model")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        sv.save_predictions(path, predictions, {})
    assert before == path.read_bytes()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", ["schema", "order", "preprocessing", "kind", "duplicate"])
def test_artifact_metadata_corruption_rejected(tmp_path, change):
    path = tmp_path / "predictions.npz"
    sv.save_predictions(path, {"one": record()}, {})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    data = json.loads(arrays["metadata"].item())
    if change == "schema":
        data["schema_version"] = 99
    elif change == "order":
        data["state_order"].reverse()
    elif change == "preprocessing":
        data["records"][0]["preprocessing"] = "other"
    elif change == "kind":
        data["records"][0]["probability_kind"] = "calibrated"
    else:
        data["records"].append(data["records"][0])
    arrays["metadata"] = np.array(json.dumps(data))
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        sv.load_predictions(path)


def test_uniform_predictions_have_analytical_metrics_and_mask_coverage():
    report = sv.evaluate_predictions(manifest(row(aa="MKK", truth="AC-")), {"one": record("MKK")})
    scores = report["summaries"][0]
    assert scores["q20"] == .5
    assert scores["coverage"] == 2 / 3
    assert scores["multiclass_brier"] == pytest.approx(.95)
    assert scores["log_loss"] == pytest.approx(np.log(20))
    assert scores["top_label_ece"] == pytest.approx(.45)
    assert scores["confusion_true_by_predicted"][0][0] == 1
    assert scores["confusion_true_by_predicted"][1][0] == 1
    assert report["structural_omega_calibrated"] is False


def test_log_loss_uses_logits_without_probability_floor():
    values = np.zeros((1, 20))
    values[0, 0] = 1000
    report = sv.evaluate_predictions(manifest(row(aa="M", truth="C")), {"one": record("M", values)})
    assert report["summaries"][0]["log_loss"] == pytest.approx(1000)


def test_family_macro_and_truth_source_stratification():
    data = manifest(row(aa="MMM", truth="AAA"), row("two", "K", "C", "family2"),
                    row("three", "L", "A", "family3", truth_source="experimental"))
    predictions = {entry["id"]: record(entry["amino_acids"]) for entry in data["records"]}
    report = sv.evaluate_predictions(data, predictions)
    scores = report["summaries"][0]
    assert scores["q20"] == .75
    assert scores["macro_family_q20"] == .5
    assert scores["scored_residues"] == 4
    assert {item["truth_source"] for item in report["summaries"]} == {"simulated", "experimental"}


def test_missing_truth_and_hard_only_are_not_scored_as_certainty():
    predictions = {"one": sv.PredictionRecord("MK", "AC", "prostt5", "v1")}
    scores = sv.evaluate_predictions(manifest(row(truth="??")), predictions)["summaries"][0]
    assert scores["q20"] is None
    assert scores["coverage"] == 0
    assert scores["multiclass_brier"] is None
    scores = sv.evaluate_predictions(manifest(row()), predictions)["summaries"][0]
    assert scores["q20"] == 1
    assert scores["probabilistic_residues"] == 0
    assert scores["log_loss"] is None


@pytest.mark.parametrize("second", [row("two", "KK", "CC", split="calibration"),
                                   row("two", "MK", "CC", "different_family", split="calibration")])
def test_family_and_identical_sequence_split_leakage_rejected(second):
    with pytest.raises(ValueError, match="leakage"):
        sv.validate_manifest(manifest(row(), second))


@pytest.mark.parametrize("changes", [{"amino_acids": "M-K"}, {"structure_3di": "A"},
                                    {"truth_source": "ground_truth"}, {"source": ""}, {"split": "train"}])
def test_invalid_reference_manifest_rejected(changes):
    with pytest.raises(ValueError):
        sv.validate_manifest(manifest(row(**changes)))


def test_evaluation_requires_exact_ids_sequences_and_model_identity():
    with pytest.raises(ValueError, match="IDs"):
        sv.evaluate_predictions(manifest(row()), {"other": record()})
    with pytest.raises(ValueError, match="sequence"):
        sv.evaluate_predictions(manifest(row()), {"one": record("MM")})
    with pytest.raises(ValueError, match="one backend"):
        sv.evaluate_predictions(manifest(row(), row("two")),
                                {"one": record(), "two": sv.PredictionRecord("MK", "AA", "prostt5", "other")})


class LogitPredictor:
    device = "cpu"
    torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    labels = tuple(reversed(sv.STATE_ORDER))

    def __init__(self, max_batch=10):
        self.calls = []
        self.max_batch = max_batch

    def predict_logits_batch(self, sequences):
        self.calls.append(list(sequences))
        if len(sequences) > self.max_batch:
            raise RuntimeError("out of memory")
        return [np.zeros((len(seq), 20)) for seq in sequences]


def test_record_api_bypasses_hard_cache_deduplicates_and_retries_oom(tmp_path, monkeypatch):
    predictor = LogitPredictor(max_batch=1)
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    monkeypatch.setattr(sp.sa, "_load_prostt5_sequence_cache", lambda *a: pytest.fail("hard cache cannot supply logits"))
    result = sp.predict_3di_records({"one": "M-K", 2: "mk", "long": "MKK", "empty": "--"}, {})
    assert list(result) == ["one", 2, "long", "empty"]
    assert [len(chunk) for chunk in predictor.calls] == [2, 1, 1]
    assert result["one"].prediction == "YY"
    assert result[2] is result["one"]
    np.testing.assert_allclose(result["one"].probabilities, .05)
    assert result["empty"].probabilities.shape == (0, 20)


def test_generator_records_explicitly_lack_probabilities(monkeypatch):
    def predict(sequences, g):
        assert g["prostt5_cache"] is False
        return {"one": "AC"}
    monkeypatch.setattr(sp.sa, "predict_3di_with_prostt5", predict)
    result = sp.predict_3di_records({"one": "M-K"}, {"sa_backend": "prostt5"})
    assert result["one"].probability_kind == "unavailable"
    assert result["one"].amino_acids == "MK"


def test_empty_encoder_record_does_not_load_model(monkeypatch):
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: pytest.fail("empty inputs need no model"))
    assert sp.predict_3di_records({}, {}) == {}
    assert sp.predict_3di_records({"empty": ""}, {})["empty"].prediction == ""


def test_encoder_logit_api_matches_original_hard_predictions():
    torch = pytest.importorskip("torch")
    labels = tuple(reversed(sv.STATE_ORDER))

    def tokenizer(prompts, **kwargs):
        length = len(prompts[0]) + 2
        return {"input_ids": torch.arange(length)[None, :], "attention_mask": torch.ones((1, length))}

    def model(input_ids, attention_mask):
        return SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids % 20, 20).float())

    predictor = sp.EncoderPredictor("esm3di-35m", torch, tokenizer, model, "cpu", labels)
    with torch.inference_mode():
        logits = predictor.predict_logits_batch(["MK"])[0]
        hard = predictor.predict_batch(["MK"])[0]
    result = sv.PredictionRecord.from_logits("MK", logits, labels, "esm3di-35m", "fixture")
    assert logits.shape == (2, 20)
    assert result.prediction == hard == labels[1] + labels[2]
    assert predictor.predict_logits_batch([]) == []


def test_shared_errors_preserve_accuracy_but_change_joint_false_events():
    confusion = np.eye(20)
    confusion[0, :2] = [.8, .2]
    truth = np.zeros((2, 50000), dtype=int)
    independent = sv.simulate_observation_errors(truth, confusion, seed=19)
    shared = sv.simulate_observation_errors(truth, confusion, seed=19, shared_fraction=1)
    assert (independent == 0).mean() == pytest.approx(.8, abs=.006)
    assert (shared == 0).mean() == pytest.approx(.8, abs=.006)
    assert np.all(independent == 1, axis=0).mean() == pytest.approx(.04, abs=.003)
    assert np.all(shared == 1, axis=0).mean() == pytest.approx(.2, abs=.006)
    np.testing.assert_array_equal(shared[0], shared[1])


def test_zero_error_identity_and_state_dependent_confusion_direction():
    truth = np.array([[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(sv.simulate_observation_errors(truth, np.eye(20), 0), truth)
    matrix = np.roll(np.eye(20), 1, axis=1)
    np.testing.assert_array_equal(sv.simulate_observation_errors(truth, matrix, 0), (truth + 1) % 20)


def test_error_blocks_and_reproducibility():
    matrix = np.full((20, 20), .05)
    truth = np.zeros((2, 12), dtype=int)
    result = sv.simulate_observation_errors(truth, matrix, 8, block_size=3)
    np.testing.assert_array_equal(result, sv.simulate_observation_errors(truth, matrix, 8, block_size=3))
    np.testing.assert_array_equal(result[:, 0::3], result[:, 1::3])
    np.testing.assert_array_equal(result[:, 0::3], result[:, 2::3])


@pytest.mark.parametrize("kwargs", [{"shared_fraction": -1}, {"shared_fraction": float("nan")},
                                   {"block_size": 0}, {"block_size": True}])
def test_invalid_error_model_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        sv.simulate_observation_errors(np.zeros((2, 3), dtype=int), np.eye(20), 0, **kwargs)
