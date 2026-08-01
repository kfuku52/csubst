import hashlib
import types

import pandas as pd
import pytest

from csubst import model_resources
from csubst import variant_effect
from csubst import vesm


def test_window_for_long_context_is_centered_and_bounded():
    sequence = "A" * 1600
    window, position, start, end = vesm._window_for_event(
        sequence=sequence,
        aa_position_1based=801,
    )
    assert len(window) == vesm.MAX_SEQUENCE_RESIDUES
    assert end - start == vesm.MAX_SEQUENCE_RESIDUES
    assert window[position] == "A"
    assert position in [vesm.MAX_SEQUENCE_RESIDUES // 2, (vesm.MAX_SEQUENCE_RESIDUES // 2) - 1]


@pytest.mark.parametrize(
    ("aa_position", "expected_start", "expected_end"),
    [(1, 0, vesm.MAX_SEQUENCE_RESIDUES), (1600, 1600 - vesm.MAX_SEQUENCE_RESIDUES, 1600)],
)
def test_window_for_long_context_clamps_at_sequence_edges(aa_position, expected_start, expected_end):
    _window, _position, start, end = vesm._window_for_event("A" * 1600, aa_position)
    assert (start, end) == (expected_start, expected_end)


def test_merge_and_load_score_cache_preserves_existing_rows(tmp_path):
    cache_file = tmp_path / "scores.tsv"
    key_a = vesm._cache_key(hashlib.sha256(b"AAAA").hexdigest(), 1, "A", "V")
    key_b = vesm._cache_key(hashlib.sha256(b"CCCC").hexdigest(), 2, "C", "G")
    vesm.merge_score_cache(str(cache_file), {key_a: -1.25})
    vesm.merge_score_cache(str(cache_file), {key_b: 0.75})
    loaded = vesm.load_score_cache(str(cache_file))
    assert loaded == {key_a: pytest.approx(-1.25), key_b: pytest.approx(0.75)}


def test_scorer_records_llr_and_window_metadata_without_sign_reversal(monkeypatch):
    events = variant_effect.empty_event_table()
    row = {column: None for column in events.columns}
    row.update(
        {
            "event_id": "b1.a2.A>V",
            "from_aa": "A",
            "to_aa": "V",
            "aa_position_ancestral": 2,
            "_context_sequence": "AAAA",
        }
    )
    events = pd.DataFrame([row], columns=events.columns)
    scorer = vesm.Vesm35mScorer(
        g={"vep_cache": False},
        components=(None, None, None, "cpu"),
    )
    monkeypatch.setattr(
        scorer,
        "_infer_windows",
        lambda window_to_records: {
            record["cache_key"]: -2.5
            for records in window_to_records.values()
            for record in records
        },
    )
    out = scorer.score(events)
    assert out.at[0, "vesm_llr"] == pytest.approx(-2.5)
    assert "vesm_deleteriousness" not in out.columns
    assert out.at[0, "score_status"] == "scored"
    assert out.at[0, "window_start_aa"] == 1
    assert out.at[0, "window_end_aa"] == 4
    assert out.at[0, "vesm_model_resource_id"] == model_resources.VESM_35M_RESOURCE_ID


def test_score_events_does_not_construct_model_for_empty_table(monkeypatch):
    monkeypatch.setattr(
        vesm,
        "Vesm35mScorer",
        lambda *_args, **_kwargs: pytest.fail("model should not be constructed"),
    )
    g = {}
    out = vesm.score_events(events=variant_effect.empty_event_table(), g=g)
    assert out.empty
    assert "_vep_scorer" not in g


def _fake_torch(cuda_available=False, mps_available=False):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available)
        ),
    )


def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu():
    assert vesm.resolve_device(_fake_torch(cuda_available=True, mps_available=True), "auto") == "cuda"
    assert vesm.resolve_device(_fake_torch(mps_available=True), "auto") == "mps"
    assert vesm.resolve_device(_fake_torch(), "auto") == "cpu"


def test_resolve_device_rejects_unavailable_accelerators():
    with pytest.raises(ValueError, match="CUDA is not available"):
        vesm.resolve_device(_fake_torch(), "cuda")
    with pytest.raises(ValueError, match="MPS is not available"):
        vesm.resolve_device(_fake_torch(), "mps")
