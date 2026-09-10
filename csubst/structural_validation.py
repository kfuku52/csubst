"""Uncalibrated 3Di prediction artifacts and held-out residue-level evaluation.

These measurements do not calibrate ancestral events or structural omegaC.
No classifier probabilities are treated as phylogenetic observation likelihoods.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray


STATE_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
PREPROCESSING = "strip-uppercase-remove-gaps-nonstandard-to-X-v1"
SCHEMA_VERSION = 1
TRUTH_SOURCES = ("experimental", "simulated", "predicted", "reconstructed")


def _nonempty_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string.".format(name))


@dataclass(frozen=True)
class PredictionRecord:
    """One gapless sequence. Logit columns always use STATE_ORDER.

    ``logits=None`` means that probabilities were not obtained, not certainty.
    Hard predictions retain the original model's tie-breaking order.
    """

    amino_acids: str
    prediction: str
    backend: str
    model_key: str
    logits: NDArray[np.float64] | None = None
    preprocessing: str = PREPROCESSING

    def __post_init__(self):
        for name in ("backend", "model_key", "preprocessing"):
            _nonempty_text(getattr(self, name), name)
        if not isinstance(self.amino_acids, str) or not set(self.amino_acids) <= set(STATE_ORDER + ("X",)):
            raise ValueError("amino_acids must be sanitized, uppercase and gapless.")
        if (not isinstance(self.prediction, str) or len(self.prediction) != len(self.amino_acids)
                or not set(self.prediction) <= set(STATE_ORDER)):
            raise ValueError("Prediction must have one valid 3Di state per amino acid.")
        if self.logits is not None:
            values = np.array(self.logits, dtype=np.float64, copy=True)
            if values.shape != (len(self.amino_acids), 20) or not np.isfinite(values).all():
                raise ValueError("Logits must be a finite residue-by-20 array.")
            with np.errstate(over="ignore"):
                if not np.isfinite(values - values.max(axis=1, keepdims=True)).all():
                    raise ValueError("Logit range exceeds finite float64 arithmetic.")
            ids = np.array([STATE_ORDER.index(c) for c in self.prediction], dtype=int)
            if len(ids) and not np.array_equal(values[np.arange(len(ids)), ids], values.max(axis=1)):
                raise ValueError("Prediction disagrees with logit maxima.")
            values.setflags(write=False)
            object.__setattr__(self, "logits", values)

    @classmethod
    def from_logits(cls, amino_acids, logits, labels, backend, model_key):
        labels = tuple(labels)
        values = np.asarray(logits, dtype=np.float64)
        if len(labels) != 20 or set(labels) != set(STATE_ORDER):
            raise ValueError("Logit labels must contain each 3Di state exactly once.")
        if values.shape != (len(amino_acids), 20) or not np.isfinite(values).all():
            raise ValueError("Logits must be a finite residue-by-20 array.")
        prediction = "".join(labels[i] for i in values.argmax(axis=1))
        order = [labels.index(label) for label in STATE_ORDER]
        return cls(amino_acids, prediction, backend, model_key, values[:, order])

    @property
    def probability_kind(self):
        return "unavailable" if self.logits is None else "uncalibrated_softmax"

    @property
    def log_probabilities(self):
        if self.logits is None:
            return None
        shifted = self.logits - self.logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    @property
    def probabilities(self):
        values = self.log_probabilities
        return None if values is None else np.exp(values)


def save_predictions(path, records, provenance):
    """Atomically publish a new, pickle-free NPZ; never overwrite a result.

    Artifacts are explicit outputs, not an automatic prediction cache. Raw
    logits and identities are retained; softmax can be reproduced losslessly.
    """
    if not records:
        raise ValueError("No prediction records to save.")
    entries, arrays = [], {}
    for index, (identifier, record) in enumerate(records.items()):
        _nonempty_text(identifier, "record id")
        entry = {"id": identifier, "amino_acids": record.amino_acids,
                 "prediction": record.prediction, "backend": record.backend,
                 "model_key": record.model_key, "preprocessing": record.preprocessing,
                 "probability_kind": record.probability_kind}
        if record.logits is not None:
            entry["logits_key"] = "logits_{}".format(index)
            arrays[entry["logits_key"]] = record.logits
        entries.append(entry)
    metadata = dict(schema_version=SCHEMA_VERSION, state_order=STATE_ORDER,
                    records=entries, provenance=provenance)
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False, sort_keys=True))
    path = Path(path).expanduser()
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        stage = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(stage, path)  # atomic publication, EEXIST protects other runs
        finally:
            stage.unlink(missing_ok=True)


def load_predictions(path, expected_model_key=None):
    """Read and validate an artifact; optionally require a particular model."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schema_version") != SCHEMA_VERSION or tuple(metadata.get("state_order", ())) != STATE_ORDER:
            raise ValueError("Unsupported 3Di prediction schema or state order.")
        result = {}
        for entry in metadata["records"]:
            identifier = entry["id"]
            _nonempty_text(identifier, "record id")
            if identifier in result:
                raise ValueError("Duplicate prediction id: {}".format(identifier))
            if entry["preprocessing"] != PREPROCESSING:
                raise ValueError("Unsupported 3Di preprocessing identity.")
            if expected_model_key is not None and entry["model_key"] != expected_model_key:
                raise ValueError("3Di prediction model identity mismatch.")
            record = PredictionRecord(
                entry["amino_acids"], entry["prediction"], entry["backend"], entry["model_key"],
                archive[entry["logits_key"]] if "logits_key" in entry else None,
                entry["preprocessing"],
            )
            if entry["probability_kind"] != record.probability_kind:
                raise ValueError("Probability kind disagrees with saved logits.")
            result[identifier] = record
    if not result:
        raise ValueError("No prediction records in artifact.")
    return result, metadata["provenance"]


def validate_manifest(manifest):
    """Validate exact residue correspondence and declared family split isolation.

    Family assignments and structure extraction must be checked by the caller;
    this cannot discover unannotated homology or model pretraining overlap.
    """
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported validation manifest schema.")
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest records must be a nonempty list.")
    ids: set[str] = set()
    family_splits: dict[str, str] = {}
    sequence_splits: dict[str, str] = {}
    for row in rows:
        for name in ("id", "amino_acids", "structure_3di", "family", "split", "group", "source"):
            _nonempty_text(row.get(name), name)
        if row["id"] in ids:
            raise ValueError("Duplicate manifest id: {}".format(row["id"]))
        ids.add(row["id"])
        aa, truth = row["amino_acids"], row["structure_3di"]
        if not set(aa) <= set(STATE_ORDER + ("X",)):
            raise ValueError("Manifest amino_acids must already be uppercase and gapless.")
        if len(aa) != len(truth) or not set(truth) <= set(STATE_ORDER + ("-", "?")):
            raise ValueError("structure_3di must correspond residue-for-residue; mask unknown truth with '-' or '?'.")
        if row.get("truth_source") not in TRUTH_SOURCES:
            raise ValueError("truth_source must be one of {}.".format(TRUTH_SOURCES))
        if row["split"] not in ("calibration", "test"):
            raise ValueError("split must be calibration or test.")
        for assignments, key in ((family_splits, row["family"]), (sequence_splits, aa)):
            previous = assignments.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError("Family or identical-sequence leakage across calibration/test splits.")
    return rows


def _score_rows(rows, predictions):
    confusion: NDArray[np.int64] = np.zeros((20, 20), dtype=np.int64)
    family_counts: dict[str, list[int]] = {}
    bins = [{"count": 0, "correct": 0, "confidence_sum": 0.0} for _ in range(10)]
    n_total = n_prob = 0
    brier_sum = log_loss_sum = 0.0
    for row in rows:
        record = predictions[row["id"]]
        truth = np.array([STATE_ORDER.index(c) if c in STATE_ORDER else -1
                          for c in row["structure_3di"]], dtype=int)
        valid = truth >= 0
        expected = truth[valid]
        observed = np.array([STATE_ORDER.index(c) for c in record.prediction], dtype=int)[valid]
        np.add.at(confusion, (expected, observed), 1)
        n_total += len(truth)
        correct = expected == observed
        totals = family_counts.setdefault(row["family"], [0, 0])
        totals[0] += int(correct.sum())
        totals[1] += len(expected)
        if record.logits is None or not len(expected):
            continue
        log_prob = record.log_probabilities[valid]
        prob = np.exp(log_prob)
        n_prob += len(expected)
        indices = np.arange(len(expected))
        error = prob.copy()
        error[indices, expected] -= 1
        brier_sum += float(np.square(error).sum())
        log_loss_sum -= float(log_prob[indices, expected].sum())
        confidence = prob.max(axis=1)
        bin_ids = np.minimum((confidence * 10).astype(int), 9)
        for bin_id, item in enumerate(bins):
            selected = bin_ids == bin_id
            item["count"] += int(selected.sum())
            item["correct"] += int(correct[selected].sum())
            item["confidence_sum"] += float(confidence[selected].sum())
    n_scored = int(confusion.sum())
    supports = confusion.sum(axis=1)
    recalls = np.divide(confusion.diagonal(), supports, out=np.zeros(20, dtype=float), where=supports > 0)
    eligible_families = [a / b for a, b in family_counts.values() if b]
    reliability = []
    for index, item in enumerate(bins):
        count = item["count"]
        reliability.append(dict(lower=index / 10, upper=(index + 1) / 10, count=count,
                                accuracy=item["correct"] / count if count else None,
                                confidence=item["confidence_sum"] / count if count else None))
    return dict(
        sequences=len(rows), families=len(family_counts), residues=n_total, scored_residues=n_scored,
        coverage=n_scored / n_total if n_total else None,
        q20=float(confusion.trace() / n_scored) if n_scored else None,
        macro_family_q20=float(np.mean(eligible_families)) if eligible_families else None,
        balanced_accuracy=float(np.mean(recalls[supports > 0])) if n_scored else None,
        confusion_true_by_predicted=confusion.tolist(),
        probabilistic_residues=n_prob,
        multiclass_brier=brier_sum / n_prob if n_prob else None,
        log_loss=log_loss_sum / n_prob if n_prob else None,
        top_label_ece=sum(abs(item["correct"] - item["confidence_sum"])
                          for item in bins) / n_prob if n_prob else None,
        reliability=reliability,
    )


def evaluate_predictions(manifest, predictions):
    """Stratify residue metrics by split and truth source; never claim event FPR."""
    rows = validate_manifest(manifest)
    if set(predictions) != {row["id"] for row in rows}:
        raise ValueError("Manifest and prediction IDs must match exactly.")
    identities = {(record.backend, record.model_key, record.preprocessing) for record in predictions.values()}
    if len(identities) != 1:
        raise ValueError("Evaluate one backend/model/preprocessing identity at a time.")
    for row in rows:
        if predictions[row["id"]].amino_acids != row["amino_acids"]:
            raise ValueError("Prediction sequence does not match manifest for {}.".format(row["id"]))
    strata: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        for group in (None, row["group"]):
            strata.setdefault((row["split"], row["truth_source"], group), []).append(row)
    summaries = [dict(split=key[0], truth_source=key[1], group=key[2], **_score_rows(selected, predictions))
                 for key, selected in strata.items()]
    backend, model_key, preprocessing = next(iter(identities))
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(schema_version=SCHEMA_VERSION, state_order=STATE_ORDER,
                backend=backend, model_key=model_key, preprocessing=preprocessing,
                manifest_sha256=manifest_hash, summaries=summaries,
                scope="residue_prediction_only", structural_omega_calibrated=False,
                limitations=["Declared family splits do not prove absence of pretraining overlap.",
                             "Softmax scores are uncalibrated, not observation likelihoods or event posteriors.",
                             "Residue metrics do not establish ancestral-event accuracy, FPR or power."])


def simulate_observation_errors(true_states, confusion, seed, shared_fraction=0.0, block_size=1):
    """Sample predicted endpoints from a specified observation model.

    Rows of confusion are true states, columns predicted states. A fraction of
    blocks use the same uniform quantile across all tips, coupling errors while
    preserving each tip's marginal confusion probabilities. Within a block the
    quantile is reused across sites. This is a stress-test copula, not a fitted
    biological model or an omegaC null generator. True states are tip-by-site
    integer indices in STATE_ORDER; missing states must be handled by callers.
    """
    states = np.asarray(true_states)
    matrix = np.asarray(confusion, dtype=float)
    if states.ndim != 2 or states.dtype.kind not in "iu" or np.any((states < 0) | (states >= 20)):
        raise ValueError("True states must be a tip-by-site array of integers in [0, 20).")
    if (matrix.shape != (20, 20) or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Confusion must be a finite, nonnegative, row-normalized 20-by-20 matrix.")
    if not np.isfinite(shared_fraction) or not 0 <= shared_fraction <= 1:
        raise ValueError("shared_fraction must be between zero and one.")
    if isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    rng = np.random.default_rng(seed)
    blocks = (states.shape[1] + block_size - 1) // block_size
    uniform = rng.random((states.shape[0], blocks))
    shared = rng.random(blocks) < shared_fraction
    common = rng.random(blocks)
    uniform[:, shared] = common[shared]
    uniform = np.repeat(uniform, block_size, axis=1)[:, :states.shape[1]]
    cumulative = matrix.cumsum(axis=1)
    cumulative[:, -1] = 1.0  # close only machine-roundoff error, after validation
    result = np.empty(states.shape, dtype=np.int64)
    for state in range(20):
        selected = states == state
        result[selected] = np.searchsorted(cumulative[state], uniform[selected], side="right")
    return result
