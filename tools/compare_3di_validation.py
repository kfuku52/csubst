#!/usr/bin/env python3
"""Descriptive comparison of a small, common-reference prediction panel."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csubst import structural_validation as sv


def compare(manifest, predictions):
    rows = sv.validate_manifest(manifest)
    if len({(r["split"], r["truth_source"]) for r in rows}) != 1:
        raise ValueError("Compare one split and reference source at a time.")
    metrics = {name: sv.evaluate_predictions(manifest, records) for name, records in predictions.items()}
    errors = {}
    for name, records in predictions.items():
        errors[name] = {}
        for row in rows:
            valid = np.array([c in sv.STATE_ORDER for c in row["structure_3di"]])
            wrong = np.array(list(records[row["id"]].prediction)) != np.array(list(row["structure_3di"]))
            errors[name][row["id"]] = (valid, wrong)
    families = sorted({row["family"] for row in rows})
    summaries = []
    for name in predictions:
        family_metrics = []
        for family in families:
            subset = [row for row in rows if row["family"] == family]
            counts = [errors[name][row["id"]] for row in subset]
            n = sum(int(valid.sum()) for valid, wrong in counts)
            mistakes = sum(int((valid & wrong).sum()) for valid, wrong in counts)
            family_metrics.append(dict(family=family, scored_residues=n, q20=1 - mistakes / n if n else None))
        eligible = [entry["q20"] for entry in family_metrics if entry["q20"] is not None]
        # Resample whole families, not correlated individual residues/structures.
        rng = np.random.default_rng(9)
        interval = (np.quantile(rng.choice(eligible, size=(10000, len(eligible))).mean(axis=1), [.025, .975]).tolist()
                    if eligible else None)
        previous_errors = joint_adjacent_errors = longest_run = 0
        for valid, wrong in errors[name].values():
            pairs = valid[:-1] & valid[1:]
            previous_errors += int((pairs & wrong[:-1]).sum())
            joint_adjacent_errors += int((pairs & wrong[:-1] & wrong[1:]).sum())
            run = 0
            for present, error in zip(valid, wrong):
                run = run + 1 if present and error else 0
                longest_run = max(longest_run, run)
        aggregate = next(item for item in metrics[name]["summaries"] if item["group"] is None)
        summaries.append(dict(backend=name, q20=aggregate["q20"], macro_family_q20=aggregate["macro_family_q20"],
                              log_loss=aggregate["log_loss"], multiclass_brier=aggregate["multiclass_brier"],
                              descriptive_family_bootstrap_percentiles=interval, family_metrics=family_metrics,
                              error_given_previous_error=joint_adjacent_errors / previous_errors if previous_errors else None,
                              eligible_previous_errors=previous_errors, longest_error_run=longest_run))
    common_errors = []
    for first, second in itertools.combinations(predictions, 2):
        n = joint = first_errors = second_errors = 0
        for row in rows:
            valid, a = errors[first][row["id"]]
            _, b = errors[second][row["id"]]
            n += int(valid.sum())
            first_errors += int((valid & a).sum())
            second_errors += int((valid & b).sum())
            joint += int((valid & a & b).sum())
        common_errors.append(dict(first=first, second=second, residues=n,
                                  both_wrong=joint / n if n else None,
                                  marginal_product=(first_errors / n) * (second_errors / n) if n else None))
    structure_pairs = []
    for first, second in itertools.combinations(rows, 2):
        if first["amino_acids"] != second["amino_acids"]:
            continue
        a, b = np.array(list(first["structure_3di"])), np.array(list(second["structure_3di"]))
        valid = np.isin(a, sv.STATE_ORDER) & np.isin(b, sv.STATE_ORDER)
        n = int(valid.sum())
        structure_pairs.append(dict(first=first["id"], second=second["id"], identical_sequence=True,
                                    comparable_residues=n, reference_disagreement=float((a[valid] != b[valid]).mean()) if n else None))
    return dict(scope="descriptive_pilot_only", families=len(families), sequences=len(rows),
                summaries=summaries, predictor_common_errors=common_errors,
                same_sequence_structure_pairs=structure_pairs, structural_omega_calibrated=False,
                bootstrap_seed=9, bootstrap_repeats=10000,
                limitations=["Few families: bootstrap percentiles are descriptive, not a population guarantee.",
                             "Training overlap is unknown; no calibrator was fitted.",
                             "Multiple structures of one sequence are correlated references.",
                             "Adjacent/common-error summaries are descriptive, not independence tests."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, help="NAME=artifact.npz (repeat)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    predictions = {}
    for value in args.predictions:
        name, path = value.split("=", 1)
        if not name or name in predictions:
            parser.error("Prediction names must be nonempty and unique")
        predictions[name], _ = sv.load_predictions(path)
    result = compare(manifest, predictions)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
