#!/usr/bin/env python3
"""Known-ancestor toy: equal residue accuracy can hide shared false changes.

Both true descendants and their known ancestor are A at every site. Only an
A-to-C observation error is added. This is not an ASR or omegaC FPR experiment.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csubst.structural_validation import simulate_observation_errors


def main():
    confusion = np.eye(20)
    confusion[0, :2] = [.8, .2]
    truth = np.zeros((2, 100000), dtype=int)
    results = []
    for shared in (0.0, 1.0):
        prediction = simulate_observation_errors(truth, confusion, seed=9, shared_fraction=shared)
        results.append(dict(shared_fraction=shared, residue_accuracy=float((prediction == truth).mean()),
                            joint_false_A_to_C=float(np.all(prediction == 1, axis=0).mean()),
                            theoretical_joint_false_A_to_C=(1 - shared) * .2 ** 2 + shared * .2))
    print(json.dumps(dict(scope="known_endpoint_observation_error_only", seed=9,
                          sites=truth.shape[1], structural_omega_calibrated=False, results=results), indent=2))


if __name__ == "__main__":
    main()
