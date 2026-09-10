#!/usr/bin/env python3
"""Known-GTR endpoint calibration under independent and misspecified errors.

This excludes fitting, codon S and omegaC tests. All outputs are synthetic.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csubst.structural_observation import fixed_gtr_posteriors, observation_likelihoods
from csubst.structural_validation import simulate_observation_errors


def run(sites=10000, seed=9):
    if sites < 1:
        raise ValueError("sites must be positive")
    parents = np.array([-1, 0, 1, 1, 0])
    lengths = np.array([0, .1, .15, .15, .2])
    q = (np.ones((4, 4)) - 4 * np.eye(4)) / 3
    pi = np.ones(4) / 4
    rng = np.random.default_rng(seed)
    truth = np.empty((5, sites), dtype=int)
    truth[0] = rng.choice(4, size=sites, p=pi)
    for child in range(1, 5):
        transition = expm(q * lengths[child])
        uniforms = rng.random(sites)
        truth[child] = (uniforms[:, None] > transition[truth[parents[child]]].cumsum(axis=1)).sum(axis=1)
    leaves = (2, 3, 4)
    true_change = truth[0] != truth[1]
    scenarios = [("zero_error", 0., 0., 1), ("independent", .15, 0., 1),
                 ("shared_across_tips", .15, 1., 1), ("shared_blocks", .15, .5, 16)]
    results = []
    for name, error, shared, block in scenarios:
        confusion = np.full((4, 4), error / 3)
        np.fill_diagonal(confusion, 1 - error)
        noise = np.eye(20)
        noise[:4, :4] = confusion
        observed = simulate_observation_errors(truth[list(leaves)], noise, seed + 1, shared, block)
        for treatment, matrix in (("ignore_error", np.eye(4)), ("independent_error_likelihood", confusion)):
            tips = {node: observation_likelihoods(observed[i], matrix) for i, node in enumerate(leaves)}
            inferred = fixed_gtr_posteriors(parents, lengths, q, pi, tips)
            posterior = inferred["node_posterior"][0]
            root_error = posterior.copy()
            root_error[np.arange(sites), truth[0]] -= 1
            joint = inferred["edge_joint"][1]
            probability = 1 - np.trace(joint, axis1=1, axis2=2)
            reliability = []
            bins = np.minimum((probability * 10).astype(int), 9)
            for index in range(10):
                mask = bins == index
                if mask.any():
                    reliability.append(dict(bin=index, count=int(mask.sum()),
                                            mean_probability=float(probability[mask].mean()),
                                            change_frequency=float(true_change[mask].mean())))
            results.append(dict(scenario=name, treatment=treatment, marginal_error=error,
                                shared_fraction=shared, block_size=block,
                                empirical_tip_accuracy=float((observed == truth[list(leaves)]).mean()),
                                root_brier=float(np.square(root_error).sum(axis=1).mean()),
                                root_log_loss=float(-np.log(posterior[np.arange(sites), truth[0]]).mean()),
                                edge_joint_log_loss=float(-np.log(joint[np.arange(sites), truth[0], truth[1]]).mean()),
                                mean_edge_change_probability=float(probability.mean()),
                                true_edge_change_frequency=float(true_change.mean()),
                                edge_change_brier=float(np.square(probability - true_change).mean()),
                                edge_change_reliability=reliability))
    return dict(scope="synthetic_fixed_gtr_endpoint_inference_only", sites=sites, seed=seed,
                parents=parents.tolist(), branch_lengths=lengths.tolist(), q=q.tolist(), pi=pi.tolist(),
                fitted_parameters=False, structural_omega_calibrated=False, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.sites, args.seed)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
