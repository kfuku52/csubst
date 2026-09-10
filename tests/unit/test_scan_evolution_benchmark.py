"""Independent checks of the simulation harness used for training/holdout."""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from scan_evolution_benchmark import foreground_q, jump_history, null_posterior, simulate
from csubst.scan_analytic import EndpointEnrichment


def test_jump_sampler_matches_transition_distribution():
    q = foreground_q(16)
    rng = np.random.default_rng(61425)
    counts = np.bincount([jump_history(0, q, .3, rng)[0] for _ in range(12000)], minlength=4)
    np.testing.assert_allclose(counts / counts.sum(), expm(q * .3)[0], atol=.015)


def test_reference_posterior_integrates_latent_categories():
    data = simulate('uncertain', 2, 5, 0, 'null', [411, 0])
    posterior, rates = null_posterior(data)
    model = data['model']
    engine = EndpointEnrichment(model, np.array([0, 0, 1, 1]))
    # Posterior at a hidden internal node via independent clamped pruning.
    node = next(n for n in model.order if n != model.root and n not in model.leaves)
    for site in range(5):
        numerators = np.zeros(4)
        expected_rate = 0.
        for category, weight in enumerate(model.weights):
            for fixed in range(4):
                partial = {}
                for current in reversed(model.order):
                    value = data['observations'][current][site].copy() if current in model.leaves else np.ones(4)
                    if current == node:
                        value *= np.eye(4)[fixed]
                    for child in model.children[current]:
                        value *= model.transition(child, category) @ partial[child]
                    partial[current] = value
                likelihood = weight * (model.pi @ partial[model.root])
                numerators[fixed] += likelihood
                expected_rate += likelihood * model.rates[category]
        likelihood = numerators.sum()
        np.testing.assert_allclose(posterior[node, site], numerators / likelihood, rtol=1e-12)
        np.testing.assert_allclose(rates[site], expected_rate / likelihood, rtol=1e-12)
        tips = {leaf: obs[site] for leaf, obs in data['observations'].items()}
        np.testing.assert_allclose(np.exp(engine.log_likelihoods(tips)[0]), likelihood, rtol=1e-12)
