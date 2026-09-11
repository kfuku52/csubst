import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import logsumexp
from csubst import endpoint, structural_observation

rng = np.random.default_rng(19011)
checks = []
for k in [3, 4, 8]:
    for scale in [1e-7, 0.1, 2.0, 30.0]:
        for mix in [False, True]:
            pi = rng.dirichlet(np.ones(k))
            x = rng.uniform(0.1, 2, (k, k))
            x = x + x.T
            q = x * pi[None, :]
            np.fill_diagonal(q, 0)
            np.fill_diagonal(q, -q.sum(1))
            q /= -pi @ q.diagonal()
            parents = np.array([-1, 0, 0, 1, 1, 2, 2])
            lengths = np.r_[0, rng.uniform(0.3, 2, 6) * scale]
            rates = np.array([0.2, 1.0, 3.0]) if mix else np.array([1.0])
            weights = np.array([0.2, 0.5, 0.3]) if mix else np.array([1.0])
            tips = {i: np.eye(k)[rng.integers(k, size=7)] for i in [3, 4, 5, 6]}
            tips[6][0] = 1
            tips[3][1] = rng.uniform(0.1, 1, k)
            for i in tips:
                tips[i][2] = 1
            refs = [
                structural_observation.fixed_gtr_posteriors(
                    parents, lengths * r, q, pi, tips
                )
                for r in rates
            ]
            log = (
                np.array([a["site_log_likelihood"] for a in refs])
                + np.log(weights)[:, None]
            )
            cw = np.exp(log - logsumexp(log, axis=0))
            model = endpoint.EndpointModel(parents, lengths, q, pi, rates, weights)
            maximum = 0.0
            for rec in model.iter_blocks(tips, block_size=3):
                sl = slice(rec.start, rec.stop)
                node = sum(
                    cw[c, sl, None] * ref["node_posterior"][rec.child, sl]
                    for c, ref in enumerate(refs)
                )
                np.testing.assert_allclose(rec.node, node, atol=1e-11, rtol=1e-9)
                maximum = max(maximum, float(np.max(abs(rec.node - node))))
                if rec.joint is not None:
                    joint = sum(
                        cw[c, sl, None, None] * ref["edge_joint"][rec.child][sl]
                        for c, ref in enumerate(refs)
                    )
                    np.testing.assert_allclose(rec.joint, joint, atol=1e-11, rtol=1e-9)
                    maximum = max(maximum, float(np.max(abs(rec.joint - joint))))
            checks.append(
                dict(states=k, scale=scale, mixture=mix, max_abs_error=maximum)
            )
print(json.dumps(checks, indent=2))
