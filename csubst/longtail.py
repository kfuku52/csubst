"""Frozen, one-sided quantile maps for long-tail sensitivity analyses.

Fitting and application have separate row populations. No random sampling or
pipeline state is owned here. The empirical S midranks and N quantiles are both
frozen; re-ranking a displayed subset would defeat that separation.
"""

from dataclasses import dataclass
import hashlib

import numpy as np


def linear_quantiles(sorted_values, quantiles):
    """NumPy's default linear quantiles, with its floating-point operation order."""
    values = np.asarray(sorted_values, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    if values.size == 1:
        return np.full(q.shape, values[0], dtype=np.float64)
    indexes = q * float(values.size - 1)
    previous = values[np.floor(indexes).astype(np.intp)]
    following = values[np.ceil(indexes).astype(np.intp)]
    weight = indexes - np.floor(indexes)
    interval = following - previous
    return np.where(weight >= 0.5, following - interval * (1.0 - weight), previous + interval * weight)


@dataclass(frozen=True)
class QuantileMap:
    source_values: np.ndarray
    source_quantiles: np.ndarray
    target_sorted: np.ndarray
    n_pairs: int
    status: str
    reference_id: str

    @classmethod
    def fit(cls, dnc, dsc, min_pairs=2):
        n = np.asarray(dnc, dtype=np.float64).reshape(-1)
        s = np.asarray(dsc, dtype=np.float64).reshape(-1)
        if n.shape != s.shape:
            raise ValueError('Calibration reference N and S shapes must match.')
        if min_pairs < 2:
            raise ValueError('A frozen calibration reference requires at least two finite pairs.')
        if np.any(n < 0) or np.any(s < 0):
            raise ValueError('Calibration rates must be nonnegative.')
        valid = np.isfinite(n) & np.isfinite(s)
        n, s = n[valid], s[valid]
        source, counts = np.unique(s, return_counts=True)
        quantiles = (np.cumsum(counts) - counts / 2.0) / max(1, s.size)
        target = np.sort(n)
        status = 'applied' if n.size >= min_pairs else 'insufficient_reference'
        digest = hashlib.sha256(b'longtail-midrank-v1')
        digest.update(str((int(n.size), int(min_pairs))).encode())
        for array in (source, quantiles, target):
            digest.update(array.astype('<f8').tobytes())
            array.setflags(write=False)
        return cls(source, quantiles, target, int(n.size), status, digest.hexdigest())

    def apply(self, dsc):
        values = np.asarray(dsc, dtype=np.float64)
        if np.any(values < 0):
            raise ValueError('Calibration rates must be nonnegative.')
        out = values.copy()
        valid = np.isfinite(values)
        if self.status != 'applied' or not valid.any():
            return out
        # Linear interpolation between S midranks; outside the support use the
        # N minimum/maximum. Exact reference values retain their fitted ranks.
        q = np.interp(values[valid], self.source_values, self.source_quantiles, left=0.0, right=1.0)
        out[valid] = np.maximum(values[valid], linear_quantiles(self.target_sorted, q))
        return out

    def diagnostics(self):
        return dict(reference=self.reference_id, n=self.n_pairs,
                    unique_S=int(self.source_values.size), status=self.status)
