"""Resampling schemes for the Stage 0 SMC sampler.

All three take normalized log-weights and a particle array `theta` of shape
(N, d) and return (theta_resampled, log_W_uniform) where log_W_uniform is
`full(N, -log(N))`. `sampler.py` uses `systematic_resample` by default;
the other two exist for the ablations §10 names this file for.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _normalized_weights(log_W: np.ndarray) -> np.ndarray:
    log_W = log_W - logsumexp(log_W)
    return np.exp(log_W)


def systematic_resample(
    theta: np.ndarray, log_W: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Single-uniform systematic resampling (lowest-variance of the three)."""
    n = log_W.shape[0]
    w = _normalized_weights(log_W)
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0  # guard against float drift
    u0 = rng.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n) / n
    idx = np.searchsorted(cumsum, positions)
    return theta[idx].copy(), np.full(n, -np.log(n))


def multinomial_resample(
    theta: np.ndarray, log_W: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Plain i.i.d. multinomial resampling (highest-variance baseline)."""
    n = log_W.shape[0]
    w = _normalized_weights(log_W)
    idx = rng.choice(n, size=n, replace=True, p=w)
    return theta[idx].copy(), np.full(n, -np.log(n))


def residual_resample(
    theta: np.ndarray, log_W: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Residual resampling: deterministic integer part + multinomial remainder."""
    n = log_W.shape[0]
    w = _normalized_weights(log_W)
    n_w = n * w
    counts = np.floor(n_w).astype(int)
    n_deterministic = counts.sum()
    idx_deterministic = np.repeat(np.arange(n), counts)

    n_remaining = n - n_deterministic
    if n_remaining > 0:
        residual_w = n_w - counts
        residual_w = residual_w / residual_w.sum()
        idx_remaining = rng.choice(n, size=n_remaining, replace=True, p=residual_w)
        idx = np.concatenate([idx_deterministic, idx_remaining])
    else:
        idx = idx_deterministic
    return theta[idx].copy(), np.full(n, -np.log(n))
