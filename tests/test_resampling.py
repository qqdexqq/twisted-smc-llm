from __future__ import annotations

import numpy as np
import pytest

from smc.resampling import multinomial_resample, residual_resample, systematic_resample


def _weighted_particles(rng, n=8):
    theta = np.arange(n).reshape(n, 1).astype(float)
    raw = rng.normal(size=n)
    log_W = raw - np.log(np.sum(np.exp(raw - raw.max()))) - raw.max()
    return theta, log_W


@pytest.mark.parametrize("resample_fn", [systematic_resample, multinomial_resample, residual_resample])
def test_output_shape_and_uniform_weights(resample_fn):
    rng = np.random.default_rng(0)
    theta, log_W = _weighted_particles(rng, n=20)
    theta_new, log_W_new = resample_fn(theta, log_W, rng)
    n = theta.shape[0]
    assert theta_new.shape == theta.shape
    assert np.allclose(log_W_new, -np.log(n))


@pytest.mark.parametrize("resample_fn", [systematic_resample, multinomial_resample, residual_resample])
def test_systematic_resample_unbiased_counts(resample_fn):
    rng = np.random.default_rng(1)
    n = 10
    theta = np.arange(n).reshape(n, 1).astype(float)
    w = np.array([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
    log_W = np.log(w)

    n_trials = 4000
    counts = np.zeros(n)
    for _ in range(n_trials):
        theta_new, _ = resample_fn(theta, log_W, rng)
        idx = theta_new[:, 0].astype(int)
        counts += np.bincount(idx, minlength=n)
    empirical_freq = counts / (n_trials * n)
    assert np.allclose(empirical_freq, w, atol=0.01)


def test_systematic_resample_lower_variance_than_multinomial():
    rng = np.random.default_rng(2)
    n = 50
    theta = np.arange(n).reshape(n, 1).astype(float)
    w = np.full(n, 1.0 / n)
    w[0] = 0.3
    w[1:] = 0.7 / (n - 1)
    log_W = np.log(w)

    def realized_count_of_particle_0(resample_fn, n_trials=3000):
        counts = np.zeros(n_trials)
        for i in range(n_trials):
            theta_new, _ = resample_fn(theta, log_W, rng)
            counts[i] = np.sum(theta_new[:, 0] == 0)
        return counts

    sys_counts = realized_count_of_particle_0(systematic_resample)
    mult_counts = realized_count_of_particle_0(multinomial_resample)
    assert np.var(sys_counts) <= np.var(mult_counts)
