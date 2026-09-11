from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from smc.llm_particle import new_particle
from smc.resampling_rules import DeterministicTopK, EntropicResample, EssTriggeredResample
from smc.sampler import effective_sample_size


def _log_W_from_raw(raw):
    raw = np.asarray(raw, dtype=float)
    return raw - logsumexp(raw)


def _particles(n):
    return np.array([new_particle(f"q{i}", lineage_id=i) for i in range(n)], dtype=object)


class TestEssTriggeredResample:
    def test_no_resample_when_ess_high(self):
        rule = EssTriggeredResample(gamma=0.5)
        n = 100
        log_W = np.full(n, -np.log(n))  # uniform -> ESS = N, well above gamma*N
        particles = _particles(n)
        rng = np.random.default_rng(0)
        new_particles, new_log_W, resampled, diag = rule.maybe_resample(particles, log_W, 1, 0.1, rng)
        assert resampled is False
        assert new_particles is particles
        assert diag["ess_before"] == pytest.approx(n, rel=1e-6)

    def test_resamples_when_ess_low(self):
        rule = EssTriggeredResample(gamma=0.5)
        n = 100
        raw = np.zeros(n)
        raw[0] = 10.0  # one dominant particle -> very low ESS
        log_W = _log_W_from_raw(raw)
        particles = _particles(n)
        rng = np.random.default_rng(0)
        new_particles, new_log_W, resampled, diag = rule.maybe_resample(particles, log_W, 1, 0.1, rng)
        assert resampled is True
        assert diag["ess_before"] < 0.5 * n
        # post-resample weights are uniform, per systematic_resample's contract
        np.testing.assert_allclose(new_log_W, np.full(n, -np.log(n)))


class TestDeterministicTopK:
    def test_prunes_to_n_keep_by_log_psi(self):
        import dataclasses

        particles = _particles(5)
        scores = [0.1, 0.9, 0.3, 0.7, 0.5]
        particles = np.array(
            [dataclasses.replace(p, log_psi_cumulative=s) for p, s in zip(particles, scores)], dtype=object
        )
        rule = DeterministicTopK(n_keep=2)
        log_W = np.full(5, -np.log(5))
        rng = np.random.default_rng(0)
        kept, new_log_W, resampled, diag = rule.maybe_resample(particles, log_W, 1, 0.1, rng)
        assert resampled is True
        assert len(kept) == 2
        kept_scores = sorted(p.log_psi_cumulative for p in kept)
        assert kept_scores == [0.7, 0.9]  # the two highest
        np.testing.assert_allclose(new_log_W, np.full(2, -np.log(2)))

    def test_ignores_log_W_entirely(self):
        # Even if log_W ranks particles in the OPPOSITE order of
        # log_psi_cumulative, DeterministicTopK must still follow
        # log_psi_cumulative -- this is the "non-probabilistic baseline"
        # property, not a detail.
        import dataclasses

        particles = _particles(2)
        particles = np.array(
            [
                dataclasses.replace(particles[0], log_psi_cumulative=0.9),
                dataclasses.replace(particles[1], log_psi_cumulative=0.1),
            ],
            dtype=object,
        )
        log_W = _log_W_from_raw([0.0, 10.0])  # heavily favors particle 1 by weight
        rule = DeterministicTopK(n_keep=1)
        rng = np.random.default_rng(0)
        kept, _, _, _ = rule.maybe_resample(particles, log_W, 1, 0.1, rng)
        assert kept[0].log_psi_cumulative == 0.9  # picked by score, not by weight


class TestEntropicResample:
    def test_no_resample_when_ess_high(self):
        rule = EntropicResample(ess_threshold=0.5, intervention_frac=0.5)
        n = 50
        log_W = np.full(n, -np.log(n))
        particles = _particles(n)
        rng = np.random.default_rng(0)
        new_particles, new_log_W, resampled, diag = rule.maybe_resample(particles, log_W, 1, 0.1, rng)
        assert resampled is False
        assert diag["annealed"] is False

    def test_annealing_restores_ess_near_threshold_inside_window(self):
        rule = EntropicResample(ess_threshold=0.5, intervention_frac=0.5, n_iter=30)
        n = 200
        raw = np.random.default_rng(1).normal(size=n) * 3  # skewed -> low ESS
        log_W = _log_W_from_raw(raw)
        ess_before = effective_sample_size(log_W)
        assert ess_before < 0.5 * n  # sanity: this case should trigger annealing

        particles = _particles(n)
        rng = np.random.default_rng(2)
        new_particles, new_log_W, resampled, diag = rule.maybe_resample(particles, log_W, 1, 0.1, rng)
        assert resampled is True
        assert diag["annealed"] is True
        b = diag["annealing_beta"]
        assert 0.0 < b < 1.0

        # Verify the *pre-resample* smoothed distribution W_i^b actually
        # has ESS close to the target -- this checks the bisection math
        # itself (solve_beta_step's reused-for-a-different-purpose
        # binding), independent of the stochastic resampling outcome.
        smoothed_log_W = _log_W_from_raw(b * log_W)
        smoothed_ess = effective_sample_size(smoothed_log_W)
        assert smoothed_ess == pytest.approx(rule.ess_threshold * n, rel=0.05)

    def test_falls_back_to_plain_resample_outside_window(self):
        rule = EntropicResample(ess_threshold=0.5, intervention_frac=0.5, n_iter=30)
        n = 100
        raw = np.zeros(n)
        raw[0] = 8.0
        log_W = _log_W_from_raw(raw)
        particles = _particles(n)
        rng = np.random.default_rng(0)
        # t_frac=0.9 > intervention_frac=0.5 -> past the window
        new_particles, new_log_W, resampled, diag = rule.maybe_resample(particles, log_W, 10, 0.9, rng)
        assert resampled is True
        assert diag["annealed"] is False
        assert diag["annealing_beta"] == 1.0

    def test_boundary_b_equals_1_recovers_true_weights_ess(self):
        # Sanity check on the algebraic claim in the class docstring:
        # ESS(b=1) via the smoothing formula equals the actual current ESS.
        n = 50
        raw = np.random.default_rng(3).normal(size=n) * 2
        log_W = _log_W_from_raw(raw)
        true_ess = effective_sample_size(log_W)
        smoothed_at_1 = effective_sample_size(_log_W_from_raw(1.0 * log_W))
        assert smoothed_at_1 == pytest.approx(true_ess, rel=1e-9)

    def test_boundary_b_equals_0_gives_full_ess(self):
        n = 50
        raw = np.random.default_rng(4).normal(size=n) * 2
        log_W = _log_W_from_raw(raw)
        smoothed_at_0 = effective_sample_size(_log_W_from_raw(0.0 * log_W))
        assert smoothed_at_0 == pytest.approx(n, rel=1e-6)
