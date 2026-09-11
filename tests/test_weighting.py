from __future__ import annotations

import numpy as np
import pytest

from smc.weighting import (
    PRM_SCORE_CLIP,
    compute_log_G_and_offset,
    compute_log_G_and_offset_batch,
    log_psi_from_prm_score,
)


def _incremental_weight(log_G, log_offset, b, beta_prev):
    """Reimplements solve_beta_step's own incremental-weight formula
    independently, so this test doesn't just re-check weighting.py
    against itself."""
    return (b - beta_prev) * log_G + log_offset


def _doc_formula(log_psi_new, log_psi_prev, b, beta_prev, log_p0_ratio=0.0):
    """The plan doc's §2 formula, transcribed directly and independently
    of compute_log_G_and_offset, as the ground truth to check against."""
    return b * log_psi_new - beta_prev * log_psi_prev + log_p0_ratio


@pytest.mark.parametrize(
    "log_psi_new,log_psi_prev,beta_prev,b,log_p0_ratio",
    [
        (-0.5, -1.2, 0.3, 0.6, 0.0),
        (-2.0, -2.0, 0.5, 1.0, 0.0),  # unchanged psi (e.g. done particle-ish)
        (-0.1, -3.0, 0.9, 0.95, 0.0),
        (-1.0, -1.5, 0.4, 0.4, 0.0),  # b == beta_prev exactly
        (-0.5, -1.2, 0.3, 0.6, -0.2),  # nonzero proposal-correction term
    ],
)
def test_matches_doc_formula_exactly(log_psi_new, log_psi_prev, beta_prev, b, log_p0_ratio):
    log_G, log_offset = compute_log_G_and_offset(log_psi_new, log_psi_prev, beta_prev, log_p0_ratio)
    got = _incremental_weight(log_G, log_offset, b, beta_prev)
    want = _doc_formula(log_psi_new, log_psi_prev, b, beta_prev, log_p0_ratio)
    assert got == pytest.approx(want, abs=1e-12)


def test_naive_wrong_decomposition_actually_differs():
    """Guards against silently reintroducing the exact bug flagged in the
    module docstring: folding in only -beta_prev*logpsi(x_1:t-1) and
    leaving log_G alone (no compensating +beta_prev*logpsi(x_1:t) cross
    term) is wrong by -beta_prev*logpsi(x_1:t). Confirm that "wrong"
    version really does diverge from the doc formula, so we know the test
    above is actually discriminating, not vacuously true.
    """
    log_psi_new, log_psi_prev, beta_prev, b = -0.5, -1.2, 0.3, 0.6
    naive_log_offset = -beta_prev * log_psi_prev  # the bug: missing +beta_prev*log_psi_new
    naive_result = _incremental_weight(log_psi_new, naive_log_offset, b, beta_prev)
    correct_result = _doc_formula(log_psi_new, log_psi_prev, b, beta_prev)
    assert naive_result != pytest.approx(correct_result, abs=1e-9)


def test_t0_convention_first_step_offset_is_zero():
    # beta_0 = 0 always; logpsi is 0.0 (neutral) before any step exists.
    log_G, log_offset = compute_log_G_and_offset(log_psi_new=-0.7, log_psi_prev=0.0, beta_prev=0.0)
    assert log_G == -0.7
    assert log_offset == pytest.approx(0.0, abs=1e-12)


def test_done_particle_frozen_score_gives_zero_offset():
    # Done particle: same frozen score passed as both new and prev.
    log_G, log_offset = compute_log_G_and_offset(log_psi_new=-0.42, log_psi_prev=-0.42, beta_prev=0.6)
    assert log_G == -0.42
    assert log_offset == pytest.approx(0.0, abs=1e-12)


def test_batch_matches_scalar_elementwise():
    log_psi_new = np.array([-0.5, -2.0, -0.1, -1.0])
    log_psi_prev = np.array([-1.2, -2.0, -3.0, -1.5])
    beta_prev = 0.4
    log_G_batch, log_offset_batch = compute_log_G_and_offset_batch(log_psi_new, log_psi_prev, beta_prev)
    for i in range(len(log_psi_new)):
        log_G_i, log_offset_i = compute_log_G_and_offset(log_psi_new[i], log_psi_prev[i], beta_prev)
        assert log_G_batch[i] == pytest.approx(log_G_i, abs=1e-12)
        assert log_offset_batch[i] == pytest.approx(log_offset_i, abs=1e-12)


def test_batch_default_log_p0_ratio_is_zero():
    log_psi_new = np.array([-0.5, -1.0])
    log_psi_prev = np.array([-1.0, -1.0])
    _, log_offset = compute_log_G_and_offset_batch(log_psi_new, log_psi_prev, beta_prev=0.5)
    # log_p0_ratio=0 => log_offset = beta_prev*(new-prev)
    expected = 0.5 * (log_psi_new - log_psi_prev)
    np.testing.assert_allclose(log_offset, expected)


def test_log_psi_from_prm_score_clips_extremes():
    lo, hi = PRM_SCORE_CLIP
    assert log_psi_from_prm_score(0.0) == pytest.approx(np.log(lo))
    assert log_psi_from_prm_score(1.0) == pytest.approx(np.log(hi))
    assert np.isfinite(log_psi_from_prm_score(0.0))
    assert np.isfinite(log_psi_from_prm_score(1.0))


def test_log_psi_from_prm_score_mid_range_unclipped():
    assert log_psi_from_prm_score(0.5) == pytest.approx(np.log(0.5))
