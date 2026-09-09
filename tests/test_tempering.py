"""Unit tests for smc.tempering.solve_beta_step, on synthetic weight arrays.

No toy target, no sampler -- these tests only exercise the bisection solver
itself, per the Stage 0 plan's "get the solver right in isolation first".
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from smc.tempering import BetaDecision, _criterion_value, solve_beta_step


def _random_case(rng, n=200, beta_prev=0.3, offset_scale=0.0):
    """A random normalized-log-weight / rate pair, plus optional log_offset."""
    raw = rng.normal(size=n)
    log_W = raw - np.log(np.sum(np.exp(raw - raw.max()))) - raw.max()
    log_G = -rng.exponential(scale=1.0, size=n)  # same-sign (<=0), realistic case
    log_offset = rng.normal(scale=offset_scale, size=n) if offset_scale else None
    return log_W, log_G, beta_prev, log_offset


def test_cess_monotonic_decreasing_in_beta():
    rng = np.random.default_rng(0)
    for _ in range(20):
        log_W, log_G, beta_prev, _ = _random_case(rng)
        grid = np.linspace(beta_prev, 1.0, 50)
        vals = [_criterion_value("cess", log_W, log_G, b, beta_prev, None) for b in grid]
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-9), "CESS must be non-increasing in beta"


def test_ess_boundary_equals_carried_ess():
    rng = np.random.default_rng(1)
    for _ in range(20):
        log_W, log_G, beta_prev, _ = _random_case(rng)
        carried_ess = 1.0 / np.sum(np.exp(2.0 * log_W))
        ess_at_boundary = _criterion_value("ess", log_W, log_G, beta_prev, beta_prev, None)
        assert ess_at_boundary == pytest.approx(carried_ess, rel=1e-9)
    # Deliberately not asserting ESS(b) is monotone in b -- it isn't, reliably
    # (see smc/tempering.py module docstring). Asserting that would be flaky.


def test_cess_boundary_equals_n_regardless_of_degenerate_weights():
    rng = np.random.default_rng(2)
    n = 300
    for skew in (1.0, 5.0, 20.0, 100.0):
        raw = rng.normal(size=n) * skew
        log_W = raw - np.log(np.sum(np.exp(raw - raw.max()))) - raw.max()
        log_G = -rng.exponential(size=n)
        beta_prev = 0.2
        cess_at_boundary = _criterion_value("cess", log_W, log_G, beta_prev, beta_prev, None)
        assert cess_at_boundary == pytest.approx(n, rel=1e-8)


def test_cess_degenerate_branch_unreachable_for_kappa_le_1():
    rng = np.random.default_rng(3)
    for _ in range(50):
        log_W, log_G, beta_prev, _ = _random_case(rng, n=100)
        kappa = rng.uniform(0.01, 1.0)
        decision = solve_beta_step(log_W, log_G, kappa, beta_prev, criterion="cess")
        assert decision.degenerate is False


def test_ess_degenerate_branch_reachable():
    rng = np.random.default_rng(4)
    # Construct badly-skewed carried weights (low carried ESS) and demand a
    # high kappa -- the ESS solver should have nowhere to go.
    n = 200
    raw = np.zeros(n)
    raw[0] = 20.0  # one particle dominates: carried ESS ~ 1
    log_W = raw - np.log(np.sum(np.exp(raw - raw.max()))) - raw.max()
    log_G = -rng.exponential(size=n)
    beta_prev = 0.1
    decision = solve_beta_step(log_W, log_G, 0.9, beta_prev, criterion="ess")
    assert decision.degenerate is True
    assert decision.beta == pytest.approx(beta_prev + 1e-4)


def test_solve_beta_step_jumps_to_one_when_cess_at_one_exceeds_target():
    rng = np.random.default_rng(5)
    n = 100
    log_W = np.full(n, -np.log(n))
    log_G = -rng.exponential(scale=1e-4, size=n)  # tiny rates -> CESS stays ~N even at b=1
    decision = solve_beta_step(log_W, log_G, 0.9, 0.0, criterion="cess")
    assert decision.jumped_to_one is True
    assert decision.beta == 1.0


def test_solve_beta_step_matches_brentq_reference():
    rng = np.random.default_rng(6)
    for _ in range(20):
        log_W, log_G, beta_prev, _ = _random_case(rng, n=150)
        kappa = 0.7
        n = log_W.shape[0]
        target = kappa * n

        def f(b):
            return _criterion_value("cess", log_W, log_G, b, beta_prev, None) - target

        # Only compare against brentq when the root genuinely lies inside
        # (beta_prev, 1) -- i.e. neither short-circuit fires.
        if f(1.0) >= 0 or f(beta_prev) < 0:
            continue
        ref_beta = brentq(f, beta_prev, 1.0, xtol=1e-10)
        decision = solve_beta_step(log_W, log_G, kappa, beta_prev, criterion="cess")
        assert decision.beta == pytest.approx(ref_beta, abs=1e-5)


def test_solve_beta_step_monotone_wrt_kappa():
    rng = np.random.default_rng(7)
    for _ in range(20):
        log_W, log_G, beta_prev, _ = _random_case(rng, n=150)
        b_low = solve_beta_step(log_W, log_G, 0.5, beta_prev, criterion="cess").beta
        b_high = solve_beta_step(log_W, log_G, 0.95, beta_prev, criterion="cess").beta
        assert b_high <= b_low + 1e-9  # higher kappa -> no larger a step


def test_solve_beta_step_beta_in_bounds():
    rng = np.random.default_rng(8)
    for criterion in ("cess", "ess"):
        for _ in range(30):
            log_W, log_G, beta_prev, _ = _random_case(rng, n=80)
            kappa = rng.uniform(0.05, 0.99)
            decision = solve_beta_step(log_W, log_G, kappa, beta_prev, criterion=criterion)
            assert beta_prev < decision.beta <= 1.0 + 1e-12


def test_default_criterion_is_cess():
    rng = np.random.default_rng(9)
    log_W, log_G, beta_prev, _ = _random_case(rng, n=100)
    d_default = solve_beta_step(log_W, log_G, 0.8, beta_prev)
    d_explicit = solve_beta_step(log_W, log_G, 0.8, beta_prev, criterion="cess")
    assert d_default == d_explicit


def test_renormalizes_log_w_prev_defensively():
    rng = np.random.default_rng(10)
    log_W, log_G, beta_prev, _ = _random_case(rng, n=100)
    unnormalized = log_W + 3.7  # no longer sums to 1 in prob space
    d_norm = solve_beta_step(log_W, log_G, 0.8, beta_prev, criterion="cess")
    d_unnorm = solve_beta_step(unnormalized, log_G, 0.8, beta_prev, criterion="cess")
    assert d_norm.beta == pytest.approx(d_unnorm.beta, abs=1e-9)


def test_bisection_iteration_budget():
    rng = np.random.default_rng(11)
    log_W, log_G, beta_prev, _ = _random_case(rng, n=150)
    n = log_W.shape[0]
    kappa = 0.7
    for n_iter in (20, 30):
        decision = solve_beta_step(
            log_W, log_G, kappa, beta_prev, criterion="cess", n_iter=n_iter
        )
        if decision.jumped_to_one or decision.degenerate:
            continue
        assert decision.n_iters_used == n_iter
        assert abs(decision.criterion_value - kappa * n) < 1e-4 * n


def test_log_offset_seam_does_not_affect_cess_boundary_shape():
    # Stage 0 never uses log_offset; this just checks the seam is wired
    # correctly (additive offset shifts the incremental weight as documented)
    # without asserting anything about model-specific semantics.
    rng = np.random.default_rng(12)
    log_W, log_G, beta_prev, log_offset = _random_case(rng, n=100, offset_scale=0.5)
    decision = solve_beta_step(
        log_W, log_G, 0.6, beta_prev, criterion="cess", log_offset=log_offset
    )
    assert beta_prev < decision.beta <= 1.0


def test_invalid_criterion_raises():
    rng = np.random.default_rng(13)
    log_W, log_G, beta_prev, _ = _random_case(rng, n=10)
    with pytest.raises(ValueError):
        solve_beta_step(log_W, log_G, 0.8, beta_prev, criterion="bogus")


def test_invalid_beta_prev_raises():
    rng = np.random.default_rng(14)
    log_W, log_G, _, _ = _random_case(rng, n=10)
    with pytest.raises(ValueError):
        solve_beta_step(log_W, log_G, 0.8, 1.0)
    with pytest.raises(ValueError):
        solve_beta_step(log_W, log_G, 0.8, -0.1)
