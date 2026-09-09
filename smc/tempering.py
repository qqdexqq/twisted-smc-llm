"""Stage 0 deliverable: the adaptive-tempering bisection solver.

This module is imported unchanged into the LLM engine in later stages, so the
one rule that matters is architectural, not mathematical: nothing in this
file may import anything model-related (no torch, no transformers, no vLLM,
no reward-model code). It knows about log-weight arrays and a scalar
temperature `beta` in [0, 1]; it does not know or care whether the particles
behind those arrays are reasoning traces or points in R^2.

Background (see docs/Adaptive_TSMC_Build_Plan.md, section 2 and 4.3):

Given particles with current normalized log-weights `log_w_prev` (= log W_{t-1})
and a per-particle "rate" `log_G` (such that the incremental log-weight for
moving from beta_prev to a candidate beta b is `(b - beta_prev) * log_G`,
optionally plus a fixed `log_offset` that does not depend on b), choose the
largest b in (beta_prev, 1] such that the conditional ESS (CESS) of the
reweighted particle set is still at least `target_cess * N`:

    CESS_t(b) = N * (sum_j W_j * w_j(b))^2 / sum_k W_k * w_k(b)^2

An "ESS" variant of the same idea replaces the denominator with the
unconditional (quadratic-in-W) ESS of the reweighted set. Both are found by
bisection on log-weights; both share this module's bisection core.

Two provable facts drive the boundary handling below (verified analytically
and numerically during design; see the plan doc / commit history for the
derivation):

  1. CESS(beta_prev) == N *exactly*, for ANY log_w_prev, whenever log_offset
     is None (Cauchy-Schwarz equality case at b=beta_prev, where all
     incremental weights collapse to a constant). So for CESS with
     target_cess <= 1, the "already below target" branch is mathematically
     unreachable -- degenerate=True can only happen for the ESS criterion.
  2. CESS(b) is provably monotone non-increasing in b (Beskos et al., cited
     in the plan doc) -- bisection on it is an exact root isolation. Plain
     ESS is NOT reliably monotone in b (small non-monotonic ripples occur
     even for same-sign log_G) -- its bisection is a well-terminating
     heuristic, not exact root isolation. Do not rely on ESS monotonicity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

_VALID_CRITERIA = ("cess", "ess")


@dataclass(frozen=True)
class BetaDecision:
    """The result of one bisection solve for the next temperature.

    Attributes:
        beta: the chosen temperature, in (beta_prev, 1].
        criterion_value: the CESS or ESS value (whichever criterion was
            used) attained at `beta`, on the particle-count scale (i.e. in
            [0, N], not normalized to [0, 1]).
        n_iters_used: number of bisection halvings actually performed (0 for
            the jump-to-one and degenerate short-circuits).
        degenerate: True if even the smallest step already dropped the
            criterion below target -- `beta` is then just `beta_prev +
            min_step`, and the caller should treat this as a flagged event,
            not a normal solve.
        jumped_to_one: True if the criterion was still above target at
            beta=1, so the schedule jumped straight to the terminal
            temperature instead of bisecting.
    """

    beta: float
    criterion_value: float
    n_iters_used: int
    degenerate: bool
    jumped_to_one: bool


def _incremental_log_weight(
    log_G: np.ndarray, b: float, beta_prev: float, log_offset: np.ndarray | None
) -> np.ndarray:
    """log w_i(b) = (b - beta_prev) * log_G_i [+ log_offset_i]."""
    lw = (b - beta_prev) * log_G
    if log_offset is not None:
        lw = lw + log_offset
    return lw


def _criterion_value(
    criterion: str,
    log_W: np.ndarray,
    log_G: np.ndarray,
    b: float,
    beta_prev: float,
    log_offset: np.ndarray | None,
) -> float:
    """Evaluate CESS(b) or ESS(b) on the particle-count scale [0, N]."""
    log_wb = _incremental_log_weight(log_G, b, beta_prev, log_offset)
    log_A = logsumexp(log_W + log_wb)  # log( sum_j W_j * w_j(b) )
    if criterion == "cess":
        log_B = logsumexp(log_W + 2.0 * log_wb)  # log( sum_k W_k * w_k(b)^2 )
        n = log_W.shape[0]
        return float(n * np.exp(2.0 * log_A - log_B))
    else:  # "ess"
        log_B = logsumexp(2.0 * log_W + 2.0 * log_wb)  # log( sum_k (W_k*w_k(b))^2 )
        return float(np.exp(2.0 * log_A - log_B))


def solve_beta_step(
    log_w_prev: np.ndarray,
    log_G: np.ndarray,
    target_cess: float,
    beta_prev: float,
    *,
    criterion: str = "cess",
    n_iter: int = 30,
    min_step: float = 1e-4,
    log_offset: np.ndarray | None = None,
) -> BetaDecision:
    """Choose the next temperature by bisection on the CESS/ESS functional.

    Args:
        log_w_prev: (N,) normalized log-weights W_{t-1} (log-sum-exp ~ 0).
            Defensively renormalized internally, so passing unnormalized
            log-weights degrades gracefully rather than silently corrupting
            the result -- but callers should still pass normalized weights.
        log_G: (N,) per-particle rate such that the incremental log-weight
            at candidate temperature b is (b - beta_prev) * log_G [+
            log_offset]. For the Stage-0 static-tempering target this is
            -V(theta_i); it must already be finite (no +-inf, no nan).
        target_cess: kappa in (0, 1]. Threshold is target_cess * N.
        beta_prev: current temperature, must satisfy 0 <= beta_prev < 1.
        criterion: "cess" (default) or "ess" -- selects the functional; both
            share this same bisection core.
        n_iter: number of bisection halvings once neither short-circuit
            applies (doc recommends 20-30; default 30).
        min_step: floor step size used only in the degenerate branch.
        log_offset: optional (N,) fixed additive term, independent of b.
            Unused by Stage 0 (always None there); reserved as the seam a
            Stage-3 LLM wrapper can use to fold in terms like
            `-beta_prev*log_psi_prev + proposal_correction` without this
            file ever importing anything model-related.

    Returns:
        A BetaDecision with beta_prev < beta <= 1.
    """
    if criterion not in _VALID_CRITERIA:
        raise ValueError(f"criterion must be one of {_VALID_CRITERIA}, got {criterion!r}")
    if not (0.0 <= beta_prev < 1.0):
        raise ValueError(f"beta_prev must be in [0, 1), got {beta_prev}")
    if not (0.0 < target_cess <= 1.0):
        raise ValueError(f"target_cess must be in (0, 1], got {target_cess}")

    log_W = np.asarray(log_w_prev, dtype=float)
    log_G = np.asarray(log_G, dtype=float)
    if log_W.shape != log_G.shape or log_W.ndim != 1:
        raise ValueError(
            f"log_w_prev and log_G must be 1-D arrays of the same shape, "
            f"got {log_W.shape} and {log_G.shape}"
        )
    if log_offset is not None:
        log_offset = np.asarray(log_offset, dtype=float)
        if log_offset.shape != log_W.shape:
            raise ValueError("log_offset must have the same shape as log_w_prev")

    n = log_W.shape[0]
    log_W = log_W - logsumexp(log_W)  # defensive renormalization
    target = target_cess * n

    def functional(b: float) -> float:
        return _criterion_value(criterion, log_W, log_G, b, beta_prev, log_offset)

    hi_val = functional(1.0)
    if hi_val >= target:
        return BetaDecision(
            beta=1.0, criterion_value=hi_val, n_iters_used=0, degenerate=False, jumped_to_one=True
        )

    lo_val = functional(beta_prev)
    if lo_val < target:
        beta = min(beta_prev + min_step, 1.0)
        return BetaDecision(
            beta=beta,
            criterion_value=functional(beta),
            n_iters_used=0,
            degenerate=True,
            jumped_to_one=False,
        )

    # Invariant going into the loop: functional(lo) >= target > functional(hi).
    lo, hi = beta_prev, 1.0
    val_at_lo = lo_val
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        mid_val = functional(mid)
        if mid_val >= target:
            lo, val_at_lo = mid, mid_val
        else:
            hi = mid

    return BetaDecision(
        beta=lo,
        criterion_value=val_at_lo,
        n_iters_used=n_iter,
        degenerate=False,
        jumped_to_one=False,
    )
