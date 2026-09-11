"""ResamplingRule: a new, minimal, ADDITIVE protocol alongside
smc.controllers.ScheduleController. Orthogonal axes -- ScheduleController
decides the temperature beta each step; ResamplingRule decides whether/how
particles get resampled or pruned. Existing ScheduleController
implementations (fixed/ess/cess) are untouched and still fully compliant
with their own protocol; nothing here changes that interface.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from smc.resampling import systematic_resample
from smc.sampler import effective_sample_size  # pure numpy, model-agnostic -- safe to reuse
from smc.selection import deterministic_topk_select
from smc.tempering import solve_beta_step


class ResamplingRule(Protocol):
    def maybe_resample(
        self,
        particles: np.ndarray,
        log_W: np.ndarray,
        t: int,
        t_frac: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """Return (particles, log_W, resampled, diagnostics)."""
        ...


class EssTriggeredResample:
    """Standard SMC: systematic resample iff ESS < gamma*N. Used by
    particle filtering and twisted-SMC-fixed (baselines #1 and #3)."""

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma

    def maybe_resample(self, particles, log_W, t, t_frac, rng):
        n = len(particles)
        ess = effective_sample_size(log_W)
        if ess < self.gamma * n:
            new_particles, new_log_W = systematic_resample(particles, log_W, rng)
            return new_particles, new_log_W, True, {"ess_before": ess}
        return particles, log_W, False, {"ess_before": ess}


class DeterministicTopK:
    """Beam search (#2): prune branch_factor*N_parents candidates down to
    n_keep, ranked by each candidate's own raw log_psi_cumulative --
    NEVER log_W. Deliberately non-probabilistic (plan doc §6: beam search
    is "a non-probabilistic search baseline") -- rng is unused, and log_W
    is threaded through only so the same instrumentation/accounting
    applies uniformly across all 4 methods, never consulted for pruning.
    """

    def __init__(self, n_keep: int):
        self.n_keep = n_keep

    def maybe_resample(self, particles, log_W, t, t_frac, rng):
        scores = np.array([p.log_psi_cumulative for p in particles])
        kept = deterministic_topk_select(particles, scores, self.n_keep)
        # Beam search carries no importance weight at all; reset to
        # uniform for consistency with what "just resampled" means
        # everywhere else (log_W stays diagnostic-only for this method).
        new_log_W = np.full(self.n_keep, -np.log(self.n_keep))
        return kept, new_log_W, True, {}


class EntropicResample:
    """Entropic Particle Filtering (#4), plan doc §1.2's published
    settings: "resamples when normalised ESS <= 0.5... intervention over
    the first 50% of steps." This is a DERIVED design choice -- the ePF
    paper's exact mechanism isn't given verbatim in the plan doc, only
    characterized at a high level -- flagged, not something to treat as
    definitionally correct without checking against the actual paper if
    precision here ever becomes load-bearing for a real comparison claim.

    Per the doc's own §1.2 table, ePF's beta stays pinned at 1 (same
    schedule as PF) -- it never tempers psi at all; its novelty is
    reactively annealing WHICH ancestors get resampled once ESS collapses,
    not proactively controlling psi's temperature (that contrast with
    "your method" is the doc's own framing). Modeled here as: when ESS
    drops below threshold AND we're still inside the intervention window,
    find the smoothing exponent b in (0, 1] that flattens the CURRENT
    normalized weights toward uniform just enough to restore ESS to
    threshold, then resample on W_i^b instead of raw W_i (b=1 recovers
    plain PF-style resampling on the true weights; b->0 flattens toward
    uniform). Past the intervention window, or if ESS hasn't collapsed,
    behaves exactly like EssTriggeredResample on the raw weights.

    Reuses smc/tempering.py::solve_beta_step UNCHANGED via an argument
    binding where the CURRENT weights themselves play the role normally
    played by an incremental rate:
        log_w_prev = uniform log(1/N)   (reference: "no info yet")
        log_G      = log_W              (current weights ARE the rate:
                                          (b-0)*log_G = b*log(W_i) = log(W_i^b))
        beta_prev  = 0.0
        criterion  = "ess"               (ePF's own published metric is
                                           normalised ESS, not CESS)
        target_cess (repurposed name)    = ess_threshold
    Verified algebraically: with this binding, solve_beta_step's internal
    ESS(b) functional reduces to exp(2*logsumexp(b*log_W) -
    logsumexp(2*b*log_W)) = (sum W_i^b)^2 / sum (W_i^b)^2 -- exactly the
    textbook ESS-of-power-tempered-weights formula, ranging from N at b=0
    (fully flattened) down to the true current ESS at b=1 (untouched
    weights). Because resampling always resets weights to uniform
    afterward (systematic_resample's own contract), the smoothing affects
    only WHO gets replicated -- no importance-weight bookkeeping is
    disturbed by reusing the solver this way.
    """

    def __init__(self, ess_threshold: float = 0.5, intervention_frac: float = 0.5, n_iter: int = 30):
        self.ess_threshold = ess_threshold
        self.intervention_frac = intervention_frac
        self.n_iter = n_iter

    def maybe_resample(self, particles, log_W, t, t_frac, rng):
        n = len(particles)
        ess = effective_sample_size(log_W)
        if ess >= self.ess_threshold * n:
            return particles, log_W, False, {"ess_before": ess, "annealed": False}

        if t_frac <= self.intervention_frac:
            uniform_log_W = np.full(n, -np.log(n))
            decision = solve_beta_step(
                uniform_log_W, log_W, self.ess_threshold, 0.0, criterion="ess", n_iter=self.n_iter
            )
            # log(W_i^b), unnormalized -- systematic_resample normalizes
            # internally regardless, so no explicit renormalization needed.
            resample_weights = decision.beta * log_W
            annealing_beta = decision.beta
        else:
            resample_weights = log_W
            annealing_beta = 1.0

        new_particles, new_log_W = systematic_resample(particles, resample_weights, rng)
        return (
            new_particles,
            new_log_W,
            True,
            {"ess_before": ess, "annealed": t_frac <= self.intervention_frac, "annealing_beta": annealing_beta},
        )
