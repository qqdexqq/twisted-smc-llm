"""Stage 2: the seam between LLM-specific quantities (PRM scores, policy
log-probs) and smc/tempering.py::solve_beta_step, which stays frozen and
model-agnostic (per Stage 0's hard rule -- it never imports anything
model-related). Pure math, no I/O, no torch/vllm import -- fully
CPU-testable.

Derivation (plan doc §2's incremental weight, matched against
solve_beta_step's contract):

    doc's incremental weight, moving (t-1, beta_prev) -> (t, b):
        log w_t(b) = b*logpsi(x_1:t) - beta_prev*logpsi(x_1:t-1)
                     + log p0(x_t|x_1:t-1) - log q(x_t|x_1:t-1)

    solve_beta_step's contract: incremental weight at candidate b is
        (b - beta_prev)*log_G + log_offset

Match the b-dependent coefficient: log_G = logpsi(x_1:t) (the NEW
prefix's PRM log-score -- the only term multiplied by b). Then evaluate
the doc's formula at b = beta_prev, where the log_G term must vanish by
construction, so the remainder AT THAT POINT must equal log_offset:

    log w_t(beta_prev) = beta_prev*logpsi(x_1:t) - beta_prev*logpsi(x_1:t-1) + log_p0_ratio
                        = beta_prev*(logpsi(x_1:t) - logpsi(x_1:t-1)) + log_p0_ratio
    => log_offset = beta_prev*(logpsi(x_1:t) - logpsi(x_1:t-1)) + log_p0_ratio

Sanity check by substitution: (b-beta_prev)*log_G + log_offset with this
log_offset expands to
    b*logpsi(x_1:t) - beta_prev*logpsi(x_1:t) + beta_prev*logpsi(x_1:t) - beta_prev*logpsi(x_1:t-1) + log_p0_ratio
  = b*logpsi(x_1:t) - beta_prev*logpsi(x_1:t-1) + log_p0_ratio
which matches the doc's formula exactly (b=beta_t). NOTE: naively folding
in only "-beta_prev*logpsi(x_1:t-1)" and leaving log_G alone (i.e.
forgetting the "+beta_prev*logpsi(x_1:t)" cross term that appears when
you expand (b-beta_prev)*log_G) is WRONG by exactly that cross term --
this was caught during design, not just an abstract worry; verify by
substitution again before touching this module.

Stage 2 never uses a twisted proposal (q ≡ p0 for all four baselines), so
log_p0_ratio is identically 0 there -- kept as an explicit parameter
(default 0.0) purely so Stage 3's twist-induced proposal can supply a
nonzero value later without this module changing shape.

t=0 convention: logpsi = 0.0 (neutral, psi=1) before any step exists, so
a particle's very first incremental weight has log_offset=0 automatically
(beta_0=0 always). A DONE particle (no new step generated this round)
reuses log_psi_new = log_psi_prev = its frozen cumulative score, which
makes log_G = that frozen score and log_offset = log_p0_ratio (0 for
Stage 2) -- falls out of the exact same formula with no special-casing,
and is the mathematically correct treatment: a finished particle's own
target still sharpens as the *global* beta climbs toward 1 using its
fixed psi score.

Caveat carried forward from design: smc/tempering.py's two proven facts
(CESS(beta_prev)=N exactly; CESS(b) monotone-decreasing) both assumed
log_offset is 0 or constant across particles. Here log_offset varies
per-particle (each one's own beta_prev*(delta logpsi)), so neither
guarantee is provably preserved in general -- solve_beta_step still runs
fine (it never hard-codes N or relies on monotonicity beyond "terminate
after n_iter halvings"), but treat CESS-controller bisection in this
setting as a well-terminating heuristic, the same status the module
already assigns plain ESS, not a proof-backed exact root-find.
"""

from __future__ import annotations

import numpy as np

# Numerical hygiene (plan doc §7.1): "PRM scores near 0 or 1 produce
# infinities that will silently poison the bisection." Clip before ever
# taking a log of a raw [0,1] PRM score.
PRM_SCORE_CLIP = (1e-6, 1.0 - 1e-6)


def log_psi_from_prm_score(score: float) -> float:
    """log(clip(score)) -- the only place a raw PRM score in [0,1] should
    ever be turned into a log-psi value, so the clip is never skipped.
    """
    lo, hi = PRM_SCORE_CLIP
    return float(np.log(min(max(score, lo), hi)))


def compute_log_G_and_offset(
    log_psi_new: float,
    log_psi_prev: float,
    beta_prev: float,
    log_p0_ratio: float = 0.0,
) -> tuple[float, float]:
    """One particle's (log_G, log_offset) for this step -- see module
    docstring for the derivation. log_psi_new/log_psi_prev are already
    logs (use log_psi_from_prm_score to get there from a raw PRM score);
    for a done particle, pass the same frozen value for both.
    """
    log_G = log_psi_new
    log_offset = beta_prev * (log_psi_new - log_psi_prev) + log_p0_ratio
    return log_G, log_offset


def compute_log_G_and_offset_batch(
    log_psi_new: np.ndarray,
    log_psi_prev: np.ndarray,
    beta_prev: float,
    log_p0_ratio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized form for a whole particle array in one call -- what
    smc/sampler_llm.py::Sampler.weight() actually uses each step.
    """
    log_psi_new = np.asarray(log_psi_new, dtype=float)
    log_psi_prev = np.asarray(log_psi_prev, dtype=float)
    if log_p0_ratio is None:
        log_p0_ratio = np.zeros_like(log_psi_new)
    else:
        log_p0_ratio = np.asarray(log_p0_ratio, dtype=float)
    log_G = log_psi_new
    log_offset = beta_prev * (log_psi_new - log_psi_prev) + log_p0_ratio
    return log_G, log_offset
