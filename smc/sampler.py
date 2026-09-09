"""The Stage 0 SMC sampler: a standard resample-move loop over a tempered
path of static targets, driven by a pluggable ScheduleController.

Ordering (see docs/Adaptive_TSMC_Build_Plan.md §4.1/§7 and the plan file):
reweight at the OLD particle positions (this is what makes the log-Z
increment an unbiased factor of Z_{beta_new}/Z_{beta_prev}) -> resample only
if ESS < gamma*N -> RWM-move, invariant to the NEW pi_beta. beta_t is chosen
only from already-realized (log_W, log_G) at the start of the step, never
from unrealized future randomness, which is the condition the cited
adaptive-SMC theory needs for the log-Z estimator to stay unbiased despite
adaptive step selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from smc.controllers import ScheduleController
from smc.resampling import systematic_resample
from smc.tempering import BetaDecision


def effective_sample_size(log_W: np.ndarray) -> float:
    """ESS of normalized log-weights: 1 / sum(W_i^2)."""
    return float(1.0 / np.sum(np.exp(2.0 * log_W)))


def rwm_rejuvenate(
    theta: np.ndarray,
    target,
    beta: float,
    rng: np.random.Generator,
    n_steps: int,
    cov_ridge: float,
) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis moves invariant to pi_beta(theta) ~ mu(theta)
    * exp(beta * log_G(theta)). Proposal covariance is the current empirical
    particle covariance scaled by (2.38)^2/d (Roberts-Gelman-Gilks), plus a
    ridge floor so a covariance-collapsed particle set (e.g. right after
    aggressive resampling) never produces a singular proposal.
    """
    n, d = theta.shape
    scale = (2.38**2) / d
    n_accept = 0

    def log_pi(x: np.ndarray) -> np.ndarray:
        return target.log_mu(x) + beta * target.log_G(x)

    log_pi_theta = log_pi(theta)
    for _ in range(n_steps):
        if n > 1:
            cov = np.cov(theta, rowvar=False)
            if d == 1:
                cov = np.array([[cov]])
        else:
            cov = np.eye(d)
        cov = scale * cov + cov_ridge * np.eye(d)
        chol = np.linalg.cholesky(cov)

        proposal = theta + rng.normal(size=theta.shape) @ chol.T
        log_pi_prop = log_pi(proposal)
        log_alpha = log_pi_prop - log_pi_theta
        u = rng.uniform(size=n)
        accept = np.log(u) < log_alpha

        theta = np.where(accept[:, None], proposal, theta)
        log_pi_theta = np.where(accept, log_pi_prop, log_pi_theta)
        n_accept += int(accept.sum())

    return theta, n_accept / (n * n_steps)


@dataclass(frozen=True)
class StepRecord:
    step: int
    beta_prev: float
    beta: float
    ess: float
    decision: BetaDecision
    resampled: bool
    rwm_acceptance_rate: float


@dataclass
class SMCResult:
    log_Z_hat: float
    theta: np.ndarray
    log_W: np.ndarray
    history: list[StepRecord] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.history)


def run_smc(
    target,
    controller: ScheduleController,
    n_particles: int,
    gamma: float,
    rng: np.random.Generator,
    max_steps: int = 3000,
    rwm_steps: int = 10,
    cov_ridge: float = 1e-6,
) -> SMCResult:
    """Run one full temperature sweep (beta: 0 -> 1) with the given controller.

    `max_steps` defaults high (3000) because the ESS-adaptive controller can
    need hundreds of steps when kappa >= gamma: once the controller pins ESS
    at kappa*N >= gamma*N, resampling stops triggering entirely, so carried
    weight degeneracy is never reset and each step's safe increment shrinks
    (empirically up to ~600 steps at kappa=0.9, gamma=0.5, N=1000, on the
    toy GMM target). CESS and fixed-linear schedules typically finish in
    under 50 steps on the same target -- this is itself the mechanism
    behind the plan doc's "Figure 1" diagnostic (see smc/tempering.py).
    """
    controller.reset()
    theta = target.sample_reference(n_particles, rng)
    log_W = np.full(n_particles, -np.log(n_particles))
    beta = 0.0
    log_Z_hat = 0.0
    history: list[StepRecord] = []

    for t in range(1, max_steps + 1):
        log_G = target.log_G(theta)
        decision = controller.next_beta(log_W, log_G, beta, rng)
        beta_new = decision.beta

        log_w_inc = (beta_new - beta) * log_G
        log_W_unnorm = log_W + log_w_inc
        # logsumexp(log_W) == 0 here by the loop invariant, so this term
        # alone is log(Z_{beta_new}/Z_beta) for this step.
        log_Z_hat += float(logsumexp(log_W_unnorm))
        log_W = log_W_unnorm - logsumexp(log_W_unnorm)

        ess = effective_sample_size(log_W)
        resampled = False
        if ess < gamma * n_particles:
            theta, log_W = systematic_resample(theta, log_W, rng)
            resampled = True

        theta, accept_rate = rwm_rejuvenate(theta, target, beta_new, rng, rwm_steps, cov_ridge)

        history.append(
            StepRecord(
                step=t,
                beta_prev=beta,
                beta=beta_new,
                ess=ess,
                decision=decision,
                resampled=resampled,
                rwm_acceptance_rate=accept_rate,
            )
        )
        beta = beta_new
        if beta >= 1.0:
            break
    else:
        raise RuntimeError(f"did not reach beta=1.0 within max_steps={max_steps}")

    assert abs(beta - 1.0) < 1e-9, f"terminal beta={beta}, expected exactly 1.0"
    return SMCResult(log_Z_hat=log_Z_hat, theta=theta, log_W=log_W, history=history)
