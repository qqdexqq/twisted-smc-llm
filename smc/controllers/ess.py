from __future__ import annotations

import numpy as np

from smc.tempering import BetaDecision, solve_beta_step


class EssAdaptiveController:
    """Bisects for the largest beta holding the *unconditional* (carried-in)
    ESS at kappa * N. Unlike CESS, this functional is NOT reliably
    monotone in b (see smc/tempering.py's module docstring) -- its
    bisection is a well-terminating heuristic, not exact root isolation.
    It is the "free ablation" the plan doc calls for: implemented for
    comparison, not because it is believed to be as principled as CESS.
    """

    # eval/compute_accounting.py: this controller solves solve_beta_step every step.
    uses_bisection = True

    def __init__(self, kappa: float, n_iter: int = 30, min_step: float = 1e-4):
        self.kappa = kappa
        self.n_iter = n_iter
        self.min_step = min_step

    def reset(self) -> None:
        pass  # stateless

    def next_beta(
        self,
        log_W: np.ndarray,
        log_G: np.ndarray,
        beta_prev: float,
        rng: np.random.Generator,
    ) -> BetaDecision:
        return solve_beta_step(
            log_W,
            log_G,
            self.kappa,
            beta_prev,
            criterion="ess",
            n_iter=self.n_iter,
            min_step=self.min_step,
        )
