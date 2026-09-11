from __future__ import annotations

import numpy as np

from smc.tempering import BetaDecision, solve_beta_step


class CessAdaptiveController:
    """Bisects for the largest beta holding conditional ESS (CESS) at
    kappa * N. Provably well-posed: CESS(b) is monotone-decreasing in b
    (Beskos et al.), so this bisection is exact root isolation -- see
    smc/tempering.py's module docstring.
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
            criterion="cess",
            n_iter=self.n_iter,
            min_step=self.min_step,
        )
