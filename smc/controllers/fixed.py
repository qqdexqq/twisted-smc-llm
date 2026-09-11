from __future__ import annotations

import numpy as np

from smc.tempering import BetaDecision


class FixedLinearController:
    """beta_i = i / n_steps, ignoring log_W / log_G entirely -- the
    non-adaptive baseline schedule everything else is compared against.
    """

    # eval/compute_accounting.py: fixed schedules never call solve_beta_step.
    uses_bisection = False

    def __init__(self, n_steps: int):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.n_steps = n_steps
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def next_beta(
        self,
        log_W: np.ndarray,
        log_G: np.ndarray,
        beta_prev: float,
        rng: np.random.Generator,
    ) -> BetaDecision:
        self._step += 1
        beta = min(self._step / self.n_steps, 1.0)
        return BetaDecision(
            beta=beta,
            criterion_value=float("nan"),
            n_iters_used=0,
            degenerate=False,
            jumped_to_one=(self._step >= self.n_steps),
        )
