"""The ScheduleController interface: fixed/ess/cess controllers are
interchangeable at exactly one call site in smc/sampler.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from smc.tempering import BetaDecision


class ScheduleController(Protocol):
    def next_beta(
        self,
        log_W: np.ndarray,
        log_G: np.ndarray,
        beta_prev: float,
        rng: np.random.Generator,
    ) -> BetaDecision:
        """Return the BetaDecision for the next temperature."""
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g. a fixed schedule's step counter)
        so the same controller instance can be reused across seeds."""
        ...
