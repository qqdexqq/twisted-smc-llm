"""Filled in for Stage 2 (was an empty stub in Stage 0 -- "ePF's real
schedule needs the Stage 2 LLM+PRM sampler to be a meaningful baseline
against").

Turns out ePF is NOT a ScheduleController at all: per the plan doc's
§1.2 comparison table, ePF's beta stays pinned at 1 throughout (it never
tempers psi -- that's specifically what distinguishes the proactive
adaptive-tempering approach from ePF's reactive annealing). Its actual
mechanism lives on the *resampling* axis instead. The real implementation
is `smc.resampling_rules.EntropicResample`; this module just re-exports
it so the repo layout planned back in Stage 0 (`smc/controllers/epf.py`)
stays discoverable rather than silently vanishing.
"""

from __future__ import annotations

from smc.resampling_rules import EntropicResample

__all__ = ["EntropicResample"]
