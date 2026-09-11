"""Compute accounting (plan doc §7.2): every Stage 2 baseline must report
accuracy alongside how much compute it spent to get there, or a
comparison between methods is meaningless (a method that just generates
more tokens/PRM calls per problem will look better for free). smc/'s
hardware-agnostic rule (smc/tempering.py's own rule, extended here)
means wall-clock/VRAM/GPU-util are measured OUTSIDE this module, by the
caller (scripts/run_stage2_baselines.py) -- ComputeAccountant only counts
things smc/sampler_llm.py::Sampler itself directly controls (tokens
generated, PRM forward passes, resample events, bisection solves), all of
which are countable identically on CPU mocks or real GPU hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComputeStats:
    n_tokens_generated: int = 0
    n_policy_calls: int = 0
    n_prm_forward_passes: int = 0
    n_prm_particles_scored: int = 0
    n_prm_steps_scored: int = 0
    n_resample_events: int = 0
    n_bisection_solves: int = 0
    # Set externally by the caller (scripts/run_stage2_baselines.py), which
    # is the only place with access to real hardware timers/counters --
    # left at their defaults (0.0 / None) for CPU-only / --dry-run usage.
    wall_clock_seconds: float = 0.0
    peak_vram_bytes: int | None = None
    gpu_util_percent: float | None = None


class ComputeAccountant:
    """Threaded through Sampler via an optional constructor arg so hook
    points stay out of the core propagate/weight/resample logic -- Sampler
    calls these record_* methods at exactly the points where the
    corresponding real cost is incurred (see smc/sampler_llm.py); this
    class only aggregates, it never decides what counts as an event.
    """

    def __init__(self):
        self.stats = ComputeStats()

    def record_tokens_generated(self, n_tokens: int) -> None:
        self.stats.n_tokens_generated += n_tokens
        self.stats.n_policy_calls += 1

    def record_prm_forward(self, n_particles: int, n_steps_total: int) -> None:
        """One call == one batched PRM forward pass (models/prm.py's
        score_batch, the real efficiency win over N separate score()
        calls -- see that module's docstring), scoring n_particles
        particles' prefixes (n_steps_total steps in aggregate, since each
        particle's step count is ragged).
        """
        self.stats.n_prm_forward_passes += 1
        self.stats.n_prm_particles_scored += n_particles
        self.stats.n_prm_steps_scored += n_steps_total

    def record_resample_event(self, diag: dict) -> None:
        self.stats.n_resample_events += 1

    def record_bisection_solve(self) -> None:
        """Only called when the controller's own class-level
        `uses_bisection` flag is True (see ScheduleController subclasses
        and Sampler.weight()) -- FixedLinearController never triggers
        this, so PF/beam/twisted-SMC-fixed/ePF (this push's four
        baselines, all fixed-schedule) report n_bisection_solves == 0;
        it's real for the Stage 3 preview's CESS-adaptive controller.
        """
        self.stats.n_bisection_solves += 1

    def as_dict(self) -> dict:
        return dict(self.stats.__dict__)
