"""Stage 2's driver for text/LLM particles. NOT an extension of
smc/sampler.py (Stage 0's toy-target resample-move loop over a
continuous-space RWM "move") -- a growing reasoning-step prefix has no
continuous-space move, so this is a new, separate driver reusing the same
frozen primitives (ScheduleController -> solve_beta_step,
ResamplingRule -> systematic_resample) at a different call-site shape.

Ordering, same principle as Stage 0's sampler.py: reweight using the
prefix state realized THIS round (propagate() already ran) before
resampling -- beta_t is chosen from already-realized (log_W, log_G), never
from unrealized future randomness.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from eval.metrics import extract_boxed_answer
from smc.controllers import ScheduleController
from smc.llm_particle import Particle, new_particle
from smc.resampling_rules import ResamplingRule
from smc.sampler import effective_sample_size
from smc.tempering import BetaDecision
from smc.weighting import compute_log_G_and_offset_batch, log_psi_from_prm_score

_BETA_EPS = 1e-9


@dataclass(frozen=True)
class StepRecord:
    step: int
    beta_prev: float
    beta: float
    ess_before_resample: float
    n_active: int
    resampled: bool
    resample_diag: dict


@dataclass
class SamplerResult:
    particles: np.ndarray
    log_W: np.ndarray
    log_Z_hat: float
    history: list[StepRecord] = field(default_factory=list)

    @property
    def n_global_steps(self) -> int:
        return len(self.history)


class Sampler:
    """Shared driver behind all four Stage 2 baselines -- which method you
    get is entirely a function of which ScheduleController/ResamplingRule
    pair and n_children you pass in (see the plan's baseline-mapping
    table); this class itself has no per-method branching.
    """

    def __init__(
        self,
        controller: ScheduleController,
        resampling_rule: ResamplingRule,
        policy,
        prm_scorer,
        n_children: int = 1,
        max_particle_steps: int = 300,
        max_tokens_per_step: int = 512,
        step_stop: str = "\n\n",
        temperature: float = 1.0,
        accountant=None,
    ):
        if n_children < 1:
            raise ValueError("n_children must be >= 1")
        self.controller = controller
        self.resampling_rule = resampling_rule
        self.policy = policy
        self.prm_scorer = prm_scorer
        self.n_children = n_children
        self.max_particle_steps = max_particle_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.step_stop = step_stop
        self.temperature = temperature
        self.accountant = accountant

    def propagate(
        self, particles: np.ndarray, prompt: str, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One batched policy call across every active (not-done)
        particle's n_children branches (never loop particle-by-particle),
        plus one batched PRM forward pass scoring every resulting
        candidate prefix in a single call.

        Returns (new_particles, was_active, source_idx):
          - new_particles: length >= len(particles) (> only when
            n_children > 1 -- beam search branches; PF/ePF/twisted-SMC use
            n_children=1, so length is always preserved for them).
          - was_active[i]: True iff new_particles[i] had a NEW step
            generated this round (an already-done particle passed through
            unchanged is False). weight() needs this to apply
            smc/weighting.py's documented "done particle, no new step
            this round" freeze correctly -- the persistent
            Particle.done flag alone can't distinguish "just finished
            this round" (should still get credit for its final delta)
            from "was already done before this round" (frozen).
          - source_idx[i]: index into the INPUT `particles`/log_W arrays
            that new_particles[i] descends from -- lets the caller expand
            log_W to match (log_W = log_W[source_idx]) with no special
            casing for n_children > 1.
        """
        n_in = len(particles)
        was_active_in = np.array([not p.done for p in particles])
        if not was_active_in.any():
            return particles.copy(), np.zeros(n_in, dtype=bool), np.arange(n_in)

        prefixes = []
        parent_idx = []
        for i, p in enumerate(particles):
            if p.done:
                continue
            full_prefix = p.prompt if not p.step_texts else p.prompt + "\n\n" + p.prefix_text
            for _ in range(self.n_children):
                prefixes.append(full_prefix)
                parent_idx.append(i)

        completions = self.policy.generate_step(
            prefixes, stop=[self.step_stop], temperature=self.temperature, max_tokens=self.max_tokens_per_step
        )
        if self.accountant is not None:
            self.accountant.record_tokens_generated(sum(c.n_tokens for c in completions))

        candidates = []  # (parent, new_step_texts, is_done, finish_reason)
        for comp, pi in zip(completions, parent_idx):
            parent = particles[pi]
            new_step_texts = parent.step_texts + (comp.text,)
            is_boxed = "\\boxed{" in comp.text
            is_eos = comp.finish_reason == "eos"
            hit_cap = len(new_step_texts) >= self.max_particle_steps
            is_done = is_boxed or is_eos or hit_cap
            if is_boxed:
                finish_reason = "boxed"
            elif is_eos:
                finish_reason = "eos"
            elif hit_cap:
                finish_reason = "max_particle_steps"
            else:
                finish_reason = None
            candidates.append((parent, new_step_texts, is_done, finish_reason))

        # One batched PRM forward pass over every candidate prefix this
        # round -- the doc's own pseudocode assumes this ("log_psi[i] =
        # prm_score(prefix_i)  # batched PRM forward"); see
        # models/prm.py::score_batch.
        steps_per_particle = [c[1] for c in candidates]
        prm_scores = self.prm_scorer.score_batch(prompt, list(steps_per_particle))
        if self.accountant is not None:
            self.accountant.record_prm_forward(
                n_particles=len(candidates), n_steps_total=sum(len(s) for s in steps_per_particle)
            )

        new_active_particles = []
        for (parent, new_step_texts, is_done, finish_reason), prm_score in zip(candidates, prm_scores):
            log_psi_new = log_psi_from_prm_score(prm_score.step_scores[-1])
            final_answer = extract_boxed_answer("\n\n".join(new_step_texts)) if is_done else None
            new_active_particles.append(
                dataclasses.replace(
                    parent,
                    step_texts=new_step_texts,
                    log_psi_cumulative=log_psi_new,
                    log_psi_prev=parent.log_psi_cumulative,
                    done=is_done,
                    finish_reason=finish_reason,
                    final_answer=final_answer,
                )
            )

        # Reassemble in input order: a done parent passes through as ONE
        # unchanged output slot; an active parent is replaced by its
        # n_children fresh children, contiguous and in order -- this is
        # what makes source_idx trivial to build alongside.
        out_particles: list[Particle] = []
        out_was_active: list[bool] = []
        out_source_idx: list[int] = []
        child_cursor = 0
        for i, p in enumerate(particles):
            if p.done:
                out_particles.append(p)
                out_was_active.append(False)
                out_source_idx.append(i)
            else:
                for _ in range(self.n_children):
                    out_particles.append(new_active_particles[child_cursor])
                    out_was_active.append(True)
                    out_source_idx.append(i)
                    child_cursor += 1

        return (
            np.array(out_particles, dtype=object),
            np.array(out_was_active, dtype=bool),
            np.array(out_source_idx, dtype=int),
        )

    def weight(
        self,
        particles: np.ndarray,
        log_W: np.ndarray,
        beta_prev: float,
        was_active: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, float, BetaDecision | None]:
        """Reweight using log_G/log_offset from smc/weighting.py (see that
        module for the derivation and the "done particle" freeze rule --
        was_active is exactly how that rule gets applied here, see
        propagate()'s docstring for why the persistent .done flag alone
        isn't enough).

        Once beta_prev is already pinned at 1.0 (PF/ePF after their very
        first step), skip calling controller.next_beta() again --
        solve_beta_step raises on beta_prev >= 1.0, and there is nothing
        left to solve for once every particle's target has fully
        sharpened. This guard belongs once here in the driver, not
        duplicated per-controller (matches the plan's design note).
        """
        log_psi_new = np.array([p.log_psi_cumulative for p in particles])
        log_psi_prev = np.array(
            [p.log_psi_prev if was_active[i] else p.log_psi_cumulative for i, p in enumerate(particles)]
        )
        log_G, log_offset = compute_log_G_and_offset_batch(log_psi_new, log_psi_prev, beta_prev)

        if beta_prev >= 1.0 - _BETA_EPS:
            beta_new = 1.0
            decision = None
        else:
            decision = self.controller.next_beta(log_W, log_G, beta_prev, rng)
            beta_new = decision.beta
            if self.accountant is not None and getattr(self.controller, "uses_bisection", False):
                self.accountant.record_bisection_solve()

        log_w_inc = (beta_new - beta_prev) * log_G + log_offset
        log_W_unnorm = log_W + log_w_inc
        # log_W is already normalized (logsumexp == 0) on entry, so this
        # term alone is the log-Z increment for this step.
        log_Z_inc = float(logsumexp(log_W_unnorm) - logsumexp(log_W))
        log_W_new = log_W_unnorm - logsumexp(log_W_unnorm)
        return particles, log_W_new, log_Z_inc, beta_new, decision

    def resample(
        self, particles: np.ndarray, log_W: np.ndarray, t: int, t_frac: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        new_particles, new_log_W, resampled, diag = self.resampling_rule.maybe_resample(
            particles, log_W, t, t_frac, rng
        )
        if resampled and self.accountant is not None:
            self.accountant.record_resample_event(diag)
        return new_particles, new_log_W, resampled, diag

    def terminal_correction(
        self, particles: np.ndarray, log_W: np.ndarray, beta_prev: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Force beta -> 1.0 if the horizon was exhausted before any
        controller naturally reached it (matches Stage 0's terminal
        correction, doc §7.1). Unreachable for this push's PF/ePF configs
        (FixedLinearController(n_steps=1) always jumps beta to 1.0 on the
        very first step), but required in general once an adaptive
        controller (Stage 3 preview) is plugged in.

        Public (not a leading-underscore helper): scripts/run_stage2_
        baselines.py drives Sampler's propagate/weight/resample loop
        directly rather than calling run() as one opaque call (so it can
        checkpoint after every global step), and needs this exact
        correction available at the end of its own loop too.
        """
        if beta_prev >= 1.0 - _BETA_EPS:
            return particles, log_W, 0.0, 1.0
        # No new steps generated at this synthetic final step -- every
        # particle's log_psi is frozen at its current cumulative value
        # (same "done, no new step this round" rule as weight()'s
        # was_active=False case), so only the (1-beta_prev)*log_G term
        # (each particle's own final PRM score) does anything.
        log_psi_cumulative = np.array([p.log_psi_cumulative for p in particles])
        log_G, log_offset = compute_log_G_and_offset_batch(log_psi_cumulative, log_psi_cumulative, beta_prev)
        log_w_inc = (1.0 - beta_prev) * log_G + log_offset
        log_W_unnorm = log_W + log_w_inc
        log_Z_inc = float(logsumexp(log_W_unnorm) - logsumexp(log_W))
        log_W_new = log_W_unnorm - logsumexp(log_W_unnorm)
        return particles, log_W_new, log_Z_inc, 1.0

    def run(self, prompt: str, n_particles: int, rng: np.random.Generator, horizon_steps: int) -> SamplerResult:
        """Global loop: break when ALL particles are done OR
        t > horizon_steps -- NOT "beta==1", since PF/beam/ePF reach
        beta=1 on step one but must keep generating for many more
        reasoning steps after that.
        """
        self.controller.reset()
        particles = np.array([new_particle(prompt, lineage_id=i) for i in range(n_particles)], dtype=object)
        log_W = np.full(n_particles, -np.log(n_particles))
        beta = 0.0
        log_Z_hat = 0.0
        history: list[StepRecord] = []

        for t in range(1, horizon_steps + 1):
            if all(p.done for p in particles):
                break

            particles, was_active, source_idx = self.propagate(particles, prompt, rng)
            log_W = log_W[source_idx]  # no-op reindex when n_children == 1

            particles, log_W, log_Z_inc, beta_new, decision = self.weight(particles, log_W, beta, was_active, rng)
            log_Z_hat += log_Z_inc
            ess_before_resample = effective_sample_size(log_W)

            t_frac = t / horizon_steps
            particles, log_W, resampled, diag = self.resample(particles, log_W, t, t_frac, rng)

            history.append(
                StepRecord(
                    step=t,
                    beta_prev=beta,
                    beta=beta_new,
                    ess_before_resample=ess_before_resample,
                    n_active=int(was_active.sum()),
                    resampled=resampled,
                    resample_diag=diag,
                )
            )
            beta = beta_new

        if beta < 1.0:
            particles, log_W, log_Z_inc, beta = self.terminal_correction(particles, log_W, beta)
            log_Z_hat += log_Z_inc

        assert abs(beta - 1.0) < 1e-9, f"terminal beta={beta}, expected exactly 1.0"

        return SamplerResult(particles=particles, log_W=log_W, log_Z_hat=log_Z_hat, history=history)
