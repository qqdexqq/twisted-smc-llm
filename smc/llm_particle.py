"""Stage 2: the LLM/text analog of Stage 0's continuous-vector `theta`
particle array. A particle here is a growing reasoning-step prefix, not a
point in R^d -- there is no continuous-space MCMC "move" for text, so
this does NOT plug into smc/sampler.py's run_smc (that stays the Stage 0
toy-target loop, untouched); smc/sampler_llm.py::Sampler is the new
driver for this representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Particle:
    """Immutable by design: after smc/resampling.py's systematic_resample
    (reused unchanged, since it just does theta[idx] fancy-indexing over
    a numpy dtype=object array), multiple array slots can reference the
    SAME Particle object. Mutating a field in place would silently
    corrupt every sibling slot pointing at that object. propagate() must
    always build a NEW Particle (dataclasses.replace(...)) rather than
    mutate -- frozen=True enforces this at the type level; mutation
    attempts raise instead of silently aliasing.
    """

    prompt: str
    # tuple, not list: frozen=True only stops *reassigning* a field, not
    # mutating a mutable field's contents in place (a list here would let
    # `.append()` silently corrupt every resampled sibling). A tuple
    # closes that gap -- extending it requires a new tuple, which forces
    # going through dataclasses.replace() like everything else.
    step_texts: tuple[str, ...] = field(default_factory=tuple)
    log_psi_cumulative: float = 0.0  # log_psi_theta(x_1:t); 0.0 (neutral, psi=1) before any step exists
    log_psi_prev: float = 0.0  # log_psi_theta(x_1:t-1)
    lineage_id: int = -1  # set once at creation, NEVER reassigned by resampling -- diversity metrics use this
    done: bool = False
    finish_reason: str | None = None  # "boxed" | "eos" | "max_particle_steps"
    final_answer: str | None = None

    @property
    def prefix_text(self) -> str:
        return "\n\n".join(self.step_texts)

    @property
    def n_steps(self) -> int:
        return len(self.step_texts)


def new_particle(prompt: str, lineage_id: int) -> Particle:
    """The t=0 particle: no steps yet, neutral log_psi (matches
    smc/weighting.py's t=0 convention -- logpsi=0.0 before any step)."""
    return Particle(prompt=prompt, lineage_id=lineage_id)
