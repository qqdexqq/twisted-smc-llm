"""Answer selection, held CONSTANT across all four Stage 2 baselines (plan
doc §7.1: "argmax under the PRM"). This is the one place that decision
gets made -- every method's own accuracy number must come from calling
this same function on its SamplerResult, never from ad hoc per-method
logic, or a numeric difference between methods could just be an artifact
of how each one happened to pick its final answer instead of a genuine
sampling-quality difference (the "classic way to accidentally manufacture
a result" §7.1 warns against).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnswerSelection:
    final_answer: str | None
    selected_index: int
    method: str  # "argmax_log_psi" -- the one reported number; others are diagnostic-only


def select_final_answer(particles: np.ndarray) -> AnswerSelection:
    """argmax_i particle[i].log_psi_cumulative over the FINAL particle set
    -- each particle's own terminal PRM score, NOT log_W. Deliberately
    invariant to each method's own weight-bookkeeping quirks: PF resets
    log_W to uniform on every resample, DeterministicTopK carries no
    importance weight at all, CESS-style methods carry non-uniform weight
    longer -- using log_W instead of log_psi_cumulative would silently
    reward whichever method happens to keep the most informative log_W
    around at the end, not whichever method finds the best answer.

    A particle that never produced a \\boxed{} answer (forced-done at
    max_particle_steps with no extractable boxed text) has
    final_answer=None -- ties in log_psi_cumulative are broken toward the
    lowest index (np.argmax's own convention), matching
    smc/selection.py::deterministic_topk_select's stable-tie-break
    philosophy: reproducible, not silently randomized.
    """
    if len(particles) == 0:
        raise ValueError("cannot select a final answer from an empty particle set")
    scores = np.array([p.log_psi_cumulative for p in particles])
    idx = int(np.argmax(scores))
    return AnswerSelection(final_answer=particles[idx].final_answer, selected_index=idx, method="argmax_log_psi")


@dataclass(frozen=True)
class DiagnosticSelection:
    """Weighted-vote / majority-vote variants -- logged as diagnostics
    ONLY (plan doc §7.1), never used for the reported accuracy number.
    Useful for sanity-checking select_final_answer isn't a fluke of one
    weird particle, not for reporting.
    """

    majority_answer: str | None
    weighted_majority_answer: str | None


def diagnostic_selections(particles: np.ndarray, log_W: np.ndarray) -> DiagnosticSelection:
    answers = [p.final_answer for p in particles if p.final_answer is not None]
    majority_answer = Counter(answers).most_common(1)[0][0] if answers else None

    weighted_votes: dict[str, float] = {}
    for p, lw in zip(particles, log_W):
        if p.final_answer is None:
            continue
        weighted_votes[p.final_answer] = weighted_votes.get(p.final_answer, 0.0) + float(np.exp(lw))
    weighted_majority_answer = max(weighted_votes, key=weighted_votes.get) if weighted_votes else None

    return DiagnosticSelection(majority_answer=majority_answer, weighted_majority_answer=weighted_majority_answer)
