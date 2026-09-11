"""Deterministic selection used by beam search (Stage 2, baseline #2) --
kept separate from smc/resampling.py's STOCHASTIC resampling schemes on
purpose: beam search is explicitly "a non-probabilistic search baseline"
(plan doc §6), and mixing in randomness or importance weights would
silently turn it into a different (weighted-beam) baseline.
"""

from __future__ import annotations

import numpy as np


def deterministic_topk_select(particles: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Keep the k particles with the highest `scores`, ties broken by
    array position (stable) so results are exactly reproducible given the
    same inputs -- no rng involved anywhere in this function.
    """
    if k > len(particles):
        raise ValueError(f"k={k} exceeds the number of particles ({len(particles)})")
    # argsort is stable (mergesort-equivalent for the default 'quicksort'
    # on ties is NOT guaranteed stable -- use kind="stable" explicitly).
    order = np.argsort(-np.asarray(scores), kind="stable")
    top_idx = order[:k]
    return particles[top_idx]
