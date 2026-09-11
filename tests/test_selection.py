from __future__ import annotations

import numpy as np
import pytest

from smc.selection import deterministic_topk_select


def test_keeps_top_k_by_score():
    particles = np.array(["a", "b", "c", "d"], dtype=object)
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    kept = deterministic_topk_select(particles, scores, k=2)
    assert list(kept) == ["b", "c"]  # highest (0.9), then next (0.5)


def test_k_equals_n_keeps_everyone_reordered_by_score():
    particles = np.array(["a", "b", "c"], dtype=object)
    scores = np.array([0.2, 0.8, 0.5])
    kept = deterministic_topk_select(particles, scores, k=3)
    assert list(kept) == ["b", "c", "a"]


def test_is_deterministic_no_randomness_involved():
    particles = np.array(list(range(10)), dtype=object)
    scores = np.array([float(i % 3) for i in range(10)])  # lots of ties
    kept1 = deterministic_topk_select(particles, scores, k=5)
    kept2 = deterministic_topk_select(particles, scores, k=5)
    np.testing.assert_array_equal(kept1, kept2)


def test_ties_broken_by_original_position_stably():
    particles = np.array(["first", "second", "third"], dtype=object)
    scores = np.array([1.0, 1.0, 1.0])  # all tied
    kept = deterministic_topk_select(particles, scores, k=2)
    # Stable sort on a full tie preserves original order.
    assert list(kept) == ["first", "second"]


def test_k_greater_than_n_raises():
    particles = np.array(["a", "b"], dtype=object)
    scores = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        deterministic_topk_select(particles, scores, k=5)
