from __future__ import annotations

from models.prm import MockPRMScorer, PRMScore


def test_score_batch_returns_one_prmscore_per_particle():
    scorer = MockPRMScorer()
    steps_per_particle = [
        ["step a1", "step a2"],
        ["step b1"],
        ["step c1", "step c2", "step c3"],
    ]
    results = scorer.score_batch("query", steps_per_particle)
    assert len(results) == 3
    assert all(isinstance(r, PRMScore) for r in results)
    assert [len(r.step_scores) for r in results] == [2, 1, 3]


def test_score_batch_matches_individual_score_calls():
    # score_batch must be equivalent to looping score() -- this is the
    # CPU-testable analogue of the real-GPU equivalence check
    # scripts/diagnose_prm.py runs against QwenMathPRMScorer.
    scorer = MockPRMScorer()
    steps_per_particle = [["alpha", "beta"], ["gamma"]]
    batched = scorer.score_batch("q", steps_per_particle)
    individual = [scorer.score("q", steps) for steps in steps_per_particle]
    assert [r.step_scores for r in batched] == [r.step_scores for r in individual]


def test_score_batch_handles_ragged_step_counts():
    scorer = MockPRMScorer()
    # Deliberately very ragged: 1, 5, 2 steps -- exercises the same
    # "different step counts per particle in one batch" shape the real
    # Sampler will hit every global step once particles start finishing
    # at different times.
    steps_per_particle = [["only step"], [f"step {i}" for i in range(5)], ["s1", "s2"]]
    results = scorer.score_batch("q", steps_per_particle)
    assert [len(r.step_scores) for r in results] == [1, 5, 2]


def test_score_batch_empty_particle_list_returns_empty():
    scorer = MockPRMScorer()
    assert scorer.score_batch("q", []) == []


def test_score_batch_is_deterministic():
    scorer = MockPRMScorer()
    steps_per_particle = [["a", "b"], ["c"]]
    r1 = scorer.score_batch("q", steps_per_particle)
    r2 = scorer.score_batch("q", steps_per_particle)
    assert [r.step_scores for r in r1] == [r.step_scores for r in r2]
