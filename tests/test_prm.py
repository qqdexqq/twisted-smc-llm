from __future__ import annotations

from models.prm import MockPRMScorer, PRMScore, QwenMathPRMScorer


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


class TestQwenScoreBatchChunking:
    """CPU-testable check on JUST the chunking/splitting logic in
    QwenMathPRMScorer.score_batch() -- added after a real Kaggle OOM
    (vLLM + PRM both resident, a 16-particle batched forward pass over
    several steps' accumulated prefix text ran out of memory). Bypasses
    __init__ entirely (it needs torch/transformers/a real GPU) via
    object.__new__, then monkeypatches _score_batch_chunk to record what
    chunks it was actually called with -- verifies the chunking driver
    itself is correct without needing the real model.
    """

    def _make_scorer_with_stub_chunk_fn(self, calls: list[list[list[str]]]):
        scorer = object.__new__(QwenMathPRMScorer)  # skip __init__ (needs a real GPU)

        def stub_chunk(query, steps_per_particle, system):
            calls.append(steps_per_particle)
            return [PRMScore(step_scores=[0.5] * len(steps)) for steps in steps_per_particle]

        scorer._score_batch_chunk = stub_chunk
        scorer.tokenizer = type("FakeTok", (), {"pad_token_id": 0})()  # score_batch touches this
        return scorer

    def test_splits_into_chunks_of_max_batch_size(self):
        calls: list[list[list[str]]] = []
        scorer = self._make_scorer_with_stub_chunk_fn(calls)
        steps_per_particle = [[f"p{i} step"] for i in range(17)]  # 17 particles
        results = scorer.score_batch("q", steps_per_particle, max_batch_size=8)
        assert len(results) == 17
        assert [len(c) for c in calls] == [8, 8, 1]  # 17 = 8 + 8 + 1

    def test_single_chunk_when_under_max_batch_size(self):
        calls: list[list[list[str]]] = []
        scorer = self._make_scorer_with_stub_chunk_fn(calls)
        steps_per_particle = [["a"], ["b"], ["c"]]
        results = scorer.score_batch("q", steps_per_particle, max_batch_size=8)
        assert len(results) == 3
        assert len(calls) == 1
        assert len(calls[0]) == 3

    def test_chunking_preserves_particle_order(self):
        calls: list[list[list[str]]] = []
        scorer = self._make_scorer_with_stub_chunk_fn(calls)
        steps_per_particle = [[f"particle-{i}"] for i in range(10)]
        scorer.score_batch("q", steps_per_particle, max_batch_size=4)
        flattened = [steps for chunk in calls for steps in chunk]
        assert flattened == steps_per_particle

    def test_empty_input_makes_no_calls(self):
        calls: list[list[list[str]]] = []
        scorer = self._make_scorer_with_stub_chunk_fn(calls)
        results = scorer.score_batch("q", [], max_batch_size=8)
        assert results == []
        assert calls == []
