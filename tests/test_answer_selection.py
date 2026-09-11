from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eval.answer_selection import diagnostic_selections, select_final_answer
from smc.llm_particle import new_particle


def _particle(lineage_id, log_psi_cumulative, final_answer):
    p = new_particle("q", lineage_id=lineage_id)
    return dataclasses.replace(p, log_psi_cumulative=log_psi_cumulative, done=True, final_answer=final_answer)


def test_selects_argmax_log_psi_not_log_W():
    particles = np.array(
        [
            _particle(0, log_psi_cumulative=-2.0, final_answer="1"),
            _particle(1, log_psi_cumulative=-0.1, final_answer="2"),  # highest log_psi
            _particle(2, log_psi_cumulative=-1.0, final_answer="3"),
        ],
        dtype=object,
    )
    result = select_final_answer(particles)
    assert result.final_answer == "2"
    assert result.selected_index == 1
    assert result.method == "argmax_log_psi"


def test_ignores_log_W_entirely():
    # log_W (not passed to select_final_answer at all) heavily favors
    # particle 0, but log_psi_cumulative favors particle 1 -- the
    # function signature itself enforces log_W can't leak in.
    particles = np.array(
        [
            _particle(0, log_psi_cumulative=-5.0, final_answer="wrong"),
            _particle(1, log_psi_cumulative=-0.01, final_answer="right"),
        ],
        dtype=object,
    )
    result = select_final_answer(particles)
    assert result.final_answer == "right"


def test_ties_broken_toward_lowest_index():
    particles = np.array(
        [
            _particle(0, log_psi_cumulative=0.5, final_answer="a"),
            _particle(1, log_psi_cumulative=0.5, final_answer="b"),
        ],
        dtype=object,
    )
    result = select_final_answer(particles)
    assert result.selected_index == 0
    assert result.final_answer == "a"


def test_particle_with_no_boxed_answer_returns_none():
    particles = np.array([_particle(0, log_psi_cumulative=0.9, final_answer=None)], dtype=object)
    result = select_final_answer(particles)
    assert result.final_answer is None


def test_empty_particle_set_raises():
    with pytest.raises(ValueError):
        select_final_answer(np.array([], dtype=object))


def test_diagnostic_majority_and_weighted_majority():
    particles = np.array(
        [
            _particle(0, log_psi_cumulative=0.1, final_answer="42"),
            _particle(1, log_psi_cumulative=0.2, final_answer="42"),
            _particle(2, log_psi_cumulative=0.3, final_answer="7"),
        ],
        dtype=object,
    )
    log_W = np.log(np.array([0.1, 0.1, 0.8]))  # weighted majority should favor "7" despite plain majority favoring "42"
    diag = diagnostic_selections(particles, log_W)
    assert diag.majority_answer == "42"
    assert diag.weighted_majority_answer == "7"


def test_diagnostic_selections_all_none_when_no_answers():
    particles = np.array([_particle(0, log_psi_cumulative=0.1, final_answer=None)], dtype=object)
    log_W = np.array([0.0])
    diag = diagnostic_selections(particles, log_W)
    assert diag.majority_answer is None
    assert diag.weighted_majority_answer is None
