from __future__ import annotations

import dataclasses

import pytest

from smc.llm_particle import Particle, new_particle


def test_new_particle_defaults():
    p = new_particle("What is 2+2?", lineage_id=5)
    assert p.prompt == "What is 2+2?"
    assert p.step_texts == ()
    assert p.log_psi_cumulative == 0.0
    assert p.log_psi_prev == 0.0
    assert p.lineage_id == 5
    assert p.done is False
    assert p.final_answer is None
    assert p.n_steps == 0
    assert p.prefix_text == ""


def test_prefix_text_joins_with_blank_line():
    p = Particle(prompt="q", step_texts=("first step.", "second step."))
    assert p.prefix_text == "first step.\n\nsecond step."
    assert p.n_steps == 2


def test_particle_is_frozen():
    p = new_particle("q", lineage_id=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.done = True  # type: ignore[misc]


def test_particle_step_texts_is_a_tuple_not_list():
    # Guards the specific aliasing-safety property called out in the
    # module docstring: a mutable list field would let .append() corrupt
    # every resampled sibling in place, bypassing frozen=True entirely.
    p = new_particle("q", lineage_id=0)
    assert isinstance(p.step_texts, tuple)


def test_extending_steps_via_replace_creates_new_object():
    p1 = new_particle("q", lineage_id=0)
    p2 = dataclasses.replace(p1, step_texts=p1.step_texts + ("a new step.",))
    assert p1.step_texts == ()  # original untouched
    assert p2.step_texts == ("a new step.",)
    assert p1 is not p2


def test_resampled_duplicates_share_identity_safely():
    # Simulates what systematic_resample produces: multiple array slots
    # referencing the SAME object. Since Particle is immutable, this is
    # safe -- confirm no shared-list-mutation footgun exists.
    p = new_particle("q", lineage_id=3)
    duplicates = [p, p, p]
    assert duplicates[0] is duplicates[1] is duplicates[2]
    # "Advancing" one duplicate must create a new object, never touch the shared one.
    advanced = dataclasses.replace(duplicates[0], step_texts=("new step.",))
    assert duplicates[1].step_texts == ()
    assert duplicates[2].step_texts == ()
    assert advanced.step_texts == ("new step.",)
