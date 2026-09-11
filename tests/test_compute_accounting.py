from __future__ import annotations

import numpy as np

from eval.compute_accounting import ComputeAccountant
from models.policy import MockPolicy
from models.prm import MockPRMScorer
from smc.controllers.fixed import FixedLinearController
from smc.resampling_rules import EssTriggeredResample
from smc.sampler_llm import Sampler


def test_record_methods_accumulate():
    acc = ComputeAccountant()
    acc.record_tokens_generated(10)
    acc.record_tokens_generated(5)
    acc.record_prm_forward(n_particles=4, n_steps_total=9)
    acc.record_resample_event({"ess_before": 1.0})
    acc.record_bisection_solve()

    assert acc.stats.n_tokens_generated == 15
    assert acc.stats.n_policy_calls == 2
    assert acc.stats.n_prm_forward_passes == 1
    assert acc.stats.n_prm_particles_scored == 4
    assert acc.stats.n_prm_steps_scored == 9
    assert acc.stats.n_resample_events == 1
    assert acc.stats.n_bisection_solves == 1


def test_as_dict_reflects_stats():
    acc = ComputeAccountant()
    acc.record_tokens_generated(3)
    d = acc.as_dict()
    assert d["n_tokens_generated"] == 3


def test_fixed_controller_flagged_as_not_using_bisection():
    assert FixedLinearController.uses_bisection is False


def test_sampler_wires_accountant_and_never_records_bisection_for_fixed_schedule():
    # End-to-end (mocked) sanity check: a full PF run must report nonzero
    # tokens/PRM calls, and exactly zero bisection solves since
    # FixedLinearController never calls solve_beta_step.
    acc = ComputeAccountant()
    sampler = Sampler(
        controller=FixedLinearController(n_steps=1),
        resampling_rule=EssTriggeredResample(gamma=0.5),
        policy=MockPolicy(seed=0),
        prm_scorer=MockPRMScorer(),
        accountant=acc,
    )
    rng = np.random.default_rng(0)
    sampler.run("Solve: 8+8", n_particles=4, rng=rng, horizon_steps=10)

    assert acc.stats.n_tokens_generated > 0
    assert acc.stats.n_policy_calls > 0
    assert acc.stats.n_prm_forward_passes > 0
    assert acc.stats.n_bisection_solves == 0
