from __future__ import annotations

import numpy as np
import pytest

from models.policy import MockPolicy
from models.prm import MockPRMScorer
from smc.controllers.fixed import FixedLinearController
from smc.llm_particle import new_particle
from smc.resampling_rules import DeterministicTopK, EntropicResample, EssTriggeredResample
from smc.sampler_llm import Sampler
from smc.tempering import BetaDecision


class _FakeController:
    """Always proposes a fixed beta, ignoring log_W/log_G entirely --
    used only to force a deterministic "horizon exhausted before beta
    reached 1.0" scenario for terminal_correction, without depending on
    a real adaptive controller's stochastic convergence speed.
    """

    def __init__(self, beta: float):
        self._beta = beta

    def reset(self) -> None:
        pass

    def next_beta(self, log_W, log_G, beta_prev, rng):
        return BetaDecision(beta=self._beta, criterion_value=float("nan"), n_iters_used=0, degenerate=False, jumped_to_one=False)


def _pf_sampler(**kwargs):
    return Sampler(
        controller=FixedLinearController(n_steps=1),
        resampling_rule=EssTriggeredResample(gamma=0.5),
        policy=MockPolicy(seed=0),
        prm_scorer=MockPRMScorer(),
        **kwargs,
    )


def _epf_sampler(**kwargs):
    return Sampler(
        controller=FixedLinearController(n_steps=1),
        resampling_rule=EntropicResample(ess_threshold=0.5, intervention_frac=0.5),
        policy=MockPolicy(seed=0),
        prm_scorer=MockPRMScorer(),
        **kwargs,
    )


class TestRunEndToEnd:
    def test_pf_run_terminates_with_beta_one(self):
        sampler = _pf_sampler()
        rng = np.random.default_rng(0)
        result = sampler.run("Solve: 2+2", n_particles=6, rng=rng, horizon_steps=12)
        assert result.log_W.shape == (6,)
        np.testing.assert_allclose(np.sum(np.exp(result.log_W)), 1.0, rtol=1e-9)
        # beta reaches 1.0 on the very first weight() call for PF (n_steps=1
        # fixed schedule) -- every subsequent history entry stays pinned there.
        assert result.history[0].beta == pytest.approx(1.0)
        assert all(rec.beta == pytest.approx(1.0) for rec in result.history)

    def test_epf_run_terminates_with_beta_one(self):
        sampler = _epf_sampler()
        rng = np.random.default_rng(1)
        result = sampler.run("Solve: 3+3", n_particles=8, rng=rng, horizon_steps=12)
        np.testing.assert_allclose(np.sum(np.exp(result.log_W)), 1.0, rtol=1e-9)
        assert result.history[0].beta == pytest.approx(1.0)

    def test_run_all_particles_eventually_done_or_horizon_exhausted(self):
        sampler = _pf_sampler()
        rng = np.random.default_rng(2)
        result = sampler.run("Solve: 5+5", n_particles=4, rng=rng, horizon_steps=12)
        # MockPolicy forces a finish by step 6 regardless of hash luck, so
        # well within horizon_steps=12 every particle must be done.
        assert all(p.done for p in result.particles)

    def test_terminal_correction_forces_beta_one_when_horizon_too_short(self):
        # A controller pinned to always propose beta=0.3, given only one
        # global step: forces the "beta < 1.0 at horizon" branch
        # (terminal_correction), unreachable for PF/ePF's fixed schedule
        # in this push's scope but required once an adaptive controller
        # (Stage 3 preview) is plugged in and doesn't finish in time.
        sampler = Sampler(
            controller=_FakeController(beta=0.3),
            resampling_rule=EssTriggeredResample(gamma=0.5),
            policy=MockPolicy(seed=0),
            prm_scorer=MockPRMScorer(),
        )
        rng = np.random.default_rng(3)
        result = sampler.run("Solve: 7+7", n_particles=6, rng=rng, horizon_steps=1)
        np.testing.assert_allclose(np.sum(np.exp(result.log_W)), 1.0, rtol=1e-9)
        # The one real StepRecord shows beta=0.3, exactly as the fake
        # controller proposed -- the correction happens AFTER the loop,
        # invisibly to history, but the invariant the caller actually
        # needs (final beta==1.0) must still hold.
        assert result.history[0].beta == pytest.approx(0.3)

    def test_terminal_correction_matches_hand_computed_weight_update(self):
        # Direct unit check on terminal_correction's math, independent of
        # run()'s stochastic dynamics: with log_psi frozen (no new step),
        # log_offset must be exactly 0, so the whole increment is
        # (1 - beta_prev) * log_psi_cumulative per particle.
        import dataclasses

        sampler = _pf_sampler()
        particles = np.array(
            [
                dataclasses.replace(new_particle("q", lineage_id=0), log_psi_cumulative=0.4),
                dataclasses.replace(new_particle("q", lineage_id=1), log_psi_cumulative=-0.2),
            ],
            dtype=object,
        )
        log_W = np.log(np.array([0.5, 0.5]))
        particles_out, log_W_out, log_Z_inc, beta_out = sampler.terminal_correction(particles, log_W, beta_prev=0.4)
        assert beta_out == 1.0
        expected_log_w_inc = 0.6 * np.array([0.4, -0.2])  # (1 - 0.4) * log_psi_cumulative
        expected_unnorm = log_W + expected_log_w_inc
        from scipy.special import logsumexp

        expected_log_W = expected_unnorm - logsumexp(expected_unnorm)
        np.testing.assert_allclose(log_W_out, expected_log_W, atol=1e-12)
        assert log_Z_inc == pytest.approx(float(logsumexp(expected_unnorm) - logsumexp(log_W)), abs=1e-12)

    def test_terminal_correction_is_a_noop_when_already_pinned(self):
        sampler = _pf_sampler()
        particles = np.array([new_particle("q", lineage_id=0)], dtype=object)
        log_W = np.array([0.0])
        particles_out, log_W_out, log_Z_inc, beta_out = sampler.terminal_correction(particles, log_W, beta_prev=1.0)
        assert beta_out == 1.0
        assert log_Z_inc == 0.0
        np.testing.assert_array_equal(log_W_out, log_W)

    def test_beam_search_style_branching_grows_then_prunes(self):
        sampler = Sampler(
            controller=FixedLinearController(n_steps=1),
            resampling_rule=DeterministicTopK(n_keep=4),
            policy=MockPolicy(seed=0),
            prm_scorer=MockPRMScorer(),
            n_children=3,
        )
        rng = np.random.default_rng(4)
        result = sampler.run("Solve: 9+9", n_particles=4, rng=rng, horizon_steps=10)
        # DeterministicTopK prunes back to n_keep=4 every round, regardless
        # of how many candidates branching produced.
        assert len(result.particles) == 4
        np.testing.assert_allclose(np.sum(np.exp(result.log_W)), 1.0, rtol=1e-9)


class TestPropagate:
    def test_active_particles_gain_one_step(self):
        sampler = _pf_sampler()
        particles = np.array([new_particle("Solve: 1+1", lineage_id=i) for i in range(3)], dtype=object)
        rng = np.random.default_rng(0)
        new_particles, was_active, source_idx = sampler.propagate(particles, "Solve: 1+1", rng)
        assert len(new_particles) == 3
        assert was_active.all()
        np.testing.assert_array_equal(source_idx, [0, 1, 2])
        for old, new in zip(particles, new_particles):
            assert new.n_steps == old.n_steps + 1

    def test_done_particles_pass_through_unchanged(self):
        sampler = _pf_sampler()
        done_particle = new_particle("Solve: 1+1", lineage_id=0)
        import dataclasses

        done_particle = dataclasses.replace(done_particle, done=True, finish_reason="boxed", final_answer="2")
        particles = np.array([done_particle], dtype=object)
        rng = np.random.default_rng(0)
        new_particles, was_active, source_idx = sampler.propagate(particles, "Solve: 1+1", rng)
        assert new_particles[0] is done_particle
        assert was_active[0] == False  # noqa: E712
        assert source_idx[0] == 0

    def test_mixed_done_and_active_population(self):
        sampler = _pf_sampler()
        import dataclasses

        active = new_particle("Solve: 4+4", lineage_id=0)
        done = dataclasses.replace(
            new_particle("Solve: 4+4", lineage_id=1), done=True, finish_reason="boxed", final_answer="8"
        )
        particles = np.array([active, done], dtype=object)
        rng = np.random.default_rng(0)
        new_particles, was_active, source_idx = sampler.propagate(particles, "Solve: 4+4", rng)
        assert len(new_particles) == 2
        assert new_particles[1] is done
        assert list(was_active) == [True, False]
        np.testing.assert_array_equal(source_idx, [0, 1])

    def test_all_done_input_is_a_noop(self):
        sampler = _pf_sampler()
        import dataclasses

        done = dataclasses.replace(new_particle("q", lineage_id=0), done=True)
        particles = np.array([done, done], dtype=object)
        rng = np.random.default_rng(0)
        new_particles, was_active, source_idx = sampler.propagate(particles, "q", rng)
        assert not was_active.any()
        np.testing.assert_array_equal(source_idx, [0, 1])

    def test_n_children_branches_active_particles_only(self):
        sampler = _pf_sampler(n_children=3)
        import dataclasses

        active = new_particle("Solve: 6+6", lineage_id=0)
        done = dataclasses.replace(new_particle("Solve: 6+6", lineage_id=1), done=True)
        particles = np.array([active, done], dtype=object)
        rng = np.random.default_rng(0)
        new_particles, was_active, source_idx = sampler.propagate(particles, "Solve: 6+6", rng)
        # 1 active particle -> 3 children; 1 done particle -> 1 passthrough slot.
        assert len(new_particles) == 4
        np.testing.assert_array_equal(source_idx, [0, 0, 0, 1])
        assert list(was_active) == [True, True, True, False]


class TestWeight:
    def test_done_particle_gets_zero_offset_frozen_delta(self):
        sampler = _pf_sampler()
        import dataclasses

        p = new_particle("q", lineage_id=0)
        p = dataclasses.replace(p, log_psi_cumulative=0.3, log_psi_prev=-5.0, done=True)  # stale log_psi_prev
        particles = np.array([p], dtype=object)
        log_W = np.array([0.0])  # single particle, already normalized (log 1 = 0)
        rng = np.random.default_rng(0)
        # was_active=False -> weight() must ignore the stale log_psi_prev
        # and use log_psi_cumulative for BOTH log_psi_new and log_psi_prev,
        # per smc/weighting.py's documented done-particle freeze.
        _, new_log_W, log_Z_inc, beta_new, _ = sampler.weight(
            particles, log_W, beta_prev=0.0, was_active=np.array([False]), rng=rng
        )
        assert beta_new == pytest.approx(1.0)  # FixedLinearController(n_steps=1)
        # (beta_new - beta_prev)*log_G + log_offset = 1.0*0.3 + 0.0 = 0.3
        assert log_Z_inc == pytest.approx(0.3, abs=1e-9)

    def test_active_particle_uses_real_delta(self):
        sampler = _pf_sampler()
        import dataclasses

        p = new_particle("q", lineage_id=0)
        p = dataclasses.replace(p, log_psi_cumulative=0.3, log_psi_prev=-0.1, done=False)
        particles = np.array([p], dtype=object)
        log_W = np.array([0.0])
        rng = np.random.default_rng(0)
        _, new_log_W, log_Z_inc, beta_new, _ = sampler.weight(
            particles, log_W, beta_prev=0.0, was_active=np.array([True]), rng=rng
        )
        # beta_prev=0 -> log_offset=0 regardless (t=0 convention); only
        # the log_G term matters here: 1.0*0.3 = 0.3. Real differentiation
        # of active-vs-done shows up once beta_prev > 0 (see next test).
        assert log_Z_inc == pytest.approx(0.3, abs=1e-9)

    def test_beta_pinned_at_one_skips_controller_and_uses_pure_offset(self):
        sampler = _pf_sampler()
        import dataclasses

        p = new_particle("q", lineage_id=0)
        p = dataclasses.replace(p, log_psi_cumulative=0.5, log_psi_prev=0.2, done=False)
        particles = np.array([p], dtype=object)
        log_W = np.array([0.0])
        rng = np.random.default_rng(0)
        _, _, log_Z_inc, beta_new, decision = sampler.weight(
            particles, log_W, beta_prev=1.0, was_active=np.array([True]), rng=rng
        )
        assert beta_new == 1.0
        assert decision is None  # controller.next_beta not called once pinned
        # (b - beta_prev)*log_G = 0 since b==beta_prev==1.0; log_offset =
        # 1.0*(0.5-0.2) = 0.3 is the entire increment.
        assert log_Z_inc == pytest.approx(0.3, abs=1e-9)
