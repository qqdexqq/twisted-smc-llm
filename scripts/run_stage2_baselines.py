#!/usr/bin/env python
"""Stage 2: run one of the four baseline samplers (PF / beam / twisted-SMC
-fixed / ePF) over a MATH500-style manifest, via the shared
smc/sampler_llm.py::Sampler driver (plan doc §6). Which baseline you get
is entirely a function of --method's (ScheduleController, ResamplingRule,
n_children) mapping -- see build_method_config below; Sampler itself has
no per-method branching.

Answer selection is held CONSTANT across all four methods
(eval/answer_selection.py's argmax-log-psi rule, doc §7.1) so a numeric
difference between methods reflects sampling quality, not selection-rule
cherry-picking.

Checkpointed at PER-GLOBAL-SMC-STEP granularity (finer than Stage 1's
per-rollout checkpointing, because one MATH500 problem here can run
dozens of SMC steps and losing all of it to a crash would repeat the
exact problem Stage 1 already hit once): one file per in-progress
problem, results/stage2/{method}/seed={seed}/{problem_id}.ckpt.json,
overwritten via write-to-.tmp + fsync + atomic rename after every global
step (not append-only like Stage 1's rollout log -- the full particle
array must be reconstructable, not just accumulated). On resume: skip a
problem entirely if its .result.json exists; resume mid-run from
.ckpt.json if present; otherwise start fresh. --fresh ignores any
existing checkpoint/result for this (method, seed) and starts over.

--dry-run swaps in MockPolicy/MockPRMScorer so the whole orchestration --
manifest loading, per-step checkpointing, resume-after-simulated-crash,
answer selection, compute accounting, Parquet aggregation -- is verified
on a CPU-only machine before ever touching a GPU (per every prior stage's
convention in this repo). Real generation/scoring is UNVERIFIED until run
for real.

Usage:
    python scripts/run_stage2_baselines.py --method pf \\
        --manifest manifests/math500_128.jsonl --dry-run --limit 3

    python scripts/run_stage2_baselines.py --method epf \\
        --manifest manifests/math500_128.jsonl \\
        --generator Qwen/Qwen2.5-1.5B-Instruct --prm Qwen/Qwen2.5-Math-PRM-7B \\
        --prm-type qwen --n-particles 16 --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from eval.answer_selection import diagnostic_selections, select_final_answer
from eval.compute_accounting import ComputeAccountant
from eval.metrics import check_correct
from models.policy import MockPolicy, VLLMPolicy
from models.prm import MockPRMScorer, QwenMathPRMScorer, RLHFlowLlamaPRMScorer
from smc.controllers import ScheduleController
from smc.controllers.fixed import FixedLinearController
from smc.llm_particle import Particle, new_particle
from smc.resampling_rules import DeterministicTopK, EntropicResample, EssTriggeredResample, ResamplingRule
from smc.sampler import effective_sample_size
from smc.sampler_llm import Sampler, StepRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "stage2"

_METHODS = ("pf", "beam", "twisted_smc", "epf")


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _safe_problem_id(problem_id: str) -> str:
    # problem_ids in these manifests contain literal "/" (e.g.
    # "test/algebra/2584.json") -- unsafe as a bare filename component,
    # same fix generate_rollouts.py applies to model ids.
    return problem_id.replace("/", "__")


def _atomic_write_json(path: Path, obj: dict) -> None:
    """write-to-.tmp + fsync + atomic rename, per the plan's checkpointing
    design -- a crash mid-write leaves the PREVIOUS .ckpt.json (or none)
    intact, never a half-written file that would corrupt resume.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)  # atomic on both POSIX and Windows


def _particle_to_dict(p: Particle) -> dict:
    return {
        "prompt": p.prompt,
        "step_texts": list(p.step_texts),
        "log_psi_cumulative": p.log_psi_cumulative,
        "log_psi_prev": p.log_psi_prev,
        "lineage_id": p.lineage_id,
        "done": p.done,
        "finish_reason": p.finish_reason,
        "final_answer": p.final_answer,
    }


def _particle_from_dict(d: dict) -> Particle:
    return Particle(
        prompt=d["prompt"],
        step_texts=tuple(d["step_texts"]),
        log_psi_cumulative=d["log_psi_cumulative"],
        log_psi_prev=d["log_psi_prev"],
        lineage_id=d["lineage_id"],
        done=d["done"],
        finish_reason=d["finish_reason"],
        final_answer=d["final_answer"],
    )


def build_method_config(method: str, args: argparse.Namespace) -> tuple[ScheduleController, ResamplingRule, int]:
    """The doc's baseline-mapping table (plan file): which
    ScheduleController/ResamplingRule/n_children triple gives you which
    of the four baselines. Sampler itself is identical across all four.
    """
    if method == "pf":
        return FixedLinearController(n_steps=1), EssTriggeredResample(gamma=args.gamma), 1
    if method == "epf":
        return (
            FixedLinearController(n_steps=1),
            EntropicResample(ess_threshold=args.ess_threshold, intervention_frac=args.intervention_frac),
            1,
        )
    if method == "twisted_smc":
        # NOTE (deferred, per the plan's scope note): --twisted-n-steps
        # defaults to a placeholder, not yet calibrated from the existing
        # Stage-1 MATH500 rollout corpus's realized n_steps distribution
        # (mean ~8.6) as the plan specifies -- structural wiring only,
        # real-GPU verification is a follow-up after PF/ePF pass the gate.
        return FixedLinearController(n_steps=args.twisted_n_steps), EssTriggeredResample(gamma=args.gamma), 1
    if method == "beam":
        return FixedLinearController(n_steps=1), DeterministicTopK(n_keep=args.n_particles), args.branch_factor
    raise ValueError(f"unknown method {method!r}, expected one of {_METHODS}")


def build_policy(dry_run: bool, generator: str, quantization: str | None):
    return MockPolicy(seed=0) if dry_run else VLLMPolicy(generator, quantization=quantization)


def build_prm_scorer(prm_type: str, prm_model: str, dry_run: bool, load_in_8bit: bool = False):
    if dry_run:
        return MockPRMScorer()
    if prm_type == "qwen":
        return QwenMathPRMScorer(prm_model, load_in_8bit=load_in_8bit)
    if prm_type == "rlhflow_llama":
        if load_in_8bit:
            raise NotImplementedError("8-bit loading not wired up for RLHFlowLlamaPRMScorer yet")
        return RLHFlowLlamaPRMScorer(prm_model)
    raise ValueError(f"unknown prm_type {prm_type!r}")


def _rng_for_problem(problem_id: str, seed: int) -> np.random.Generator:
    # Deterministic per-(problem, seed) RNG, independent of manifest
    # ordering or how many OTHER problems have already been processed --
    # important for resumability: a resumed problem's remaining steps use
    # a fresh Generator seeded the same way a from-scratch run would,
    # not a continuation of some global stream whose position depends on
    # unrelated prior problems.
    import hashlib

    h = hashlib.sha256(f"{problem_id}|{seed}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def run_one_problem(
    sampler: Sampler,
    problem: dict,
    n_particles: int,
    horizon_steps: int,
    seed: int,
    ckpt_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, list[StepRecord]]:
    """Drives Sampler's propagate/weight/resample loop directly (rather
    than calling Sampler.run() as one opaque call) so a checkpoint can be
    written after every single global step -- see module docstring.
    """
    rng = _rng_for_problem(problem["problem_id"], seed)

    if ckpt_path.exists():
        state = json.loads(ckpt_path.read_text(encoding="utf-8"))
        t_start = state["t"]
        beta = state["beta"]
        log_Z_hat = state["log_Z_hat"]
        log_W = np.array(state["log_W"], dtype=float)
        particles = np.array([_particle_from_dict(d) for d in state["particles"]], dtype=object)
        history = [StepRecord(**h) for h in state["history"]]
        print(f"  resuming {problem['problem_id']} from checkpointed step {t_start}", flush=True)
    else:
        t_start = 0
        beta = 0.0
        log_Z_hat = 0.0
        particles = np.array(
            [new_particle(problem["prompt"], lineage_id=i) for i in range(n_particles)], dtype=object
        )
        log_W = np.full(n_particles, -np.log(n_particles))
        history = []

    sampler.controller.reset()
    if t_start > 0 and isinstance(sampler.controller, FixedLinearController):
        # FixedLinearController's ONLY internal state is a step counter
        # (next_beta ignores log_W/log_G entirely) -- resync it directly
        # rather than replaying t_start dummy calls. The ESS/CESS-adaptive
        # controllers used elsewhere in this repo are stateless
        # (reset() is a no-op for both), so this is the one case that
        # needs it; generalize if a stateful controller is added later.
        sampler.controller._step = t_start

    for t in range(t_start + 1, horizon_steps + 1):
        if all(p.done for p in particles):
            break

        particles, was_active, source_idx = sampler.propagate(particles, problem["prompt"], rng)
        log_W = log_W[source_idx]

        particles, log_W, log_Z_inc, beta_new, _decision = sampler.weight(particles, log_W, beta, was_active, rng)
        log_Z_hat += log_Z_inc
        ess_before_resample = effective_sample_size(log_W)

        t_frac = t / horizon_steps
        particles, log_W, resampled, diag = sampler.resample(particles, log_W, t, t_frac, rng)

        history.append(
            StepRecord(
                step=t,
                beta_prev=beta,
                beta=beta_new,
                ess_before_resample=ess_before_resample,
                n_active=int(was_active.sum()),
                resampled=resampled,
                resample_diag=diag,
            )
        )
        beta = beta_new

        _atomic_write_json(
            ckpt_path,
            {
                "problem_id": problem["problem_id"],
                "t": t,
                "beta": beta,
                "log_Z_hat": log_Z_hat,
                "log_W": log_W.tolist(),
                "particles": [_particle_to_dict(p) for p in particles],
                "history": [dataclasses.asdict(h) for h in history],
            },
        )

    if beta < 1.0:
        particles, log_W, log_Z_inc, beta = sampler.terminal_correction(particles, log_W, beta)
        log_Z_hat += log_Z_inc

    assert abs(beta - 1.0) < 1e-9, f"terminal beta={beta}, expected exactly 1.0"
    return particles, log_W, log_Z_hat, history


def run(args: argparse.Namespace) -> dict:
    dataset_name = args.dataset_name or args.manifest.stem
    problems = load_manifest(args.manifest)
    if args.limit is not None:
        problems = problems[: args.limit]

    out_dir = RESULTS_DIR / dataset_name / args.method / f"seed={args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for f in out_dir.glob("*.ckpt.json"):
            f.unlink()
        for f in out_dir.glob("*.result.json"):
            f.unlink()

    policy = build_policy(args.dry_run, args.generator, args.quantization)
    if not args.dry_run:
        import gc

        import torch

    prm = None  # loaded lazily below, AFTER checking what's actually left to do

    n_already_done = sum(1 for p in problems if (out_dir / f"{_safe_problem_id(p['problem_id'])}.result.json").exists())
    remaining = [p for p in problems if not (out_dir / f"{_safe_problem_id(p['problem_id'])}.result.json").exists()]
    if n_already_done:
        print(f"{n_already_done}/{len(problems)} problems already have a .result.json, skipping those")

    if remaining:
        prm = build_prm_scorer(args.prm_type, args.prm, args.dry_run, load_in_8bit=args.prm_8bit)

        controller, resampling_rule, n_children = build_method_config(args.method, args)
        accountant = ComputeAccountant()
        sampler = Sampler(
            controller=controller,
            resampling_rule=resampling_rule,
            policy=policy,
            prm_scorer=prm,
            n_children=n_children,
            max_particle_steps=args.max_particle_steps,
            max_tokens_per_step=args.max_tokens_per_step,
            temperature=args.temperature,
            accountant=accountant,
        )

        for i, problem in enumerate(remaining):
            safe_id = _safe_problem_id(problem["problem_id"])
            ckpt_path = out_dir / f"{safe_id}.ckpt.json"
            result_path = out_dir / f"{safe_id}.result.json"

            accountant.stats = type(accountant.stats)()  # fresh per-problem compute stats
            particles, log_W, log_Z_hat, history = run_one_problem(
                sampler, problem, args.n_particles, args.horizon_steps, args.seed, ckpt_path
            )

            selection = select_final_answer(particles)
            diag_sel = diagnostic_selections(particles, log_W)
            is_correct = (
                check_correct(selection.final_answer, problem["gold_answer"])
                if selection.final_answer is not None
                else False
            )

            result_row = {
                "problem_id": problem["problem_id"],
                "method": args.method,
                "seed": args.seed,
                "final_answer": selection.final_answer,
                "is_correct": is_correct,
                "gold_answer": problem["gold_answer"],
                "majority_answer": diag_sel.majority_answer,
                "weighted_majority_answer": diag_sel.weighted_majority_answer,
                "log_Z_hat": log_Z_hat,
                "n_global_steps": len(history),
                "n_particles": args.n_particles,
                "horizon_steps": args.horizon_steps,
                **{f"compute_{k}": v for k, v in accountant.as_dict().items()},
            }
            _atomic_write_json(result_path, result_row)
            if ckpt_path.exists():
                ckpt_path.unlink()  # superseded by the .result.json

            print(
                f"[{n_already_done + i + 1}/{len(problems)}] {problem['problem_id']}: "
                f"answer={selection.final_answer!r} correct={is_correct} "
                f"steps={len(history)} tokens={accountant.stats.n_tokens_generated}",
                flush=True,
            )

        if not args.dry_run:
            del policy, prm
            gc.collect()
            torch.cuda.empty_cache()

    return _aggregate_results(out_dir, args.method, args.seed, len(problems))


def _aggregate_results(out_dir: Path, method: str, seed: int, n_problems: int) -> dict:
    import pandas as pd

    rows = []
    for f in sorted(out_dir.glob("*.result.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "results.parquet", index=False)

    accuracy = float(df["is_correct"].mean()) if len(df) else float("nan")
    return {
        "method": method,
        "seed": seed,
        "n_problems": n_problems,
        "n_results": len(df),
        "accuracy": accuracy,
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=_METHODS)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prm", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--prm-type", default="qwen", choices=["qwen", "rlhflow_llama"])
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, default=64)
    parser.add_argument("--max-particle-steps", type=int, default=300)
    parser.add_argument("--max-tokens-per-step", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.5, help="ESS-triggered resample threshold (PF, twisted_smc)")
    parser.add_argument("--ess-threshold", type=float, default=0.5, help="ePF's own ESS threshold")
    parser.add_argument("--intervention-frac", type=float, default=0.5, help="ePF's annealing-window fraction")
    parser.add_argument(
        "--twisted-n-steps", type=int, default=10, help="twisted_smc's fixed schedule length (UNCALIBRATED placeholder)"
    )
    parser.add_argument("--branch-factor", type=int, default=4, help="beam search's children per active beam")
    parser.add_argument("--dataset-name", default=None, help="defaults to the manifest filename stem")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N problems")
    parser.add_argument("--dry-run", action="store_true", help="use CPU-only mocks, no GPU/model needed")
    parser.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint/result and start over")
    parser.add_argument("--quantization", default=None, help="generator quantization, e.g. 'bitsandbytes'")
    parser.add_argument("--prm-8bit", action="store_true", help="load the PRM in 8-bit (needs bitsandbytes)")
    args = parser.parse_args()

    summary = run(args)
    print(
        f"\n{summary['method']} (seed={summary['seed']}): "
        f"accuracy={summary['accuracy']:.3f} over {summary['n_results']}/{summary['n_problems']} problems"
    )
    print(f"wrote {summary['out_dir']}/results.parquet")


if __name__ == "__main__":
    main()
