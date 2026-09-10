#!/usr/bin/env python
"""Stage 1, Asset 3: generate the rollout corpus (plan doc §5.3).

For each problem in a manifest, sample N independent rollouts from the
policy under fixed decoding settings, split each into reasoning steps,
score every step with the PRM, and check final-answer correctness with
math-verify. Writes two Parquet tables, partitioned by
(dataset, generator_model, prm_model, seed):

    data/rollouts/{dataset}/{generator}/{prm}/seed={seed}/steps.parquet
    data/rollouts/{dataset}/{generator}/{prm}/seed={seed}/rollouts.parquet

Also prints the three baselines the plan doc says come "for free" from
this same corpus: base sampling (pass@1), best-of-N, and self-consistency
(majority vote) -- all offline, zero extra GPU time.

--dry-run swaps in MockPolicy/MockPRMScorer (models/policy.py,
models/prm.py) so the whole pipeline -- manifest loading, step-splitting,
answer-checking, Parquet schema -- is verified on a CPU-only machine
before ever touching a GPU. Real generation/scoring is UNVERIFIED until
run for real; sanity-check the printed accuracy numbers and a few raw
rows before trusting a full 128-problem x 32-rollout batch.

Checkpointed: each problem's rows are appended to
{steps,rollouts}.checkpoint.jsonl immediately after that problem's
rollouts finish, not batched into one write at the end. Re-running the
same command resumes automatically -- already-complete problems (exactly
--n rollout rows checkpointed) are skipped and only the remainder is
generated/scored. Pass --fresh to ignore any existing checkpoint and
start over. JSONL, not Parquet, for the checkpoint itself: a crash
mid-write leaves at most one truncated trailing line (safely discarded
on resume), whereas an unclosed Parquet writer can leave an unreadable
file. The final steps.parquet/rollouts.parquet are rebuilt from the
complete checkpoint every run, matching the schema below unchanged.

Usage:
    python scripts/generate_rollouts.py --manifest manifests/math500_128.jsonl --dry-run --n 4

    python scripts/generate_rollouts.py --manifest manifests/math500_128.jsonl \\
        --generator Qwen/Qwen2.5-1.5B-Instruct --prm Qwen/Qwen2.5-Math-PRM-7B \\
        --prm-type qwen --n 32 --temperature 0.8 --max-tokens 2048 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

from eval.metrics import check_correct, extract_boxed_answer
from models.policy import MockPolicy, VLLMPolicy, split_into_steps
from models.prm import MockPRMScorer, QwenMathPRMScorer, RLHFlowLlamaPRMScorer

REPO_ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS_DIR = REPO_ROOT / "data" / "rollouts"


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _read_jsonl_safe(path: Path) -> list[dict]:
    """Read a checkpoint JSONL file, silently discarding a truncated
    trailing line -- the only way a crash mid-write can corrupt this
    format, since every complete line before it stays independently
    valid regardless of what happens after.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    rows = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue  # truncated trailing line from a mid-write crash
            raise
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append and flush to durable storage immediately -- this is the
    actual checkpoint boundary. Called once per completed problem, not
    once per rollout: a crash mid-problem discards that whole problem's
    partial rows on the next resume (see _load_checkpoint), which is a
    deliberately simple tradeoff -- a few minutes of reprocessing beats
    the complexity of resuming at the rollout level.
    """
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_checkpoint(
    steps_ckpt_path: Path, rollouts_ckpt_path: Path, n: int, fresh: bool
) -> tuple[list[dict], list[dict]]:
    """Load and compact an existing checkpoint: keep only problems with
    exactly `n` rollout rows (a problem with fewer was left mid-write by
    a crash and is discarded, to be fully reprocessed rather than
    resumed at the rollout level). Rewrites both checkpoint files to
    that compacted state so a further crash this run doesn't compound
    stale partial data on top of what's already been recovered.
    """
    if fresh:
        _write_jsonl(steps_ckpt_path, [])
        _write_jsonl(rollouts_ckpt_path, [])
        return [], []

    step_rows = _read_jsonl_safe(steps_ckpt_path)
    rollout_rows = _read_jsonl_safe(rollouts_ckpt_path)

    rollout_counts = collections.Counter(r["problem_id"] for r in rollout_rows)
    complete_ids = {pid for pid, count in rollout_counts.items() if count == n}
    step_rows = [r for r in step_rows if r["problem_id"] in complete_ids]
    rollout_rows = [r for r in rollout_rows if r["problem_id"] in complete_ids]

    _write_jsonl(steps_ckpt_path, step_rows)
    _write_jsonl(rollouts_ckpt_path, rollout_rows)
    return step_rows, rollout_rows


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


def run(args: argparse.Namespace) -> dict:
    dataset_name = args.dataset_name or args.manifest.stem
    problems = load_manifest(args.manifest)

    out_dir = (
        ROLLOUTS_DIR
        / dataset_name
        / args.generator.replace("/", "__")
        / args.prm.replace("/", "__")
        / f"seed={args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_ckpt_path = out_dir / "steps.checkpoint.jsonl"
    rollouts_ckpt_path = out_dir / "rollouts.checkpoint.jsonl"

    step_rows, rollout_rows = _load_checkpoint(
        steps_ckpt_path, rollouts_ckpt_path, args.n, args.fresh
    )
    complete_ids = {r["problem_id"] for r in rollout_rows}
    remaining_problems = [p for p in problems if p["problem_id"] not in complete_ids]
    if complete_ids:
        print(
            f"checkpoint found: {len(complete_ids)}/{len(problems)} problems already "
            f"complete, {len(remaining_problems)} remaining"
        )

    if remaining_problems:
        # Generate with the policy FIRST, then explicitly free it before ever
        # touching the PRM. Found the hard way on a real T4: vLLM's engine
        # eagerly pre-allocates and PERMANENTLY holds its weights + full KV
        # cache pool (~11+ GiB on a 16GB card at gpu_memory_utilization=0.85)
        # for as long as the LLM object is alive -- it does not release that
        # just because generation finished. Loading a second, separate model
        # (the PRM) while that's still resident starves it of GPU memory (it
        # can silently partial-offload to CPU via device_map="auto", then
        # OOM later inside the forward pass once real activation memory is
        # needed). Never have both models resident on the GPU at once.
        policy = MockPolicy(seed=args.seed) if args.dry_run else VLLMPolicy(
            args.generator, quantization=args.quantization
        )
        prompts = [p["prompt"] for p in remaining_problems]
        completions_per_prompt = policy.generate(
            prompts, n=args.n, temperature=args.temperature, max_tokens=args.max_tokens
        )

        if not args.dry_run:
            import gc

            import torch

            del policy
            gc.collect()
            torch.cuda.empty_cache()

        prm = build_prm_scorer(args.prm_type, args.prm, args.dry_run, load_in_8bit=args.prm_8bit)

        for problem, completions in zip(remaining_problems, completions_per_prompt):
            problem_step_rows = []
            problem_rollout_rows = []
            for rollout_idx, completion in enumerate(completions):
                steps = split_into_steps(completion.text) or [completion.text.strip() or " "]
                prm_scores = prm.score(problem["prompt"], steps).step_scores
                prefix = ""
                for step_idx, (step_text, prm_score) in enumerate(zip(steps, prm_scores)):
                    prefix = f"{prefix}\n\n{step_text}" if prefix else step_text
                    problem_step_rows.append(
                        {
                            "problem_id": problem["problem_id"],
                            "rollout_id": rollout_idx,
                            "step_idx": step_idx,
                            "text": step_text,
                            "n_tokens": len(step_text.split()),
                            "sum_logprob_p0": completion.sum_logprob,
                            "prm_score": prm_score,
                            "prm_model_id": prm.model_id,
                            "prefix_hash": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
                            "timestamp": time.time(),
                        }
                    )
                final_answer = extract_boxed_answer(completion.text)
                problem_rollout_rows.append(
                    {
                        "problem_id": problem["problem_id"],
                        "rollout_id": rollout_idx,
                        "final_answer": final_answer,
                        "is_correct": check_correct(final_answer, problem["gold_answer"]),
                        "total_tokens": completion.n_tokens,
                        "n_steps": len(steps),
                    }
                )

            # Checkpoint boundary: this problem is now fully done. A crash
            # from here on loses at most the next in-progress problem, not
            # the whole run -- see _load_checkpoint / _append_jsonl.
            _append_jsonl(steps_ckpt_path, problem_step_rows)
            _append_jsonl(rollouts_ckpt_path, problem_rollout_rows)
            step_rows.extend(problem_step_rows)
            rollout_rows.extend(problem_rollout_rows)

    import pandas as pd

    steps_df = pd.DataFrame(step_rows)
    rollouts_df = pd.DataFrame(rollout_rows)
    steps_df.to_parquet(out_dir / "steps.parquet", index=False)
    rollouts_df.to_parquet(out_dir / "rollouts.parquet", index=False)

    summary = summarize(problems, rollouts_df, args.n)
    summary["out_dir"] = str(out_dir)
    summary["n_step_rows"] = len(step_rows)
    summary["n_rollout_rows"] = len(rollout_rows)
    return summary


def summarize(problems: list[dict], rollouts_df, n: int) -> dict:
    """The "baselines for free" from plan doc §5.3: base pass@1,
    best-of-N, self-consistency -- all computed offline from the corpus
    that was just generated, zero extra GPU time.
    """
    n_rollouts = len(rollouts_df)
    pass_at_1 = float(rollouts_df["is_correct"].mean()) if n_rollouts else float("nan")

    bon_correct = 0
    maj_correct = 0
    for problem in problems:
        sub = rollouts_df[rollouts_df["problem_id"] == problem["problem_id"]]
        if len(sub) == 0:
            continue
        if sub["is_correct"].any():
            bon_correct += 1
        majority_answer = collections.Counter(sub["final_answer"]).most_common(1)[0][0]
        if check_correct(majority_answer, problem["gold_answer"]):
            maj_correct += 1

    n_problems = len(problems)
    return {
        "pass_at_1": pass_at_1,
        "best_of_n": bon_correct / n_problems if n_problems else float("nan"),
        "self_consistency": maj_correct / n_problems if n_problems else float("nan"),
        "n": n,
        "n_problems": n_problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prm", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--prm-type", default="qwen", choices=["qwen", "rlhflow_llama"])
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-name", default=None, help="defaults to the manifest filename stem")
    parser.add_argument("--dry-run", action="store_true", help="use CPU-only mocks, no GPU/model needed")
    parser.add_argument(
        "--fresh", action="store_true", help="ignore any existing checkpoint and start over"
    )
    parser.add_argument("--quantization", default=None, help="generator quantization, e.g. 'bitsandbytes'")
    parser.add_argument(
        "--prm-8bit", action="store_true", help="load the PRM in 8-bit (needs bitsandbytes) -- see models/prm.py"
    )
    args = parser.parse_args()

    summary = run(args)
    print(f"wrote {summary['n_step_rows']} step-rows, {summary['n_rollout_rows']} rollout-rows to {summary['out_dir']}")
    print(f"pass@1 (N={summary['n']}):        {summary['pass_at_1']:.3f}")
    print(f"best-of-{summary['n']}:              {summary['best_of_n']:.3f}")
    print(f"self-consistency (maj): {summary['self_consistency']:.3f}")


if __name__ == "__main__":
    main()
