#!/usr/bin/env python
"""Stage 1 Asset 3 consumer: PRM calibration curve (plan doc §5.3).

Plots PRM score at step t against eventual rollout correctness, bucketed
by normalized step position t/T -- the doc's own required evidence for
the "early reward signals are unreliable" premise the whole adaptive-
tempering thesis leans on: "you need it, because [that premise] is a
premise of your method, and a referee will ask you to demonstrate it
rather than cite it."

For each t/T bucket (t=0 is the first step, t/T=1 is the last step of a
rollout), steps are further binned by their own PRM score, and for each
score-bin we plot (mean PRM score in that bin) against (the empirical
fraction of steps in that bin whose PARENT ROLLOUT was eventually
correct) -- a reliability diagram. Perfect calibration is the diagonal.
The premise predicts: early t/T buckets sit further from the diagonal
(PRM score doesn't track eventual correctness well yet) than late
buckets (which should hug the diagonal much more closely, since by the
final step the PRM is essentially scoring a near-complete solution).

Usage:
    python scripts/plot_prm_calibration.py \\
        --steps data/rollouts/math500_128/.../seed=0/steps.parquet \\
        --rollouts data/rollouts/math500_128/.../seed=0/rollouts.parquet \\
        --out figures/prm_calibration.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_calibration(
    steps_df: pd.DataFrame,
    rollouts_df: pd.DataFrame,
    n_t_buckets: int = 5,
    n_score_bins: int = 10,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], np.ndarray]:
    merged = steps_df.merge(
        rollouts_df[["problem_id", "rollout_id", "is_correct", "n_steps"]],
        on=["problem_id", "rollout_id"],
        how="inner",
    )
    # Position within the rollout: first step -> 0.0, last step -> 1.0.
    # n_steps==1 rollouts all land at 0.0 (no meaningful "position").
    denom = (merged["n_steps"] - 1).clip(lower=1)
    merged["t_frac"] = np.where(merged["n_steps"] > 1, merged["step_idx"] / denom, 0.0)

    t_edges = np.linspace(0.0, 1.0, n_t_buckets + 1)
    score_edges = np.linspace(0.0, 1.0, n_score_bins + 1)
    merged["t_bucket"] = pd.cut(
        merged["t_frac"], t_edges, include_lowest=True, labels=False
    ).astype("Int64")
    merged["score_bucket"] = pd.cut(
        merged["prm_score"].clip(0.0, 1.0), score_edges, include_lowest=True, labels=False
    ).astype("Int64")

    results: dict[int, pd.DataFrame] = {}
    for t_bucket in range(n_t_buckets):
        sub = merged[merged["t_bucket"] == t_bucket]
        rows = []
        for score_bucket in range(n_score_bins):
            cell = sub[sub["score_bucket"] == score_bucket]
            if len(cell) == 0:
                continue
            rows.append(
                {
                    "score_bucket": score_bucket,
                    "mean_prm_score": cell["prm_score"].mean(),
                    "empirical_accuracy": cell["is_correct"].mean(),
                    "n": len(cell),
                }
            )
        results[t_bucket] = pd.DataFrame(rows)
    return merged, results, t_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", required=True, type=Path)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures/prm_calibration.png"))
    parser.add_argument("--n-t-buckets", type=int, default=5)
    parser.add_argument("--n-score-bins", type=int, default=10)
    args = parser.parse_args()

    steps_df = pd.read_parquet(args.steps)
    rollouts_df = pd.read_parquet(args.rollouts)

    merged, results, t_edges = compute_calibration(
        steps_df, rollouts_df, args.n_t_buckets, args.n_score_bins
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, args.n_t_buckets, figsize=(4 * args.n_t_buckets, 4), sharey=True)
    if args.n_t_buckets == 1:
        axes = [axes]

    print(f"{'t/T bucket':<14} {'n steps':>8} {'mean |gap|':>12} {'corr(score,correct)':>20}")
    for t_bucket, ax in enumerate(axes):
        df = results[t_bucket]
        lo, hi = t_edges[t_bucket], t_edges[t_bucket + 1]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
        if len(df):
            ax.scatter(df["mean_prm_score"], df["empirical_accuracy"], s=df["n"].clip(upper=50) * 2, alpha=0.7)
            ax.plot(df["mean_prm_score"], df["empirical_accuracy"], alpha=0.5)
        ax.set_title(f"t/T in [{lo:.1f}, {hi:.1f})")
        ax.set_xlabel("mean PRM score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        sub = merged[merged["t_bucket"] == t_bucket]
        if len(sub) > 1 and sub["prm_score"].std() > 0:
            corr = sub["prm_score"].corr(sub["is_correct"].astype(float))
        else:
            corr = float("nan")
        mean_gap = (df["mean_prm_score"] - df["empirical_accuracy"]).abs().mean() if len(df) else float("nan")
        print(f"[{lo:.2f},{hi:.2f})    {len(sub):>8} {mean_gap:>12.3f} {corr:>20.3f}")

    axes[0].set_ylabel("empirical P(rollout eventually correct)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("PRM calibration by position in rollout (t/T)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
