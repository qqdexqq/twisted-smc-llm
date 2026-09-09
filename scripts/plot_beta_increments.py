#!/usr/bin/env python
"""Produce the Stage 0 "Figure 1" reproduction: beta_t - beta_{t-1} vs.
beta_t, for the ESS and CESS schedules, at two resampling thresholds.

Runs the sampler directly (rather than reading scripts/run_stage0_experiment.py
output) so the two gamma values can be swept without needing two separate
saved config files. Saves figures/figure1_beta_increments.png.

Usage:
    python scripts/plot_beta_increments.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smc.controllers.cess import CessAdaptiveController
from smc.controllers.ess import EssAdaptiveController
from smc.sampler import run_smc
from toy.gmm2d import DEFAULT_TARGET

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"

KAPPA = 0.9
# Must straddle kappa: below it (no resampling, ESS needs ~370 steps on
# this target) vs above it (resampling every step, ESS collapses to ~9
# steps, same as CESS) -- see tests/test_toy_gmm.py for the empirical
# check that motivated this choice over the plan doc's illustrative
# (0.2, 0.9) example pair.
GAMMAS = (0.5, 0.95)
N_PARTICLES = 1000
N_SEEDS = 10
N_BINS = 20


def collect_binned_increments(controller_factory, gamma: float, n_seeds: int, n_bins: int):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        result = run_smc(DEFAULT_TARGET, controller_factory(), N_PARTICLES, gamma, rng)
        for rec in result.history:
            incr = rec.beta - rec.beta_prev
            b = np.clip(np.searchsorted(bin_edges, rec.beta, side="right") - 1, 0, n_bins - 1)
            bin_sums[b] += incr
            bin_counts[b] += 1
    with np.errstate(invalid="ignore"):
        means = np.where(bin_counts > 0, bin_sums / np.maximum(bin_counts, 1), np.nan)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, means


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    factories = {
        "cess": lambda: CessAdaptiveController(KAPPA),
        "ess": lambda: EssAdaptiveController(KAPPA),
    }
    styles = {"cess": "-", "ess": "--"}

    for criterion, factory in factories.items():
        for gamma in GAMMAS:
            centers, means = collect_binned_increments(factory, gamma, N_SEEDS, N_BINS)
            ax.plot(centers, means, styles[criterion], label=f"{criterion} (gamma={gamma})")
            print(f"{criterion} gamma={gamma}: mean increments per bin computed")

    ax.set_xlabel("beta_t")
    ax.set_ylabel("beta_t - beta_{t-1}")
    ax.legend()
    ax.set_title("Stage 0 Figure 1 reproduction: increment vs. beta")
    fig.tight_layout()
    out_path = FIGURES_DIR / "figure1_beta_increments.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
