#!/usr/bin/env python
"""CLI: load a Stage 0 config, run the SMC sampler over many seeds, and
dump per-seed results (log_Z_hat, n_steps, per-step beta trajectory) plus a
hash of the config used, to results/<config_stem>.npz.

Usage:
    python scripts/run_stage0_experiment.py --config configs/stage0_gmm_cess.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from smc.controllers.cess import CessAdaptiveController
from smc.controllers.ess import EssAdaptiveController
from smc.controllers.fixed import FixedLinearController
from smc.sampler import run_smc
from toy.gmm2d import DEFAULT_TARGET

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

_TARGETS = {"gmm2d": DEFAULT_TARGET}


def _config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _build_controller(cfg: dict):
    schedule = cfg["schedule"]
    if schedule == "fixed":
        return lambda: FixedLinearController(cfg["n_steps"])
    if schedule == "ess":
        return lambda: EssAdaptiveController(cfg["kappa"], n_iter=cfg.get("n_iter", 30))
    if schedule == "cess":
        return lambda: CessAdaptiveController(cfg["kappa"], n_iter=cfg.get("n_iter", 30))
    raise ValueError(f"unknown schedule {schedule!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg_hash = _config_hash(cfg)
    target = _TARGETS[cfg["target"]]
    controller_factory = _build_controller(cfg)

    n_seeds = cfg["n_seeds"]
    seed_start = cfg.get("seed_start", 0)
    log_Z_hats = np.empty(n_seeds)
    n_steps = np.empty(n_seeds, dtype=int)
    betas_per_seed: list[np.ndarray] = []
    increments_per_seed: list[np.ndarray] = []

    for i in range(n_seeds):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        result = run_smc(
            target,
            controller_factory(),
            cfg["n_particles"],
            cfg["gamma"],
            rng,
            max_steps=cfg.get("max_steps", 3000),
            rwm_steps=cfg.get("rwm_steps", 10),
            cov_ridge=cfg.get("cov_ridge", 1e-6),
        )
        log_Z_hats[i] = result.log_Z_hat
        n_steps[i] = result.n_steps
        betas_per_seed.append(np.array([h.beta for h in result.history]))
        increments_per_seed.append(np.array([h.beta - h.beta_prev for h in result.history]))

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.config.stem}.npz"
    np.savez(
        out_path,
        config_hash=cfg_hash,
        config_json=json.dumps(cfg, sort_keys=True),
        log_Z_hats=log_Z_hats,
        n_steps=n_steps,
        betas_per_seed=np.array(betas_per_seed, dtype=object),
        increments_per_seed=np.array(increments_per_seed, dtype=object),
        true_log_Z1=target.log_Z1,
    )

    print(f"config hash: {cfg_hash}")
    print(f"n_steps: mean={n_steps.mean():.1f} min={n_steps.min()} max={n_steps.max()}")
    print(f"log_Z_hat: mean={log_Z_hats.mean():.4f} std={log_Z_hats.std(ddof=1):.4f}")
    print(f"true log_Z1: {target.log_Z1:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
