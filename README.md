# Adaptive-Tempering SMC — Stage 0

Synthetic testbed for the CESS-controlled adaptive tempering project (see
`docs/Adaptive_TSMC_Build_Plan.md`). No GPU, no LLM — this stage exists to
get the tempering solver and weight bookkeeping right on a cheap, known
problem before spending anything on cloud compute.

## Setup

```
conda env create -f environment.yml
conda activate adaptive-tsmc
pip install -e .
```

Note: `environment.yml` pins only `python=3.12` via conda and installs
numpy/scipy/matplotlib/pytest/pyyaml via pip underneath it (deliberately —
pinning exact conda-channel builds of numpy+scipy+matplotlib together
against `defaults` failed to solve on this machine; letting pip resolve
those avoids conda's solver entirely).

## Running the tests

```
pytest tests/ -v
```

`tests/test_tempering.py` and `tests/test_resampling.py` are fast (seconds).
`tests/test_toy_gmm.py` is the integration suite and is slower — roughly
5-10 minutes total on a laptop CPU, dominated by the ESS-schedule
unbiasedness check (see `smc/sampler.py`'s `run_smc` docstring for why the
ESS-adaptive controller needs far more steps than CESS/fixed on this
target). Run just the fast subset while iterating:

```
pytest tests/test_tempering.py tests/test_resampling.py -v
```

## Running an experiment / reproducing the Figure-1 plot

```
python scripts/run_stage0_experiment.py --config configs/stage0_gmm_cess.yaml
python scripts/run_stage0_experiment.py --config configs/stage0_gmm_ess.yaml
python scripts/run_stage0_experiment.py --config configs/stage0_gmm_fixed.yaml
python scripts/plot_beta_increments.py
```

The last command writes `figures/figure1_beta_increments.png` (also
produced automatically by `tests/test_toy_gmm.py::test_figure1_ess_diverges_more_than_cess_across_gamma`).

## What's here vs. deferred

Built: `smc/tempering.py` (the bisection solver — the module Stage 3 will
import unchanged), `smc/resampling.py`, `smc/sampler.py`, the three
`smc/controllers/*` schedules, and `toy/gmm2d.py` (a 2-D Gaussian-mixture
target with a closed-form `Z_1`, tempered against a Gaussian reference).

Deferred (empty stub files, not gating Stage 0): `toy/latin_square.py`,
`smc/controllers/epf.py` (ePF's real schedule needs the Stage 2 LLM+PRM
sampler to be a meaningful baseline against), and cross-checking against
the external `particles`/`blackjax` packages (neither installed; the
corresponding test is `skipif`-guarded).

## An honest finding from building this

`tests/test_toy_gmm.py::test_cess_variance_vs_tuned_fixed`'s docstring
documents that the literature's ~20% CESS-vs-tuned-fixed variance
reduction did not reproduce robustly on this particular toy target with a
step-count-matched "tuned" fixed baseline, despite the tempering solver's
core invariants (CESS boundary/monotonicity) being independently verified
in `tests/test_tempering.py`. Read that docstring before relying on the
variance-reduction claim going into Stage 1+.
