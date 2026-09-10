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

## Stage 1 (no-GPU part): frozen manifests + model selection

```
python scripts/build_manifests.py
```

Pulls GSM8K / MATH500 / DeepMath / Omni-MATH / AIME 2024+2025 from the HF
hub, draws a 128-problem random subset per dataset (seed=0; AIME uses all
30 problems per year, per the plan doc), and writes
`manifests/{name}.jsonl` with fields `problem_id`, `prompt`, `gold_answer`,
`source_split`. Repo IDs and field names were verified against the live
hub rather than trusted from the plan doc (see `scripts/build_manifests.py`'s
docstring and `configs/models.yaml` for the verification notes and a couple
of real schema surprises, e.g. `math-ai/aime24`'s answer living in a field
literally called `solution`). These manifests are committed — never
regenerate them; every experiment from Stage 2 onward reads from them.

`configs/models.yaml` pins the exact generator/PRM/verifier repo IDs for
Stage 1+ (no weights downloaded yet -- that's the GPU dev step, the actual
first cloud spend).

Note: `load_dataset` downloads full source files even when only sampling a
subset -- building all six manifests pulled ~6.7GB into `~/.cache/huggingface`
(mostly DeepMath-103K's 2.15GB parquet). One-time cost; safe to clear that
cache afterward since the frozen `.jsonl` manifests are all any later stage
reads from.

## Stage 1 (GPU part): rollout corpus generation

Not runnable on this laptop (no GPU). `notebooks/stage1_rollout_generation.ipynb`
is a ready-to-upload Kaggle notebook: clone this repo, run the CPU-only
`--dry-run` smoke test first (catches repo/env problems for free), install
the GPU extras (`pip install -e ".[gpu]"` -- torch/transformers/vllm/
bitsandbytes), run a tiny 3-problem real-GPU sanity check, then the full
128-problem x 32-rollout MATH500 corpus with the 1.5B generator + strong
PRM (loaded in 8-bit to fit alongside the generator on a 16GB free-tier
GPU). See the notebook's own markdown cells for the free-tier setup steps
(accelerator, internet, "Save & Run All" for unattended runs).

`scripts/generate_rollouts.py` (+ `models/policy.py`, `models/prm.py`,
`eval/metrics.py`) is the underlying pipeline. Its plumbing -- manifest
loading, step-splitting, answer-checking, Parquet schema -- is fully
tested on this laptop via `--dry-run` (CPU-only mocks). The real
vLLM-generation + PRM-scoring path has now been verified end to end on a
live Kaggle T4 (3-problem sanity check: pass@1 = 0.667, a plausible real
result, not the dry-run mock's structural 0).

**Checkpointed, since free-tier GPU sessions have proven unreliable in
practice:** each problem's rows are written to
`{steps,rollouts}.checkpoint.jsonl` as soon as that problem finishes, not
batched into one write at the end. Re-running the same command after a
disconnect resumes automatically (`checkpoint found: X/128 problems
already complete, Y remaining`) instead of losing all progress; pass
`--fresh` to discard an existing checkpoint and start over.

Real bugs this surfaced, several only visible on actual GPU hardware
(see git log for the full trail): `math-verify`'s per-call timeout spawns
a fresh `multiprocessing.Process` on Windows (no `signal.alarm` there),
which fails silently and made every answer look wrong regardless of
correctness -- worked around by disabling that timeout, which uses the
(working) signal-based path on Linux instead; Qwen's PRM ships custom
`trust_remote_code` written against an older `transformers`, whose
`PretrainedConfig`/`Cache` API has since dropped several methods that
code still calls (`pad_token_id` defaulting, `DynamicCache.from_legacy_cache`
/ `.get_usable_length`) -- patched via `use_cache=False` (we only ever
need a single forward pass, no cache at all) plus a couple of narrow
compatibility shims; `bitsandbytes`' int8 kernel errors on the PRM's tiny
2-output classification head on Turing GPUs -- excluded from
quantization via `llm_int8_skip_modules`; holding both the generator and
the PRM resident on the GPU at once OOMs (vLLM never releases its
reserved memory until the object is explicitly freed) -- fixed by
generating first, then freeing vLLM's engine before loading the PRM; and
`QwenMathPRMScorer.score()` was missing `torch.no_grad()`, silently
building an unneeded backward graph on every call and inflating memory
enough to OOM partway through a long run.

## An honest finding from building this

`tests/test_toy_gmm.py::test_cess_variance_vs_tuned_fixed`'s docstring
documents that the literature's ~20% CESS-vs-tuned-fixed variance
reduction did not reproduce robustly on this particular toy target with a
step-count-matched "tuned" fixed baseline, despite the tempering solver's
core invariants (CESS boundary/monotonicity) being independently verified
in `tests/test_tempering.py`. Read that docstring before relying on the
variance-reduction claim going into Stage 1+.
