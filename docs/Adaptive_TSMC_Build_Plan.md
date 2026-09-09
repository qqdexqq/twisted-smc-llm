**ESS-Controlled Adaptive Tempering for Twisted SMC in LLMs**

A build plan: from zero to a first defensible result

_Working document — data acquisition, pipeline construction, and evaluation protocol_

# 0\. What this document is

This is an execution plan, not a survey. It assumes you have already read the core papers in your project folder and have settled on the research direction. Its job is to answer three questions in order: what exactly are you claiming, what data do you need before you can test the claim, and in what order do you build the pipeline so that each stage produces something checkable.

One structural change from the roadmap you got earlier: that plan started with a 7B model on a rented GPU. This one does not. Stage 0 uses no LLM at all and no GPU, because roughly 80% of the bugs in an adaptive SMC sampler are in the tempering solver and the weight bookkeeping, and those bugs are invisible when they are hidden behind a 7B model and a noisy reward. You will find them in an afternoon on a Gaussian mixture and never see them again.

# 1\. The claim you are actually testing

Write this down and keep it visible, because every design choice below is downstream of it.

**Thesis.** In reward-guided particle inference for LLMs, the schedule by which the reward signal is allowed to influence the particle system should be chosen adaptively, by controlling the per-step discrepancy between successive intermediate targets, rather than fixed in advance. Doing so yields a better accuracy-per-unit-compute curve than fixed schedules, and does so without per-benchmark tuning.

## 1.1 Three testable hypotheses

- **H1 (efficiency).** At matched generation compute (tokens generated, not particle count), CESS-controlled tempering achieves higher top-1 accuracy than a fixed schedule on MATH500 / DeepMath / AIME, with the gap widening as the problem difficulty increases and the particle budget shrinks.
- **H2 (diversity mechanism).** The accuracy gain is mediated by particle diversity: adaptive tempering keeps normalised ESS and unique-trajectory ratio higher through the first half of generation, and reduces the variance of the resampling distribution in the early, reward-unreliable regime.
- **H3 (robustness).** The advantage is larger with a weaker, less-calibrated process reward model, because adaptive step selection automatically shrinks the step size exactly when the reward signal is over-confident.

H2 is the one that makes the paper. Anyone can show a number going up; showing that it goes up for the reason your theory predicts is what makes it publishable and what makes negative results still informative.

## 1.2 Where the novelty actually sits

You must be precise here, because two of the papers in your folder are uncomfortably close to your idea and a referee will notice.

| **Existing work**                               | **What it does**                                                                                                                                                                 | **What is left open for you**                                                                                                                                                                         |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Twisted SMC for LMs (Zhao et al., 2024)         | Learns intermediate twist functions via CTL for a target defined by a terminal potential; introduces bidirectional log-Z bounds as an evaluation tool.                           | The schedule is implicit and fixed — the twist is applied at full strength from token one. No mechanism controls how fast the target sequence moves.                                                  |
| Entropic Particle Filtering (Giannone et al.)   | Detects diversity collapse via entropy and anneals the resampling softmax reactively; resamples when normalised ESS ≤ 0.5; applies the intervention over the first 50% of steps. | The annealing is a heuristic response to a symptom, with a hand-set threshold and a hand-set 50% window. It does not solve for a step size that equalises the discrepancy between successive targets. |
| Adaptive SMC with CESS (Zhou, Johansen & Aston) | Solves on the fly for the tempering increment that holds conditional ESS at a target, giving roughly 20% variance reduction versus a hand-tuned schedule at negligible cost.     | Formulated for static Bayesian targets with MCMC rejuvenation. Never transferred to autoregressive generation, where the state space grows and there is no invariant MCMC kernel.                     |
| Adaptive tempering theory (Beskos et al.)       | Proves the ESS functional is continuous and strictly decreasing in the temperature increment, so bisection is well-posed; gives WLLN and CLT for the adaptive scheme.            | Gives you the theoretical licence to use bisection inside an LLM decoder — cite it, do not re-derive it.                                                                                              |

**Your one-sentence delta.** ePF anneals reactively when diversity has already collapsed; you choose the reward-tempering increment proactively so that the discrepancy between consecutive intermediate targets is held constant — a criterion imported from the adaptive SMC-sampler literature and never applied to LLM decoding.

# 2\. The formal object you are building

Fix notation now so the code matches the write-up later. Let p0 be the base LLM, x_{1:t} a prefix (a prefix is a reasoning step, not a token — see §2.2), phi the terminal potential, and psi_theta the learned twist.

### Target

```
sigma(x_1:T)  ∝  p0(x_1:T) · phi(x_1:T)
```

```
optimal twist:  psi*_t(x_1:t) = E_{p0(x_t+1:T | x_1:t)} [ phi(x_1:T) ]
```

### Tempered intermediate targets — the new part

Instead of applying the twist at full strength at every step, introduce an inverse temperature beta_t on the twist:

```
pi_t^beta(x_1:t)  ∝  p0(x_1:t) · psi_theta(x_1:t)^{beta_t},      0 = beta_0 ≤ beta_1 ≤ ... ≤ beta_T = 1
```

```
incremental weight moving (t-1, beta_{t-1}) -> (t, beta_t):
  log w_t = beta_t · log psi_theta(x_1:t) − beta_{t-1} · log psi_theta(x_1:t-1)
            + log p0(x_t | x_1:t-1) − log q(x_t | x_1:t-1)
```

**Non-negotiable constraint.** beta_T must equal 1. If your schedule ends anywhere else, you are sampling from a different distribution than the baselines and every accuracy comparison is confounded. Enforce it with a terminal correction step and assert it in the code.

### The adaptive rule

At each step, choose the largest beta_t you can get away with, subject to the particle system not degenerating:

```
CESS_t(b) = N · ( sum_j W_j · w_j(b) )^2  /  sum_k W_k · w_k(b)^2
            where W = normalised weights at t-1, w(b) = incremental weights
```

```
beta_t = max { b ∈ (beta_{t-1}, 1] : CESS_t(b) ≥ kappa · N }     # by bisection
resample (systematic) only if  ESS_t < gamma · N
```

Two knobs, deliberately decoupled: kappa controls how hard the reward is allowed to bite per step; gamma controls when you pay the resampling cost. ePF conflates them by only intervening at resampling time. Monotonicity of the ESS functional in beta (Beskos et al., Lemma 5.1) guarantees the bisection is well-posed; use 20–30 iterations of bisection on log-weights, it costs nothing next to a forward pass.

## 2.1 Why use CESS and not plain ESS

Plain ESS measures accumulated proposal/target mismatch since the last resampling event, so fixing its decrease does not give a uniform discrepancy between successive targets unless you resample every step. CESS is the quantity that does. Zhou, Johansen & Aston report roughly 20% variance reduction from CESS-based schedules versus a manually tuned one, while their ESS-based variant gave little improvement over linear unless resampling happened every iteration. Since you specifically want adaptive resampling (resampling every step is expensive with an LLM), CESS is the correct import. Implement both; the ESS variant becomes a free ablation that supports the design choice.

## 2.2 Step granularity

Do not run SMC at token granularity. Follow the convention in the particle-filtering-for-reasoning literature: a "step" is a reasoning step, capped at ~512 tokens, with a max of ~300 steps. This is what makes PRM scoring affordable and what makes your results comparable to published baselines. It also means T is 10–60, not 4000, which keeps the number of bisection solves trivial.

# 3\. Scope discipline: do not build both contributions at once

Your stated plan has two independent contributions: (a) ESS/CESS-controlled adaptive tempering, and (b) alpha=2 twist learning. Building both simultaneously is the single most likely way this project fails, because when the numbers disappoint you will not know which half is at fault.

- **Contribution A — adaptive tempering.** Inference-time only. No training. Works with an off-the-shelf PRM as the potential. This is a complete, self-contained paper and it is the one you should finish first.
- **Contribution B — alpha=2 twist learning.** Requires a training loop, a rollout corpus, AIS machinery, and careful weight normalisation. Add it only after A produces a measurable effect.

**Rule.** Stage 4 (twist learning) does not begin until Stage 3 has produced a signed, reproducible accuracy-versus-compute curve for adaptive tempering. If A shows nothing, B will not rescue it, and you will have learned that in six weeks instead of six months.

# 4\. Stage 0 — the synthetic testbed (week 1, no GPU, no LLM)

This is your actual first step. It is cheap, it is fast, and it produces the single most reused artefact in the project.

## 4.1 What to build

1. A minimal SMC sampler over a tempered path of static targets: pi_t(theta) ∝ mu(theta) · exp(−lambda_t V(theta)), with random-walk Metropolis rejuvenation using the empirical particle covariance scaled by (2.38)^2/d.
2. Three schedules behind one interface: (i) fixed linear in beta, (ii) ESS-adaptive by bisection, (iii) CESS-adaptive by bisection.
3. Two targets where the truth is known: a 2-D Gaussian mixture with analytic normalising constant, and — if you want a harder one — the Latin-square counting problem from the waste-free SMC slides, where the exact count is known for d = 11.

## 4.2 What to check

- log-Z estimates are unbiased across 50 seeds for all three schedules.
- The CESS schedule reduces the variance of the log-Z estimator relative to the tuned fixed schedule, at comparable numbers of intermediate distributions. If you cannot reproduce this qualitative result on a Gaussian mixture, your solver is wrong, full stop.
- Plot beta_t − beta_{t−1} against beta_t for both criteria at two resampling thresholds. You should see the ESS-based increments behave differently depending on resampling frequency while CESS is stable — this is Figure 1 of the Zhou et al. paper and reproducing it is your correctness test.
- Cross-check against an independent implementation: the particles package (github.com/nchopin/particles) or Blackjax both implement adaptive tempering SMC.

## 4.3 The artefact

A single module, tempering.py, exposing solve_beta_step(log_w_prev, log_G, target_cess, beta_prev) with unit tests. This exact function is imported unchanged into the LLM engine in Stage 3. Nothing about it knows or cares that the particles are reasoning traces. That separation is what lets you debug the statistics independently of the language model.

# 5\. Stage 1 — getting the data (weeks 2–3)

This is the part your earlier roadmap skipped, and it is the part that determines whether you can run experiments cheaply for the next three months. There are four distinct data assets and they have different lifetimes.

## 5.1 Asset 1 — the frozen problem manifests

Pull the benchmark problem sets, take a fixed random subset, and never regenerate it. Comparability with published baselines depends on this.

| **Dataset**        | **Role**                 | **Notes**                                                                                                                                                             |
| ------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GSM8K (test split) | Sanity / saturated       | Everything scores high here; use it only to confirm nothing is broken.                                                                                                |
| MATH500            | Primary development set  | The standard 500-problem subset. This is where you iterate.                                                                                                           |
| DeepMath           | Difficulty scaling       | Harder; the gap between PF and adaptive methods widens here.                                                                                                          |
| Omni-MATH          | Difficulty scaling       | Harder still; accuracies in the 5–11% range, so you need many seeds.                                                                                                  |
| AIME 2024 + 2025   | Long-horizon stress test | Small (~30 problems each) but the trajectories are an order of magnitude longer — this is where premature collapse actually bites and where your method should shine. |

- **Protocol.** Draw a 128-problem random subset per dataset with a pinned seed (AIME: use all of them). Write it to manifests/{dataset}\_128.jsonl with fields: problem_id, prompt, gold_answer, source_split. Commit this file. Every experiment forever reads from it.
- **Verify repo IDs before use.** Hugging Face dataset identifiers move around. Check the exact repo id on the hub rather than trusting any id written in a plan document, including this one.

## 5.2 Asset 2 — the models

| **Component**    | **Choice**                          | **Why**                                                                                                                      |
| ---------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Generator (dev)  | Qwen2.5-1.5B-Instruct or Qwen3-0.6B | Fast iteration; published PF/ePF numbers exist for the 1.5B, so you can verify your reimplementation.                        |
| Generator (main) | Qwen2.5-7B-Instruct, Qwen3-1.7B     | Main results. Two model families guards against a single-model artefact.                                                     |
| PRM (strong)     | Qwen2.5-Math-PRM-7B                 | The reference reward model used across this line of work.                                                                    |
| PRM (weak)       | A Llama-3.1-8B-based PRM            | Required for H3. The robustness claim is only meaningful if you test against a badly calibrated reward.                      |
| Verifier         | math_verify (HuggingFace)           | Deterministic answer checking. Use a strict configuration so your numbers are conservative.                                  |
| Serving          | vLLM                                | Batched generation across particles is the whole performance story. Do not use bare transformers.generate for the main runs. |

## 5.3 Asset 3 — the rollout corpus (the data you generate)

This is the expensive asset and the one to build carefully, because four separate parts of the project consume it.

**Generation spec.** For each problem in each manifest, sample N = 32 independent rollouts from the base model under the same decoding settings you will use at inference (temperature, max 512 tokens per step, max 300 steps). Score every step with the PRM as you go.

Record, per step, one row:

```
{ problem_id, rollout_id, step_idx,
  text, n_tokens, sum_logprob_p0,
  prm_score, prm_model_id,
  prefix_hash, timestamp }
```

```
plus one row per rollout:
{ problem_id, rollout_id, final_answer, is_correct, total_tokens, n_steps }
```

Store as Parquet, partitioned by (dataset, generator_model, prm_model, seed). Rough size: 128 problems × 32 rollouts × ~30 steps ≈ 120k step-rows per dataset-model pair; with text, a few GB. Trivial to keep, enormously expensive to regenerate.

### The four consumers of this corpus

- **Baselines for free.** Base sampling, self-consistency by majority vote, and best-of-N with an outcome reward are all computed offline from these rollouts. Zero extra GPU time.
- **PRM calibration curve.** Plot PRM score at step t against eventual correctness, bucketed by t/T. This reproduces the over-confidence finding in your own setup — and you need it, because "early reward signals are unreliable" is a premise of your method, and a referee will ask you to demonstrate it rather than cite it.
- **Realistic weight traces for the solver.** Feed real log-psi values into your Stage 0 bisection to check numerical behaviour at realistic scales and dynamic ranges, before any of it runs inside a decoder.
- **Positive samples for Stage 4.** Correct traces are your approximate sigma-samples for the CTL positive phase. If correct traces are too rare on the hard sets, that is the positive-sample bottleneck, and the multilevel construction (progressively rarer intermediate events) is the documented fix.

## 5.4 Asset 4 — the toy verification task

On math benchmarks you cannot verify that your sampler targets the right distribution — you only see accuracy, which can improve for wrong reasons. So keep one small task where the ground truth is computable: sentiment steering or infilling with GPT-2-small and short output lengths (T = 2–5 tokens), where exact log-probabilities and exact target samples are available.

On this task, compute the bidirectional bounds on log Z (the SMC lower bound, plus the upper bound available when an exact target sample is available). The gap between them upper-bounds the symmetrised KL between your sampler and the target. This is your correctness certificate. Run it in CI on every change to the sampler.

## 5.5 Stage 1 exit criteria

- Manifests committed and frozen.
- Rollout corpus generated for MATH500 with the 1.5B generator + strong PRM, and stored.
- Offline baselines (base / self-consistency / BoN) computed from it and within a couple of points of published values.
- PRM calibration plot produced from your own data.

# 6\. Stage 2 — reproduce the baselines you intend to beat (weeks 3–4)

**Hard gate.** If you cannot reproduce standard particle filtering and ePF on MATH500 with Qwen2.5-1.5B to within about 2 accuracy points of the published numbers, do not proceed. Every subsequent comparison is meaningless until this holds.

Implement, in this order:

1. Particle filtering with PRM guidance and softmax resampling — the direct predecessor of your method.
2. Beam search with PRM, as a non-probabilistic search baseline.
3. Twisted SMC in the step-level reasoning formulation (Feng et al.) — the fixed-schedule twisted baseline your H1 is stated against.
4. Entropic Particle Filtering with its published settings: intervention over the first 50% of steps, normalised-ESS threshold of 0.5. This is your strongest competitor and the most informative ablation.

Build all four behind one interface: a Sampler class with propagate(), weight(), and a pluggable ScheduleController. Your method then becomes one more controller rather than a fork of the codebase, which is what makes matched-compute comparison honest.

# 7\. Stage 3 — the adaptive engine (weeks 5–7)

Now you plug the Stage 0 solver into the Stage 2 sampler. The loop:

```
initialise N particles from prompt, beta = 0, log_W = uniform
for t in 1..T:
    propose step for each particle       # vLLM batched generation
    log_psi[i] = prm_score(prefix_i)     # batched PRM forward
    if t <= warmup_steps:
        beta_t = beta_floor              # noise-dominated regime: do not commit
    else:
        beta_t = bisect_for_target_cess(log_W, log_psi, log_psi_prev, kappa, beta_prev)
    log_w  = beta_t*log_psi - beta_prev*log_psi_prev   (+ proposal correction)
    log_W  = normalise(log_W + log_w)
    if ESS(log_W) < gamma*N:  systematic_resample()
    log Z_hat += logsumexp(...)          # keep the running estimate; it is free and diagnostic
assert beta_T == 1.0
```

## 7.1 Design decisions that matter

- **Warm-up floor.** Hold beta at or near zero for the first tau steps. The rationale is exactly the PRM over-confidence you measured in Stage 1: resampling on noise early is the failure mode. Sweep tau ∈ {0, 2, 5} and also let the CESS criterion discover it automatically — if the adaptive criterion picks small increments early on its own, that is a result worth reporting, and it is a cleaner story than a hand-set warm-up.
- **Terminal correction.** If beta has not reached 1 by the final step, apply the remaining increment as one last reweighting before answer selection. Assert it.
- **Numerical hygiene.** Everything in log-space, logsumexp everywhere, and clip log-psi to a sane range. PRM scores near 0 or 1 produce infinities that will silently poison the bisection.
- **Answer selection held constant.** Use the same final-answer selection rule across all methods (argmax under the PRM). Changing selection between arms is a classic way to accidentally manufacture a result.

## 7.2 Compute accounting — read this before running anything

Your entire claim is about a compute–accuracy tradeoff, so "compute" must be measured, not assumed. Comparing methods at equal N is not comparing them at equal compute: adaptive methods change how many steps get generated and how many PRM calls happen.

Instrument every run to log, per problem:

- total tokens generated by the policy model
- number of PRM forward passes and tokens scored
- number of resampling events and number of bisection solves
- wall-clock time, peak GPU memory, mean GPU utilisation

Then plot accuracy against total generated tokens as the primary figure, with accuracy-against-N as a secondary. If your method wins on N but loses on tokens, you have learned something important and you need to know it early.

# 8\. Stage 4 — alpha=2 twist learning (weeks 8–11, conditional)

Only if Stage 3 delivered. The motivation is clean: minimising the alpha=2 divergence is equivalent to minimising the variance of the importance weights, which at the optimum makes weights constant — directly attacking weight degeneracy rather than treating its symptoms. It is mass-covering, which is what you want for preserving alternative reasoning paths.

## 8.1 The construction

- Parameterise psi_theta as a scalar head over a LoRA adapter on the base model, shared across steps rather than a separate function per step index. A per-step parameterisation is memory-hostile and does not generalise across trace lengths.
- Train with an AIS-bootstrapped estimator of D_{alpha=2}(sigma || psi), targeting p^2/q, with a prioritised replay buffer.
- Use self-normalised importance weights in the surrogate loss. This is not optional — it is the reported difference between stable and unstable training in the FAB work, and it is the specific failure mode Gemini flagged.
- Stop gradients through the AIS samples and weights.
- Start with a small number of intermediate AIS distributions (1–8). More costs compute for diminishing variance reduction.

## 8.2 Comparisons

Train the same architecture with CTL, with SIXO, and with alpha=2, and compare on: (i) downstream accuracy, (ii) variance of the importance weights during inference — the quantity alpha=2 is supposed to minimise, so measure it directly, and (iii) the bidirectional log-Z gap on the toy task. If alpha=2 reduces weight variance but not accuracy, that is still a clean, reportable finding.

## 8.3 The known blocker

CTL-style objectives need positive samples from the target. On hard benchmarks correct traces are rare, so the positive phase starves. The documented remedy is a multilevel construction: learn twists over a sequence of progressively rarer intermediate events, where each level supplies informative positives for the next. Budget time for this; assume you will need it on AIME.

# 9\. Evaluation protocol

| **Metric**                       | **What it establishes**    | **How reported**                                                         |
| -------------------------------- | -------------------------- | ------------------------------------------------------------------------ |
| Top-1 accuracy / pass@1          | H1 — the headline claim    | vs total generated tokens (primary) and vs N ∈ {2,4,8,16,32} (secondary) |
| Normalised ESS over t            | H2 — the mechanism         | Mean trajectory with band, split by t/T                                  |
| Variance of resampling weights   | H2 — early over-commitment | V\[w_t\] against t/T, your method vs PF vs ePF                           |
| Unique-trajectory ratio, entropy | H2 — diversity preserved   | At 25%, 50%, 75% of the horizon                                          |
| Number of resampling events      | Cost of the mechanism      | Mean per problem                                                         |
| Bidirectional log-Z gap          | Correctness of the sampler | Toy task only; smaller is better                                         |
| Wall-clock, peak VRAM, GPU util. | Overhead is negligible     | Per configuration                                                        |

## 9.1 Statistics

- Minimum three seeds per configuration; report mean ± SD.
- 128-problem subsets are small. Use a paired bootstrap over problems (same problems, both methods) and report confidence intervals on the difference, not just on each arm.
- On Omni-MATH and AIME the absolute accuracies are single-digit to low-double-digit. A two-point "improvement" there is noise unless the paired test says otherwise. Be ruthless about this — it is the most common way this class of result fails to replicate.

## 9.2 Ablation grid

- Schedule: fixed linear | fixed tuned | ESS-adaptive | CESS-adaptive
- kappa (CESS target) ∈ {0.5, 0.7, 0.9}; gamma (resample threshold) ∈ {0.3, 0.5, 0.7}
- Warm-up tau ∈ {0, 2, 5} steps
- PRM: strong vs weak (this is H3)
- Generator: 1.5B vs 7B, Qwen2.5 vs Qwen3

The kappa sweep matters for a subtle reason: if performance is highly sensitive to kappa, you have replaced one hand-tuned schedule with one hand-tuned threshold and the "no per-benchmark tuning" part of your thesis collapses. Demonstrating flatness in kappa is therefore a load-bearing experiment, not a nice-to-have.

# 10\. Repository layout

```
adaptive-tsmc/
  manifests/           frozen problem subsets (committed)
  data/
    rollouts/          parquet, partitioned by dataset/model/prm/seed
    calibration/       PRM calibration artefacts
  smc/
    tempering.py       Stage 0 solver — no LLM imports allowed in this file
    resampling.py      systematic / multinomial / residual
    sampler.py         propagate / weight / resample loop
    controllers/       fixed.py, ess.py, cess.py, epf.py
  models/
    policy.py          vLLM wrapper
    prm.py             batched PRM scoring + caching
    twist.py           Stage 4 LoRA twist head
  train/
    ctl.py  sixo.py  alpha2.py
  eval/
    metrics.py  compute_accounting.py  bidirectional_bounds.py
  toy/                 Stage 0 Gaussian mixture + Latin square + GPT-2 infilling
  configs/             one YAML per experiment; every run logs its config hash
  scripts/
```

The one architectural rule: smc/tempering.py must never import anything model-related. If it does, you have lost the ability to test your statistics without a GPU.

# 11\. Compute plan and cost

| **Phase**        | **Hardware**                   | **Rough cost**  | **What runs there**                                                |
| ---------------- | ------------------------------ | --------------- | ------------------------------------------------------------------ |
| Stage 0          | Your laptop, CPU               | \$0             | Synthetic testbed, solver unit tests                               |
| Stage 1 dev      | 1× 24GB (RTX 4090 / A10G / L4) | ~\$0.40–0.80/hr | 1.5B generator + quantised PRM, rollout generation for MATH500     |
| Stage 2–3        | 1× 48GB (L40S / A40)           | ~\$0.80–1.20/hr | 7B generator + 7B PRM in bf16, main SMC runs                       |
| Stage 4          | 1× 80GB (A100 / H100)          | ~\$1.80–2.60/hr | Twist training with AIS; needs headroom for LoRA + optimiser state |
| Toy verification | CPU or 1× 24GB                 | negligible      | GPT-2 bidirectional bounds, runs in CI                             |

- Use a persistent network volume for the HF cache and the rollout corpus. Re-downloading 7B weights on every spin-up is the most common way to waste money on rented GPUs.
- Load the frozen base model in 4-bit or 8-bit for development runs; keep the twist adapter in bf16. Roughly halves VRAM and lets you develop on 24GB.
- Cache PRM scores keyed by prefix hash. Across an ablation grid, the same prefixes recur constantly and PRM calls are a large fraction of the bill.
- Prefer hourly on-demand (RunPod, Lambda, Vast) over serverless for interactive development; SMC runs are long and stateful, and cold-start-per-call pricing fits them badly.

# 12\. Risk register

| **Risk**                                | **Signal it is happening**                | **Mitigation**                                                                                                                                                     |
| --------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ePF already captures most of the gain   | Your method matches but does not beat ePF | Reframe around the mechanism: show CESS control removes ePF's two hand-set hyperparameters at equal accuracy. "Principled and tuning-free" is a real contribution. |
| Sensitivity to kappa                    | Accuracy swings across the kappa sweep    | Report the sensitivity honestly; if it is real, the tuning-free claim must be dropped from the thesis.                                                             |
| PRM is the bottleneck, not the schedule | Weak-PRM and strong-PRM results converge  | This is H3 failing. Still reportable; pivot the framing toward reward-model calibration.                                                                           |
| Long-horizon collapse persists          | AIME results flat regardless of schedule  | Expected — diversity can still vanish before the planning horizon. Combine with look-ahead style modulation, or restrict claims to medium horizons.                |
| alpha=2 training diverges               | Loss spikes, weight variance explodes     | Self-normalised weights, gradient stopping, fewer AIS distributions, heavier-tailed proposal component.                                                            |
| Positive-sample starvation              | CTL positive phase sees no correct traces | Multilevel construction with progressively rarer intermediate events.                                                                                              |
| Compute accounting reversal             | Wins at equal N, loses at equal tokens    | Catch this in week 6, not week 16 — that is why accounting is instrumented from the start.                                                                         |

# 13\. Timeline and gates

| **Week** | **Milestone**                                | **Gate to pass**                                                                         |
| -------- | -------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 1        | Stage 0 synthetic testbed                    | CESS beats tuned fixed schedule on log-Z variance; solver unit-tested                    |
| 2–3      | Manifests + rollout corpus + PRM calibration | Offline baselines match published numbers; over-confidence demonstrated in your own data |
| 3–4      | Baseline reimplementation                    | PF and ePF reproduced on MATH500 to within ~2 points                                     |
| 5–6      | Adaptive engine running end-to-end           | beta_T = 1 assertion holds; toy-task log-Z gap not worse than fixed schedule             |
| 6–7      | First accuracy-vs-tokens curves              | DECISION POINT: is there a signal on MATH500 / DeepMath?                                 |
| 7–8      | Ablation grid + AIME long-horizon            | Paired bootstrap CIs exclude zero, or a clear negative result                            |
| 8–11     | alpha=2 twist learning (conditional)         | Stable training; weight variance measurably reduced                                      |
| 11–12    | Write-up, figures, code release              | —                                                                                        |

**The week-6/7 decision point is the most important date in this plan.** By then you will know whether adaptive tempering does anything at all in this setting, and the cost of finding out will have been a few hundred dollars of GPU time rather than a semester.

# 14\. Reading map — which paper for which stage

| **Stage**  | **Papers in your folder**                                                                   | **What to take**                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Stage 0    | Zhou, Johansen & Aston (1303.3123); Beskos et al. (1306.6462); Chopin waste-free SMC slides | The CESS definition and bisection scheme; the monotonicity lemma licensing bisection; the resample–move template and variance estimators |
| Stage 1    | Entropic Particle Filtering (2510.05825)                                                    | Benchmark selection, model choices, PRM over-confidence analysis, evaluation conventions                                                 |
| Stage 2    | Zhao et al. twisted SMC (2404.17546 / zhao24c); ePF (2510.05825)                            | Twist definitions, twist-induced proposal, the baselines and their settings                                                              |
| Stage 3    | Beskos et al.; ePF; Funnel-SMC for diffusion (2508.12361)                                   | Adaptive temperature inside a generative search loop; the selection/transition abstraction; resampling-timing analysis                   |
| Stage 4    | FAB (2208.01893); Zhao et al.; Adaptive Multilevel Twisted SMC (2608.21736)                 | alpha=2 with AIS bootstrap and self-normalised weights; CTL gradient; the multilevel fix for positive-sample starvation                  |
| Evaluation | Zhao et al. §5                                                                              | Bidirectional log-Z bounds as a sampler-correctness certificate                                                                          |
| Framing    | Levine, RL as probabilistic inference (1805.00909)                                          | The soft-RL reading of twists as Q-values — useful for connecting to readers from an RL background                                       |

# 15\. What to do tomorrow

1. Create the repo with the layout in §10. Empty files are fine.
2. Write smc/tempering.py and its tests against a 2-D Gaussian mixture with known Z. No GPU, no LLM.
3. Reproduce the beta-increment-versus-beta plot for ESS and CESS at two resampling thresholds.
4. Freeze the MATH500 128-problem manifest and commit it.
5. Only then rent a GPU.

Steps 1–4 cost nothing and eliminate most of the ways this project can quietly go wrong.