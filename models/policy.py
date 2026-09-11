"""Batched generation for the rollout corpus (Stage 1, Asset 3 / plan doc
§5.3) and for Stage 2's incremental per-step SMC sampler. Real generation
goes through vLLM -- "batched generation across particles is the whole
performance story. Do not use bare transformers.generate for the main
runs" (plan doc §5.2). VLLMPolicy is NOT importable/testable on this
machine (no GPU, and vllm itself needs CUDA to import on most builds) --
MockPolicy is the CPU-testable stand-in used by --dry-run to exercise the
surrounding pipeline.

Two distinct generation modes, two distinct methods (not one overloaded):
  - generate(): Stage 1's mode -- N full completions per prompt in one
    shot, later split into steps post-hoc via split_into_steps().
  - generate_step(): Stage 2's mode -- ONE incremental step per (already
    growing) prefix, stopping at a step boundary via vLLM's `stop`
    parameter, so the SMC sampler can reweight/resample between steps.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Callable

MAX_TOKENS_PER_STEP = 512
MAX_STEPS = 300


def split_into_steps(
    text: str,
    max_tokens_per_step: int = MAX_TOKENS_PER_STEP,
    count_tokens: Callable[[str], int] | None = None,
) -> list[str]:
    """Split a full completion into reasoning steps on blank-line
    boundaries (the PRM800K/Math-Shepherd convention the plan doc's
    §2.2 step-granularity discussion assumes), capped at
    max_tokens_per_step tokens each and MAX_STEPS steps total.

    count_tokens defaults to a whitespace-split approximation so this is
    testable without a real tokenizer; pass the actual generator's
    tokenizer.encode-based counter for real runs if token-exactness
    matters (the 512 cap is a soft engineering bound, not load-bearing
    the way beta_T=1 is, so the approximation is an acceptable default).
    """
    if count_tokens is None:
        count_tokens = lambda s: len(s.split())

    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    steps: list[str] = []
    for para in paragraphs:
        if count_tokens(para) <= max_tokens_per_step:
            steps.append(para)
            continue
        # Overlong paragraph (rare): split greedily on sentence boundaries.
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current = ""
        for sent in sentences:
            if count_tokens(sent) > max_tokens_per_step:
                # A single "sentence" is still too big on its own (e.g. no
                # punctuation at all in this stretch) -- sentence-boundary
                # splitting has nothing left to grab onto, so fall back to
                # a hard word-count split rather than silently emitting an
                # oversized step.
                if current:
                    steps.append(current)
                    current = ""
                steps.extend(_hard_split_by_words(sent, max_tokens_per_step))
                continue
            candidate = f"{current} {sent}".strip() if current else sent
            if count_tokens(candidate) > max_tokens_per_step and current:
                steps.append(current)
                current = sent
            else:
                current = candidate
        if current:
            steps.append(current)
    return steps[:MAX_STEPS]


def _hard_split_by_words(text: str, max_words: int) -> list[str]:
    """Last-resort fallback when a chunk has no sentence-ending
    punctuation to split on: chop by raw word count instead of leaving
    one oversized step. Uses word count directly rather than the
    (possibly custom) count_tokens passed to split_into_steps -- acceptable
    since this path is rare and just needs *some* reasonable cap, not an
    exact one.
    """
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


@dataclass
class RolloutCompletion:
    text: str
    n_tokens: int
    sum_logprob: float


@dataclass
class StepCompletion:
    """One incremental reasoning step for ONE particle's growing prefix
    (Stage 2). `text` is this step's NEW content only, not the full
    prefix -- callers append it (smc/llm_particle.py's step_texts).
    """

    text: str
    n_tokens: int
    sum_logprob: float
    finish_reason: str  # "stop_string" | "eos" | "length"


class MockPolicy:
    """Deterministic, CPU-only stand-in for VLLMPolicy. Exercises
    generate_rollouts.py's orchestration/schema code (manifest -> steps ->
    PRM scoring -> Parquet) without needing a GPU or model weights.
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def generate(
        self, prompts: list[str], n: int | list[int], temperature: float, max_tokens: int
    ) -> list[list[RolloutCompletion]]:
        n_list = [n] * len(prompts) if isinstance(n, int) else n
        results = []
        for prompt, n_i in zip(prompts, n_list):
            completions = []
            for _ in range(n_i):
                n_steps = self._rng.randint(1, 4)
                paras = [
                    f"Mock reasoning step {j} for prompt hash {hash(prompt) % 10000}."
                    for j in range(n_steps)
                ]
                text = (
                    "\n\n".join(paras)
                    + f"\nThe final answer is \\boxed{{{self._rng.randint(0, 999)}}}."
                )
                n_tokens = len(text.split())
                completions.append(
                    RolloutCompletion(text=text, n_tokens=n_tokens, sum_logprob=-float(n_tokens))
                )
            results.append(completions)
        return results

    def generate_step(
        self, prefixes: list[str], stop: list[str], temperature: float, max_tokens: int
    ) -> list[StepCompletion]:
        """One incremental step per prefix. Deterministic per-prefix
        "does this lineage finish now" decision, seeded from a hash of
        the prefix TEXT itself -- not self._rng's sequential state -- so
        the same prefix always gets the same next-step behavior
        regardless of call order. This lets a single --dry-run run
        exercise reproducible MIXED populations within one global step:
        some prefixes finishing early, some continuing, some finishing
        late (forced after 6 steps so dry-run always terminates in
        bounded time even in the unlucky-hash worst case).
        """
        completions = []
        for prefix in prefixes:
            h = int(hashlib.sha256(prefix.encode("utf-8")).hexdigest(), 16)
            local_rng = random.Random(h)
            n_steps_so_far = prefix.count("\n\n") + (1 if prefix.strip() else 0)
            finishes_now = local_rng.random() < 0.3 or n_steps_so_far >= 6
            if finishes_now:
                text = f"So the final answer is \\boxed{{{h % 1000}}}."
                finish_reason = "eos"
            else:
                text = f"Mock reasoning step {n_steps_so_far} (hash {h % 10000})."
                finish_reason = "stop_string"
            n_tokens = len(text.split())
            completions.append(
                StepCompletion(
                    text=text, n_tokens=n_tokens, sum_logprob=-float(n_tokens), finish_reason=finish_reason
                )
            )
        return completions


class VLLMPolicy:
    """Thin batched-generation wrapper around vLLM, written to the
    documented vLLM API. Unverified until run on a real GPU -- sanity-check
    the first few outputs by hand before trusting a full batch.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = True,
    ):
        """enforce_eager=True by default -- skips vLLM's CUDA-graph
        capture/warmup entirely. Found necessary on a real Kaggle T4 run:
        that warmup step (profile_cudagraph_memory -> _warmup_and_capture)
        triggers FlashInfer's JIT kernel compilation, which failed there
        with "cannot find -lcuda" (a missing libcuda.so linker stub in
        that specific container image, unrelated to this code). Separately
        justified for Stage 2 regardless of that failure: CUDA graphs
        assume fixed batch shapes, but generate_step()'s batch composition
        changes every global step as particles finish at different times
        -- the shape-matching win graphs exist for barely applies here,
        so eager mode trades a bit of raw throughput for much simpler,
        more robust startup. Set False to re-enable if a future
        environment's graph capture works and the throughput matters.
        """
        from vllm import LLM  # local import: keeps this module importable on CPU-only machines

        self.model_id = model_id
        kwargs = dict(
            model=model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )
        if quantization:
            kwargs["quantization"] = quantization
        self._llm = LLM(**kwargs)

    def generate(
        self, prompts: list[str], n: int | list[int], temperature: float, max_tokens: int
    ) -> list[list[RolloutCompletion]]:
        """One batched vLLM call across ALL prompts -- never loop
        prompt-by-prompt for the main runs (plan doc §5.2). `n` may be a
        single int (same rollout count for every prompt) or a per-prompt
        list (e.g. when resuming a partially-scored problem and only the
        remaining few rollouts are actually needed) -- vLLM accepts a
        list of SamplingParams matching len(prompts) for exactly this.
        """
        from vllm import SamplingParams

        if isinstance(n, int):
            params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, logprobs=0)
        else:
            params = [
                SamplingParams(n=n_i, temperature=temperature, max_tokens=max_tokens, logprobs=0)
                for n_i in n
            ]
        outputs = self._llm.generate(prompts, params)

        results = []
        for out in outputs:
            per_prompt = []
            for completion in out.outputs:
                sum_logprob = (
                    float(completion.cumulative_logprob)
                    if completion.cumulative_logprob is not None
                    else float("nan")
                )
                per_prompt.append(
                    RolloutCompletion(
                        text=completion.text,
                        n_tokens=len(completion.token_ids),
                        sum_logprob=sum_logprob,
                    )
                )
            results.append(per_prompt)
        return results

    def generate_step(
        self, prefixes: list[str], stop: list[str], temperature: float, max_tokens: int
    ) -> list[StepCompletion]:
        """One batched vLLM call, ONE completion per (distinct) prefix,
        stopping at a step boundary (stop=["\\n\\n"] matches
        split_into_steps' paragraph convention, for comparability with
        Stage 1). Unverified until run on a real GPU -- in particular the
        finish_reason classification below (vLLM's documented convention:
        finish_reason="stop" + stop_reason=None means the model's own EOS
        token fired; finish_reason="stop" + a non-None stop_reason means
        one of our `stop` strings matched; finish_reason="length" means
        max_tokens truncation) -- confirm this against the actual
        installed vLLM version's CompletionOutput before trusting it
        inside the Sampler loop, same as this class's other methods.
        """
        from vllm import SamplingParams

        params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens, stop=stop, logprobs=0)
        outputs = self._llm.generate(prefixes, params)

        completions = []
        for out in outputs:
            completion = out.outputs[0]
            sum_logprob = (
                float(completion.cumulative_logprob) if completion.cumulative_logprob is not None else float("nan")
            )
            if completion.finish_reason == "length":
                finish_reason = "length"
            elif completion.finish_reason == "stop":
                finish_reason = "eos" if completion.stop_reason is None else "stop_string"
            else:
                finish_reason = completion.finish_reason or "eos"
            completions.append(
                StepCompletion(
                    text=completion.text,
                    n_tokens=len(completion.token_ids),
                    sum_logprob=sum_logprob,
                    finish_reason=finish_reason,
                )
            )
        return completions
