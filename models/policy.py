"""Batched generation for the rollout corpus (Stage 1, Asset 3 / plan doc
§5.3). Real generation goes through vLLM -- "batched generation across
particles is the whole performance story. Do not use bare
transformers.generate for the main runs" (plan doc §5.2). VLLMPolicy is
NOT importable/testable on this machine (no GPU, and vllm itself needs
CUDA to import on most builds) -- MockPolicy is the CPU-testable stand-in
used by --dry-run to exercise the surrounding pipeline.
"""

from __future__ import annotations

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


class MockPolicy:
    """Deterministic, CPU-only stand-in for VLLMPolicy. Exercises
    generate_rollouts.py's orchestration/schema code (manifest -> steps ->
    PRM scoring -> Parquet) without needing a GPU or model weights.
    """

    def __init__(self, seed: int = 0):
        import random

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
    ):
        from vllm import LLM  # local import: keeps this module importable on CPU-only machines

        self.model_id = model_id
        kwargs = dict(model=model_id, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization)
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
