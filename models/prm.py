"""Batched PRM scoring + caching (Stage 1, Asset 2/3 / plan doc §5.2).

Two real scorers, each following its own model's documented convention
(verified against the live model cards -- see the commit message for the
fetch trail). They are NOT the same mechanism:

  - QwenMathPRMScorer: a dedicated reward head. Steps are joined by a
    literal "<extra_0>" separator token in a SINGLE forward pass; the
    per-step score is the softmax positive-class probability at each
    "<extra_0>" position. Code below started as the model card's example
    verbatim (channel [:, 1] as "positive"), but empirically the correct
    channel has flipped TWICE across this project's history depending on
    the transformers version loading the checkpoint -- currently channel
    1, re-confirmed on transformers==4.57.6 (pinned via pyproject.toml's
    gpu extras). See _make_step_rewards for the full history and why any
    future transformers/vllm version change must re-run
    scripts/diagnose_prm.py before trusting a score again. Real corpus
    data generated under a different channel assumption has its PRM
    scores inverted.
  - RLHFlowLlamaPRMScorer: a chat-formatted classifier convention. Each
    step becomes a user turn; the score is P("+") vs P("-") over the
    next-token logits right after that turn -- ONE forward pass PER STEP
    (more expensive; that's this model family's documented convention,
    not an inefficiency introduced here). Still UNVERIFIED -- never
    exercised on a real GPU, unlike QwenMathPRMScorer. Before trusting a
    real run: score a hand-picked correct step and a hand-picked wrong
    step from the same problem (see scripts/diagnose_prm.py's pattern)
    and confirm the correct one scores higher -- exactly the check that
    caught QwenMathPRMScorer's channel-index bug above.

MockPRMScorer is the CPU-testable stand-in that exercises the
surrounding pipeline via --dry-run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def _patch_dynamic_cache_compat() -> None:
    """Qwen2.5-Math-PRM-7B's custom remote code (modeling_qwen2_rm.py) was
    written against an older transformers whose Cache API had methods
    since renamed or removed entirely (the legacy tuple-based KV-cache
    format was dropped in favor of always using Cache objects). Each
    method here restores one such removed API, with the exact behavior a
    plain DynamicCache (no length cap) used to have. Idempotent / safe to
    call repeatedly -- each patch no-ops if the method already exists
    (i.e. on an older transformers where it was never removed).

    Found by running the real model on a live GPU one AttributeError at a
    time -- there may be more of these further into the same forward()
    call; add here if so, same pattern.
    """
    from transformers import DynamicCache

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx in range(len(past_key_values)):
                    key_states, value_states = past_key_values[layer_idx]
                    cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(DynamicCache, "get_usable_length"):
        # DynamicCache has no max-length cap, so the old get_usable_length
        # (which only trims when the cache would exceed a cap) always
        # reduced to plain get_seq_length for this cache type.
        def get_usable_length(self, new_seq_length, layer_idx: int = 0) -> int:
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = get_usable_length


@dataclass
class PRMScore:
    step_scores: list[float]  # one per step, in order, each in [0, 1]


class MockPRMScorer:
    """Deterministic pseudo-scores hashed from step text -- CPU-testable,
    exercises the pipeline's dtype/shape/caching expectations without a
    real model.
    """

    model_id = "mock-prm"

    def score(self, query: str, steps: list[str]) -> PRMScore:
        scores = []
        for step in steps:
            h = int(hashlib.sha256(step.encode("utf-8")).hexdigest(), 16)
            scores.append((h % 1000) / 1000.0)
        return PRMScore(step_scores=scores)

    def score_batch(self, query: str, steps_per_particle: list[list[str]]) -> list[PRMScore]:
        """Trivial loop over score() -- just needs to exist with the right
        call shape so Sampler's CPU-testable path can exercise the
        "one batched PRM call per global step" call site without a real
        model. The real batching win (one forward pass, not N) only
        matters for QwenMathPRMScorer.
        """
        return [self.score(query, steps) for steps in steps_per_particle]


class QwenMathPRMScorer:
    """Qwen/Qwen2.5-Math-PRM-7B -- see the model card's Python usage
    example (huggingface.co/Qwen/Qwen2.5-Math-PRM-7B), reproduced here
    near-verbatim.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """load_in_8bit/4bit (needs bitsandbytes): per the plan doc §11,
        "load the frozen base model in 4-bit or 8-bit for development
        runs" -- this PRM is 7B, so on a 16GB free-tier GPU (T4/P100)
        running it alongside even a 1.5B generator in bf16 is tight
        (~17GB > 16GB); 8-bit brings the PRM down to ~7-8GB.
        """
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if load_in_8bit or load_in_4bit:
            # bitsandbytes re-emits this warning on EVERY forward pass (once
            # per quantized Linear layer -- hundreds of times for a 28-layer
            # 7B model), not once like a normal Python warning. Harmless
            # (it's just describing its own internal dtype cast) but it
            # drowns out actually useful output -- found the hard way when a
            # real Kaggle run's pasted output got truncated by this spam
            # before the score_batch() equivalence check result it needed
            # to show ever appeared.
            import warnings

            warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast")

        _patch_dynamic_cache_compat()

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Qwen2.5-Math-PRM-7B ships custom modeling code (trust_remote_code)
        # written against an older transformers whose PretrainedConfig
        # always defaulted pad_token_id to None. Recent transformers raises
        # AttributeError on truly-unset config attributes instead, and this
        # model's custom Qwen2RMConfig never sets pad_token_id explicitly --
        # crashes inside their own __init__ (self.padding_idx =
        # config.pad_token_id) before we ever get a chance to touch it.
        # Fix: load the config first, force pad_token_id to exist (falling
        # back to eos_token_id, matching the common Qwen convention of no
        # separate pad token), then hand that patched config in explicitly.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = getattr(config, "eos_token_id", None) or 0

        kwargs = dict(config=config, device_map=device_map, trust_remote_code=True)
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                # "score" is the final [hidden_dim -> 2] classification
                # head (modeling_qwen2_rm.py: self.score(hidden_states)).
                # bitsandbytes' int8 CUDA kernel errors on such a tiny
                # output dimension on Turing-class GPUs (T4: "cublasLt ran
                # into an error", found on a live run) -- found empirically,
                # not documented anywhere obvious. This head is a trivial
                # fraction of the model's size, so skipping its
                # quantization costs effectively nothing.
                llm_int8_skip_modules=["score"],
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        self._step_sep_id = self.tokenizer.encode("<extra_0>")[0]

    @staticmethod
    def _make_step_rewards(logits, token_masks):
        import torch.nn.functional as F

        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            # Channel 1, not 0 -- and this index has now flipped TWICE across
            # this project's history, which is itself the important finding:
            # this checkpoint's positive/negative class ordering is NOT a
            # fixed property of the weights, it depends on the transformers
            # version loading them. First diagnosis (an unpinned, since-
            # drifted environment) found channel 0 correct. After pinning
            # transformers<5.0 (pyproject.toml's gpu extras -- see that
            # pin's own commit message for why) to kill the instability
            # transformers 5.x introduced, a clean re-run of
            # scripts/diagnose_prm.py on transformers 4.57.6 found the
            # OPPOSITE: channel 1 now gives a near-binary-clean signal
            # (e.g. a self-contradicting step scored ~0.001 on channel 1 vs
            # ~1.000 for staying correct) while channel 0 is backwards.
            # CONSEQUENCE: any future change to the pinned transformers/vllm
            # versions must re-run scripts/diagnose_prm.py and re-check this
            # index before trusting a single PRM score -- do not assume it
            # carries over. (This also means Stage 1's MATH500 rollout
            # corpus, generated under the OLD channel-0 assumption, has its
            # PRM scores inverted again and needs regenerating before its
            # calibration finding can be trusted as-is.)
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores_res.append(positive_probs.cpu().tolist())
        return all_scores_res

    def score(self, query: str, steps: list[str], system: str = DEFAULT_SYSTEM_PROMPT) -> PRMScore:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
        conversation_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(conversation_str, return_tensors="pt").to(self.model.device)
        # use_cache=False: this is a single-shot scoring pass, not
        # autoregressive generation -- no KV cache is ever needed. Found
        # on a live GPU run that skipping it also sidesteps a chain of
        # transformers version-skew AttributeErrors in Qwen's custom
        # remote code (from_legacy_cache / get_usable_length /
        # to_legacy_cache, all removed Cache-API methods this custom
        # code still calls) -- more robust than patching each one.
        #
        # torch.no_grad(): .eval() only disables dropout/batchnorm, NOT
        # autograd -- without this, every call built a full backward
        # graph and retained every layer's activations for a gradient we
        # never take. Found the hard way: survived a tiny 3-problem
        # sanity check (small enough to fit despite the waste) but OOM'd
        # partway into the real 128x32 run on a longer completion. The
        # bnb.matmul autograd.Function.apply calls visible in that OOM's
        # traceback were the tell -- gradients were being tracked at all.
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)
        token_masks = input_ids == self._step_sep_id
        step_rewards = self._make_step_rewards(outputs[0], token_masks)
        return PRMScore(step_scores=step_rewards[0])

    def score_batch(
        self,
        query: str,
        steps_per_particle: list[list[str]],
        system: str = DEFAULT_SYSTEM_PROMPT,
        max_batch_size: int = 8,
    ) -> list[PRMScore]:
        """True batched PRM forward pass -- scores every particle's
        (ragged) step list, chunked into groups of at most
        max_batch_size, instead of score()'s one call per particle
        (still far fewer, larger calls than that). This is the real gap
        the plan doc's pseudocode assumed ("log_psi[i] = prm_score(
        prefix_i)  # batched PRM forward") and that Stage 2's
        per-global-step accounting (§7.2, PRM-forward-pass count) needs
        to be a meaningful metric.

        max_batch_size caps how many particles go through ONE forward
        pass, chunking internally rather than batching everything the
        caller hands in at once. Found necessary on a real Kaggle T4
        run: activation memory for a single batched forward pass scales
        with batch_size * sequence_length, and vLLM + this PRM are BOTH
        resident on the GPU for the whole Sampler run (unlike Stage 1) --
        a real 16-particle batch, several global steps into a problem
        (long accumulated prefixes), OOM'd with only ~500 MiB free
        (CUDA out of memory allocating 674 MiB; vLLM + PRM already using
        ~14 of 14.56 GiB). Chunking keeps total FLOPs and the actual
        scores identical (each chunk is scored independently and
        correctly -- this is not an approximation), it only caps PEAK
        memory. NOTE: this means the true number of GPU forward passes
        can exceed what eval/compute_accounting.py's n_prm_forward_passes
        counts, since Sampler.propagate() calls this method once per
        global step regardless of how many chunks it uses internally --
        see that counter's own docstring.

        Right-padding, not left: this is a single-shot scoring pass, not
        autoregressive generation, so there's no "next token must be at
        the end" requirement that left-padding exists for. An explicit
        attention_mask keeps padded positions from contributing to any
        other position's attention.

        _make_step_rewards's existing per-row masking
        (`probabilities * token_masks.unsqueeze(-1)`, then
        `sample[sample != 0]`) already generalizes to ragged step counts
        with no changes needed: each row is masked and filtered
        independently, and padding never collides with a real <extra_0>
        position since pad_token_id is forced to differ from it (see
        __init__). Believed correct by inspection, but per the plan doc's
        own caveat this must be verified empirically, not assumed --
        that's scripts/diagnose_prm.py's score_batch()-vs-score()
        equivalence check, a prerequisite before this is trusted inside
        the Sampler loop, not optional.
        """
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            )

        results: list[PRMScore] = []
        for start in range(0, len(steps_per_particle), max_batch_size):
            chunk = steps_per_particle[start : start + max_batch_size]
            results.extend(self._score_batch_chunk(query, chunk, system))
        return results

    def _score_batch_chunk(
        self, query: str, steps_per_particle: list[list[str]], system: str
    ) -> list[PRMScore]:
        """One actual forward pass over <= max_batch_size particles --
        the part score_batch() chunks over. Split out purely so
        max_batch_size-driven chunking doesn't duplicate this logic.
        """
        import torch

        conversation_strs = []
        for steps in steps_per_particle:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
            ]
            conversation_strs.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )

        encoded = self.tokenizer(
            conversation_strs, return_tensors="pt", padding=True, padding_side="right"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=False
            )
        token_masks = encoded["input_ids"] == self._step_sep_id
        step_rewards = self._make_step_rewards(outputs[0], token_masks)
        return [PRMScore(step_scores=rewards) for rewards in step_rewards]


class RLHFlowLlamaPRMScorer:
    """RLHFlow/Llama3.1-8B-PRM-Deepseek-Data -- see
    github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm: each step
    is a user turn, scored by P("+") vs P("-") over the next-token
    logits right after that turn.
    """

    def __init__(self, model_id: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data", device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch.bfloat16
        ).eval()
        self._plus_id = self.tokenizer.encode("+", add_special_tokens=False)[-1]
        self._minus_id = self.tokenizer.encode("-", add_special_tokens=False)[-1]
        self._torch = torch

    def score(self, query: str, steps: list[str]) -> PRMScore:
        import torch.nn.functional as F

        messages: list[dict] = []
        scores = []
        for i, step in enumerate(steps):
            content = f"{query}\n\n{step}" if i == 0 else step
            messages.append({"role": "user", "content": content})
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            with self._torch.no_grad():
                logits = self.model(input_ids=input_ids, use_cache=False).logits[0, -1]
            pair = logits[[self._plus_id, self._minus_id]]
            p_plus = F.softmax(pair, dim=-1)[0].item()
            scores.append(p_plus)
            # Feed back the model's own call so multi-turn context is
            # consistent for the next step's forward pass.
            messages.append({"role": "assistant", "content": "+" if p_plus >= 0.5 else "-"})
        return PRMScore(step_scores=scores)
