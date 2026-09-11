#!/usr/bin/env python
"""Diagnostic: does the PRM actually rank a correct step above a wrong one?

v2 -- prints BOTH softmax channels explicitly (not just whichever one
models/prm.py currently commits to), plus a sum-to-1 sanity check and the
raw number of "<extra_0>" positions found, so there's no ambiguity about
which channel means what or whether the masking itself found the right
number of step positions. The first version of this script (single
channel only) produced inconsistent-looking results across two separate
runs/sessions that didn't look like a clean complement of each other,
which this version is meant to resolve definitively.

v3 -- adds the score_batch()-vs-score() equivalence check Stage 2's plan
requires as a PREREQUISITE, not optional, before score_batch() is trusted
inside smc/sampler_llm.py::Sampler's loop (models/prm.py's own docstring:
"believed correct by inspection, but... must be verified empirically").
Ragged per-particle step counts are exercised deliberately, since that's
the actual shape every real Sampler.propagate() call produces once
particles start finishing at different times.

Usage (on the GPU box, weights already cached from an earlier run):
    python scripts/diagnose_prm.py --prm-8bit
    python scripts/diagnose_prm.py --prm-8bit --skip-batch-check  # ranking-only, faster
"""

from __future__ import annotations

import argparse


def check_score_batch_equivalence(prm, atol: float = 0.02) -> bool:
    """score_batch() must return the same per-step scores as N separate
    score() calls, for a batch of RAGGED (different-length) step lists --
    exactly what Sampler.propagate() hands it every real global step. A
    generous atol (not exact-equality) is deliberate: batched vs
    single-row matmul kernels can legitimately differ at the bf16-
    precision level even when both are "correct"; a large gap (not a
    rounding-level one) is the actual failure signature to watch for.
    """
    query = "What is 12 + 7?"
    steps_per_particle = [
        ["12 + 7 = 19.", "The final answer is \\boxed{19}."],  # 2 steps
        ["12 + 7 = 21.", "The final answer is \\boxed{21}.", "Wait, let me redo that."],  # 3 steps, ragged
        ["Just one step, no boxed answer yet."],  # 1 step
    ]

    individual = [prm.score(query, steps) for steps in steps_per_particle]
    batched = prm.score_batch(query, steps_per_particle)

    print(f"\nscore_batch() vs score() equivalence check (ragged batch of {len(steps_per_particle)}):")
    all_ok = True
    for i, (ind, bat) in enumerate(zip(individual, batched)):
        if len(ind.step_scores) != len(bat.step_scores):
            print(f"  particle {i}: SHAPE MISMATCH individual={len(ind.step_scores)} batched={len(bat.step_scores)}")
            all_ok = False
            continue
        diffs = [abs(a - b) for a, b in zip(ind.step_scores, bat.step_scores)]
        max_diff = max(diffs) if diffs else 0.0
        ok = max_diff <= atol
        all_ok = all_ok and ok
        print(
            f"  particle {i} ({len(ind.step_scores)} steps): individual={[f'{s:.4f}' for s in ind.step_scores]} "
            f"batched={[f'{s:.4f}' for s in bat.step_scores]} max_diff={max_diff:.5f} {'OK' if ok else 'MISMATCH'}"
        )

    print(
        "PASS: score_batch() matches score() within tolerance -- safe to use inside Sampler."
        if all_ok
        else f"FAIL: score_batch() diverges from score() by more than atol={atol} -- "
        "do NOT trust it inside the Sampler loop yet; check padding_side, pad_token_id, "
        "and the token_masks nonzero-filtering logic in models/prm.py::_make_step_rewards."
    )
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--prm-8bit", action="store_true")
    parser.add_argument(
        "--skip-batch-check", action="store_true", help="skip the score_batch()-vs-score() equivalence check"
    )
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from models.prm import _patch_dynamic_cache_compat

    if args.prm_8bit:
        # See models/prm.py::QwenMathPRMScorer.__init__ for why this is
        # here at all -- this script's own raw AutoModel diagnostic (cases
        # A/B/C below) triggers the same per-forward-pass warning spam,
        # bypassing that class entirely, so it needs the same suppression.
        import warnings

        warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast")

    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
    import transformers

    print(f"transformers: {transformers.__version__}")

    _patch_dynamic_cache_compat()

    tokenizer = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.prm, trust_remote_code=True)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(config, "eos_token_id", None) or 0

    kwargs = dict(config=config, device_map="auto", trust_remote_code=True)
    if args.prm_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=["score"]
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModel.from_pretrained(args.prm, **kwargs).eval()
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    print(f"model.config revision info: {getattr(config, '_commit_hash', 'n/a')}")

    query = "What is 12 + 7?"
    system = "Please reason step by step, and put your final answer within \\boxed{}."

    cases = {
        "A (all correct)": ["12 + 7 = 19.", "The final answer is \\boxed{19}."],
        "B (wrong arithmetic)": ["12 + 7 = 21.", "The final answer is \\boxed{21}."],
        "C (correct step then contradicts it)": ["12 + 7 = 19.", "So the final answer is \\boxed{21}."],
    }

    for label, steps in cases.items():
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
        conversation_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)
        n_sep_found = int((input_ids == step_sep_id).sum().item())

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs[0]  # [1, seq_len, 2]
        probs = F.softmax(logits, dim=-1)[0]  # [seq_len, 2]
        mask = (input_ids[0] == step_sep_id)
        step_probs = probs[mask]  # [n_steps, 2]

        print(f"\n{label}  (expected {len(steps)} step positions, found {n_sep_found})")
        for i, step in enumerate(steps):
            ch0, ch1 = step_probs[i, 0].item(), step_probs[i, 1].item()
            print(f"  step {i}: channel0={ch0:.4f}  channel1={ch1:.4f}  sum={ch0+ch1:.4f}  {step!r}")

    print(
        "\nIf channel0+channel1 isn't ~1.000 for every row, the 2-class softmax "
        "assumption itself is wrong (more than 2 classes, or the mask is picking "
        "up extra/wrong positions) -- that would be the real bug, not a simple "
        "channel swap. If it does sum to ~1, compare channel0 vs channel1 across "
        "A/B/C by hand: whichever one gives A>B and makes C's second step drop "
        "below its first is the correct channel."
    )

    if not args.skip_batch_check:
        # Free the raw AutoModel used above before loading a SECOND copy of
        # the same weights via QwenMathPRMScorer -- both resident at once on
        # a 16GB free-tier card would risk exactly the OOM Stage 1 already
        # hit once from holding two models on GPU simultaneously.
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        from models.prm import QwenMathPRMScorer

        prm = QwenMathPRMScorer(args.prm, load_in_8bit=args.prm_8bit)
        check_score_batch_equivalence(prm)


if __name__ == "__main__":
    main()
