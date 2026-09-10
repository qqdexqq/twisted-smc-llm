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

Usage (on the GPU box, weights already cached from an earlier run):
    python scripts/diagnose_prm.py --prm-8bit
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--prm-8bit", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from models.prm import _patch_dynamic_cache_compat

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


if __name__ == "__main__":
    main()
