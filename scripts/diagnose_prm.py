#!/usr/bin/env python
"""Diagnostic: does the PRM actually rank a correct step above a wrong one?

The full MATH500 corpus showed PRM scores narrowly clustered (median
0.17, 95th pct 0.33) and NEGATIVELY correlated with eventual correctness
(-0.14 to -0.22 across min/mean/max/last-step aggregates) -- backwards
from what a working process reward model should show, and a much
narrower range than Qwen's own model-card example output
([1.0, 0.19, 0.98, 1.0], clearly bimodal). This is the direct check that
was supposed to catch exactly this before trusting a full batch (see
models/prm.py's module docstring) but never actually got run.

Hand-constructed problem with an unambiguous right step and an
unambiguous wrong step, scored together (matching real usage --
QwenMathPRMScorer.score() takes a list of steps for one response, so we
build one response with both variants back to back and read off which
position each ends up ranked at) so any systematic bug (e.g. a swapped
softmax channel) shows up as clearly as possible.

Usage (on the GPU box, weights already cached from the full run):
    python scripts/diagnose_prm.py --prm-8bit
"""

from __future__ import annotations

import argparse

from models.prm import QwenMathPRMScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--prm-8bit", action="store_true")
    args = parser.parse_args()

    query = "What is 12 + 7?"

    # Response A: correct step, then a correct final answer.
    steps_a = [
        "12 + 7 = 19.",
        "The final answer is \\boxed{19}.",
    ]
    # Response B: same problem, but the first step is a clear arithmetic
    # error, propagated to a wrong final answer.
    steps_b = [
        "12 + 7 = 21.",
        "The final answer is \\boxed{21}.",
    ]
    # Response C: correct first step, but then a wrong (contradicting)
    # second step -- checks the PRM actually penalizes the SPECIFIC bad
    # step, not just "vibes" of the whole response.
    steps_c = [
        "12 + 7 = 19.",
        "So the final answer is \\boxed{21}.",
    ]

    prm = QwenMathPRMScorer(args.prm, load_in_8bit=args.prm_8bit)

    for label, steps in [("A (all correct)", steps_a), ("B (wrong arithmetic)", steps_b), ("C (correct step then contradicts it)", steps_c)]:
        scores = prm.score(query, steps).step_scores
        print(f"\n{label}:")
        for step, score in zip(steps, scores):
            print(f"  {score:.4f}  {step!r}")

    print(
        "\nExpected if the PRM/pipeline is working: A's scores clearly higher than "
        "B's; in C, the second step's score should be clearly lower than the "
        "first step's (and lower than A's second step). If A/B/C all look "
        "similar, or B/C score HIGHER than A, that confirms a real bug in how "
        "we're calling the PRM (not just 'the PRM is weak here')."
    )


if __name__ == "__main__":
    main()
