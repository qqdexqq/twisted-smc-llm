"""Answer extraction and verification (plan doc §5.2: "Verifier: math_verify
(HuggingFace). Deterministic answer checking. Use a strict configuration
so your numbers are conservative."). This one IS fully testable on this
laptop -- no model, no GPU, just math-verify + regex -- and is covered by
tests/test_metrics.py.
"""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str:
    """Return the content of the LAST \\boxed{...} in text (handles one
    level of nested braces). Falls back to the trailing ~100 characters
    if no \\boxed{} is found, so a malformed completion still gets some
    answer string rather than crashing the pipeline.
    """
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return text.strip()[-100:]
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip()


def check_correct(model_answer: str, gold_answer: str) -> bool:
    """Deterministic correctness check via math-verify. Strict: any
    parse/verify exception counts as incorrect rather than raising, so a
    malformed model answer never crashes a batch run -- conservative per
    the plan doc's instruction.
    """
    from math_verify import parse, verify

    # parsing_timeout=None: math-verify's default per-call timeout uses
    # signal.alarm() on Unix but falls back to spawning a fresh
    # multiprocessing.Process per call on Windows (its own docs: "this
    # will incur a huge performance penalty"). On this machine that
    # spawn path itself fails (WinError 6, a handle-inheritance issue in
    # this sandboxed shell), and the wrapper silently swallows the
    # failure and returns an empty parse -- every answer looks wrong
    # regardless of correctness. Disabling the timeout sidesteps the
    # broken path entirely; on Linux (e.g. the actual GPU rental box)
    # the signal-based timeout works fine and could be re-enabled there
    # if a real hang on pathological input becomes a problem.
    try:
        gold_parsed = parse(_normalize_latex_macros(gold_answer), parsing_timeout=None)
        model_parsed = parse(_normalize_latex_macros(model_answer), parsing_timeout=None)
        return bool(verify(gold_parsed, model_parsed, timeout_seconds=None))
    except Exception:
        return False


def _normalize_latex_macros(text: str) -> str:
    """math-verify does not treat \\dfrac/\\tfrac as equivalent to \\frac
    out of the box (found while testing against this repo's own DeepMath
    manifest, which uses \\dfrac in its gold answers) -- they are the same
    fraction, just a different display-style macro. Normalize before
    parsing rather than silently under-counting correct rollouts.
    """
    return text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
