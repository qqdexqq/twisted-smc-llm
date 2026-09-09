from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.metrics import check_correct, extract_boxed_answer

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_extract_boxed_answer_simple():
    assert extract_boxed_answer("blah blah \\boxed{204}") == "204"


def test_extract_boxed_answer_last_one_wins():
    text = "First I think \\boxed{5}, no wait, \\boxed{7}."
    assert extract_boxed_answer(text) == "7"


def test_extract_boxed_answer_nested_braces():
    assert extract_boxed_answer("\\boxed{\\dfrac{1}{6}}") == "\\dfrac{1}{6}"


def test_extract_boxed_answer_no_box_falls_back():
    result = extract_boxed_answer("the answer is 42")
    assert "42" in result


@pytest.mark.parametrize(
    "model_answer,gold_answer,expected",
    [
        ("204", "204", True),
        ("204", "205", False),
        ("1/2", "\\frac{1}{2}", True),
        ("\\dfrac{1}{6}", "\\frac{1}{6}", True),
        ("3.0", "3", True),
    ],
)
def test_check_correct(model_answer, gold_answer, expected):
    assert check_correct(model_answer, gold_answer) is expected


def test_check_correct_never_raises_on_garbage():
    assert check_correct("not a math expression @#$%", "204") is False


@pytest.mark.parametrize(
    "manifest_name", ["math500_128", "aime2024_all", "aime2025_all", "gsm8k_128"]
)
def test_gold_answers_self_verify(manifest_name):
    """Sanity check the frozen manifests themselves: every gold_answer
    must at least verify against its own (identical) string -- a
    trivial check, but it catches math-verify being unable to parse the
    gold answer format at all (e.g. a stray LaTeX construct it chokes
    on), which would silently make every rollout on that problem count
    as wrong regardless of correctness.
    """
    path = REPO_ROOT / "manifests" / f"{manifest_name}.jsonl"
    n_unparseable = 0
    n_total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_total += 1
            if not check_correct(row["gold_answer"], row["gold_answer"]):
                n_unparseable += 1
    # A handful of unparseable gold answers (e.g. free-form text like
    # "p - q") is expected and fine; flag it if it's most of the file.
    assert n_unparseable / n_total < 0.15, (
        f"{manifest_name}: {n_unparseable}/{n_total} gold answers didn't even "
        f"self-verify -- math-verify likely can't parse this format"
    )
