#!/usr/bin/env python
"""Stage 1, Asset 1: freeze the problem manifests (plan doc §5.1).

Pulls each benchmark, draws a 128-problem random subset with a pinned seed
(AIME: uses all problems, per the doc), and writes
manifests/{dataset}_{n}.jsonl with fields: problem_id, prompt, gold_answer,
source_split. No GPU, no model -- pure data curation.

Repo IDs and field names below were verified against the live HF hub
(see the conversation / commit message for the search trail) rather than
trusted from the plan doc, per its own instruction: "Hugging Face dataset
identifiers move around. Check the exact repo id on the hub rather than
trusting any id written in a plan document, including this one."

Usage:
    python scripts/build_manifests.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "manifests"

SEED = 0
N_SUBSET = 128


def _sample_indices(n_total: int, n_subset: int, seed: int) -> list[int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    n = min(n_subset, n_total)
    return sorted(rng.choice(n_total, size=n, replace=False).tolist())


def _extract_boxed(text: str) -> str:
    """Extract the content of the last \\boxed{...} in a string (handles
    one level of nested braces, which is enough for AIME's short answers).
    """
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return text.strip()
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


def _extract_gsm8k_answer(text: str) -> str:
    return text.split("####")[-1].strip()


def _write_manifest(name: str, rows: list[dict]) -> Path:
    MANIFESTS_DIR.mkdir(exist_ok=True)
    out_path = MANIFESTS_DIR / f"{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows)")
    return out_path


def build_gsm8k() -> None:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = _sample_indices(len(ds), N_SUBSET, SEED)
    rows = [
        {
            "problem_id": f"gsm8k-{i}",
            "prompt": ds[i]["question"],
            "gold_answer": _extract_gsm8k_answer(ds[i]["answer"]),
            "source_split": "test",
        }
        for i in idx
    ]
    _write_manifest("gsm8k_128", rows)


def build_math500() -> None:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = _sample_indices(len(ds), N_SUBSET, SEED)
    rows = [
        {
            "problem_id": ds[i]["unique_id"],
            "prompt": ds[i]["problem"],
            "gold_answer": ds[i]["answer"],
            "source_split": "test",
        }
        for i in idx
    ]
    _write_manifest("math500_128", rows)


def build_deepmath() -> None:
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    idx = _sample_indices(len(ds), N_SUBSET, SEED)
    rows = [
        {
            "problem_id": f"deepmath-{i}",
            "prompt": ds[i]["question"],
            "gold_answer": ds[i]["final_answer"],
            "source_split": "train",  # only split this dataset has
        }
        for i in idx
    ]
    _write_manifest("deepmath_128", rows)


def build_omnimath() -> None:
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    idx = _sample_indices(len(ds), N_SUBSET, SEED)
    rows = [
        {
            "problem_id": f"omnimath-{i}",
            "prompt": ds[i]["problem"],
            "gold_answer": ds[i]["answer"],
            "source_split": "test",
        }
        for i in idx
    ]
    _write_manifest("omnimath_128", rows)


def build_aime2024() -> None:
    # All 30 problems, per the plan doc ("AIME: use all of them"). Named
    # without "_128" since it isn't a 128-subset.
    ds = load_dataset("math-ai/aime24", split="test")
    rows = [
        {
            "problem_id": f"aime2024-{row['id']}",
            "prompt": row["problem"],
            "gold_answer": _extract_boxed(row["solution"]),  # field is
            # named "solution" in this dataset but only ever contains a
            # short \boxed{...} final answer -- verified against the live
            # dataset viewer, not assumed from the column name.
            "source_split": "test",
        }
        for row in ds
    ]
    _write_manifest("aime2024_all", rows)


def build_aime2025() -> None:
    ds = load_dataset("math-ai/aime25", split="test")
    rows = [
        {
            "problem_id": f"aime2025-{row['id']}",
            "prompt": row["problem"],
            "gold_answer": row["answer"],  # unlike aime24, this dataset
            # already has a plain "answer" field, not "solution".
            "source_split": "test",
        }
        for row in ds
    ]
    _write_manifest("aime2025_all", rows)


BUILDERS: dict[str, Callable[[], None]] = {
    "gsm8k": build_gsm8k,
    "math500": build_math500,
    "deepmath": build_deepmath,
    "omnimath": build_omnimath,
    "aime2024": build_aime2024,
    "aime2025": build_aime2025,
}


def main() -> None:
    for name, builder in BUILDERS.items():
        print(f"--- {name} ---")
        builder()


if __name__ == "__main__":
    main()
