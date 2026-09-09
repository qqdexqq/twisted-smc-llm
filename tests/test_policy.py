from __future__ import annotations

from models.policy import MAX_STEPS, MockPolicy, split_into_steps


def test_split_into_steps_basic():
    text = "Step one.\n\nStep two.\n\nStep three."
    assert split_into_steps(text) == ["Step one.", "Step two.", "Step three."]


def test_split_into_steps_strips_blank_paragraphs():
    text = "Step one.\n\n\n\nStep two."
    assert split_into_steps(text) == ["Step one.", "Step two."]


def test_split_into_steps_single_paragraph():
    assert split_into_steps("Just one step.") == ["Just one step."]


def test_split_into_steps_caps_overlong_paragraph():
    long_para = " ".join(f"word{i}" for i in range(2000))
    result = split_into_steps(long_para, max_tokens_per_step=100)
    assert len(result) > 1
    assert all(len(chunk.split()) <= 100 for chunk in result)
    # No words dropped in the split.
    assert " ".join(result).split() == long_para.split()


def test_split_into_steps_respects_max_steps_cap():
    text = "\n\n".join(f"step {i}" for i in range(MAX_STEPS + 50))
    result = split_into_steps(text)
    assert len(result) == MAX_STEPS


def test_mock_policy_returns_n_completions_per_prompt():
    policy = MockPolicy(seed=0)
    prompts = ["prompt A", "prompt B"]
    results = policy.generate(prompts, n=5, temperature=0.8, max_tokens=100)
    assert len(results) == 2
    assert all(len(completions) == 5 for completions in results)
    for completions in results:
        for c in completions:
            assert c.n_tokens > 0
            assert "\\boxed{" in c.text


def test_mock_policy_deterministic_given_seed():
    p1 = MockPolicy(seed=42).generate(["same prompt"], n=3, temperature=0.8, max_tokens=100)
    p2 = MockPolicy(seed=42).generate(["same prompt"], n=3, temperature=0.8, max_tokens=100)
    assert [c.text for c in p1[0]] == [c.text for c in p2[0]]
