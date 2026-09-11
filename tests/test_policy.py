from __future__ import annotations

from models.policy import MAX_STEPS, MockPolicy, StepCompletion, split_into_steps


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


def test_generate_step_returns_one_completion_per_prefix():
    policy = MockPolicy(seed=0)
    prefixes = ["", "step one.", "step one.\n\nstep two."]
    completions = policy.generate_step(prefixes, stop=["\n\n"], temperature=0.8, max_tokens=512)
    assert len(completions) == 3
    assert all(isinstance(c, StepCompletion) for c in completions)
    assert all(c.finish_reason in ("stop_string", "eos", "length") for c in completions)


def test_generate_step_is_deterministic_by_prefix_content():
    # Same prefix text -> same next step, regardless of call order/rng
    # state -- checked by calling with the SAME prefix in two separate
    # policy instances (fresh rng state each time).
    p1 = MockPolicy(seed=0).generate_step(["a prefix"], stop=["\n\n"], temperature=0.8, max_tokens=100)
    p2 = MockPolicy(seed=999).generate_step(["a prefix"], stop=["\n\n"], temperature=0.8, max_tokens=100)
    assert p1[0].text == p2[0].text
    assert p1[0].finish_reason == p2[0].finish_reason


def test_generate_step_forces_finish_by_step_six():
    policy = MockPolicy(seed=0)
    long_prefix = "\n\n".join(f"step {i}." for i in range(6))  # 6 steps already generated
    completion = policy.generate_step([long_prefix], stop=["\n\n"], temperature=0.8, max_tokens=100)[0]
    assert completion.finish_reason == "eos"
    assert "\\boxed{" in completion.text


def test_generate_step_mixed_population_within_one_call():
    # A single call should be able to produce BOTH still-generating and
    # finished completions in the same batch -- exercises the mixed
    # done/active path the Sampler needs to handle every global step.
    policy = MockPolicy(seed=0)
    prefixes = [f"prefix {i}" for i in range(50)]  # enough prefixes to hit both outcomes with ~30% finish rate
    completions = policy.generate_step(prefixes, stop=["\n\n"], temperature=0.8, max_tokens=100)
    finish_reasons = {c.finish_reason for c in completions}
    assert "eos" in finish_reasons
    assert "stop_string" in finish_reasons
