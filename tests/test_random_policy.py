"""随机搜索策略：采样必须落在搜索空间内，且不同轮次不应完全相同。"""

from policy.random import RandomPolicy, generate_random_config

SPACE = {
    "lr0": [0.001, 0.01],
    "box": [5.0, 12.0],
    "mosaic": [0.0, 0.8],
    "dropout": [0.0, 0.15],
}


def test_all_values_within_bounds():
    for _ in range(100):
        cfg = generate_random_config(SPACE)
        for key, (lo, hi) in SPACE.items():
            assert lo <= cfg[key] <= hi, f"{key}={cfg[key]} 超出 [{lo}, {hi}]"


def test_probability_keys_can_be_zeroed():
    """增强类参数有 30% 概率被置 0，跑 200 次应该观察到。"""
    zeros = sum(
        1 for _ in range(200) if generate_random_config(SPACE)["mosaic"] == 0.0
    )
    assert zeros > 0


def test_configs_are_not_constant():
    """random.seed() 保证换新种子：连续采样不应总是一模一样。"""
    seen = {tuple(sorted(generate_random_config(SPACE).items())) for _ in range(20)}
    assert len(seen) > 1


def test_policy_returns_description():
    decision = RandomPolicy(SPACE).decide(history=[], best=None)
    assert decision.description
    assert set(decision.config).issubset(set(SPACE))
