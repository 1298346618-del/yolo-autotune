"""简化原则与早停逻辑的单元测试（纯函数，不需要 GPU）。"""

from core.loop_utils import apply_simplicity_criterion, should_continue


# ------------------------------------------------------------
# apply_simplicity_criterion
# ------------------------------------------------------------

def test_first_experiment_is_kept():
    keep, reason = apply_simplicity_criterion({"map50": 0.5}, None, {}, None)
    assert keep is True
    assert "首个" in reason


def test_significant_gain_is_kept():
    keep, _ = apply_simplicity_criterion(
        {"map50": 0.80}, {"map50": 0.75}, {}, None, threshold=0.002
    )
    assert keep is True


def test_regression_is_discarded():
    keep, reason = apply_simplicity_criterion(
        {"map50": 0.74}, {"map50": 0.75}, {}, None, threshold=0.002
    )
    assert keep is False
    assert "下降" in reason


def test_tie_with_more_complexity_is_discarded():
    complex_cfg = {"mosaic": 0.8, "mixup": 0.2, "dropout": 0.1}
    simple_cfg = {"mosaic": 0.0}
    keep, reason = apply_simplicity_criterion(
        {"map50": 0.750}, {"map50": 0.750}, complex_cfg, simple_cfg, threshold=0.002
    )
    assert keep is False
    assert "复杂" in reason


def test_tie_with_less_complexity_is_kept():
    simple_cfg = {"mosaic": 0.0}
    complex_cfg = {"mosaic": 0.8, "mixup": 0.2, "dropout": 0.1}
    keep, _ = apply_simplicity_criterion(
        {"map50": 0.750}, {"map50": 0.750}, simple_cfg, complex_cfg, threshold=0.002
    )
    assert keep is True


def test_small_gain_with_equal_complexity_is_kept():
    cfg = {"mosaic": 0.3}
    keep, _ = apply_simplicity_criterion(
        {"map50": 0.7505}, {"map50": 0.750}, cfg, cfg, threshold=0.002
    )
    assert keep is True


# ------------------------------------------------------------
# should_continue（早停）
# ------------------------------------------------------------

def _hist(scores):
    return [{"map50": s} for s in scores]


def test_short_history_always_continues():
    assert should_continue(_hist([0.5]), patience=5) is True


def test_continuous_improvement_continues():
    scores = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56]
    assert should_continue(_hist(scores), patience=5) is True


def test_plateau_stops():
    # 前 5 轮提升，之后 5 轮完全没有刷新最佳
    scores = [0.50, 0.51, 0.52, 0.53, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54]
    assert should_continue(_hist(scores), patience=5) is False


def test_noise_below_best_still_counts_as_no_improvement():
    scores = [0.50, 0.52, 0.51, 0.515, 0.518, 0.519, 0.5195]
    # 最佳在第二轮(0.52)，此后 5 轮都没超过它
    assert should_continue(_hist(scores), patience=5) is False


def test_zero_patience_disables_early_stop():
    assert should_continue(_hist([0.5, 0.5, 0.5]), patience=0) is True
