"""
主循环辅助工具
包含简化原则判断、早停逻辑、旧实验清理
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple


def apply_simplicity_criterion(
    new_result: Dict,
    best_result: Optional[Dict],
    new_config: Dict,
    best_config: Optional[Dict],
    threshold: float = 0.002
) -> Tuple[bool, str]:
    """
    简化原则：如果新配置更复杂但 mAP 提升很小，优先保留简化版

    Returns:
        (是否保留新配置, 原因说明)
    """
    if best_result is None:
        return True, "首个实验，直接保留"

    map50_diff = new_result["map50"] - best_result["map50"]

    # 计算配置复杂度（非零增强的数量）
    def complexity(cfg: Dict) -> int:
        complex_keys = ["mosaic", "mixup", "copy_paste", "degrees", "shear", "dropout"]
        return sum(1 for k in complex_keys if cfg.get(k, 0) > 0)

    new_complex = complexity(new_config)
    best_complex = complexity(best_config) if best_config else 0

    # 如果 mAP 显著提升，无条件保留
    if map50_diff > threshold:
        return True, f"mAP 提升 {map50_diff:.4f} > 阈值 {threshold}，保留"

    # 如果 mAP 下降，不保留
    if map50_diff < 0:
        return False, f"mAP 下降 {abs(map50_diff):.4f}，丢弃"

    # 如果 mAP 持平或微升，但新配置更复杂，保留简化版
    if 0 <= map50_diff <= threshold and new_complex > best_complex:
        return False, f"mAP 仅提升 {map50_diff:.4f} 但配置更复杂（{new_complex} vs {best_complex}），丢弃"

    # 其他情况（简化或持平）保留
    return True, f"mAP 持平/微升 {map50_diff:.4f}，配置不更复杂，保留"


def should_continue(tsv_history: List[Dict], patience: int) -> bool:
    """早停逻辑：连续 patience 轮无提升则停止"""
    if len(tsv_history) < patience:
        return True

    # 只看 status=keep 的实验
    kept = [h for h in tsv_history if h.get("status") == "keep"]
    if len(kept) < patience:
        return True

    recent = kept[-patience:]
    recent_best = max(h["map50"] for h in recent)
    earlier_best = max((h["map50"] for h in kept[:-patience]), default=0)

    return recent_best > earlier_best


def cleanup_old_runs(project: str = "yolo_autotune_runs", keep: int = 5):
    """清理旧的实验目录"""
    if not os.path.exists(project):
        return
    runs = sorted(
        [d for d in os.listdir(project) if d.startswith("experiment_")],
        key=lambda x: os.path.getmtime(os.path.join(project, x))
    )
    for old_run in runs[:-keep]:
        try:
            shutil.rmtree(os.path.join(project, old_run))
            print(f"[清理] 删除旧实验: {old_run}")
        except Exception as e:
            print(f"[清理] 删除失败 {old_run}: {e}")
