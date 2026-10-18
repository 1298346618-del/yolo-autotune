"""
随机搜索策略（Phase 2）
包含随机采样函数和 RandomPolicy 类
"""

import random
from typing import Dict, List, Optional

from core.interfaces import BasePolicy, PolicyDecision


def generate_random_config(search_space: Dict) -> Dict:
    """随机采样生成一组超参数配置（基线策略）"""
    # 每次采样前重新播种，避免被 ultralytics 训练内的 set_seed() 重置导致重复配置
    random.seed()
    config = {}
    for key, (min_val, max_val) in search_space.items():
        if key in ["mosaic", "mixup", "copy_paste", "dropout"]:
            # 概率类参数：30% 概率设为 0（关闭增强）
            if random.random() < 0.3:
                config[key] = 0.0
            else:
                config[key] = round(random.uniform(min_val, max_val), 4)
        else:
            config[key] = round(random.uniform(min_val, max_val), 6)
    return config


class RandomPolicy(BasePolicy):
    """
    随机搜索策略
    在搜索空间内均匀随机采样超参数配置
    """

    def decide(self, history: List[Dict], best: Optional[Dict] = None) -> PolicyDecision:
        config = generate_random_config(self.search_space)
        return PolicyDecision(config=config, description="随机搜索")
