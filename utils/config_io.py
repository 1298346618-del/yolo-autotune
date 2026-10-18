"""
超参数配置文件的读写操作
"""

import os
import yaml
from typing import Dict

from config.settings import CONFIG_PATH


def load_config(config_path: str = CONFIG_PATH) -> Dict:
    """加载当前超参数配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    # 默认配置（YOLOv8 官方默认值）
    return {
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "fliplr": 0.5,
        "erasing": 0.4,
        "dropout": 0.0,
    }


def save_config(config: Dict, config_path: str = CONFIG_PATH):
    """保存超参数配置"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
