"""
全局常量配置
所有模块共享的配置项和路径常量
"""

import os

# ============================================================
# 主配置字典
# ============================================================

CONFIG = {
    # 数据集配置（改成你自己的数据集；data_example.yaml 是模板）
    "data_yaml": "data_example.yaml",
    "model": "yolo26n.pt",   # 预训练权重或 .yaml 结构文件；scratch 训练时用 *.yaml
    "pretrained": True,

    # 实验预算
    "epochs": 100,
    "time_budget_seconds": 0,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "SGD",
    "device": 0,

    # Agent 配置
    "max_iterations": 12,
    "patience": 5,
    "top_k_checkpoint": 3,

    # 简化原则阈值（验证集仅 36 张，mAP 噪声大，阈值放宽）
    "simplification_threshold": 0.005,

    # 超参数搜索空间（默认面向"小数据集微调"；从零训练请自行放宽 lr0 / 增强上限）
    "search_space": {
        "lr0": [0.001, 0.01],
        "lrf": [0.001, 0.1],
        "momentum": [0.85, 0.95],
        "weight_decay": [0.0001, 0.001],
        "warmup_epochs": [1.0, 3.0],
        "box": [5.0, 12.0],
        "cls": [0.3, 1.0],
        "dfl": [1.0, 2.5],
        "hsv_h": [0.0, 0.02],
        "hsv_s": [0.0, 0.3],
        "hsv_v": [0.0, 0.2],
        "degrees": [0.0, 15.0],
        "translate": [0.0, 0.15],
        "scale": [0.0, 0.3],
        "shear": [0.0, 5.0],
        "mosaic": [0.0, 0.8],
        "mixup": [0.0, 0.2],
        "fliplr": [0.0, 0.5],
        "erasing": [0.0, 0.4],
        "dropout": [0.0, 0.15],
    },

    # LLM API 配置
    # 密钥只从环境变量读取，绝不要写进代码（.env.example 有示例）
    "llm_api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "llm_base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "llm_model": os.environ.get("LLM_MODEL", "deepseek-chat"),

    # 任务画像：注入 LLM 提示词的任务描述（换成自己的任务时只改这个 yaml）
    "task_profile": os.environ.get("TASK_PROFILE", "config/task_profile.yaml"),

    # Git 版本控制
    "use_git": False,
}

# ============================================================
# 路径常量
# ============================================================

TSV_PATH = "results.tsv"
CONFIG_PATH = "yolo_autotune_config.yaml"
HISTORY_JSON_PATH = "yolo_experiment_history.json"
