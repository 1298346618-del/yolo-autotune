# Auto-Tune 实验记录

---

## Run 1: yolo26s 微调 (已完成)

| 项目 | 值 |
|---|---|
| 基础模型 | `runs/detect/train2/weights/best.pt`（本地路径已脱敏） |
| 模型架构 | YOLOv8s (122层, 9.5M参数, 20.5 GFLOPs) |
| 训练方式 | 在已训100epoch的best.pt上继续微调50epoch |
| Agent策略 | 随机搜索 (无LLM) |
| 总实验轮数 | 22轮 (前10轮OOM, 后12轮成功) |
| 早停轮次 | 连续10轮无提升自动停止 |

| 指标 | 原始 best.pt | auto-tune 最佳 | 提升 |
|---|---|---|---|
| **mAP50** | 0.8549 | **0.9032** | +0.048 (+5.6%) |
| **mAP50-95** | 0.5658 | **0.6479** | +0.082 (+14.5%) |
| Precision | 0.815 | 0.851 | +0.036 |
| Recall | 0.802 | 0.834 | +0.032 |

最佳超参数:
```yaml
lr0: 0.012084
lrf: 0.090903
momentum: 0.900469
weight_decay: 0.000354
warmup_epochs: 4.267413
box: 11.18369
cls: 0.475354
dfl: 2.819493
hsv_h: 0.049139
hsv_s: 0.648174
hsv_v: 0.5413
degrees: 3.101476
translate: 0.145966
scale: 0.449419
shear: 3.41992
mosaic: 0.1007
mixup: 0.1833
copy_paste: 0.29
dropout: 0.0865
```

---

## Run 2: yolo26n scratch → fine-tune

### Stage 1: 从原始 scratch best.pt 微调 (已完成)

| 项目 | 值 |
|---|---|
| 基础模型 | `yolo26n_scratch/weights/best.pt` |
| 模型架构 | YOLOv8n (122 fused layers, 2.5M params, 5.2 GFLOPs) |
| 原始 scratch mAP50 | 0.6815 (epoch 132, fitness best) |
| 原始 scratch mAP50-95 | 0.4229 |
| Agent策略 | 第1轮随机搜索命中, 后续LLM因imgsz误导未超越 |
| 总实验轮数 | 17轮 |

| 指标 | 原始 scratch best.pt | stage1 最佳 | 提升 |
|---|---|---|---|
| **mAP50** | 0.6815 | **0.7465** | +0.065 (+9.5%) |
| **mAP50-95** | 0.4229 | **0.5082** | +0.085 (+20.1%) |
| Precision | 0.670 | 0.716 | +0.046 |
| Recall | 0.663 | 0.707 | +0.044 |

各类别 mAP50 提升:
| 类别 | 原始 | stage1 | 变化 |
|---|---|---|---|
| open | 0.759 | 0.786 | +0.027 |
| short | 0.558 | 0.573 | +0.015 |
| mousebite | 0.712 | 0.779 | +0.067 |
| spur | 0.641 | 0.695 | +0.054 |
| **pinhole** | **0.515** | **0.728** | **+0.213** |
| spurious_copper | 0.871 | 0.917 | +0.046 |

### Stage 2: 从 stage1 最佳继续微调 (待开始)

| 项目 | 值 |
|---|---|
| 基础模型 | `experiment_000_0610_0952/weights/best.pt` (stage1最佳) |
| Agent策略 | DeepSeek LLM (已修复: 不再调imgsz) |

---
