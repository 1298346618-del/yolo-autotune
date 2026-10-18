# YOLO Auto-Tune — 让 LLM 自己给 YOLO 调超参数

> **English** — A closed-loop hyperparameter tuning agent for YOLO object detection, inspired by
> [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The LLM reads the experiment
> log, proposes a hypothesis, runs a real training job, keeps or rolls back, and repeats. Includes
> a simplicity criterion, an LLM→perturbation→random fallback chain, and a 5-level OOM ladder.
> Validated on a PCB defect-detection case study (mAP50 0.676 → 0.784).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.x-green.svg)](https://github.com/ultralytics/ultralytics)

---

## 一句话说明

给大模型一个**真实的训练环境**，而不是一段提示词：它自己提假设 → 跑 YOLO 训练 → 看验证集 mAP
→ 保留或回滚 → 继续下一轮，整夜无人值守地跑下去。

```text
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │  历史实验记录 │─────▶│  LLM 提假设   │─────▶│  YOLO 训练    │
   │ results.tsv   │      │ (system prompt)│     │ (ultralytics) │
   └──────────────┘      └──────────────┘      └──────┬───────┘
          ▲                                           │
          │                                           ▼
   ┌──────┴───────┐                            ┌──────────────┐
   │ 保留 / 回滚   │◀───────────────────────── │  验证集 mAP   │
   │ 简化原则筛选  │                            └──────────────┘
   └──────────────┘
```

这个项目把 autoresearch 的「**你写 program.md，而不是改 Python 文件**」搬到目标检测：
人类负责写 [program.md](program.md)（Agent 的行为准则）和
[config/task_profile.yaml](config/task_profile.yaml)（任务先验），Agent 负责所有超参数实验。

### 与 autoresearch 的对应关系

| 维度 | autoresearch | 本项目 |
|---|---|---|
| Agent 操作对象 | 直接编辑 `train.py` 源码 | 改超参数字典，传给 YOLO API |
| 实验记录 | `results.tsv` | `results.tsv`（格式完全一致） |
| 评估指标 | `val_bpb`（越低越好） | `mAP@0.5`（越高越好） |
| 版本控制 | `git commit` / `git reset` | 可选（`use_git`），同时用 TSV 记录 |
| 简化原则 | 有 | 有，并带"提升 < 阈值但更复杂 → 丢弃"的判据 |
| 适用场景 | LLM 预训练研究 | YOLO 目标检测调优 |

---

## 架构

三层解耦，换任何一层都不影响其余两层：

```
main.py                  组装 Policy + Environment + Kernel
├── core/
│   ├── agent.py         AgentKernel：提议→训练→评估→keep/discard→早停 主循环
│   ├── interfaces.py    BasePolicy / BaseEnvironment 抽象协议
│   ├── history.py       results.tsv 读写（断点续跑）
│   ├── rollback.py      git commit / git reset
│   └── loop_utils.py    简化原则、早停、旧 run 清理
├── policy/
│   ├── llm.py           LLM 策略 + YAML 解析 + 三级降级
│   └── random.py        随机搜索基线
├── env/
│   └── yolo.py          ultralytics 封装、显存预检、OOM 5 级降级
├── config/
│   ├── settings.py      CONFIG：数据集、预算、搜索空间、LLM
│   └── task_profile.yaml 任务画像（注入 prompt，换任务只改这里）
└── utils/config_io.py   YAML 读写
```

### 几个关键设计

- **LLM 输出不信任**：解析失败 → 正则逐行兜底 → 越界值裁剪到搜索空间 → 无效键跳过；
  整体失败则退化为"在历史最佳附近扰动"，再失败才纯随机。**Agent 不会因为 LLM 抽风而中断**。
- **简化原则**：mAP 提升 `< simplification_threshold` 却引入了更多复杂度时，丢弃新配置。
- **OOM 5 级降级**：`batch → imgsz` 逐级下调并在 TSV 描述里标记，避免整轮实验浪费。
- **任务画像外置**：换数据集不用改代码，只改 `config/task_profile.yaml`。

---

## 效果：PCB 缺陷检测案例

工业 PCB 数据集（449 训练 / 100 验证，6 类缺陷，均为小目标），YOLO26n 从已训练权重微调。

| 阶段 | 策略 | mAP50 | mAP50-95 |
|---|---|---|---|
| 起点（scratch 模型） | — | 0.676 | 0.424 |
| 人工初始配置 | — | 0.7568 | 0.5047 |
| 随机搜索 ×15 | 随机采样 | 0.7499（未超过初始配置） | 0.5366 |
| **LLM Agent ×16** | **历史驱动** | **0.7836** | **0.5558** |

- 相对 scratch 起点：**mAP50 +15.0%**（0.6815 → 0.7836，按整体验证集口径）
- 相对人工初始配置：**+3.5%**（0.7568 → 0.7836）
- 6 类缺陷**全部提升**，短路（short）与缺口（mousebite）提升最大（+0.165 / +0.149）
- LLM 自主收敛到的规律：**box 权重拉满 + 增强压低 + lr0 收紧**——与"小目标检测"的人类经验一致，
  但它是从实验数据里自己找到的

完整分析（含每轮 LLM 推理原文、逐类指标、失败复盘）见 **[docs/case_study_pcb.md](docs/case_study_pcb.md)**。
可视化对比见 `docs/demo/index.html` 与 `docs/demo3/`。

> ⚠️ **关于"LLM vs 随机搜索"这组对比的诚实说明**：存档的那轮随机搜索在 15 轮里给出了**完全相同的
> 配置**（见 `experiments/results_pcb_backup.tsv` 第 2–16 行），原因是当时 ultralytics 训练内部调用
> `set_seed()` 重置了全局随机数状态——该 bug 已在 [policy/random.py](policy/random.py) 中修复。
> 因此这个对比**只能说明"随机基线当时是失效的"**，不能当作受控对照实验。真正站得住的是：
> LLM 在 16 轮内把人工初始配置从 0.7568 推到 0.7836。

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/1298346618-del/yolo-autotune.git
cd yolo-autotune

conda create -n yolo-autotune python=3.10 -y
conda activate yolo-autotune
pip install torch --index-url https://download.pytorch.org/whl/cu118   # 按你的 CUDA 版本
pip install -r requirements.txt
```

### 2. 配密钥

```bash
cp .env.example .env      # 填入 DEEPSEEK_API_KEY
```

或者直接用环境变量（Windows PowerShell）：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
$env:LLM_BASE_URL="https://api.deepseek.com/v1"
$env:LLM_MODEL="deepseek-chat"
```

> **不填 key 也能跑**——会自动退化为纯随机搜索，适合先验证流程通不通。

### 3. 准备数据集

按 [data_example.yaml](data_example.yaml) 建一个 `data.yaml`：

```yaml
path: /your/dataset/root
train: images/train
val: images/val
names:
  0: open
  1: short
```

数据集**不随仓库分发**，请自行准备。

### 4. 改配置并运行

编辑 [config/settings.py](config/settings.py) 顶部的 `CONFIG`（`data_yaml` / `model` / `epochs` /
`max_iterations`），然后：

```bash
python main.py
```

跑起来后 `results.tsv` 会持续累积，中断后重跑会自动从历史继续。

### 5. 监控

```bash
python -c "import pandas as pd; print(pd.read_csv('results.tsv', sep='\t'))"
```

---

## 换成你自己的任务

**只改两个文件，不动代码**：

1. `config/settings.py` → `CONFIG["data_yaml"]`、`CONFIG["model"]`、`CONFIG["search_space"]`
2. `config/task_profile.yaml` → 任务名称、数据特点、领域经验、调参优先级

`task_profile.yaml` 会被原样注入 LLM 的 system prompt。例如换成"无人机小目标检测"，
只需把 `dataset_notes` 和 `domain_hints` 改掉即可，`policy/llm.py` 一行都不用改。

想调整"Agent 的性格"（更激进/更保守/只调某几个参数），编辑 [program.md](program.md)。

---

## 目录结构

```
yolo-autotune/
├── main.py                 入口
├── program.md              Agent 行为准则（人类编辑）
├── config/
│   ├── settings.py         搜索空间、预算、LLM 配置
│   └── task_profile.yaml   任务画像（注入 prompt）
├── core/                   Agent 内核、历史、回滚、简化原则
├── policy/                 LLM 策略 / 随机策略
├── env/                    ultralytics 训练封装
├── experiments/            真实实验记录（TSV）
├── docs/
│   ├── case_study_pcb.md   PCB 案例完整分析
│   ├── demo/               调优前后检测效果对比（HTML）
│   └── demo3/              PR 曲线、逐图预测
├── data_example.yaml       数据集配置模板
├── .env.example            密钥模板
└── requirements.txt
```

## 实验记录

| 文件 | 内容 | 结论 |
|---|---|---|
| `experiments/results_pcb_backup.tsv` | PCB / YOLO26n，初始配置 → 随机搜索 → LLM | 0.7568 → **0.7836** |
| `experiments/yolo26n_stage1/results.tsv` | PCB / 从 scratch 微调，17 轮 | 0.6815 → **0.7465** |
| `experiments/yolo26s_finetune/results.tsv` | 更大模型，随机搜索 22 轮 | 0.8549 → **0.9032** |
| `experiments/results_expression_trial.tsv` | 小样本场景（84 训练图），11 轮 | **无提升**（10/11 被丢弃） |

最后一行是很有价值的**负结果**：当数据集只有几十张图时，验证集 mAP 的噪声大于超参数带来的真实差异，
Agent 会把运气当成提升。**先修数据，再调参**。

---

## 已知局限（请在使用前读一遍）

1. **验证集小 → 指标噪声大**。项目用"最后 5 个 epoch 平均"来抗噪，但在 <100 张验证图时仍建议
   手动复核最佳配置，而不是盲信 TSV。
2. **随机基线不可复现**。`generate_random_config()` 里用 `random.seed()` 重新播种，
   因此每次运行的随机序列不同。要做受控对比请自己传入固定 seed。
3. **搜索空间偏窄**。默认范围面向"小数据微调"（如 `lr0 ∈ [0.001, 0.01]`），从零训练请自行放宽。
4. **LLM 会被非搜索空间参数带偏**。开发过程中出现过 Agent 反复调 `imgsz`（800→960→1024→1280）
   却不动学习率的情况；现已在 prompt 中显式约束参数范围，但换模型时仍建议检查前几轮输出。
5. **成本**：每轮一次训练 + 一次 LLM 调用。PCB 案例每轮约 6–10 分钟（RTX 4060 8GB）。

---

## 致谢

- 核心思想来自 **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)**——
  "把研究流程写成 program.md 交给 Agent" 的范式。
- 训练与评估基于 [Ultralytics](https://github.com/ultralytics/ultralytics)。
- LLM 通过任意 OpenAI 兼容接口调用（开发时使用 DeepSeek）。

## License

[MIT](LICENSE)
