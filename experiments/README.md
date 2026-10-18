# 实验记录（真实运行数据，未做修饰）

这里保存的是 Agent 实际跑出来的 `results.tsv`。字段含义与
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 完全一致：

```
commit      map50       map50_95    status      description
iter17      0.783617    0.555783    keep        增大box至15.0，降低mosaic至0.1，微调lr0至0.008
```

- `commit`：有 git 时是 commit hash，否则是迭代号
- `status`：`keep`（保留为新最佳）/ `discard`（丢弃并回滚）/ `crash`（训练失败）
- `description`：当轮改了什么的自然语言说明（LLM 自己写的）

---

## 文件索引

| 文件 | 实验 | 轮数 | 结果 |
|---|---|---|---|
| `results_pcb_backup.tsv` | **PCB 缺陷检测**，YOLO26n，初始配置 → 随机搜索 → LLM Agent | 33 | 0.7568 → **0.7836**（mAP50），+3.5% |
| `yolo26n_stage1/results.tsv` | PCB，从 scratch 权重微调（stage 1） | 17 | 0.7465 最佳（相对 scratch +9.5%） |
| `yolo26s_finetune/results.tsv` | 更大模型 YOLO26s，**纯随机搜索** | 22（前 10 轮 OOM） | 0.8549 → **0.9032**，+5.6% |
| `results_expression_trial.tsv` | 小样本对照（84 训练图 / 36 验证图） | 11 | **无提升**，10/11 被丢弃 |
| `results_expr_trial_20260817_1252.tsv` | 同一小样本实验的另一次存档 | — | 同上 |
| `yolo_autotune_config_20260817_1252.yaml` | 上述实验对应的超参数快照 | — | — |
| `RUN_NOTES.md` | 作者当时手写的两轮实验总结（含最佳超参数全文） | — | — |

---

## 阅读这些数据时请注意三件事

### 1. `results_pcb_backup.tsv` 里第 2–16 行是**同一个数字**

```
0.741739  0.741739  0.741739 ...  (连续 15 行完全相同)
```

这不是巧合，而是当时随机搜索**失效**：ultralytics 在训练内部调用了 `set_seed()`，
把全局 `random` 的状态重置成同一个值，于是每轮采样出完全相同的配置。
该问题已在 [policy/random.py](../policy/random.py) 中修复（每次采样前重新 `random.seed()`）。

**因此这张表里"随机搜索 15 轮停滞"不能作为"随机搜索不行"的证据**——
它只能说明当时的随机基线是坏的。第 17 行之后才是真正有效的 LLM 决策记录。

### 2. `results_expression_trial.tsv` 是刻意的负结果

数据集只有 84 张训练图 / 36 张验证图时，验证集 mAP 的波动幅度（±0.05）远大于超参数带来的
真实差异（通常 0.002–0.02）。Agent 会把噪声当成提升，于是连"看起来更好"的配置也大多被判为
`discard`。**这不是框架坏了，而是数据量撑不起超参数搜索**——先扩数据，再调参。

### 3. `yolo26s_finetune/results.tsv` 前 10 轮是 OOM

`0.000000 / crash` 的连续记录是 8GB 显存下的 OOM，不是模型收敛失败。
OOM 5 级降级机制就是在这之后加上的。

---

## 想自己复现？

```bash
python main.py          # 会在仓库根目录生成新的 results.tsv
```

生成的 `results.tsv` 已被 `.gitignore` 排除，不会污染这些存档。想归档自己的实验时，
把文件复制到本目录并在此表中登记一行即可。
