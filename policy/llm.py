"""
LLM 智能策略（Phase 2）
包含 LLM 调用函数和 LLMPolicy 类
"""

import os
import yaml
from typing import Dict, List, Optional, Tuple

from core.interfaces import BasePolicy, PolicyDecision
from policy.random import generate_random_config


# ============================================================
# 任务画像：把"任务相关的先验"从代码里剥离出来
# ============================================================

DEFAULT_TASK_PROFILE: Dict = {
    "task_name": "目标检测任务",
    "task_description": "通用的 YOLO 目标检测超参数调优任务。",
    "dataset_notes": "- 数据规模与类别分布请参考数据集本身的统计。",
    "training_mode": "finetune",
    "domain_hints": (
        "- 分析历史趋势：哪些参数变化时 mAP 上升？\n"
        "- 数据量小时优先防过拟合，增强强度不宜过激。\n"
        "- 微调模式下 lr0 保持保守，warmup 可缩短。"
    ),
    "priorities": ["lr0", "box", "cls", "数据增强强度", "其他参数"],
}


def load_task_profile(path: Optional[str] = None) -> Dict:
    """读取任务画像 YAML；文件缺失或解析失败时回退到通用默认值。"""
    profile = dict(DEFAULT_TASK_PROFILE)
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                profile.update(loaded)
        except Exception as e:  # 配置坏了不应该让整个 Agent 挂掉
            print(f"[Agent] 任务画像读取失败({e})，使用通用默认提示词")
    return profile


def build_system_prompt(profile: Dict) -> str:
    """用任务画像组装 LLM 的 system prompt。"""
    priorities = " > ".join(profile.get("priorities") or DEFAULT_TASK_PROFILE["priorities"])
    mode = profile.get("training_mode", "finetune")
    mode_hint = (
        "微调模式（在已训练权重上继续训练，起点较好）"
        if mode == "finetune"
        else "从零训练模式（没有预训练起点，学习率可以更大）"
    )
    return f"""你是一个目标检测模型调优专家 Agent。
你的任务是基于历史实验记录，提出新的超参数假设并生成配置。

当前任务背景：
- 任务：{profile.get('task_name', '目标检测任务')}
- 描述：{str(profile.get('task_description', '')).strip()}
- 训练模式：{mode_hint}
- 数据特点：
{str(profile.get('dataset_notes', '')).strip()}
- 领域经验：
{str(profile.get('domain_hints', '')).strip()}

核心原则：
1. 分析历史趋势：哪些参数增大时 mAP 提升？哪些改动反而变差？
2. 尊重领域经验，但以实验证据为准：如果数据与经验冲突，相信数据
3. 简化原则：如果减少某个增强后 mAP 不变，优先简化
4. 每次只改动少数几个参数，保证能归因
5. 优先级：{priorities}

输出要求：
- 先写 1-2 行分析（你的假设是什么）
- 再输出 YAML 格式的超参数配置
- 最后写 1 行简短的实验描述（用于 results.tsv）"""


def fallback_from_best(search_space: Dict, best: Optional[Dict], n_perturb: int = 4) -> Dict:
    """LLM 失败时基于当前最佳配置做小幅随机扰动，替代纯随机搜索"""
    import random
    random.seed()  # 避免被 ultralytics 训练内的 set_seed() 重置导致重复配置
    base = dict(best.get("config", {})) if best and best.get("config") else {}
    config = {}
    for key, (min_v, max_v) in search_space.items():
        if key in base:
            v = float(base[key])
        else:
            v = random.uniform(min_v, max_v)
        # 在 best 值附近扰动（取搜索空间的 10%~20% 作为扰动幅度）
        span = max_v - min_v
        v = v + random.uniform(-0.15 * span, 0.15 * span)
        if key in ["mosaic", "mixup", "copy_paste", "dropout"] and random.random() < 0.3:
            v = 0.0
        config[key] = round(max(min_v, min(max_v, v)), 6)
    return config


def generate_llm_config(
    search_space: Dict,
    tsv_history: List[Dict],
    current_best: Optional[Dict] = None,
    api_key: str = "",
    base_url: str = "",
    model: str = "deepseek-chat",
    task_profile: Optional[Dict] = None
) -> Tuple[Dict, str]:
    """
    调用 LLM API 生成新的超参数配置（Agent 模式）
    返回: (新配置, 实验描述)

    task_profile: 由 config/task_profile.yaml 加载的任务画像，
                  决定 system prompt 里的任务背景与领域经验
    """
    task_profile = task_profile or DEFAULT_TASK_PROFILE
    if not api_key:
        print("[Agent] 未配置 LLM API，回退到随机搜索")
        return generate_random_config(search_space), "随机搜索（无 LLM）"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 搜索空间描述
        space_desc = "\n".join([
            f"  {k}: [{v[0]}, {v[1]}]"
            for k, v in search_space.items()
        ])

        # TSV 历史摘要（最近 5 轮，跳过 crash 记录避免误导）
        valid_history = [h for h in tsv_history if h.get("status") != "crash"]
        recent_history = valid_history[-5:] if len(valid_history) > 5 else valid_history
        history_summary = "\n".join([
            f"  轮次 {i+1}: mAP50={h['map50']:.4f}, status={h['status']}, desc={h['description']}"
            for i, h in enumerate(recent_history)
        ]) if recent_history else "  无历史记录"
        crash_count = len([h for h in tsv_history[-10:] if h.get("status") == "crash"])
        if crash_count > 0:
            history_summary += f"\n  [注意] 最近 10 轮中有 {crash_count} 轮因 OOM/错误失败]"

        # 当前最佳
        best_summary = ""
        if current_best:
            best_summary = f"\n当前最佳配置: mAP50={current_best['map50']:.4f}, config={yaml.dump(current_best.get('config', {}), allow_unicode=True).strip()}"

        system_prompt = build_system_prompt(task_profile)

        user_prompt = f"""=== 超参数搜索空间 ===
{space_desc}

=== 历史实验记录（最近5轮） ===
{history_summary}
{best_summary}

=== 任务 ===
请基于以上信息，生成一组新的超参数配置（所有参数都必须在搜索空间范围内）。

输出格式：
分析: <你的分析和假设>
```yaml
lr0: 0.01
lrf: 0.01
momentum: 0.937
...
```
描述: <简短描述，如"增大 lr0 到 0.02"，用于 results.tsv>"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        content = response.choices[0].message.content

        # 提取分析、YAML 和描述
        analysis = ""
        description = "LLM 生成"
        yaml_str = ""

        lines = content.split('\n')
        in_yaml = False
        yaml_lines = []

        for line in lines:
            if line.startswith("分析:"):
                analysis = line.replace("分析:", "").strip()
            elif line.startswith("描述:"):
                description = line.replace("描述:", "").strip()
            elif line.strip() == "```yaml":
                in_yaml = True
            elif line.strip() == "```" and in_yaml:
                in_yaml = False
            elif in_yaml:
                yaml_lines.append(line)

        if yaml_lines:
            yaml_str = '\n'.join(yaml_lines)
        else:
            # 回退：直接提取代码块
            if "```yaml" in content:
                yaml_str = content.split("```yaml")[1].split("```")[0].strip()
            elif "```" in content:
                yaml_str = content.split("```")[1].split("```")[0].strip()
            else:
                yaml_str = content.strip()

        new_config = yaml.safe_load(yaml_str)
        if not isinstance(new_config, dict):
            # 兜底：正则逐行提取 key: value（兼容 LLM 格式跑偏）
            import re
            new_config = {}
            for line in yaml_str.split('\n'):
                m = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(-?[\d.]+)', line)
                if m:
                    new_config[m.group(1)] = float(m.group(2))
            if not new_config:
                raise ValueError("LLM 输出不是有效的 YAML 字典")

        # 验证并截断到搜索空间范围内（逐键容错：坏键跳过，不整段作废）
        validated_config = {}
        skipped_keys = []
        for key, value in new_config.items():
            if key not in search_space:
                skipped_keys.append(key)
                continue
            try:
                num_value = float(value)
            except (TypeError, ValueError):
                skipped_keys.append(key)
                continue
            min_v, max_v = search_space[key]
            validated_config[key] = max(min_v, min(max_v, num_value))

        if skipped_keys:
            print(f"[LLM] 忽略无效键: {skipped_keys}")
        if not validated_config:
            raise ValueError("LLM 输出中没有有效的搜索空间参数")

        print(f"[LLM] 分析: {analysis}")
        return validated_config, description

    except Exception as e:
        print(f"[Agent] LLM 调用失败: {e}，回退到基于最佳配置微调")
        if current_best and current_best.get("config"):
            config = fallback_from_best(search_space, current_best)
            return config, f"基于最佳微调（LLM 失败: {str(e)[:30]}）"
        return generate_random_config(search_space), f"随机搜索（LLM 失败: {str(e)[:30]}）"


class LLMPolicy(BasePolicy):
    """
    LLM 智能策略
    调用大模型 API 基于历史实验记录生成新的超参数配置
    """

    def __init__(
        self,
        search_space: Dict,
        api_key: str = "",
        base_url: str = "",
        model: str = "deepseek-chat",
        task_profile_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(search_space, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.task_profile = load_task_profile(task_profile_path)
        print(f"[LLM] 任务画像: {self.task_profile.get('task_name', '未命名')}")

    def decide(self, history: List[Dict], best: Optional[Dict] = None) -> PolicyDecision:
        config, description = generate_llm_config(
            self.search_space,
            history,
            best,
            self.api_key,
            self.base_url,
            self.model,
            self.task_profile
        )
        return PolicyDecision(config=config, description=description)
