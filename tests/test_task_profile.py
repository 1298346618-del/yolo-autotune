"""任务画像（task_profile.yaml）与 prompt 组装。"""

from policy.llm import (
    DEFAULT_TASK_PROFILE,
    build_system_prompt,
    load_task_profile,
)


def test_load_profile_from_yaml(tmp_path):
    p = tmp_path / "task.yaml"
    p.write_text(
        "task_name: 无人机小目标检测\n"
        "training_mode: scratch\n"
        "priorities:\n  - lr0\n  - box\n",
        encoding="utf-8",
    )
    profile = load_task_profile(str(p))
    assert profile["task_name"] == "无人机小目标检测"
    assert profile["training_mode"] == "scratch"
    assert profile["priorities"] == ["lr0", "box"]


def test_missing_file_falls_back_to_default():
    profile = load_task_profile("/definitely/not/here.yaml")
    assert profile["task_name"] == DEFAULT_TASK_PROFILE["task_name"]


def test_broken_yaml_falls_back(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("task_name: [unclosed\n", encoding="utf-8")
    profile = load_task_profile(str(p))
    assert profile["task_name"] == DEFAULT_TASK_PROFILE["task_name"]


def test_prompt_contains_task_specifics():
    profile = {
        "task_name": "PCB 缺陷检测",
        "task_description": "6 类缺陷",
        "dataset_notes": "449 张训练图",
        "training_mode": "finetune",
        "domain_hints": "box 权重对定位很重要",
        "priorities": ["lr0", "box"],
    }
    prompt = build_system_prompt(profile)
    assert "PCB 缺陷检测" in prompt
    assert "6 类缺陷" in prompt
    assert "449 张训练图" in prompt
    assert "微调模式" in prompt
    assert "lr0 > box" in prompt
    # 输出格式约束必须存在，否则 LLM 返回的内容无法解析
    assert "```yaml" not in prompt  # 这是 system prompt，格式要求用文字描述
    assert "YAML" in prompt


def test_scratch_mode_changes_wording():
    prompt = build_system_prompt({"task_name": "t", "training_mode": "scratch"})
    assert "从零训练" in prompt


def test_prompt_with_missing_optional_fields_still_works():
    prompt = build_system_prompt({})
    assert isinstance(prompt, str) and len(prompt) > 100
