"""
Git 版本控制快照与回滚

⚠️ 设计原则（重要）
本模块**只允许回滚 Agent 自己创建的快照提交**。
早期实现直接执行 `git reset --hard HEAD~1`，如果用户在自己的代码仓库里打开
`use_git=True`，一次 discard 就会把用户自己的提交和未提交改动一起抹掉。
现在所有回滚都必须通过 `_is_agent_commit()` 校验，否则一律拒绝。
"""

import os
import subprocess

from config.settings import CONFIG_PATH

# Agent 创建的提交都带这个前缀，用于区分人类提交
AGENT_COMMIT_PREFIX = "exp:"


def _run(args, timeout=10):
    """执行 git 命令，返回 (returncode, stdout, stderr)。"""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def _is_agent_commit(rev: str = "HEAD") -> bool:
    """判断某个提交是否是 Agent 自己创建的（靠提交信息前缀识别）。"""
    code, out, _ = _run(["git", "log", "-1", "--format=%s", rev])
    return code == 0 and out.startswith(AGENT_COMMIT_PREFIX)


def git_commit(description: str, use_git: bool = False, config_path: str = CONFIG_PATH) -> str:
    """
    把当前超参数配置提交为一个快照，返回短 hash。
    未启用 git / 不在 git 仓库 / 没有变更时返回空字符串。
    """
    if not use_git:
        return ""

    code, _, _ = _run(["git", "rev-parse", "--git-dir"], timeout=5)
    if code != 0:
        print("[Git] 当前目录不是 git 仓库，跳过快照")
        return ""

    # 用绝对路径，避免 cwd 不同导致 add 静默失败
    abs_config = os.path.abspath(config_path)
    if not os.path.exists(abs_config):
        print(f"[Git] 配置文件不存在，跳过快照: {abs_config}")
        return ""

    _run(["git", "add", abs_config])
    code, _, _ = _run(
        ["git", "commit", "-m", f"{AGENT_COMMIT_PREFIX} {description}", "--", abs_config]
    )
    if code != 0:
        # 通常是"没有变更"，不是错误
        return ""

    code, out, _ = _run(["git", "rev-parse", "--short", "HEAD"])
    return out if code == 0 else ""


def git_reset(use_git: bool = False, expected_commit: str = "") -> bool:
    """
    回滚 Agent 自己创建的最近一个快照提交。

    安全约束（缺一不可）：
      1. use_git 必须为 True
      2. HEAD 必须是 Agent 创建的提交（信息以 "exp:" 开头）
      3. 如果传了 expected_commit，HEAD 必须与它匹配

    任何一条不满足都只打印提示并返回 False，**绝不触碰用户的提交**。
    """
    if not use_git:
        return False

    code, head_hash, _ = _run(["git", "rev-parse", "--short", "HEAD"])
    if code != 0:
        print("[Git] 无法读取 HEAD，跳过回滚")
        return False

    if not _is_agent_commit("HEAD"):
        print("[Git] HEAD 不是 Agent 创建的快照，拒绝回滚（保护你的提交）")
        return False

    if expected_commit and not head_hash.startswith(expected_commit.strip()[:7]):
        print(f"[Git] HEAD({head_hash}) 与预期快照({expected_commit}) 不一致，拒绝回滚")
        return False

    code, _, err = _run(["git", "reset", "--hard", "HEAD~1"])
    if code != 0:
        print(f"[Git] 回滚失败: {err[:120]}")
        return False
    return True
