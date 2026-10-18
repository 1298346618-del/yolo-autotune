"""
Git 版本控制快照与回滚
用于实验失败时恢复到上一个成功的代码状态
"""

import subprocess

from config.settings import CONFIG_PATH


def git_commit(description: str, use_git: bool = False) -> str:
    """
    将当前超参数配置 git commit，返回 commit hash
    如果未启用 git 或 git 不可用，返回空字符串
    """
    if not use_git:
        return ""
    try:
        # 检查是否在 git 仓库中
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return ""

        # 添加配置文件并提交
        subprocess.run(["git", "add", CONFIG_PATH], check=False, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"exp: {description}"],
            check=False, capture_output=True
        )

        # 获取最新 commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"[Git] 警告: {e}")
    return ""


def git_reset(use_git: bool = False):
    """回滚到上一个 commit"""
    if not use_git:
        return False
    try:
        subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            check=False, capture_output=True, timeout=10
        )
        return True
    except Exception as e:
        print(f"[Git] 回滚失败: {e}")
        return False
