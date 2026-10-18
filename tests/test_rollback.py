"""Git 回滚的安全护栏：绝不能动用户的提交。

这些测试需要本机有 git；没有则整体跳过。
"""

import shutil
import subprocess

import pytest

from core.rollback import git_commit, git_reset

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git 未安装")


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _make_user_repo(tmp_path):
    """造一个只有一条"用户自己的提交"的仓库。"""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "work.txt").write_text("important user work", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "my own important commit")
    return tmp_path


def test_refuses_to_reset_user_commit(tmp_path, monkeypatch):
    repo = _make_user_repo(tmp_path)
    monkeypatch.chdir(repo)
    before = _git(repo, "rev-parse", "HEAD")

    assert git_reset(True) is False

    assert _git(repo, "rev-parse", "HEAD") == before
    assert (repo / "work.txt").read_text(encoding="utf-8") == "important user work"


def test_agent_commit_can_be_rolled_back(tmp_path, monkeypatch):
    repo = _make_user_repo(tmp_path)
    monkeypatch.chdir(repo)
    user_head = _git(repo, "rev-parse", "HEAD")

    cfg = repo / "yolo_autotune_config.yaml"
    cfg.write_text("lr0: 0.01\n", encoding="utf-8")
    agent_hash = git_commit("test experiment", use_git=True, config_path=str(cfg))

    assert agent_hash, "应该产生了快照"
    assert _git(repo, "rev-parse", "--short", "HEAD") == agent_hash
    assert _git(repo, "log", "-1", "--format=%s").startswith("exp:")

    assert git_reset(True, agent_hash) is True
    assert _git(repo, "rev-parse", "HEAD") == user_head


def test_reset_rejects_hash_mismatch(tmp_path, monkeypatch):
    repo = _make_user_repo(tmp_path)
    monkeypatch.chdir(repo)

    cfg = repo / "yolo_autotune_config.yaml"
    cfg.write_text("lr0: 0.01\n", encoding="utf-8")
    git_commit("snapshot A", use_git=True, config_path=str(cfg))

    # 传入一个不存在的 hash，应当拒绝回滚
    assert git_reset(True, "deadbeef") is False


def test_disabled_git_is_noop(tmp_path, monkeypatch):
    repo = _make_user_repo(tmp_path)
    monkeypatch.chdir(repo)
    before = _git(repo, "rev-parse", "HEAD")

    assert git_commit("x", use_git=False) == ""
    assert git_reset(False) is False
    assert _git(repo, "rev-parse", "HEAD") == before
