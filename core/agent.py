"""
AgentKernel（Phase 2）
将主循环封装为可复用的 Agent 核心，与具体策略和环境解耦
"""

import os
import shutil
import yaml
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import CONFIG_PATH, TSV_PATH
from core.interfaces import BasePolicy, BaseEnvironment, PolicyDecision, ExperimentResult
from core.history import init_tsv, load_tsv_history, log_tsv
from core.rollback import git_commit, git_reset
from core.loop_utils import apply_simplicity_criterion, should_continue, cleanup_old_runs
from utils.config_io import load_config, save_config


class AgentKernel:
    """
    Agent 核心内核

    职责：
    1. 协调 Policy（决策者）和 Environment（执行者）
    2. 管理实验历史、最佳结果、回滚逻辑
    3. 实现提议 → 执行 → 评估 → 保留/回滚 的主循环

    使用方式：
        agent = AgentKernel(policy=LLMPolicy(...), env=YOLOEnvironment(...))
        agent.run()
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseEnvironment,
        max_iterations: int = 50,
        patience: int = 10,
        simplification_threshold: float = 0.002,
        top_k_checkpoint: int = 3,
        config_path: str = CONFIG_PATH,
        use_git: bool = False,
    ):
        self.policy = policy
        self.env = env
        self.max_iterations = max_iterations
        self.patience = patience
        self.simplification_threshold = simplification_threshold
        self.top_k_checkpoint = top_k_checkpoint
        self.config_path = config_path
        self.use_git = use_git

        self.best_experiment: Optional[Dict] = None
        self._history: List[Dict] = []

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def _setup(self):
        """初始化 TSV、加载历史、恢复最佳实验"""
        init_tsv()
        self._history = load_tsv_history()
        print(f"[历史] 已加载 {len(self._history)} 轮实验记录")

        crash_recent = len([h for h in self._history[-10:] if h.get("status") == "crash"])
        if crash_recent > 3:
            print(f"[警告] 最近 10 轮中有 {crash_recent} 轮失败，建议检查显存或降低初始配置")

        # 恢复当前最佳实验
        kept_experiments = [h for h in self._history if h["status"] == "keep"]
        if kept_experiments:
            self.best_experiment = max(kept_experiments, key=lambda x: x["map50"])
            print(f"[历史] 当前最佳 mAP50: {self.best_experiment['map50']:.4f}")

            # 如果 TSV 中没有 config（重启后），尝试从 best 目录恢复
            if "config" not in self.best_experiment:
                best_dir = "yolo_autotune_best"
                if os.path.exists(best_dir):
                    # 按迭代序号数字排序（best_iter2 < best_iter10），避免字典序错乱
                    config_files = sorted(
                        [f for f in os.listdir(best_dir) if f.endswith('_config.yaml')],
                        key=lambda f: int(f.split('_iter')[1].split('_config')[0])
                    )
                    if config_files:
                        latest_cfg_path = os.path.join(best_dir, config_files[-1])
                        self.best_experiment["config"] = load_config(latest_cfg_path)
                        print(f"[历史] 从 {latest_cfg_path} 恢复最佳配置")

    # --------------------------------------------------------
    # 主循环
    # --------------------------------------------------------

    def run(self):
        """运行实验主循环"""
        print("=" * 70)
        print("YOLO Auto-Tune Agent")
        print("基于 autoresearch 思想的自动调优框架")
        print("=" * 70)
        print(f"\n[环境] {self.env.describe()}")

        self._setup()

        start_iteration = len(self._history)
        end_iteration = start_iteration + self.max_iterations

        for iteration in range(start_iteration, end_iteration):
            print(f"\n{'=' * 70}")
            print(f"[Agent] 第 {iteration + 1}/{end_iteration} 轮实验")
            print(f"{'=' * 70}")

            # 1. Policy 决策：生成新的超参数配置
            current_config = load_config(self.config_path)
            decision = self.policy.decide(self._history, self.best_experiment)
            new_config = decision.config
            description = decision.description

            # 合并配置并保存
            merged_config = {**current_config, **new_config}
            save_config(merged_config, self.config_path)
            print(f"[Agent] 新配置:\n{yaml.dump(merged_config, allow_unicode=True)}")

            # 2. Environment 执行：运行训练实验
            exp_name = f"experiment_{iteration:03d}_{datetime.now().strftime('%m%d_%H%M')}"
            result = self.env.execute(
                merged_config,
                context={"iteration": iteration, "experiment_name": exp_name}
            )

            # 如果 OOM 恢复触发了，在描述中标记
            actual_batch = result.artifacts.get("actual_batch")
            actual_imgsz = result.artifacts.get("actual_imgsz")
            default_imgsz = result.artifacts.get("default_imgsz", 640)
            default_batch = getattr(self.env, "default_batch", 16)
            if actual_batch and actual_batch != default_batch:
                description += f" [OOM恢复 batch={actual_batch}]"
            if actual_imgsz and actual_imgsz != default_imgsz:
                description += f" [OOM恢复 imgsz={actual_imgsz}]"

            # 3. 评估与决策：保留 vs 丢弃
            status = self._evaluate_and_decide(result, merged_config, description, iteration)

            # 4. 记录到 TSV
            commit_id = f"iter{iteration}"
            log_tsv(
                commit_id,
                result.metrics.get("map50", 0.0),
                result.metrics.get("map50_95", 0.0),
                status,
                description
            )
            self._history = load_tsv_history()

            # 5. 早停检查
            if not should_continue(self._history, self.patience):
                print(f"\n[早停] 连续 {self.patience} 轮无提升，停止实验")
                break

            # 6. 清理旧实验
            runs_project = os.path.join("runs", "detect", getattr(self.env, "project", "yolo_autotune_runs"))
            cleanup_old_runs(project=runs_project, keep=self.top_k_checkpoint + 2)

        # 最终报告
        self._report()

    # --------------------------------------------------------
    # 评估与决策
    # --------------------------------------------------------

    def _evaluate_and_decide(
        self,
        result: ExperimentResult,
        merged_config: Dict,
        description: str,
        iteration: int
    ) -> str:
        """
        根据实验结果决定保留或丢弃
        返回: status ("keep" / "discard" / "crash")
        """
        if not result.success:
            print(f"[错误] 训练失败: {result.error or '未知错误'}")
            self._rollback_config()
            git_reset(self.use_git)
            return "crash"

        # 训练成功
        print(f"[结果] mAP50: {result.metrics.get('map50', 0):.4f} | "
              f"mAP50-95: {result.metrics.get('map50_95', 0):.4f} | "
              f"耗时: {result.elapsed_seconds / 60:.1f} 分钟")

        # 简化原则判断
        result_dict = {
            "map50": result.metrics.get("map50", 0.0),
            "map50_95": result.metrics.get("map50_95", 0.0),
        }
        best_result = None
        if self.best_experiment:
            best_result = {
                "map50": self.best_experiment["map50"],
                "map50_95": self.best_experiment["map50_95"],
            }
        best_config = self.best_experiment.get("config") if self.best_experiment else None

        keep_new, reason = apply_simplicity_criterion(
            result_dict, best_result,
            merged_config, best_config,
            self.simplification_threshold
        )
        print(f"[决策] {reason}")

        if keep_new:
            self._keep_result(result, merged_config, description, iteration)
            return "keep"
        else:
            self._rollback_config()
            git_reset(self.use_git)
            return "discard"

    def _keep_result(
        self,
        result: ExperimentResult,
        config: Dict,
        description: str,
        iteration: int
    ):
        """保留新结果为当前最佳"""
        self.best_experiment = {
            "commit": "",
            "map50": result.metrics.get("map50", 0.0),
            "map50_95": result.metrics.get("map50_95", 0.0),
            "status": "keep",
            "description": description,
            "config": config,
        }

        # Git commit（如果启用）
        commit_hash = git_commit(description, self.use_git)
        if commit_hash:
            self.best_experiment["commit"] = commit_hash

        # 保存最佳权重
        self._save_checkpoint(iteration, result, config)

    def _rollback_config(self):
        """回滚配置到历史最佳"""
        if self.best_experiment and "config" in self.best_experiment:
            save_config(self.best_experiment["config"], self.config_path)
            print("[回滚] 恢复到历史最佳配置")
        else:
            print("[回滚] 无历史最佳配置，保持当前配置")

    def _save_checkpoint(self, iteration: int, result: ExperimentResult, config: Dict):
        """保存最佳权重和配置"""
        best_dir = "yolo_autotune_best"
        os.makedirs(best_dir, exist_ok=True)

        weights_path = result.artifacts.get("weights_path")
        if weights_path and os.path.exists(weights_path):
            shutil.copy2(weights_path, os.path.join(best_dir, f"best_iter{iteration}.pt"))
            with open(os.path.join(best_dir, f"best_iter{iteration}_config.yaml"), 'w') as f:
                yaml.dump(config, f, allow_unicode=True)

    # --------------------------------------------------------
    # 报告
    # --------------------------------------------------------

    def _report(self):
        """输出最终实验报告"""
        print(f"\n{'=' * 70}")
        print("[完成] YOLO Auto-Tune 实验结束")
        print(f"{'=' * 70}")

        if self.best_experiment:
            print(f"\n最佳实验:")
            print(f"  mAP50: {self.best_experiment['map50']:.4f}")
            print(f"  mAP50-95: {self.best_experiment['map50_95']:.4f}")
            print(f"  描述: {self.best_experiment['description']}")
            print(f"  配置:\n{yaml.dump(self.best_experiment.get('config', {}), allow_unicode=True)}")
            print(f"\n最佳权重保存在: yolo_autotune_best/")
        else:
            print("未能完成任何成功的实验")

        print(f"\n实验记录保存在: {TSV_PATH}")
