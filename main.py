"""
YOLO Auto-Tune Agent —— Phase 2 入口

极简入口：组装 Policy + Environment + AgentKernel，然后运行
"""

from config.settings import CONFIG
from core.agent import AgentKernel
from env.yolo import YOLOEnvironment
from policy.llm import LLMPolicy
from policy.random import RandomPolicy


def main():
    # --------------------------------------------------------
    # 1. 创建训练环境
    # --------------------------------------------------------
    env = YOLOEnvironment(
        data_yaml=CONFIG["data_yaml"],
        model_path=CONFIG["model"],
        epochs=CONFIG["epochs"],
        time_budget_seconds=CONFIG["time_budget_seconds"],
        imgsz=CONFIG["imgsz"],
        device=CONFIG["device"],
        pretrained=CONFIG.get("pretrained", True),
        default_batch=CONFIG.get("batch", 16),
        optimizer=CONFIG.get("optimizer", "SGD"),
    )

    # --------------------------------------------------------
    # 2. 创建决策策略
    # --------------------------------------------------------
    if CONFIG["llm_api_key"]:
        policy = LLMPolicy(
            search_space=CONFIG["search_space"],
            api_key=CONFIG["llm_api_key"],
            base_url=CONFIG["llm_base_url"],
            model=CONFIG["llm_model"],
            task_profile_path=CONFIG.get("task_profile"),
        )
    else:
        policy = RandomPolicy(search_space=CONFIG["search_space"])

    # --------------------------------------------------------
    # 3. GPU 显存预检查
    # --------------------------------------------------------
    from env.yolo import check_gpu_memory
    gpu_ok, gpu_info = check_gpu_memory(device=CONFIG["device"], min_required_mb=4096)
    print(f"\n{gpu_info}")
    if not gpu_ok:
        print("[警告] GPU 显存可能不足，OOM 恢复机制将尝试降级 batch/imgsz")
        print("[建议] 如需避免 OOM，可手动将 CONFIG['imgsz'] 改为 480 或 320")

    # --------------------------------------------------------
    # 4. 创建 Agent 核心并运行
    # --------------------------------------------------------
    agent = AgentKernel(
        policy=policy,
        env=env,
        max_iterations=CONFIG["max_iterations"],
        patience=CONFIG["patience"],
        simplification_threshold=CONFIG["simplification_threshold"],
        top_k_checkpoint=CONFIG["top_k_checkpoint"],
        use_git=CONFIG.get("use_git", False),
    )
    agent.run()


if __name__ == "__main__":
    main()
