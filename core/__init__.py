"""
core 包：Agent 核心基础设施

包含抽象接口、主循环内核、历史记录、回滚机制等
"""

from core.interfaces import BasePolicy, BaseEnvironment, PolicyDecision, ExperimentResult
from core.agent import AgentKernel

__all__ = [
    "BasePolicy",
    "BaseEnvironment",
    "PolicyDecision",
    "ExperimentResult",
    "AgentKernel",
]
