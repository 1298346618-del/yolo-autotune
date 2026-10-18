"""
核心抽象接口（Phase 2）
定义 Policy 和 Environment 的契约，使主循环与具体实现解耦
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


# ============================================================
# 数据对象
# ============================================================

@dataclass
class PolicyDecision:
    """Policy 的决策结果"""
    config: Dict[str, Any]           # 新的超参数配置
    description: str                 # 实验描述


@dataclass
class ExperimentResult:
    """一次实验的执行结果"""
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 抽象基类
# ============================================================

class BasePolicy(ABC):
    """
    策略抽象基类
    职责：基于历史实验记录，生成新的超参数假设
    """

    def __init__(self, search_space: Dict[str, Tuple[float, float]], **kwargs):
        self.search_space = search_space

    @abstractmethod
    def decide(self, history: List[Dict], best: Optional[Dict] = None) -> PolicyDecision:
        """
        基于历史实验记录，生成新的超参数配置

        Args:
            history: TSV 历史记录列表（Dict 列表）
            best: 当前最佳实验记录（含 config），可能为 None

        Returns:
            PolicyDecision: 包含新配置和实验描述
        """


class BaseEnvironment(ABC):
    """
    环境抽象基类
    职责：执行训练实验并返回结构化结果
    """

    @abstractmethod
    def describe(self) -> str:
        """
        返回环境描述，供 LLM 或其他决策者理解当前任务上下文
        """

    @abstractmethod
    def execute(self, hyperparams: Dict[str, Any], context: Optional[Dict] = None) -> ExperimentResult:
        """
        使用给定的超参数执行一次训练实验

        Args:
            hyperparams: 超参数字典（由 Policy 生成）
            context: 实验上下文（如 iteration、experiment_name 等）

        Returns:
            ExperimentResult: 标准化的实验结果
        """
