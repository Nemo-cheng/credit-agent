"""
基础类和数据模型定义
定义智能体框架中使用的基础数据结构和抽象基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CreditData:
    """信用数据容器，存储各阶段的信用相关数据"""
    raw_data: Dict[str, Any] = field(default_factory=dict)  # 原始数据
    structured_data: pd.DataFrame = field(default_factory=pd.DataFrame)  # 结构化数据
    feature_data: pd.DataFrame = field(default_factory=pd.DataFrame)  # 特征数据
    risk_labels: Dict[str, Any] = field(default_factory=dict)  # 风险标签
    credit_score: Optional[float] = None  # 信用分数
    credit_rating: Optional[str] = None  # 信用等级(AAA到D)
    industry: Optional[str] = None  # 行业类型
    company_id: Optional[str] = None  # 企业唯一标识

@dataclass
class AnalysisResult:
    """分析结果容器"""
    features: pd.DataFrame = field(default_factory=pd.DataFrame)  # 特征数据
    weights: Dict[str, float] = field(default_factory=dict)  # 各指标权重
    consistency_ratio: Optional[float] = None  # AHP一致性比率
    selected_features: List[str] = field(default_factory=list)  # 筛选后的特征列表

@dataclass
class RiskAnalysisResult:
    """风险分析结果容器"""
    risk_labels: Dict[str, Any] = field(default_factory=dict)  # 风险标签
    risk_level: str = "unknown"  # 风险等级(高/中/低)
    risk_details: Dict[str, Any] = field(default_factory=dict)  # 风险详情
    credit_rating: Optional[str] = None  # 信用等级

@dataclass
class ReportGenerationResult:
    """报告生成结果容器"""
    report: Dict[str, Any] = field(default_factory=dict)  # 生成的报告

@dataclass
class FeedbackResult:
    """反馈结果容器"""
    error_metrics: Dict[str, float] = field(default_factory=dict)  # 误差指标
    hidden_risk_rate: float = 0.0  # 隐性风险漏判率
    industry_bias_rate: float = 0.0  # 行业适配偏差率
    weight_adjustments: Dict[str, float] = field(default_factory=dict)  # 权重调整建议
    rule_adjustments: Dict[str, Any] = field(default_factory=dict)  # 规则调整建议

class BaseModule(ABC):
    """模块基类，所有模块都需要继承此类并实现抽象方法"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化模块"""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """处理数据的核心方法"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> Tuple[bool, str]:
        """验证输入数据的有效性"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置模块状态"""
        pass
