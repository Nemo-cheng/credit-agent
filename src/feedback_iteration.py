"""
反馈迭代模块
负责误差修正与模型逻辑优化
"""
from typing import Dict, Any, Tuple, List, Optional
from events import Event, EventType, EventBus
import pandas as pd
import numpy as np
from datetime import datetime
# 移除pyanp依赖，使用简化的权重调整

from base import BaseModule, CreditData, RiskAnalysisResult, FeedbackResult, logger

class FeedbackIterationModule(BaseModule):
    """反馈迭代模块实现类"""
    
    def __init__(self, config: Dict[str, Any] = None, event_bus: Optional[EventBus] = None):
        """初始化反馈迭代模块"""
        self.config = config or {
            "default_thresholds": {
                "hidden_risk_rate": 0.1,  # 隐性风险漏判率阈值
                "industry_bias_rate": 0.08  # 行业适配偏差率阈值
            },
            "adjustment_strength": 0.1,  # 调整强度（0-1）
            "min_weight": 0.05,  # 最小权重值
            "max_weight": 0.6,   # 最大权重值
            "default_major_lawsuit_threshold": 500000  # 默认重大诉讼阈值
        }
        self.initialized = False
        self.historical_data = pd.DataFrame(columns=[
            "company_id", "predicted_rating", "actual_outcome", "industry",
            "predicted_risk_level", "actual_risk_level", "analysis_date"
        ])
    
    def initialize(self) -> bool:
        """初始化模块"""
        try:
            # 验证配置参数
            if not (0 < self.config["adjustment_strength"] < 1):
                raise ValueError("调整强度必须在0和1之间")
            
            if self.config["min_weight"] >= self.config["max_weight"]:
                raise ValueError("最小权重必须小于最大权重")
            
            # 预填充历史数据以便计算行业偏差
            self._initialize_historical_data()
            
            self.initialized = True
            logger.info("反馈迭代模块初始化完成")
            return True
        except Exception as e:
            logger.error(f"反馈迭代模块初始化失败: {str(e)}")
            self.initialized = False
            return False
    
    def process(self, 
                credit_data: CreditData, 
                risk_result: RiskAnalysisResult, 
                actual_outcome: Dict[str, Any]) -> FeedbackResult:
        """
        处理反馈信息，计算误差并生成调整建议
        
        参数:
            credit_data: 信用数据对象
            risk_result: 风险分析结果
            actual_outcome: 实际结果，包含实际违约情况等
        
        返回:
            反馈结果，包含误差指标和调整建议
        """
        # 验证输入
        valid, message = self.validate_input((credit_data, risk_result, actual_outcome))
        if not valid:
            raise ValueError(f"输入数据无效: {message}")
        
        try:
            logger.info(f"开始处理企业 {credit_data.raw_data.get('enterprise_info', {}).get('company_name')} 的反馈")
            
            # 1. 记录历史数据
            self._record_historical_data(credit_data, risk_result, actual_outcome)
            
            # 2. 计算误差指标
            error_metrics = self._calculate_error_metrics(credit_data, risk_result, actual_outcome)
            
            # 3. 计算隐性风险漏判率
            hidden_risk_rate = self._calculate_hidden_risk_rate(credit_data, actual_outcome)
            
            # 4. 计算行业适配偏差率
            industry_bias_rate = self._calculate_industry_bias_rate(credit_data.industry)
            
            # 5. 生成权重调整建议
            weight_adjustments = self._generate_weight_adjustments(
                credit_data, risk_result, actual_outcome, hidden_risk_rate, industry_bias_rate
            )
            
            # 6. 生成规则调整建议
            rule_adjustments = self._generate_rule_adjustments(
                credit_data, actual_outcome, hidden_risk_rate
            )
            
            # 构建反馈结果
            feedback_result = FeedbackResult(
                error_metrics=error_metrics,
                hidden_risk_rate=hidden_risk_rate,
                industry_bias_rate=industry_bias_rate,
                weight_adjustments=weight_adjustments,
                rule_adjustments=rule_adjustments
            )
            
            logger.info("反馈处理完成")
            
            return feedback_result
            
        except Exception as e:
            logger.error(f"反馈处理过程出错: {str(e)}", exc_info=True)
            raise
    
    def validate_input(self, input_data: Tuple[CreditData, RiskAnalysisResult, Dict[str, Any]]) -> Tuple[bool, str]:
        """验证输入数据的有效性"""
        if not isinstance(input_data, tuple) or len(input_data) != 3:
            return False, "输入必须是包含CreditData、RiskAnalysisResult和实际结果的元组"
        
        credit_data, risk_result, actual_outcome = input_data
        
        if not isinstance(credit_data, CreditData):
            return False, "第一个输入必须是CreditData类型"
        
        if not isinstance(risk_result, RiskAnalysisResult):
            return False, "第二个输入必须是RiskAnalysisResult类型"
        
        if not isinstance(actual_outcome, dict):
            return False, "第三个输入必须是字典类型"
        
        required_actual_fields = ["is_default", "actual_risk_level"]
        for field in required_actual_fields:
            if field not in actual_outcome:
                return False, f"实际结果中缺少必要字段: {field}"
        
        if credit_data.credit_rating is None:
            return False, "CreditData中没有信用评级结果"
        
        return True, "输入验证通过"
    
    def reset(self) -> None:
        """重置模块状态"""
        self.historical_data = self.historical_data.iloc[0:0]  # 清空历史数据
        # 重新初始化历史数据
        if self.initialized:
            self._initialize_historical_data()
        logger.info("反馈迭代模块已重置")
    
    def _initialize_historical_data(self) -> None:
        """预填充历史数据以便计算行业偏差率"""
        # 创建模拟的历史数据
        historical_records = [
            # 信息技术行业数据
            {"company_id": "IT001", "predicted_rating": "AA", "actual_outcome": 0, "industry": "信息技术", 
             "predicted_risk_level": "low", "actual_risk_level": "low", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "IT002", "predicted_rating": "BBB", "actual_outcome": 0, "industry": "信息技术", 
             "predicted_risk_level": "medium", "actual_risk_level": "medium", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "IT003", "predicted_rating": "A", "actual_outcome": 1, "industry": "信息技术", 
             "predicted_risk_level": "low", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 0.0},
            {"company_id": "IT004", "predicted_rating": "CCC", "actual_outcome": 1, "industry": "信息技术", 
             "predicted_risk_level": "high", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "IT005", "predicted_rating": "AA", "actual_outcome": 0, "industry": "信息技术", 
             "predicted_risk_level": "low", "actual_risk_level": "low", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            
            # 制造业数据
            {"company_id": "MF001", "predicted_rating": "BBB", "actual_outcome": 0, "industry": "制造业", 
             "predicted_risk_level": "medium", "actual_risk_level": "medium", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "MF002", "predicted_rating": "B", "actual_outcome": 1, "industry": "制造业", 
             "predicted_risk_level": "high", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "MF003", "predicted_rating": "A", "actual_outcome": 0, "industry": "制造业", 
             "predicted_risk_level": "low", "actual_risk_level": "medium", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "MF004", "predicted_rating": "BB", "actual_outcome": 1, "industry": "制造业", 
             "predicted_risk_level": "medium", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 0.0},
            {"company_id": "MF005", "predicted_rating": "CCC", "actual_outcome": 1, "industry": "制造业", 
             "predicted_risk_level": "high", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            
            # 批发零售业数据
            {"company_id": "RT001", "predicted_rating": "BBB", "actual_outcome": 0, "industry": "批发零售", 
             "predicted_risk_level": "medium", "actual_risk_level": "low", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "RT002", "predicted_rating": "A", "actual_outcome": 0, "industry": "批发零售", 
             "predicted_risk_level": "low", "actual_risk_level": "low", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "RT003", "predicted_rating": "BB", "actual_outcome": 0, "industry": "批发零售", 
             "predicted_risk_level": "medium", "actual_risk_level": "medium", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "RT004", "predicted_rating": "B", "actual_outcome": 1, "industry": "批发零售", 
             "predicted_risk_level": "high", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "RT005", "predicted_rating": "CCC", "actual_outcome": 1, "industry": "批发零售", 
             "predicted_risk_level": "high", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            
            # 服务业数据
            {"company_id": "SV001", "predicted_rating": "AA", "actual_outcome": 0, "industry": "服务业", 
             "predicted_risk_level": "low", "actual_risk_level": "low", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "SV002", "predicted_rating": "A", "actual_outcome": 0, "industry": "服务业", 
             "predicted_risk_level": "low", "actual_risk_level": "low", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "SV003", "predicted_rating": "BBB", "actual_outcome": 0, "industry": "服务业", 
             "predicted_risk_level": "medium", "actual_risk_level": "medium", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0},
            {"company_id": "SV004", "predicted_rating": "BB", "actual_outcome": 1, "industry": "服务业", 
             "predicted_risk_level": "medium", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 0.0},
            {"company_id": "SV005", "predicted_rating": "B", "actual_outcome": 1, "industry": "服务业", 
             "predicted_risk_level": "high", "actual_risk_level": "high", "analysis_date": datetime.now(), "default_prediction_accuracy": 1.0}
        ]
        
        self.historical_data = pd.DataFrame(historical_records)
        logger.info(f"预填充了 {len(historical_records)} 条历史数据")
    
    def _record_historical_data(self, 
                                credit_data: CreditData, 
                                risk_result: RiskAnalysisResult, 
                                actual_outcome: Dict[str, Any]) -> None:
        """记录历史数据用于后续分析"""
        new_record = {
            "company_id": credit_data.company_id,
            "predicted_rating": credit_data.credit_rating,
            "actual_outcome": 1 if actual_outcome.get("is_default", False) else 0,
            "industry": credit_data.industry,
            "predicted_risk_level": risk_result.risk_level,
            "actual_risk_level": actual_outcome.get("actual_risk_level", "unknown"),
            "analysis_date": datetime.now()
        }
        
        # 添加到历史数据
        self.historical_data = pd.concat(
            [self.historical_data, pd.DataFrame([new_record])],
            ignore_index=True
        )
        
        # 限制历史数据量，保持性能
        if len(self.historical_data) > 10000:
            self.historical_data = self.historical_data.tail(10000)
    
    def _calculate_error_metrics(self, 
                                credit_data: CreditData, 
                                risk_result: RiskAnalysisResult, 
                                actual_outcome: Dict[str, Any]) -> Dict[str, float]:
        """计算误差指标"""
        metrics = {}
        
        # 违约预测准确率
        predicted_default = credit_data.credit_rating in ["CCC", "CC", "C", "D"]
        actual_default = actual_outcome.get("is_default", False)
        metrics["default_prediction_accuracy"] = 1.0 if predicted_default == actual_default else 0.0
        
        # 风险等级预测准确率
        predicted_risk = risk_result.risk_level
        actual_risk = actual_outcome.get("actual_risk_level", "unknown")
        metrics["risk_level_accuracy"] = 1.0 if predicted_risk == actual_risk else 0.0
        
        # 信用等级偏差（绝对值）
        rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
        try:
            predicted_index = rating_order.index(credit_data.credit_rating)
            actual_rating = actual_outcome.get("actual_rating", "unknown")
            actual_index = rating_order.index(actual_rating) if actual_rating in rating_order else predicted_index
            metrics["rating_deviation"] = abs(predicted_index - actual_index)
        except ValueError:
            metrics["rating_deviation"] = -1  # 无法计算偏差
        
        return metrics
    
    def _calculate_hidden_risk_rate(self, 
                                   credit_data: CreditData, 
                                   actual_outcome: Dict[str, Any]) -> float:
        """计算隐性风险漏判率 - 改进版本，考虑所有风险等级的预测偏差"""
        industry = credit_data.industry
        
        # 如果没有历史数据，基于当前案例计算
        if len(self.historical_data) == 0:
            predicted_risk = credit_data.credit_rating
            actual_risk_level = actual_outcome.get("actual_risk_level", "unknown")
            
            # 定义风险等级映射
            risk_mapping = {
                "AAA": "low", "AA": "low", "A": "low",
                "BBB": "medium", "BB": "medium", "B": "medium",
                "CCC": "high", "CC": "high", "C": "high", "D": "high"
            }
            
            predicted_risk_level = risk_mapping.get(predicted_risk, "unknown")
            
            # 计算风险等级预测偏差
            if predicted_risk_level == "low" and actual_risk_level == "high":
                return 0.8  # 严重漏判
            elif predicted_risk_level == "low" and actual_risk_level == "medium":
                return 0.4  # 中等漏判
            elif predicted_risk_level == "medium" and actual_risk_level == "high":
                return 0.3  # 轻微漏判
            else:
                return 0.0  # 无漏判
        
        # 基于历史数据计算行业漏判率
        if industry and len(self.historical_data) > 0:
            industry_data = self.historical_data[
                self.historical_data["industry"] == industry
            ]
            
            if len(industry_data) >= 3:  # 降低数据要求
                # 计算所有风险等级的误判情况
                total_cases = len(industry_data)
                
                # 严重漏判：预测低风险但实际高风险
                severe_misjudged = len(industry_data[
                    (industry_data["predicted_risk_level"] == "low") &
                    (industry_data["actual_risk_level"] == "high")
                ])
                
                # 中等漏判：预测低风险但实际中风险，或预测中风险但实际高风险
                moderate_misjudged = len(industry_data[
                    ((industry_data["predicted_risk_level"] == "low") &
                     (industry_data["actual_risk_level"] == "medium")) |
                    ((industry_data["predicted_risk_level"] == "medium") &
                     (industry_data["actual_risk_level"] == "high"))
                ])
                
                # 加权计算漏判率
                weighted_misjudged = severe_misjudged * 1.0 + moderate_misjudged * 0.5
                hidden_risk_rate = min(1.0, weighted_misjudged / total_cases)
                
                return round(hidden_risk_rate, 4)
        
        # 默认返回基于当前案例的简单计算
        predicted_risk_level = getattr(credit_data, 'risk_level', 'unknown')
        actual_risk_level = actual_outcome.get("actual_risk_level", "unknown")
        
        if predicted_risk_level == "low" and actual_risk_level == "high":
            return 0.15  # 15%的漏判率
        elif predicted_risk_level != actual_risk_level:
            return 0.08  # 8%的一般偏差率
        
        return 0.0
    
    def _calculate_industry_bias_rate(self, industry: Optional[str]) -> float:
        """计算行业适配偏差率 - 改进版本，降低数据要求并优化计算逻辑"""
        if not industry or len(self.historical_data) < 3:
            # 基于行业风险特征的基础偏差率
            high_risk_industries = ["制造业", "建筑业", "采矿业"]
            medium_risk_industries = ["批发零售", "交通运输"]
            low_risk_industries = ["信息技术", "服务业", "金融业"]
            
            if industry in high_risk_industries:
                return 0.12  # 高风险行业基础偏差
            elif industry in low_risk_industries:
                return 0.06  # 低风险行业基础偏差
            else:
                return 0.08  # 中等风险行业基础偏差
        
        # 计算该行业的准确率
        industry_data = self.historical_data[
            self.historical_data["industry"] == industry
        ]
        
        if len(industry_data) < 2:  # 降低数据要求
            return 0.08  # 数据不足时返回默认偏差率
        
        industry_accuracy = self._calculate_prediction_accuracy(industry_data)
        
        # 计算所有其他行业的平均准确率
        other_industries_data = self.historical_data[
            self.historical_data["industry"] != industry
        ]
        
        if len(other_industries_data) < 2:  # 降低数据要求
            # 仅基于该行业数据计算内部偏差
            return max(0.0, 1.0 - industry_accuracy)  # 偏差率 = 1 - 准确率
        
        other_accuracy = self._calculate_prediction_accuracy(other_industries_data)
        
        # 计算相对偏差率
        if other_accuracy > 0:
            relative_bias = abs(industry_accuracy - other_accuracy) / max(other_accuracy, 0.1)
            return min(0.5, round(relative_bias, 4))  # 限制最大偏差率为50%
        
        return 0.05  # 默认偏差率
    
    def _calculate_prediction_accuracy(self, data) -> float:
        """计算预测准确率"""
        if len(data) == 0:
            return 0.0
        
        # 使用现有的default_prediction_accuracy字段
        if "default_prediction_accuracy" in data.columns:
            return data["default_prediction_accuracy"].mean()
        
        # 计算风险等级预测准确率
        correct_predictions = 0
        total_predictions = len(data)
        
        for _, row in data.iterrows():
            predicted = row.get("predicted_risk_level", "unknown")
            actual = row.get("actual_risk_level", "unknown")
            
            if predicted == actual:
                correct_predictions += 1
            elif (predicted == "low" and actual == "medium") or \
                 (predicted == "medium" and actual == "low"):
                correct_predictions += 0.5  # 部分正确
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _generate_weight_adjustments(self, 
                                    credit_data: CreditData, 
                                    risk_result: RiskAnalysisResult, 
                                    actual_outcome: Dict[str, Any],
                                    hidden_risk_rate: float,
                                    industry_bias_rate: float) -> Dict[str, float]:
        """生成权重调整建议"""
        adjustments = {}
        
        # 如果隐性风险漏判率过高，需要调整风险因素权重
        if hidden_risk_rate > self.config["default_thresholds"]["hidden_risk_rate"]:
            logger.info(f"隐性风险漏判率 {hidden_risk_rate} 超过阈值，建议调整权重")
            
            # 简化的权重调整逻辑
            if actual_outcome.get("is_default", False):
                # 实际违约，增加诉讼和处罚风险的权重
                adjustments["lawsuit_risk"] = self.config["adjustment_strength"] * 0.1
                adjustments["penalty_risk"] = self.config["adjustment_strength"] * 0.08
                adjustments["operation_risk"] = -self.config["adjustment_strength"] * 0.05
            else:
                # 误判为高风险，降低相关权重
                adjustments["lawsuit_risk"] = -self.config["adjustment_strength"] * 0.05
                adjustments["penalty_risk"] = -self.config["adjustment_strength"] * 0.03
        
        # 如果行业适配偏差率过高，需要调整行业相关权重
        if industry_bias_rate > self.config["default_thresholds"]["industry_bias_rate"]:
            logger.info(f"行业适配偏差率 {industry_bias_rate} 超过阈值，建议调整权重")
            
            industry = credit_data.industry
            if industry:
                # 根据不同行业调整权重
                if "科技" in industry or "信息" in industry:
                    adjustments["operation_risk"] = self.config["adjustment_strength"] * 0.1
                    adjustments["industry_risk"] = -self.config["adjustment_strength"] * 0.05
                elif "制造" in industry:
                    adjustments["penalty_risk"] = self.config["adjustment_strength"] * 0.08
                    adjustments["industry_risk"] = -self.config["adjustment_strength"] * 0.04
                elif "服务" in industry:
                    adjustments["lawsuit_risk"] = self.config["adjustment_strength"] * 0.06
                    adjustments["industry_risk"] = -self.config["adjustment_strength"] * 0.03
        
        # 确保权重调整后不会超出合理范围
        for factor, adj in adjustments.items():
            current = risk_result.risk_labels.get(f"{factor}_weight", 0.25)
            new_weight = current + adj
            new_weight = max(self.config["min_weight"], 
                           min(self.config["max_weight"], new_weight))
            adjustments[factor] = new_weight - current  # 返回调整量
        
        return adjustments
    
    def _generate_rule_adjustments(self, 
                                  credit_data: CreditData, 
                                  actual_outcome: Dict[str, Any],
                                  hidden_risk_rate: float) -> Dict[str, Any]:
        """生成规则调整建议"""
        adjustments = {}
        
        # 如果隐性风险漏判率过高，需要调整风险判断规则
        if hidden_risk_rate > self.config["default_thresholds"]["hidden_risk_rate"]:
            logger.info(f"隐性风险漏判率 {hidden_risk_rate} 超过阈值，建议调整风险判断规则")
            
            # 重大诉讼阈值调整
            current_threshold = self.config["default_major_lawsuit_threshold"]
            threshold_adj = self.config["adjustment_strength"] * 0.2
            new_threshold = max(100000, int(current_threshold * (1 - threshold_adj)))
            adjustments["major_lawsuit_threshold"] = {
                "current": current_threshold,
                "suggested": new_threshold,
                "reason": "降低重大诉讼判定阈值以减少隐性风险漏判"
            }
            
            # 近期风险权重调整
            recent_weight_adj = self.config["adjustment_strength"] * 0.15
            adjustments["recent_risk_weight"] = {
                "suggested_increase": recent_weight_adj,
                "reason": "增加近期风险权重以提高对新出现风险的敏感度"
            }
        
        # 检查是否存在实际违约但预测为低风险的情况
        if actual_outcome.get("is_default", False) and credit_data.credit_rating in ["AAA", "AA", "A", "BBB"]:
            penalty_adj = self.config["adjustment_strength"] * 0.2
            adjustments["penalty_amount_weight"] = {
                "suggested_increase": penalty_adj,
                "reason": "实际违约但预测为低风险，增加处罚金额权重"
            }
        
        return adjustments
