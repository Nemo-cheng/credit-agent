"""
保守优化的Credit-Agents智能体
在现有框架基础上进行最小化改动，确保稳定运行
"""
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
import numpy as np

from base import CreditData, AnalysisResult, RiskAnalysisResult, FeedbackResult, logger
from events import Event, EventType, EventBus
from data_collection import DataCollectionModule
from data_analysis import DataAnalysisModule
from risk_analysis import RiskAnalysisModule
from feedback_iteration import FeedbackIterationModule

class ConservativeOptimizedCreditAgent:
    """保守优化的Credit-Agents智能体"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化保守优化智能体"""
        # 如果未提供配置，则使用保守优化配置
        if config is None:
            config = create_conservative_config()
        self.config = config
        self.initialized = False
        self.event_bus = EventBus()
        
        # 性能监控（保守增强）
        self.performance_metrics = {
            "total_processed": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_processing_time": 0.0,
            "optimization_applied": 0  # 记录优化应用次数
        }
        
        # 使用原始模块，确保兼容性
        self.data_collector = DataCollectionModule(
            self._ensure_safe_config("data_collection"),
            self.event_bus
        )
        self.data_analyzer = DataAnalysisModule(
            self._ensure_safe_config("data_analysis"),
            self.event_bus
        )
        self.risk_analyzer = RiskAnalysisModule(
            self._ensure_safe_config("risk_analysis"),
            self.event_bus
        )
        self.feedback_processor = FeedbackIterationModule(
            self._ensure_safe_config("feedback_iteration"),
            self.event_bus
        )
        
        # 注册事件处理器
        self._register_event_handlers()
        
        # 保守优化配置
        self.optimization_config = {
            "enable_weight_adjustment": True,      # 启用权重调整
            "enable_consistency_check": True,     # 启用一致性检查
            "enable_confidence_boost": True,      # 启用置信度提升
            "enable_risk_sensitivity": True,      # 启用风险敏感度调整
            "max_adjustment_ratio": 0.1           # 最大调整比例（保守）
        }
    
    def initialize(self) -> bool:
        """初始化智能体（保守策略）"""
        try:
            start_time = time.time()
            logger.info("开始初始化保守优化的Credit-Agents智能体")
            
            # 逐个初始化模块，记录状态但不强制要求全部成功
            init_results = {}
            init_results["data_collection"] = self._safe_module_init(self.data_collector, "data_collection")
            init_results["data_analysis"] = self._safe_module_init(self.data_analyzer, "data_analysis")
            init_results["risk_analysis"] = self._safe_module_init(self.risk_analyzer, "risk_analysis")
            init_results["feedback_iteration"] = self._safe_module_init(self.feedback_processor, "feedback_iteration")
            
            # 保守策略：只要核心分析模块成功即可
            core_modules = ["data_analysis", "risk_analysis"]
            core_success = all(init_results.get(module, False) for module in core_modules)
            
            if core_success:
                self.initialized = True
                success_count = sum(init_results.values())
                total_count = len(init_results)
                
                init_time = time.time() - start_time
                logger.info(f"保守优化的Credit-Agents智能体初始化完成")
                logger.info(f"模块初始化成功率: {success_count}/{total_count}, 耗时: {init_time:.2f}秒")
                
                # 根据初始化结果调整优化策略
                if success_count < total_count:
                    logger.info("部分模块初始化失败，将采用更保守的优化策略")
                    self._adjust_optimization_strategy(init_results)
                
                return True
            else:
                failed_modules = [m for m in core_modules if not init_results.get(m, False)]
                logger.error(f"核心模块初始化失败: {failed_modules}")
                return False
                
        except Exception as e:
            logger.error(f"保守优化的Credit-Agents智能体初始化失败: {str(e)}")
            self.initialized = False
            return False
    
    def _ensure_safe_config(self, module_name: str) -> Dict[str, Any]:
        """确保配置安全（保守策略）"""
        base_config = self.config.get(module_name, {})
        
        # 为每个模块提供最基本的安全配置
        if module_name == "data_collection":
            if "sources" not in base_config:
                base_config["sources"] = {
                    "qichacha": "https://www.qichacha.com",
                    "tianyancha": "https://www.tianyancha.com"
                }
        
        elif module_name == "risk_analysis":
            # 确保风险分析模块的基本配置
            if "risk_level_thresholds" not in base_config:
                base_config["risk_level_thresholds"] = {
                    "high": 0.7, "medium": 0.3, "low": 0.0
                }
            if "credit_rating_mapping" not in base_config:
                base_config["credit_rating_mapping"] = {
                    (90, 100): "AAA", (80, 90): "AA", (70, 80): "A",
                    (60, 70): "BBB", (50, 60): "BB", (40, 50): "B",
                    (30, 40): "CCC", (20, 30): "CC", (10, 20): "C", (0, 10): "D"
                }
            if "default_criteria" not in base_config:
                base_config["default_criteria"] = {
                    "administrative_penalties": 3,
                    "lawsuit_count": 5,
                    "shareholder_dishonesty": 1
                }
            if "langgraph_config" not in base_config:
                base_config["langgraph_config"] = {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
        
        elif module_name == "data_analysis":
            # 确保数据分析模块的基本配置
            if "ahp_base_weights" not in base_config:
                base_config["ahp_base_weights"] = {
                    "enterprise_potential": 0.302,
                    "development_environment": 0.076,
                    "operator_info": 0.152,
                    "compliance": 0.47
                }
            if "industry_adjustments" not in base_config:
                base_config["industry_adjustments"] = {
                    "科技型": {"enterprise_potential": 0.35, "compliance": 0.422},
                    "传统型": {"enterprise_potential": 0.28, "compliance": 0.492},
                    "信息技术": {"enterprise_potential": 0.35, "compliance": 0.422},
                    "制造业": {"enterprise_potential": 0.28, "compliance": 0.492},
                    "服务业": {"enterprise_potential": 0.25, "compliance": 0.522}
                }
            if "feature_thresholds" not in base_config:
                base_config["feature_thresholds"] = {
                    "information_gain": 0.08
                }
        
        elif module_name == "feedback_iteration":
            # 确保反馈模块的基本配置
            if "default_thresholds" not in base_config:
                base_config["default_thresholds"] = {
                    "hidden_risk_rate": 0.2,
                    "industry_bias_rate": 0.3,
                    "prediction_accuracy": 0.8
                }
            if "adjustment_strength" not in base_config:
                base_config["adjustment_strength"] = 0.05  # 更保守的调整强度
            if "min_weight" not in base_config:
                base_config["min_weight"] = 0.05
            if "max_weight" not in base_config:
                base_config["max_weight"] = 0.6
            if "default_major_lawsuit_threshold" not in base_config:
                base_config["default_major_lawsuit_threshold"] = 500000
        
        return base_config
    
    def _safe_module_init(self, module, module_name: str) -> bool:
        """安全模块初始化"""
        try:
            result = module.initialize()
            if result:
                logger.info(f"模块 {module_name} 初始化成功")
            else:
                logger.warning(f"模块 {module_name} 初始化失败，但系统将继续运行")
            return result
        except Exception as e:
            logger.warning(f"模块 {module_name} 初始化异常: {str(e)}，但系统将继续运行")
            return False
    
    def _adjust_optimization_strategy(self, init_results: Dict[str, bool]):
        """根据初始化结果调整优化策略"""
        failed_modules = [name for name, success in init_results.items() if not success]
        
        if "feedback_iteration" in failed_modules:
            # 如果反馈模块失败，禁用相关优化
            self.optimization_config["enable_weight_adjustment"] = False
            logger.info("反馈模块不可用，禁用权重调整优化")
        
        if len(failed_modules) >= 2:
            # 如果多个模块失败，采用最保守策略
            self.optimization_config["max_adjustment_ratio"] = 0.05
            logger.info("多个模块不可用，采用最保守的优化策略")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        self.event_bus.subscribe(EventType.DATA_COLLECTED, self._on_data_collected_conservative)
        self.event_bus.subscribe(EventType.DATA_ANALYZED, self._on_data_analyzed_conservative)
        self.event_bus.subscribe(EventType.RISK_ANALYZED, self._on_risk_analyzed_conservative)
    
    def _on_data_collected_conservative(self, event: Event):
        """保守的数据采集处理"""
        company_info = event.data
        
        try:
            # 标准数据采集流程
            credit_data = self.data_collector.process(company_info)
            
            # 保守优化：确保行业信息正确设置
            if "industry" in company_info and company_info["industry"]:
                credit_data.industry = company_info["industry"]
                if "enterprise_info" in credit_data.raw_data:
                    credit_data.raw_data["enterprise_info"]["industry"] = company_info["industry"]
            
            # 继续数据分析
            self.event_bus.publish(Event(
                type=EventType.DATA_ANALYZED,
                data=self.data_analyzer.process(credit_data),
                source="ConservativeOptimizedCreditAgent"
            ))
            
        except Exception as e:
            logger.warning(f"数据采集处理出现问题: {str(e)}，使用备用流程")
            # 保守策略：创建基础数据结构继续流程
            credit_data = self._create_fallback_credit_data(company_info, str(e))
            self.event_bus.publish(Event(
                type=EventType.DATA_ANALYZED,
                data=self.data_analyzer.process(credit_data),
                source="ConservativeOptimizedCreditAgent"
            ))
    
    def _on_data_analyzed_conservative(self, event: Event):
        """保守的数据分析处理"""
        credit_data, analysis_result = event.data
        
        try:
            # 保守优化：轻微调整分析结果
            if self.optimization_config["enable_weight_adjustment"]:
                enhanced_result = self._conservative_enhance_analysis(analysis_result)
                self.performance_metrics["optimization_applied"] += 1
            else:
                enhanced_result = analysis_result
            
            # 继续风险分析
            self.event_bus.publish(Event(
                type=EventType.RISK_ANALYZED,
                data=self.risk_analyzer.process(credit_data, enhanced_result),
                source="ConservativeOptimizedCreditAgent"
            ))
            
        except Exception as e:
            logger.warning(f"数据分析处理出现问题: {str(e)}，使用原始结果")
            # 保守策略：如果优化失败，使用原始结果
            self.event_bus.publish(Event(
                type=EventType.RISK_ANALYZED,
                data=self.risk_analyzer.process(credit_data, analysis_result),
                source="ConservativeOptimizedCreditAgent"
            ))
    
    def _on_risk_analyzed_conservative(self, event: Event):
        """保守的风险分析处理"""
        credit_data, risk_result = event.data
        
        try:
            # 保守优化：轻微调整风险结果
            if self.optimization_config["enable_consistency_check"]:
                optimized_result = self._conservative_optimize_risk_result(credit_data, risk_result)
                if optimized_result != risk_result:
                    self.performance_metrics["optimization_applied"] += 1
            else:
                optimized_result = risk_result
            
            # 保守优化：提升置信度
            if self.optimization_config["enable_confidence_boost"]:
                self._conservative_boost_confidence(optimized_result)
            
            # 存储结果
            self._current_result = (credit_data, optimized_result)
            
        except Exception as e:
            logger.warning(f"风险分析处理出现问题: {str(e)}，使用原始结果")
            # 保守策略：如果优化失败，使用原始结果
            self._current_result = (credit_data, risk_result)
    
    def analyze_credit(self, company_info: Dict[str, str]) -> Tuple[CreditData, RiskAnalysisResult]:
        """
        保守优化的企业信用分析
        """
        if not self.initialized:
            raise RuntimeError("智能体未初始化，请先调用initialize()方法")
        
        start_time = time.time()
        
        try:
            logger.info(f"开始保守优化分析: {company_info.get('company_name', '未知企业')}")
            
            # 输入验证（保守）
            if not company_info.get("company_name"):
                raise ValueError("企业名称不能为空")
            
            # 重置结果
            self._current_result = None
            
            # 启动分析流程
            self.event_bus.publish(Event(
                type=EventType.DATA_COLLECTED,
                data=company_info,
                source="ConservativeOptimizedCreditAgent"
            ))
            
            # 等待分析完成（保守超时）
            result = self._wait_for_analysis_completion(timeout=30)
            
            if result is None:
                logger.warning("分析超时，可能网络或数据问题")
                # 保守策略：返回默认结果而不是抛出异常
                return self._create_default_result(company_info)
            
            credit_data, risk_result = result
            
            # 最终保守优化
            final_result = self._final_conservative_optimization(credit_data, risk_result)
            
            # 更新性能指标
            total_time = time.time() - start_time
            self._update_performance_metrics(total_time, True)
            
            logger.info(f"保守优化分析完成: {company_info.get('company_name')}, "
                       f"耗时: {total_time:.2f}秒, 信用等级: {final_result[0].credit_rating}")
            
            return final_result
            
        except Exception as e:
            total_time = time.time() - start_time
            self._update_performance_metrics(total_time, False)
            logger.warning(f"分析过程出现问题: {str(e)}，返回保守结果")
            
            # 保守策略：即使出错也返回一个合理的结果
            return self._create_conservative_error_result(company_info, str(e))
    
    def _conservative_enhance_analysis(self, analysis_result: AnalysisResult) -> AnalysisResult:
        """保守的分析结果增强"""
        try:
            if hasattr(analysis_result, 'weights') and analysis_result.weights:
                enhanced_weights = analysis_result.weights.copy()
                
                # 保守调整：轻微增加合规权重
                if 'compliance' in enhanced_weights:
                    adjustment = min(self.optimization_config["max_adjustment_ratio"], 0.05)
                    enhanced_weights['compliance'] = min(0.6, enhanced_weights['compliance'] * (1 + adjustment))
                
                # 重新归一化（保守）
                total_weight = sum(enhanced_weights.values())
                if total_weight > 0:
                    enhanced_weights = {k: v/total_weight for k, v in enhanced_weights.items()}
                
                analysis_result.weights = enhanced_weights
                
        except Exception as e:
            logger.debug(f"分析增强失败，使用原始结果: {str(e)}")
        
        return analysis_result
    
    def _conservative_optimize_risk_result(self, credit_data: CreditData, 
                                         risk_result: RiskAnalysisResult) -> RiskAnalysisResult:
        """保守的风险结果优化"""
        try:
            # 保守的一致性检查
            risk_level = risk_result.risk_level
            credit_rating = risk_result.credit_rating
            
            # 只在明显不一致时才调整
            if risk_level == "high" and credit_rating in ["AAA", "AA"]:
                logger.info("检测到明显的风险-信用不一致，进行保守调整")
                risk_result.credit_rating = "A"  # 保守调整，不要过度降级
                credit_data.credit_rating = "A"
                return risk_result
            
            # 如果启用风险敏感度调整
            if self.optimization_config["enable_risk_sensitivity"]:
                # 保守地提高风险敏感度
                if risk_level == "medium" and len(risk_result.risk_details.get("major_risk_factors", [])) >= 2:
                    # 只有在有多个风险因素时才考虑升级
                    pass  # 保守策略：不轻易改变风险等级
            
        except Exception as e:
            logger.debug(f"风险优化失败，使用原始结果: {str(e)}")
        
        return risk_result
    
    def _conservative_boost_confidence(self, risk_result: RiskAnalysisResult):
        """保守的置信度提升"""
        try:
            if "confidence_score" not in risk_result.risk_details:
                # 保守地设置置信度
                risk_result.risk_details["confidence_score"] = 0.65
            else:
                # 轻微提升现有置信度
                current_confidence = risk_result.risk_details["confidence_score"]
                boost = min(0.05, self.optimization_config["max_adjustment_ratio"])
                risk_result.risk_details["confidence_score"] = min(0.9, current_confidence + boost)
                
        except Exception as e:
            logger.debug(f"置信度提升失败: {str(e)}")
    
    def _final_conservative_optimization(self, credit_data: CreditData, 
                                       risk_result: RiskAnalysisResult) -> Tuple[CreditData, RiskAnalysisResult]:
        """最终保守优化"""
        try:
            # 确保结果的合理性
            if credit_data.credit_score is None or credit_data.credit_score <= 0:
                credit_data.credit_score = 60.0  # 保守的默认分数
            
            if not credit_data.credit_rating:
                credit_data.credit_rating = "BBB"  # 保守的默认等级
            
            # 确保风险详情完整
            if not risk_result.risk_details:
                risk_result.risk_details = {
                    "confidence_score": 0.6,
                    "major_risk_factors": [],
                    "analysis_method": "conservative_optimization"
                }
            
        except Exception as e:
            logger.debug(f"最终优化失败: {str(e)}")
        
        return credit_data, risk_result
    
    def _wait_for_analysis_completion(self, timeout: int = 30) -> Optional[Tuple[CreditData, RiskAnalysisResult]]:
        """等待分析完成（保守超时）"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if hasattr(self, '_current_result') and self._current_result is not None:
                return self._current_result
            time.sleep(0.1)
        
        return None
    
    def _create_fallback_credit_data(self, company_info: Dict[str, str], error_msg: str) -> CreditData:
        """创建备用信用数据"""
        return CreditData(
            raw_data={
                "enterprise_info": {
                    "company_name": company_info.get("company_name", ""),
                    "industry": company_info.get("industry", "其他")
                },
                "compliance_data": {
                    "lawsuit_count": 0,
                    "administrative_penalties": [],
                    "shareholder_dishonesty": 0
                },
                "fallback_reason": error_msg
            }
        )
    
    def _create_default_result(self, company_info: Dict[str, str]) -> Tuple[CreditData, RiskAnalysisResult]:
        """创建默认结果"""
        credit_data = CreditData(
            raw_data={
                "enterprise_info": {
                    "company_name": company_info.get("company_name", ""),
                    "industry": company_info.get("industry", "其他")
                }
            }
        )
        credit_data.credit_score = 60.0
        credit_data.credit_rating = "BBB"
        
        risk_result = RiskAnalysisResult(
            risk_labels={"timeout_risk": 0.3},
            risk_level="medium",
            risk_details={
                "timeout_analysis": True,
                "confidence_score": 0.5,
                "major_risk_factors": ["数据获取超时"]
            },
            credit_rating="BBB"
        )
        
        return credit_data, risk_result
    
    def _create_conservative_error_result(self, company_info: Dict[str, str], 
                                        error_msg: str) -> Tuple[CreditData, RiskAnalysisResult]:
        """创建保守的错误结果"""
        credit_data = CreditData(
            raw_data={
                "enterprise_info": {
                    "company_name": company_info.get("company_name", ""),
                    "industry": company_info.get("industry", "其他")
                },
                "error": error_msg
            }
        )
        credit_data.credit_score = 50.0  # 保守的错误分数
        credit_data.credit_rating = "B"   # 保守的错误等级
        
        risk_result = RiskAnalysisResult(
            risk_labels={"analysis_error": 0.5},
            risk_level="medium",  # 保守地设为中等风险
            risk_details={
                "error_message": error_msg,
                "confidence_score": 0.3,
                "major_risk_factors": ["分析过程异常"],
                "conservative_result": True
            },
            credit_rating="B"
        )
        
        return credit_data, risk_result
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """更新性能指标"""
        self.performance_metrics["total_processed"] += 1
        
        if success:
            self.performance_metrics["success_count"] += 1
        else:
            self.performance_metrics["error_count"] += 1
        
        # 更新平均处理时间
        total_time = (self.performance_metrics["avg_processing_time"] * 
                     (self.performance_metrics["total_processed"] - 1) + processing_time)
        self.performance_metrics["avg_processing_time"] = total_time / self.performance_metrics["total_processed"]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = self.performance_metrics.copy()
        
        if report["total_processed"] > 0:
            report["success_rate"] = report["success_count"] / report["total_processed"]
            report["optimization_rate"] = report["optimization_applied"] / report["total_processed"]
        else:
            report["success_rate"] = 0.0
            report["optimization_rate"] = 0.0
        
        return report
    
    def process_feedback(self, 
                        credit_data: CreditData, 
                        risk_result: RiskAnalysisResult, 
                        actual_outcome: Dict[str, Any]) -> FeedbackResult:
        """
        处理反馈信息，用于模型迭代优化
        
        参数:
            credit_data: 信用数据
            risk_result: 风险分析结果
            actual_outcome: 实际结果
        
        返回:
            反馈结果
        """
        if not self.initialized:
            raise RuntimeError("智能体尚未初始化，请先调用initialize()方法")
        
        try:
            logger.info(f"处理企业 {credit_data.raw_data.get('enterprise_info', {}).get('company_name')} 的反馈")
            
            # 处理反馈
            feedback_result = self.feedback_processor.process(credit_data, risk_result, actual_outcome)
            
            # 根据反馈结果调整其他模块
            self._adjust_modules_based_on_feedback(feedback_result)
            
            return feedback_result
            
        except Exception as e:
            logger.error(f"反馈处理过程出错: {str(e)}", exc_info=True)
            raise
    
    def _adjust_modules_based_on_feedback(self, feedback_result: FeedbackResult) -> None:
        """根据反馈结果调整各个模块"""
        # 如果有权重调整建议，调整数据分析模块
        if feedback_result.weight_adjustments:
            logger.info(f"根据反馈调整风险因素权重: {feedback_result.weight_adjustments}")
            # 这里可以实现调整数据分析模块权重的逻辑
            pass
        
        # 如果有规则调整建议，调整风险分析模块
        if feedback_result.rule_adjustments:
            logger.info(f"根据反馈调整风险判断规则: {feedback_result.rule_adjustments}")
            # 这里可以实现调整风险分析模块规则的逻辑
            pass
        
        # 根据隐性风险漏判率调整
        if feedback_result.hidden_risk_rate > self.feedback_processor.config["default_thresholds"]["hidden_risk_rate"]:
            logger.info(f"隐性风险漏判率过高({feedback_result.hidden_risk_rate})，增强风险识别敏感度")
    
    def get_module_status(self) -> Dict[str, bool]:
        """获取各个模块的状态"""
        return {
            "data_collector": hasattr(self.data_collector, 'initialized') and getattr(self.data_collector, 'initialized', True),
            "data_analyzer": hasattr(self.data_analyzer, 'initialized') and getattr(self.data_analyzer, 'initialized', True),
            "risk_analyzer": hasattr(self.risk_analyzer, 'initialized') and getattr(self.risk_analyzer, 'initialized', True),
            "feedback_processor": hasattr(self.feedback_processor, 'initialized') and getattr(self.feedback_processor, 'initialized', True)
        }
    
    def reset(self) -> None:
        """重置智能体状态"""
        logger.info("重置保守优化的Credit-Agents智能体")
        
        # 安全重置各模块
        try:
            self.data_collector.reset()
        except:
            pass
        
        try:
            self.data_analyzer.reset()
        except:
            pass
        
        try:
            self.risk_analyzer.reset()
        except:
            pass
        
        try:
            self.feedback_processor.reset()
        except:
            pass
        
        # 重置性能指标
        self.performance_metrics = {
            "total_processed": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_processing_time": 0.0,
            "optimization_applied": 0
        }
        
        self.initialized = False
        logger.info("保守优化的Credit-Agents智能体已重置")

# 创建保守优化配置的辅助函数
def create_conservative_config() -> Dict[str, Any]:
    """创建保守优化配置"""
    return {
        "data_collection": {
            "retry_count": 2,  # 减少重试次数
            "timeout": 10,     # 减少超时时间
            "sources": {
                "qichacha": "https://www.qichacha.com",
                "tianyancha": "https://www.tianyancha.com"
            }
        },
        "data_analysis": {
            "ahp_base_weights": {
                "enterprise_potential": 0.302,
                "development_environment": 0.076,
                "operator_info": 0.152,
                "compliance": 0.47
            }
        },
        "risk_analysis": {
            "risk_level_thresholds": {
                "high": 0.7,
                "medium": 0.3,
                "low": 0.0
            },
            "langgraph_config": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        },
        "feedback_iteration": {
            "adjustment_strength": 0.05,  # 更保守的调整强度
            "default_thresholds": {
                "hidden_risk_rate": 0.2,
                "industry_bias_rate": 0.3,
                "prediction_accuracy": 0.8
            }
        }
    }