"""
报告生成模块
负责根据风险分析结果和信用评级，生成结构化的信用报告
"""
from typing import Dict, Any, Optional
from base import BaseModule, CreditData, AnalysisResult, RiskAnalysisResult, logger

class ReportGenerationAgent(BaseModule):
    """报告生成智能体"""

    def __init__(self, config: Dict[str, Any] = None):
        """初始化报告生成智能体"""
        self.config = config or {}
        self.initialized = False

    def initialize(self) -> bool:
        """初始化模块"""
        self.initialized = True
        logger.info("报告生成智能体初始化完成")
        return True

    def process(self, credit_data: CreditData, analysis_result: AnalysisResult, risk_analysis_result: RiskAnalysisResult) -> Dict[str, Any]:
        """
        生成信用报告

        参数:
            credit_data: 信用数据对象
            analysis_result: 分析结果对象
            risk_analysis_result: 风险分析结果对象

        返回:
            包含信用报告内容的字典
        """
        if not self.initialized:
            raise RuntimeError("报告生成智能体尚未初始化")

        logger.info(f"开始为企业 {credit_data.company_id} 生成信用报告")

        report = self._generate_report_structure(credit_data, analysis_result, risk_analysis_result)

        logger.info(f"信用报告生成完毕")

        return report

    def _generate_report_structure(self, credit_data: CreditData, analysis_result: AnalysisResult, risk_analysis_result: RiskAnalysisResult) -> Dict[str, Any]:
        """
        构建报告结构

        返回:
            结构化的报告字典
        """
        report = {
            "company_info": {
                "company_name": credit_data.raw_data.get("enterprise_info", {}).get("company_name", "N/A"),
                "company_id": credit_data.company_id,
                "industry": credit_data.industry,
            },
            "credit_rating": {
                "score": credit_data.credit_score,
                "rating": credit_data.credit_rating,
                "risk_level": risk_analysis_result.risk_level,
            },
            "risk_analysis": {
                "summary": self._get_risk_summary(risk_analysis_result),
                "details": risk_analysis_result.risk_details,
            },
            "data_analysis": {
                "weighted_scores": analysis_result.weights,
                "key_features": analysis_result.selected_features,
            },
            "recommendations": self._generate_recommendations(risk_analysis_result),
            "timestamp": self._get_timestamp(),
        }
        return report

    def _get_risk_summary(self, risk_analysis_result: RiskAnalysisResult) -> str:
        """
        生成风险摘要

        返回:
            风险摘要字符串
        """
        major_risks = risk_analysis_result.risk_details.get("major_risk_factors", [])
        if major_risks:
            return f"主要风险因素包括: {', '.join(major_risks)}。"
        return "未发现重大风险因素。"

    def _generate_recommendations(self, risk_analysis_result: RiskAnalysisResult) -> str:
        """
        根据风险等级生成建议

        返回:
            建议字符串
        """
        risk_level = risk_analysis_result.risk_level
        if risk_level == "high":
            return "建议采取严格的风险控制措施，并考虑进一步的尽职调查。"
        elif risk_level == "medium":
            return "建议保持关注，并定期审查其信用状况。"
        else:
            return "信用状况良好，建议正常合作。"

    def _get_timestamp(self) -> str:
        """
        获取当前时间戳

        返回:
            格式化的时间戳字符串
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def validate_input(self, input_data: Any) -> tuple[bool, str]:
        """验证输入数据的有效性"""
        return True, ""

    def reset(self) -> None:
        """重置模块状态"""
        pass