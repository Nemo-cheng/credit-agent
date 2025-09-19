"""
风险解析模块
负责非结构化数据深度挖掘与风险映射
"""
from typing import Dict, Any, Tuple, List, Optional
from events import Event, EventType, EventBus
import pandas as pd
import numpy as np
import re
from datetime import datetime
from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

from base import BaseModule, CreditData, AnalysisResult, RiskAnalysisResult, logger

class RiskAnalysisModule(BaseModule):
    """风险解析模块实现类"""
    
    def __init__(self, config: Dict[str, Any] = None, event_bus: Optional[EventBus] = None):
        """初始化风险解析模块"""
        load_dotenv()
        self.config = config or {
            "risk_level_thresholds": {
                "high": 0.7,
                "medium": 0.3,
                "low": 0.0
            },
            "lawsuit_risk_criteria": {
                "major": {"amount": 500000, "recent_days": 365},  # 50万以上且近一年的诉讼视为重大
                "medium": {"amount": 100000, "recent_days": 730}  # 10万以上且近两年的诉讼视为中等
            },
            "credit_rating_mapping": {
                (90, 100): "AAA",
                (80, 90): "AA",
                (70, 80): "A",
                (60, 70): "BBB",
                (50, 60): "BB",
                (40, 50): "B",
                (30, 40): "CCC",
                (20, 30): "CC",
                (10, 20): "C",
                (0, 10): "D"
            },
            "langgraph_config": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "entity_types": [
                    "PERSON", "ORG", "LAWSUIT", "PENALTY", 
                    "FINANCIAL_TERM", "RISK_TERM"
                ]
            },
            "default_criteria": {
                "administrative_penalties": 3,  # 行政处罚≥3次
                "lawsuit_count": 5,  # 法律诉讼≥5次
                "shareholder_dishonesty": 1  # 主要股东失信≥1次
            }
        }
        self.initialized = False
        # 风险因素权重
        self.risk_factor_weights = {
            "lawsuit_risk": 0.35,
            "penalty_risk": 0.3,
            "operation_risk": 0.2,
            "industry_risk": 0.15
        }
        try:
            self.llm = ChatDeepSeek(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL")
            )
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek LLM: {e}")
            self.llm = None
    
    def initialize(self) -> bool:
        """初始化模块"""
        try:
            # 验证风险等级阈值配置
            thresholds = self.config["risk_level_thresholds"]
            if not (thresholds["high"] > thresholds["medium"] > thresholds["low"]):
                raise ValueError("风险等级阈值配置不合理，应该是high > medium > low")
            
            # 验证信用评级映射
            rating_bounds = list(self.config["credit_rating_mapping"].keys())
            for i in range(1, len(rating_bounds)):
                if rating_bounds[i][0] >= rating_bounds[i-1][0]:
                    raise ValueError("信用评级映射的区间配置不合理，应该按降序排列")
            
            self.initialized = True
            logger.info("风险解析模块初始化完成")
            return True
        except Exception as e:
            logger.error(f"风险解析模块初始化失败: {str(e)}")
            self.initialized = False
            return False
    
    def process(self, credit_data: CreditData, analysis_result: AnalysisResult) -> Tuple[CreditData, RiskAnalysisResult]:
        """
        解析风险并映射到信用等级
        
        参数:
            credit_data: 信用数据对象
            analysis_result: 分析结果对象
        
        返回:
            更新后的信用数据和风险分析结果
        """
        # 验证输入
        valid, message = self.validate_input((credit_data, analysis_result))
        if not valid:
            raise ValueError(f"输入数据无效: {message}")
        
        try:
            logger.info(f"开始分析企业 {credit_data.raw_data.get('enterprise_info', {}).get('company_name')} 的风险")
            
            # 1. 提取非结构化文本中的风险实体
            risk_entities = self._extract_risk_entities(credit_data.raw_data)
            
            # 2. 计算各类风险分数
            lawsuit_risk = self._calculate_lawsuit_risk(
                credit_data.raw_data.get("compliance_data", {}).get("lawsuit_details", []),
                risk_entities.get("lawsuit_entities", {})
            )
            
            penalty_risk = self._calculate_penalty_risk(
                credit_data.raw_data.get("compliance_data", {}).get("administrative_penalties", []),
                risk_entities.get("penalty_entities", {})
            )
            
            operation_risk = self._calculate_operation_risk(credit_data.feature_data)
            
            industry_risk = self._calculate_industry_risk(
                credit_data.raw_data.get("industry_data", {}),
                credit_data.industry
            )
            
            # 3. 综合风险评分
            overall_risk_score = self._calculate_overall_risk_score({
                "lawsuit_risk": lawsuit_risk,
                "penalty_risk": penalty_risk,
                "operation_risk": operation_risk,
                "industry_risk": industry_risk
            })
            
            # 4. 确定风险等级
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # 5. 计算信用分数和等级
            credit_score = 100 - (overall_risk_score * 100)  # 风险分数反向映射为信用分数
            credit_rating = self._map_to_credit_rating(credit_score, credit_data.industry)
            
            # 更新信用数据
            credit_data.credit_score = credit_score
            credit_data.credit_rating = credit_rating
            credit_data.risk_labels = {
                "lawsuit_risk": lawsuit_risk,
                "penalty_risk": penalty_risk,
                "operation_risk": operation_risk,
                "industry_risk": industry_risk,
                "overall_risk_score": overall_risk_score,
                "risk_level": risk_level
            }
            
            # 构建风险分析结果
            risk_result = RiskAnalysisResult(
                risk_labels=credit_data.risk_labels,
                risk_level=risk_level,
                risk_details={
                    "major_risk_factors": self._identify_major_risk_factors({
                        "lawsuit_risk": lawsuit_risk,
                        "penalty_risk": penalty_risk,
                        "operation_risk": operation_risk,
                        "industry_risk": industry_risk
                    }),
                    "risk_entities": risk_entities
                },
                credit_rating=credit_rating
            )
            
            logger.info(f"风险分析完成，企业信用等级: {credit_rating}，风险等级: {risk_level}")
            
            return credit_data, risk_result
            
        except Exception as e:
            logger.error(f"风险分析过程出错: {str(e)}", exc_info=True)
            raise
    
    def validate_input(self, input_data: Tuple[CreditData, AnalysisResult]) -> Tuple[bool, str]:
        """验证输入数据的有效性"""
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            return False, "输入必须是包含CreditData和AnalysisResult的元组"
        
        credit_data, analysis_result = input_data
        
        if not isinstance(credit_data, CreditData):
            return False, "第一个输入必须是CreditData类型"
        
        if not isinstance(analysis_result, AnalysisResult):
            return False, "第二个输入必须是AnalysisResult类型"
        
        if credit_data.structured_data.empty:
            return False, "CreditData中没有结构化数据"
        
        if credit_data.feature_data.empty:
            return False, "CreditData中没有特征数据"
        
        return True, "输入验证通过"
    
    def reset(self) -> None:
        """重置模块状态"""
        logger.info("风险解析模块已重置")
    
    def _extract_risk_entities(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LangGraph从非结构化文本中提取风险实体
        
        参数:
            raw_data: 原始数据字典
        
        返回:
            提取的风险实体
        """
        risk_entities = {
            "lawsuit_entities": {},
            "penalty_entities": {},
            "other_entities": {}
        }

        if not self.llm:
            logger.warning("LLM not initialized, skipping entity extraction.")
            return risk_entities

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["langgraph_config"]["chunk_size"],
            chunk_overlap=self.config["langgraph_config"]["chunk_overlap"]
        )

        lawsuit_texts = [f"案件类型:{d.get('case_type','')} 金额:{d.get('amount',0)} 日期:{d.get('date','')} 结果:{d.get('result','')}"
                         for d in raw_data.get("compliance_data", {}).get("lawsuit_details", [])]
        penalty_texts = [f"处罚原因:{d.get('reason','')} 金额:{d.get('amount',0)} 日期:{d.get('date','')}"
                         for d in raw_data.get("compliance_data", {}).get("administrative_penalties", [])]
        
        all_texts = lawsuit_texts + penalty_texts
        documents = text_splitter.create_documents(all_texts)

        extracted_entities = {}

        for doc in documents:
            prompt = f"""从以下文本中提取风险实体。请以JSON格式返回结果，包含以下实体类型: {self.config["langgraph_config"]["entity_types"]}

            文本: {doc.page_content}

            JSON输出:
            """
            try:
                response = self.llm.invoke(prompt)
                # The response from the LLM is a string, so we need to parse it as JSON.
                # We will use a regex to find the JSON part of the response.
                import json
                json_match = re.search(r'```json\n(.*)```', response.content, re.DOTALL)
                if json_match:
                    entities_str = json_match.group(1)
                    entities = json.loads(entities_str)
                    for key, values in entities.items():
                        if key not in extracted_entities:
                            extracted_entities[key] = []
                        extracted_entities[key].extend(values)
            except Exception as e:
                logger.error(f"Error during entity extraction with LLM: {e}")

        # 整理提取结果
        if "LAWSUIT" in extracted_entities:
            total_amount = sum(int(amount) for _, amount in extracted_entities["LAWSUIT"])
            case_types = list(set(case_type for case_type, _ in extracted_entities["LAWSUIT"]))
            
            risk_entities["lawsuit_entities"] = {
                "total_amount": total_amount,
                "case_types": case_types,
                "recent_cases_count": self._count_recent_cases(
                    raw_data.get("compliance_data", {}).get("lawsuit_details", [])),
                "major_case_count": sum(1 for _, amount in extracted_entities["LAWSUIT"] 
                                      if int(amount) >= self.config["lawsuit_risk_criteria"]["major"]["amount"])
            }
        
        if "PENALTY" in extracted_entities:
            penalty_reasons = extracted_entities["PENALTY"]
            violation_types = self._classify_violations(penalty_reasons)
            
            risk_entities["penalty_entities"] = {
                "total_amount": sum(p.get("amount", 0) for p in 
                                  raw_data.get("compliance_data", {}).get("administrative_penalties", [])),
                "reason_types": violation_types,
                "recent_penalty_count": self._count_recent_penalties(
                    raw_data.get("compliance_data", {}).get("administrative_penalties", []))
            }
        
        # 其他风险实体
        shareholder_dishonesty = raw_data.get("compliance_data", {}).get("shareholder_dishonesty", 0)
        if shareholder_dishonesty > 0:
            risk_entities["other_entities"]["shareholder_dishonesty"] = shareholder_dishonesty
        
        # 添加提取的人员和组织信息
        if "PERSON" in extracted_entities:
            risk_entities["other_entities"]["key_persons"] = extracted_entities["PERSON"]
        if "ORG" in extracted_entities:
            risk_entities["other_entities"]["related_orgs"] = extracted_entities["ORG"]
        
        return risk_entities
    
    def _calculate_lawsuit_risk(self, lawsuits: List[Dict[str, Any]], lawsuit_entities: Dict[str, Any]) -> float:
        """计算诉讼风险分数（0-1，越高风险越大）"""
        if not lawsuits or not lawsuit_entities:
            return 0.0
        
        lawsuit_count = len(lawsuits)
        
        # 根据违约标准调整风险计算
        default_threshold = self.config["default_criteria"]["lawsuit_count"]
        
        # 诉讼数量风险 - 超过违约阈值时风险急剧增加
        if lawsuit_count >= default_threshold:
            count_risk = 1.0  # 达到违约标准，最高风险
        else:
            count_risk = min(1.0, lawsuit_count / default_threshold)  # 按比例计算风险
        
        # 重大诉讼风险
        major_case_ratio = lawsuit_entities.get("major_case_count", 0) / max(1, lawsuit_count)
        major_risk = major_case_ratio
        
        # 近期诉讼风险
        recent_case_ratio = lawsuit_entities.get("recent_cases_count", 0) / max(1, lawsuit_count)
        recent_risk = recent_case_ratio
        
        # 诉讼金额风险
        total_amount = lawsuit_entities.get("total_amount", 0)
        amount_risk = min(1.0, total_amount / 10000000)  # 1000万以上诉讼金额视为最高风险
        
        # 综合诉讼风险（加权平均）
        weights = {
            "count_risk": 0.4,  # 增加数量权重，因为这是违约的核心标准
            "major_risk": 0.3,
            "recent_risk": 0.2,
            "amount_risk": 0.1
        }
        
        lawsuit_risk = (count_risk * weights["count_risk"] +
                       major_risk * weights["major_risk"] +
                       recent_risk * weights["recent_risk"] +
                       amount_risk * weights["amount_risk"])
        
        return round(lawsuit_risk, 4)
    
    def _calculate_penalty_risk(self, penalties: List[Dict[str, Any]], penalty_entities: Dict[str, Any]) -> float:
        """计算处罚风险分数（0-1，越高风险越大）"""
        if not penalties or not penalty_entities:
            return 0.0
        
        penalty_count = len(penalties)
        
        # 根据违约标准调整风险计算
        default_threshold = self.config["default_criteria"]["administrative_penalties"]
        
        # 处罚数量风险 - 超过违约阈值时风险急剧增加
        if penalty_count >= default_threshold:
            count_risk = 1.0  # 达到违约标准，最高风险
        else:
            count_risk = min(1.0, penalty_count / default_threshold)  # 按比例计算风险
        
        # 处罚金额风险
        total_amount = penalty_entities.get("total_amount", 0)
        amount_risk = min(1.0, total_amount / 1000000)  # 100万以上处罚金额视为最高风险
        
        # 近期处罚风险
        recent_ratio = penalty_entities.get("recent_penalty_count", 0) / max(1, penalty_count)
        recent_risk = recent_ratio
        
        # 严重违规类型风险
        violation_types = penalty_entities.get("reason_types", {})
        severe_violation_count = sum(count for type_name, count in violation_types.items() 
                                    if type_name in ["financial", "fraud", "safety"])
        severe_risk = min(1.0, severe_violation_count / max(1, penalty_count))
        
        # 综合处罚风险（加权平均）
        weights = {
            "count_risk": 0.4,  # 增加数量权重，因为这是违约的核心标准
            "amount_risk": 0.2,
            "recent_risk": 0.2,
            "severe_risk": 0.2
        }
        
        penalty_risk = (count_risk * weights["count_risk"] +
                       amount_risk * weights["amount_risk"] +
                       recent_risk * weights["recent_risk"] +
                       severe_risk * weights["severe_risk"])
        
        return round(penalty_risk, 4)
    
    def _calculate_operation_risk(self, feature_data: pd.DataFrame) -> float:
        """计算经营风险分数（0-1，越高风险越大）"""
        if feature_data.empty:
            return 0.5  # 无法评估时返回中等风险
        
        # 从特征数据中提取相关指标
        # 注意：这里假设特征数据已经标准化，值越高表示该方面表现越好
        capital = feature_data.get("registered_capital", [0])[0]
        establishment_years = feature_data.get("establishment_years", [0])[0]
        employee_count = feature_data.get("employee_count", [0])[0]
        patent_count = feature_data.get("patent_count", [0])[0]
        
        # 转换为风险分数（值越高风险越大）
        capital_risk = 1.0 - min(1.0, (capital + 1) / 2)  # 资本越少风险越大
        age_risk = 1.0 - min(1.0, (establishment_years + 1) / 10)  # 成立时间越短风险越大
        size_risk = 1.0 - min(1.0, (employee_count + 1) / 100)  # 规模越小风险越大
        innovation_risk = 1.0 - min(1.0, (patent_count + 1) / 20)  # 创新性越弱风险越大
        
        # 检查股东失信情况
        shareholder_dishonesty = feature_data.get("shareholder_dishonesty", [0])[0]
        # 股东失信风险 - 如果有股东失信记录，风险显著增加
        shareholder_risk = 1.0 if shareholder_dishonesty > 0 else 0.0
        
        # 综合经营风险（加权平均）
        weights = {
            "capital_risk": 0.25,
            "age_risk": 0.15,
            "size_risk": 0.25,
            "innovation_risk": 0.15,
            "shareholder_risk": 0.2  # 增加股东失信权重
        }
        
        operation_risk = (capital_risk * weights["capital_risk"] +
                         age_risk * weights["age_risk"] +
                         size_risk * weights["size_risk"] +
                         innovation_risk * weights["innovation_risk"] +
                         shareholder_risk * weights["shareholder_risk"])
        
        return round(operation_risk, 4)
    
    def _calculate_industry_risk(self, industry_data: Dict[str, Any], industry: Optional[str]) -> float:
        """计算行业风险分数（0-1，越高风险越大）"""
        # 行业景气度风险（景气度越低风险越大）
        outlook = industry_data.get("industry_outlook", 0.5)
        outlook_risk = 1.0 - float(outlook) if isinstance(outlook, (int, float)) else 0.5
        
        # GDP增长率风险（增长率越低风险越大）
        gdp_growth = industry_data.get("gdp_growth_rate", 0.0)
        gdp_risk = 1.0 - min(1.0, float(gdp_growth) / 10)  # 假设10%为高增长率
        
        # 政策影响风险（政策越不利风险越大）
        policy_impact = industry_data.get("policy_impact", 0.5)
        policy_risk = 1.0 - float(policy_impact) if isinstance(policy_impact, (int, float)) else 0.5
        
        # 行业特定风险
        industry_specific_risk = 0.0
        if industry:
            # 根据行业类型进行更精细的风险评估
            tech_industries = ["信息", "科技", "软件", "互联网", "电子", "通信", "生物", "医药", "新材料", "新能源"]
            traditional_industries = ["制造", "建筑", "贸易", "零售", "餐饮", "物流"]
            high_risk_industries = ["房地产", "金融", "矿业", "P2P", "小贷"]
            
            if any(ind in industry for ind in high_risk_industries):
                industry_specific_risk = 0.7  # 高风险行业
            elif any(ind in industry for ind in tech_industries):
                industry_specific_risk = 0.3  # 科技型行业风险较低（政策支持）
            elif any(ind in industry for ind in traditional_industries):
                industry_specific_risk = 0.5  # 传统行业中等风险
            else:
                industry_specific_risk = 0.4  # 其他行业默认风险
        
        # 综合行业风险（加权平均）
        weights = {
            "outlook_risk": 0.3,
            "gdp_risk": 0.2,
            "policy_risk": 0.3,
            "industry_specific_risk": 0.2
        }
        
        industry_risk = (outlook_risk * weights["outlook_risk"] +
                        gdp_risk * weights["gdp_risk"] +
                        policy_risk * weights["policy_risk"] +
                        industry_specific_risk * weights["industry_specific_risk"])
        
        return round(industry_risk, 4)
    
    def _calculate_overall_risk_score(self, risk_scores: Dict[str, float]) -> float:
        """计算综合风险分数（0-1，越高风险越大）"""
        overall_score = 0.0
        
        for risk_type, score in risk_scores.items():
            if risk_type in self.risk_factor_weights:
                overall_score += score * self.risk_factor_weights[risk_type]
        
        return round(overall_score, 4)
    
    def _determine_risk_level(self, overall_risk_score: float) -> str:
        """根据综合风险分数确定风险等级"""
        thresholds = self.config["risk_level_thresholds"]
        
        if overall_risk_score >= thresholds["high"]:
            return "high"
        elif overall_risk_score >= thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _map_to_credit_rating(self, credit_score: float, industry: Optional[str] = None) -> str:
        """将信用分数映射到信用等级，考虑行业因素"""
        # 确保分数在0-100范围内
        credit_score = max(0, min(100, credit_score))
        
        # 基础信用等级映射
        base_rating = "D"
        for (lower, upper), rating in self.config["credit_rating_mapping"].items():
            if lower < credit_score <= upper:
                base_rating = rating
                break
        
        # 行业调整（示例：高风险行业降级，低风险行业升级）
        if industry:
            # 根据行业类型进行更精细的调整
            tech_industries = ["信息", "科技", "软件", "互联网", "电子", "通信", "生物", "医药", "新材料", "新能源"]
            traditional_industries = ["制造", "建筑", "贸易", "零售", "餐饮", "物流"]
            high_risk_industries = ["房地产", "金融", "矿业", "P2P", "小贷"]
            
            if any(ind in industry for ind in high_risk_industries):
                # 高风险行业降一级
                rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
                try:
                    idx = rating_order.index(base_rating)
                    base_rating = rating_order[min(idx + 1, len(rating_order) - 1)]
                except ValueError:
                    pass
            elif any(ind in industry for ind in tech_industries):
                # 科技型企业升一级（政策支持）
                rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
                try:
                    idx = rating_order.index(base_rating)
                    base_rating = rating_order[max(idx - 1, 0)]
                except ValueError:
                    pass
        
        return base_rating
    
    def _identify_major_risk_factors(self, risk_scores: Dict[str, float]) -> List[str]:
        """识别主要风险因素"""
        # 计算每个风险因素的加权分数
        weighted_scores = {
            risk_type: score * self.risk_factor_weights.get(risk_type, 0)
            for risk_type, score in risk_scores.items()
        }
        
        # 按加权分数排序
        sorted_risks = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前两个主要风险因素
        return [risk_type for risk_type, _ in sorted_risks[:2]]
    
    def _count_recent_cases(self, lawsuits: List[Dict[str, Any]]) -> int:
        """计算近期（一年内）的诉讼数量"""
        recent_count = 0
        today = datetime.today()
        
        for lawsuit in lawsuits:
            date_str = lawsuit.get("date")
            if date_str:
                try:
                    lawsuit_date = datetime.strptime(date_str, "%Y-%m-%d")
                    days_diff = (today - lawsuit_date).days
                    if days_diff <= self.config["lawsuit_risk_criteria"]["major"]["recent_days"]:
                        recent_count += 1
                except Exception:
                    continue
        
        return recent_count
    
    def _classify_violations(self, reasons: List[str]) -> Dict[str, int]:
        """将处罚原因分类"""
        violation_types = {
            "financial": 0,  # 财务相关违规
            "operational": 0,  # 经营相关违规
            "safety": 0,  # 安全相关违规
            "fraud": 0,  # 欺诈相关违规
            "other": 0  # 其他类型
        }
        
        # 关键词匹配规则
        financial_keywords = ["税", "财务", "资金", "发票", "账户"]
        operational_keywords = ["经营", "许可", "资质", "广告", "合同"]
        safety_keywords = ["安全", "环境", "卫生", "质量", "标准"]
        fraud_keywords = ["欺诈", "虚假", "假冒", "侵权", "违法"]
        
        for reason in reasons:
            if not reason:
                violation_types["other"] += 1
                continue
                
            reason_lower = reason.lower()
            classified = False
            
            for keyword in financial_keywords:
                if keyword in reason_lower:
                    violation_types["financial"] += 1
                    classified = True
                    break
                    
            if classified:
                continue
                
            for keyword in operational_keywords:
                if keyword in reason_lower:
                    violation_types["operational"] += 1
                    classified = True
                    break
                    
            if classified:
                continue
                
            for keyword in safety_keywords:
                if keyword in reason_lower:
                    violation_types["safety"] += 1
                    classified = True
                    break
                    
            if classified:
                continue
                
            for keyword in fraud_keywords:
                if keyword in reason_lower:
                    violation_types["fraud"] += 1
                    classified = True
                    break
                    
            if not classified:
                violation_types["other"] += 1
        
        return violation_types
    
    def _count_recent_penalties(self, penalties: List[Dict[str, Any]]) -> int:
        """计算近期（一年内）的处罚数量"""
        recent_count = 0
        today = datetime.today()
        
        for penalty in penalties:
            date_str = penalty.get("date")
            if date_str:
                try:
                    penalty_date = datetime.strptime(date_str, "%Y-%m-%d")
                    days_diff = (today - penalty_date).days
                    if days_diff <= 365:  # 近一年
                        recent_count += 1
                except Exception:
                    continue
        
        return recent_count
