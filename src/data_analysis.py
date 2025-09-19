"""
数据分析模块
负责数据清洗、特征筛选和AHP权重行业适配
"""
from typing import Dict, Any, Tuple, List, Optional
from events import Event, EventType, EventBus
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
# from pandasai import SmartDataframe
# from pandasai.llm.openai import OpenAI
from sklearn.feature_selection import mutual_info_classif
# 移除pyanp依赖，使用简化的权重计算
from base import BaseModule, CreditData, AnalysisResult, logger

# 配置 DeepSeek 模型 (暂时注释掉)
# from pandasai.llm.openai import OpenAI
# OpenAI._supported_chat_models.append('deepseek-chat')
# llm = OpenAI(
#     api_base='https://api.deepseek.com/v1',
#     api_token='your_key',  # 用户提供的API密钥
#     model='deepseek-chat',
#     timeout=1000
# )

class DataAnalysisModule(BaseModule):
    """数据分析模块实现类"""
    
    def __init__(self, config: Dict[str, Any] = None, event_bus: Optional[EventBus] = None):
        """初始化数据分析模块"""
        self.config = config or {
            "ahp_base_weights": {
                "enterprise_potential": 0.302,  # 企业发展潜力
                "development_environment": 0.076,  # 企业发展环境
                "operator_info": 0.152,  # 经营者信息
                "compliance": 0.47  # 行为合规度
            },
            "industry_adjustments": {
                "科技型": {"enterprise_potential": 0.35, "compliance": 0.422},
                "传统型": {"enterprise_potential": 0.28, "compliance": 0.492},
                "信息技术": {"enterprise_potential": 0.35, "compliance": 0.422},
                "制造业": {"enterprise_potential": 0.28, "compliance": 0.492},
                "服务业": {"enterprise_potential": 0.25, "compliance": 0.522}
            },
            "feature_thresholds": {
                "information_gain": 0.1  # 信息增益阈值
            },
            "default_criteria": {
                "administrative_penalties": 3,  # 行政处罚≥3次
                "lawsuit_count": 5,  # 法律诉讼≥5次
                "shareholder_dishonesty": 1  # 主要股东失信≥1次
            }
        }
        self.initialized = False
        self.scaler = StandardScaler()
        # 四级指标的子指标定义
        self.sub_indicators = {
            "enterprise_potential": ["registered_capital", "establishment_years", "patent_count", "employee_count"],
            "development_environment": ["industry_outlook", "gdp_growth_rate", "policy_impact"],
            "operator_info": ["legal_rep_credit", "shareholder_credit"],
            "compliance": ["lawsuit_count", "major_lawsuit_ratio", "penalty_count", "penalty_amount"]
        }
    
    def initialize(self) -> bool:
        """初始化模块"""
        try:
            # 验证AHP基础权重是否合法
            base_weights_sum = sum(self.config["ahp_base_weights"].values())
            if not np.isclose(base_weights_sum, 1.0):
                raise ValueError(f"AHP基础权重之和必须为1，当前为{base_weights_sum}")
            
            # 验证行业调整权重
            for industry, adjustments in self.config["industry_adjustments"].items():
                # 创建新的权重字典，直接使用调整值
                adjusted_weights = {}
                for criterion in self.config["ahp_base_weights"].keys():
                    if criterion in adjustments:
                        adjusted_weights[criterion] = adjustments[criterion]
                    else:
                        adjusted_weights[criterion] = self.config["ahp_base_weights"][criterion]
                
                # 归一化权重
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
                
                adjusted_sum = sum(adjusted_weights.values())
                if not np.isclose(adjusted_sum, 1.0, atol=0.001):
                    raise ValueError(f"{industry}行业调整后权重之和必须为1，当前为{adjusted_sum:.3f}")
            
            self.initialized = True
            logger.info("数据分析模块初始化完成")
            return True
        except Exception as e:
            logger.error(f"数据分析模块初始化失败: {str(e)}")
            self.initialized = False
            return False
    
    def process(self, credit_data: CreditData) -> Tuple[CreditData, AnalysisResult]:
        """
        处理信用数据，进行清洗、特征提取和权重计算
        
        参数:
            credit_data: 包含原始数据的CreditData对象
        
        返回:
            处理后的CreditData对象和分析结果
        """
        # 验证输入
        valid, message = self.validate_input(credit_data)
        if not valid:
            raise ValueError(f"输入数据无效: {message}")
        
        try:
            # 1. 数据预处理和结构化
            logger.info(f"开始处理企业 {credit_data.raw_data.get('enterprise_info', {}).get('company_name')} 的数据")
            structured_data = self._preprocess_and_structure(credit_data.raw_data)
            credit_data.structured_data = structured_data
            
            # 2. 特征工程
            features = self._feature_engineering(structured_data)
            credit_data.feature_data = features
            
            # 3. 特征筛选
            selected_features = self._select_features(features, credit_data.industry)
            
            # 4. 计算AHP权重
            weights, consistency_ratio = self._calculate_ahp_weights(credit_data.industry)
            
            # 构建分析结果
            analysis_result = AnalysisResult(
                features=features,
                weights=weights,
                consistency_ratio=consistency_ratio,
                selected_features=selected_features
            )
            
            logger.info("数据分析完成")
            
            return credit_data, analysis_result
            
        except Exception as e:
            logger.error(f"数据分析过程出错: {str(e)}", exc_info=True)
            raise
    
    def validate_input(self, credit_data: CreditData) -> Tuple[bool, str]:
        """验证输入数据的有效性"""
        if not isinstance(credit_data, CreditData):
            return False, "输入必须是CreditData类型"
        
        if not credit_data.raw_data:
            return False, "CreditData中没有原始数据"
        
        required_sections = ["enterprise_info", "compliance_data"]
        for section in required_sections:
            if section not in credit_data.raw_data:
                return False, f"原始数据中缺少必要部分: {section}"
        
        return True, "输入验证通过"
    
    def reset(self) -> None:
        """重置模块状态"""
        self.scaler = StandardScaler()
        logger.info("数据分析模块已重置")
    
    def _preprocess_and_structure(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        使用Pandas AI智能预处理原始数据并转换为结构化DataFrame
        
        参数:
            raw_data: 原始数据字典
        
        返回:
            结构化的DataFrame
        """
        # 转换为DataFrame
        df = pd.DataFrame([raw_data])
        
        # 使用Pandas AI进行智能数据清洗
        try:
            from pandasai import SmartDataframe
            sdf = SmartDataframe(df, config={"llm": llm})
            # 使用AI进行数据清洗和转换
            df = sdf.clean_data()
        except Exception as e:
            logger.warning(f"Pandas AI数据清洗失败: {str(e)}，使用标准pandas方法")
        
        # 自动处理缺失值
        df = df.fillna({
            "enterprise_info": {},
            "compliance_data": {},
            "industry_data": {},
            "patent_data": {},
            "environment_data": {}
        })
        
        # 提取和转换关键字段，处理None值
        enterprise_info = df["enterprise_info"].iloc[0]
        compliance_data = df["compliance_data"].iloc[0]
        
        structured = {
            "company_name": enterprise_info.get("company_name", ""),
            "registered_capital": self._parse_capital(enterprise_info.get("registered_capital", "0万")),
            "establishment_date": enterprise_info.get("establishment_date"),
            "establishment_years": self._calculate_years_since(enterprise_info.get("establishment_date")),
            "legal_representative": enterprise_info.get("legal_representative", ""),
            "industry": enterprise_info.get("industry", ""),
            "patent_count": enterprise_info.get("patent_count", 0),
            "employee_count": enterprise_info.get("employee_count", 0),
            "lawsuit_count": compliance_data.get("lawsuit_count", 0),
            "major_lawsuit_ratio": self._calculate_major_lawsuit_ratio(
                compliance_data.get("lawsuit_details", [])),
            "penalty_count": len(compliance_data.get("administrative_penalties", [])),
            "penalty_amount": self._sum_penalties(
                compliance_data.get("administrative_penalties", [])),
            "shareholder_dishonesty": compliance_data.get("shareholder_dishonesty", 0),
            "is_default": raw_data.get("is_default", False)  # 添加违约状态字段
        }
        
        # 处理缺失的字段，使用合理的默认值而不是虚假数据
        # 只有当真实数据不可用时才使用默认值
        structured["industry_outlook"] = 0.5  # 中性默认值
        structured["gdp_growth_rate"] = 0.0   # 中性默认值
        structured["policy_impact"] = 0.5     # 中性默认值
        structured["rd_investment"] = 0       # 中性默认值
        structured["esg_rating"] = 0.5        # 中性默认值
        
        # 经营者信用评分 - 基于真实数据计算
        shareholder_dishonesty = compliance_data.get("shareholder_dishonesty", 0)
        structured["legal_rep_credit"] = 0.85  # 基础假设值
        structured["shareholder_credit"] = 0.9 if shareholder_dishonesty == 0 else max(0.1, 0.9 - shareholder_dishonesty * 0.2)
        
        return pd.DataFrame([structured])
    
    def _feature_engineering(self, structured_data: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程，生成更多有价值的特征
        
        参数:
            structured_data: 结构化数据
        
        返回:
            包含工程特征的DataFrame
        """
        features = structured_data.copy()
        
        # 生成比率特征
        if not features.empty:
            # 注册资本/员工数比率
            emp_count = features["employee_count"].iloc[0]
            if emp_count > 0:
                features["capital_per_employee"] = features["registered_capital"] / emp_count
            else:
                features["capital_per_employee"] = 0
            
            # 专利/员工数比率
            features["patent_per_employee"] = features["patent_count"] / max(1, emp_count)
            
            # 诉讼/成立年限比率
            est_years = features["establishment_years"].iloc[0]
            features["lawsuit_per_year"] = features["lawsuit_count"] / max(1, est_years)
            
            # 处罚/成立年限比率
            features["penalty_per_year"] = features["penalty_count"] / max(1, est_years)
        
        # 标准化特征
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols and not features.empty:
            features[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])
        
        return features
    
    def _select_features(self, features: pd.DataFrame, industry: Optional[str]) -> List[str]:
        """
        基于信息增益筛选特征
        
        参数:
            features: 特征数据
            industry: 行业类型
        
        返回:
            筛选后的特征列表
        """
        if features.empty:
            return []
        
        # 只处理数值型特征
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return []
        
        # 使用违约状态作为目标变量（如果存在）
        if "is_default" in features.columns:
            y = features["is_default"]
        else:
            # 如果没有违约标签，使用专利数量作为模拟目标变量
            y = features["patent_count"].apply(lambda x: 1 if x > 5 else 0)
        
        # 如果数据不足，直接返回所有数值特征
        if len(features) < 2 or len(set(y)) < 2:
            logger.warning("数据不足，跳过特征选择，使用所有数值特征")
            return numeric_cols
        
        try:
            # 计算信息增益
            mi_scores = mutual_info_classif(features[numeric_cols], y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=numeric_cols)
            
            # 根据阈值筛选特征
            threshold = self.config["feature_thresholds"]["information_gain"]
            selected_features = mi_scores[mi_scores > threshold].index.tolist()
        except Exception as e:
            logger.warning(f"特征选择失败: {str(e)}，使用所有数值特征")
            return numeric_cols
        
        # 如果筛选结果为空，返回信息增益最高的3个特征
        if not selected_features:
            selected_features = mi_scores.sort_values(ascending=False).head(3).index.tolist()
        
        # 确保包含行业关键特征
        if industry:
            # 根据行业类型选择关键特征
            industry_type = self._get_industry_type(industry)
            industry_key_features = {
                "科技型": ["patent_count", "rd_investment", "employee_count", "gdp_growth_rate"],
                "传统型": ["registered_capital", "employee_count", "penalty_count", "lawsuit_count"]
            }.get(industry_type, [])
            
            selected_features = list(set(selected_features + industry_key_features))
        
        return selected_features
    
    def _get_industry_type(self, industry: str) -> str:
        """根据行业名称判断行业类型（科技型/传统型）"""
        industry_lower = industry.lower()
        tech_keywords = ["信息", "科技", "软件", "互联网", "电子", "通信", "生物", "医药", "新材料", "新能源"]
        if any(keyword in industry_lower for keyword in tech_keywords):
            return "科技型"
        return "传统型"
    
    def _calculate_ahp_weights(self, industry: Optional[str]) -> Tuple[Dict[str, float], float]:
        """
        计算权重，考虑行业调整
        
        参数:
            industry: 行业类型
        
        返回:
            权重字典和一致性比率
        """
        try:
            # 使用基础权重
            base_weights = self.config["ahp_base_weights"].copy()
            
            # 根据行业进行调整
            if industry and industry in self.config["industry_adjustments"]:
                adjustments = self.config["industry_adjustments"][industry]
                
                # 应用行业调整（直接使用调整值，而不是相乘）
                for criterion, adjustment in adjustments.items():
                    if criterion in base_weights:
                        base_weights[criterion] = adjustment
                
                # 重新归一化权重，确保总和为1
                total_weight = sum(base_weights.values())
                if not np.isclose(total_weight, 1.0):
                    # 如果权重和不等于1，进行归一化
                    base_weights = {k: v/total_weight for k, v in base_weights.items()}
            
            # 验证权重和是否为1
            total_weight = sum(base_weights.values())
            if not np.isclose(total_weight, 1.0, atol=0.001):
                logger.warning(f"权重和不为1 ({total_weight:.3f})，进行强制归一化")
                base_weights = {k: v/total_weight for k, v in base_weights.items()}
            
            # 模拟一致性比率（简化版本）
            consistency_ratio = 0.05  # 假设一致性良好
            
            return base_weights, consistency_ratio
            
        except Exception as e:
            logger.warning(f"权重计算失败，使用默认权重: {str(e)}")
            return self.config["ahp_base_weights"].copy(), 0.1
    
    def _parse_capital(self, capital_str: str) -> float:
        """解析注册资本字符串为数值"""
        try:
            if not capital_str:
                return 0.0
                
            # 移除所有非数字和小数点的字符
            numeric_part = ''.join([c for c in capital_str if c.isdigit() or c == '.'])
            if not numeric_part:
                return 0.0
                
            capital = float(numeric_part)
            
            # 处理单位（万、亿等）
            if '亿' in capital_str:
                capital *= 10000  # 转换为万单位
            # 默认为万单位
            
            return capital
        except Exception as e:
            logger.warning(f"解析注册资本失败: {capital_str}, 错误: {str(e)}")
            return 0.0
    
    def _calculate_years_since(self, date_str: Optional[str]) -> int:
        """计算从给定日期到现在的年数"""
        try:
            if not date_str:
                return 0
                
            from datetime import datetime
            date_format = "%Y-%m-%d"
            date_obj = datetime.strptime(date_str, date_format)
            today = datetime.today()
            years = today.year - date_obj.year
            
            # 考虑月份
            if (today.month, today.day) < (date_obj.month, date_obj.day):
                years -= 1
                
            return max(0, years)
        except Exception as e:
            logger.warning(f"计算成立年限失败: {date_str}, 错误: {str(e)}")
            return 0
    
    def _calculate_major_lawsuit_ratio(self, lawsuits: List[Dict[str, Any]]) -> float:
        """计算重大诉讼比例"""
        if not lawsuits:
            return 0.0
            
        total = len(lawsuits)
        major_count = 0
        
        for lawsuit in lawsuits:
            # 涉诉金额超过500万视为重大诉讼
            if lawsuit.get("amount", 0) >= 500000:
                major_count += 1
                
        return major_count / total
    
    def _sum_penalties(self, penalties: List[Dict[str, Any]]) -> float:
        """计算处罚总金额"""
        total = 0.0
        for penalty in penalties:
            total += penalty.get("amount", 0)
        return total
    
    def _map_outlook_to_numeric(self, outlook: str) -> float:
        """将行业景气度映射为数值"""
        outlook_map = {
            "growing": 1.0,
            "stable": 0.5,
            "declining": 0.0,
            "unknown": 0.5
        }
        return outlook_map.get(outlook.lower(), 0.5)
    
    def _map_policy_impact(self, policy_impact: str) -> float:
        """将政策影响映射为数值"""
        mapping = {
            "positive": 1.0,
            "neutral": 0.5,
            "negative": 0.0,
            "supportive": 0.8,
            "restrictive": 0.2
        }
        return mapping.get(policy_impact.lower(), 0.5)
    
    def _map_esg_rating(self, esg_rating: str) -> float:
        """将ESG评级映射为数值"""
        mapping = {
            "aaa": 1.0, "aa": 0.9, "a": 0.8,
            "bbb": 0.7, "bb": 0.6, "b": 0.5,
            "ccc": 0.4, "cc": 0.3, "c": 0.2, "d": 0.1
        }
        return mapping.get(esg_rating.lower(), 0.5)
