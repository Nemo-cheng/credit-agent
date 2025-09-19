'''
数据转换模块
将企业风险数据转换为Credit-Agents可用的数据格式
'''

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from base import CreditData, RiskAnalysisResult

def convert_enterprise_risk_data(enterprise_data: Dict[str, Any]) -> CreditData:
    """
    将企业风险数据转换为CreditData格式
    
    参数:
        enterprise_data: 包含企业风险数据的字典
        
    返回:
        CreditData对象
    """
    credit_data = CreditData()
    
    # 设置企业基本信息
    credit_data.company_id = enterprise_data.get("company_id")
    credit_data.industry = enterprise_data.get("industry")
    
    # 设置原始数据
    raw_data = {
        "enterprise_info": {
            "company_name": enterprise_data.get("company_name"),
            "legal_representative": enterprise_data.get("legal_representative"),
            "establishment_date": enterprise_data.get("establishment_date"),
            "registered_capital": enterprise_data.get("registered_capital"),
            "industry": enterprise_data.get("industry"),
            "address": enterprise_data.get("address"),
            "business_scope": enterprise_data.get("business_scope"),
            "registration_status": enterprise_data.get("registration_status"),
        },
        "compliance_data": {
            "lawsuit_count": enterprise_data.get("lawsuit_count", 0),
            "lawsuit_details": enterprise_data.get("lawsuit_details", []),
            "administrative_penalties": enterprise_data.get("administrative_penalties", []),
            "shareholder_dishonesty": enterprise_data.get("shareholder_dishonesty", 0),
            "environmental_violations": enterprise_data.get("environmental_violations", []),
            "tax_violations": enterprise_data.get("tax_violations", []),
            "credit_blacklist": enterprise_data.get("credit_blacklist", False),
            "risk_indicators": enterprise_data.get("risk_indicators", {}),
        },
        "patent_data": {
            "patent_count": enterprise_data.get("patent_count", 0),
            "patent_details": enterprise_data.get("patent_details", []),
        },
    }
    
    credit_data.raw_data = raw_data
    
    # 设置风险标签
    credit_data.risk_labels = {
        "self_risk_count": enterprise_data.get("self_risk_count", 0),
        "related_risk_count": enterprise_data.get("related_risk_count", 0),
        "high_risk_count": enterprise_data.get("high_risk_count", 0),
        "medium_risk_count": enterprise_data.get("medium_risk_count", 0),
        "low_risk_count": enterprise_data.get("low_risk_count", 0),
    }
    
    return credit_data

def convert_to_risk_analysis_result(risk_data: Dict[str, Any]) -> RiskAnalysisResult:
    """
    将风险数据转换为RiskAnalysisResult格式
    
    参数:
        risk_data: 包含风险数据的字典
        
    返回:
        RiskAnalysisResult对象
    """
    risk_result = RiskAnalysisResult()
    
    # 设置风险标签
    risk_result.risk_labels = {
        "self_risk_count": risk_data.get("self_risk_count", 0),
        "related_risk_count": risk_data.get("related_risk_count", 0),
        "lawsuit_count": risk_data.get("lawsuit_count", 0),
        "administrative_penalties_count": len(risk_data.get("administrative_penalties", [])),
        "shareholder_dishonesty": risk_data.get("shareholder_dishonesty", 0),
    }
    
    # 设置风险等级
    if risk_data.get("high_risk_count", 0) > 0:
        risk_result.risk_level = "high"
    elif risk_data.get("medium_risk_count", 0) > 0:
        risk_result.risk_level = "medium"
    else:
        risk_result.risk_level = "low"
    
    # 设置风险详情
    risk_result.risk_details = {
        "lawsuit_risk": {
            "count": risk_data.get("lawsuit_count", 0),
            "details": risk_data.get("lawsuit_details", []),
        },
        "penalty_risk": {
            "count": len(risk_data.get("administrative_penalties", [])),
            "details": risk_data.get("administrative_penalties", []),
        },
        "dishonesty_risk": {
            "count": risk_data.get("shareholder_dishonesty", 0),
        },
        "major_risk_factors": risk_data.get("major_risk_factors", []),
    }
    
    # 设置信用等级
    risk_result.credit_rating = risk_data.get("credit_rating", "C")
    
    return risk_result

def parse_image_data() -> Dict[str, Any]:
    """
    解析图片中的企业风险数据
    
    返回:
        包含企业风险数据的字典
    """
    # 根据图片内容提取数据
    enterprise_data = {
        "company_name": "广东龙泉科技有限公司",
        "company_id": "91440000707650621",
        "legal_representative": "周志刚",
        "establishment_date": "1998-01-23",
        "registered_capital": "1,001万元",
        "industry": "批发业",
        "address": "广州市海珠区敦和路189号大院第3栋905-909室",
        "business_scope": "软件开发;智能水务系统开发;信息系统集成服务;计算机系统服务;信息系统运行维护服务;信息技术咨询服务;物联网技术服务;智能控制系统集成;人工智能通用应用系统;人工智能应用软件开发;人工智能理论与算法软件开发;大数据服务;数据处理服务;地理遥感信息服务;卫星遥感数据处理;工程技术服务",
        "registration_status": "开业",
        
        # 风险数据
        "self_risk_count": 9,
        "related_risk_count": 100,
        "high_risk_count": 3,
        "medium_risk_count": 6,
        "lawsuit_count": 3,
        "lawsuit_details": [
            {"case_type": "执行类案件", "amount": 21795, "date": "2025-09-10", "result": "被执行人", "court": "广州市海珠区人民法院", "case_number": "(2025)粤0105执12782号"},
            {"case_type": "执行类案件", "amount": 22192, "date": "2025-08-26", "result": "被执行人", "court": "广州市海珠区人民法院", "case_number": "(2025)粤0105执11980号"},
            {"case_type": "执行类案件", "amount": 30110, "date": "2025-08-20", "result": "被执行人", "court": "广州市海珠区人民法院", "case_number": "(2025)粤0105执11612号"},
        ],
        "administrative_penalties": [],
        "shareholder_dishonesty": 0,
        "environmental_violations": [],
        "tax_violations": [],
        "credit_blacklist": False,
        
        # 专利数据
        "patent_count": 16,
        "patent_details": [
            {"patent_name": "一种基于图像识别与多参数建模的混凝剂智能投加控制方法及系统", "patent_number": "CN120535101A", "patent_type": "发明专利", "application_date": "2025-08-26"},
            {"patent_name": "一种基于机器视觉和AI算法的供水厂滤池运行异常识别方法和系统", "patent_number": "CN119649368A", "patent_type": "发明专利", "application_date": "2025-03-18"},
            {"patent_name": "基于语义标签构建的河湖水环境污染方法及系统", "patent_number": "CN119025926B", "patent_type": "发明专利", "application_date": "2025-03-11"},
        ],
        
        # 诉讼关系
        "litigation_relationships": [
            {"counterparty": "江苏柳松市政工程有限公司", "case_type": "民事案件", "case_reason": "承揽合同纠纷", "count": 2},
            {"counterparty": "东莞市德仁管道工程检测有限公司", "case_type": "民事案件", "case_reason": "承揽合同纠纷", "count": 1},
        ],
        
        # 风险指标
        "risk_indicators": {
            "litigation_risk": 0.3,
            "penalty_risk": 0.0,
            "environmental_risk": 0.0,
            "tax_risk": 0.0,
            "dishonesty_risk": 0.0,
            "overall_compliance_risk": 0.15
        },
        
        # 信用评级
        "credit_rating": "BB",
    }
    
    return enterprise_data

def main():
    # 解析图片数据
    enterprise_data = parse_image_data()
    
    # 转换为CreditData格式
    credit_data = convert_enterprise_risk_data(enterprise_data)
    
    # 转换为RiskAnalysisResult格式
    risk_result = convert_to_risk_analysis_result(enterprise_data)
    
    # 打印结果
    print("企业名称:", enterprise_data["company_name"])
    print("风险等级:", risk_result.risk_level)
    print("信用评级:", risk_result.credit_rating)
    print("诉讼数量:", credit_data.raw_data["compliance_data"]["lawsuit_count"])
    print("专利数量:", credit_data.raw_data["patent_data"]["patent_count"])
    
    return credit_data, risk_result

if __name__ == "__main__":
    main()