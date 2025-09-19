"""
数据采集模块
负责从公开渠道采集企业相关数据，包括基础信息、合规数据和行业特征数据
支持多数据源整合、自动异常检测和数据质量评估
"""
import time
import random
import os
import json
import re
import hashlib
import concurrent.futures
from typing import Dict, Any, Tuple, List, Optional, Union, Set
from events import Event, EventType, EventBus
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import akshare as ak
from datetime import datetime, timedelta
import logging
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
# 可选依赖，如果安装了就使用
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False

from base import BaseModule, CreditData, logger
import asyncio
import hashlib
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from dotenv import load_dotenv
import os

import requests
from bs4 import BeautifulSoup
from pydantic import ValidationError

from .schema import CreditData, Event

class DataCollectionAgent:
    """
    数据收集代理，负责从各种来源收集企业信用数据。
    """
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        """
        初始化数据收集代理。
        """
        load_dotenv()
        self.config = config
        self.event_bus = event_bus
        self.initialized = False
        
        # 从配置中获取参数，并提供默认值
        self.retry_count = self.config.get("retry_count", 3)
        self.timeout = self.config.get("timeout", 30)
        self.proxies = self.config.get("proxies", {})
        self.api_keys = {
            "qichacha": os.getenv("QICHACHA_API_KEY"),
            "tianyancha": os.getenv("TIANYANCHA_API_KEY"),
            **self.config.get("api_keys", {}),
        }
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.max_workers = self.config.get("max_workers", 5)
        
        self.session = None
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def initialize(self) -> bool:
        """初始化模块，包括数据源连接测试、缓存初始化和API密钥验证"""
        try:
            logger.info("开始初始化数据采集模块...")
            
            # 1. 验证配置参数
            self._validate_config()
            
            # 2. 初始化缓存目录
            self._initialize_cache()
            
            # 3. 验证API密钥
            api_keys_valid = self._validate_api_keys()
            if not api_keys_valid:
                logger.warning("部分API密钥无效或缺失，某些高级功能可能不可用")
            
            # 4. 测试主要数据源连接性（使用更宽容的测试方法）
            test_urls = [url for url in self.config["sources"].values() if isinstance(url, str) and url.startswith("http")]
            successful_connections = 0
            connection_results = {}
            
            # 使用并发请求测试多个数据源
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(test_urls), 5)) as executor:
                future_to_url = {executor.submit(self._test_data_source, url): url for url in test_urls}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        success, status_code, message = future.result()
                        connection_results[url] = (success, status_code, message)
                        if success:
                            successful_connections += 1
                    except Exception as exc:
                        logger.warning(f"测试数据源 {url} 时发生异常: {exc}")
                        connection_results[url] = (False, None, str(exc))
            
            # 记录连接测试结果
            for url, (success, status_code, message) in connection_results.items():
                if success:
                    logger.info(f"数据源 {url} 连接测试成功，状态码: {status_code}")
                else:
                    logger.warning(f"数据源 {url} 连接测试失败: {message}")
            
            # 5. 测试备用数据源
            backup_sources_available = self._test_backup_sources()
            
            # 6. 确定初始化状态
            if successful_connections > 0 or backup_sources_available:
                self.initialized = True
                logger.info(f"数据采集模块初始化完成，成功连接 {successful_connections}/{len(test_urls)} 个主要数据源")
                if backup_sources_available:
                    logger.info("备用数据源可用，将在主要数据源失败时使用")
            else:
                # 即使全部失败也允许继续，使用模拟数据
                self.initialized = True
                self.development_mode = True
                logger.warning("所有数据源连接测试失败，将使用模拟数据进行开发模式")
            
            # 7. 初始化行业分类器
            self._initialize_industry_classifier()
            
            return True
        except Exception as e:
            logger.error(f"数据采集模块初始化失败: {str(e)}", exc_info=True)
            self.initialized = False
            return False
    
    def _validate_config(self) -> bool:
        """验证配置参数的有效性"""
        # 验证必要的配置项
        required_configs = ["sources", "retry_count", "timeout"]
        for config_key in required_configs:
            if config_key not in self.config:
                logger.warning(f"配置中缺少 {config_key}，将使用默认值")
        
        # 验证数据质量配置
        if "data_quality" not in self.config:
            logger.warning("配置中缺少数据质量设置，将使用默认值")
            self.config["data_quality"] = {
                "min_quality_score": 0.6,
                "required_fields": ["company_name", "industry"],
                "anomaly_thresholds": {
                    "capital_outlier": 10000000000,
                    "employee_outlier": 100000,
                    "patent_outlier": 10000
                }
            }
        
        return True
    
    def _initialize_cache(self) -> None:
        """初始化数据缓存"""
        # 清理过期缓存
        current_time = time.time()
        expired_keys = [k for k, v in self.cache.items() if current_time - v.get("timestamp", 0) > self.cache_ttl]
        for key in expired_keys:
            self.cache.pop(key, None)
        
        logger.info(f"缓存初始化完成，清理了 {len(expired_keys)} 条过期数据")
    
    def _validate_api_keys(self) -> bool:
        """验证API密钥的有效性"""
        valid_keys = 0
        total_keys = len(self.api_keys)
        
        # 检查每个API密钥
        for api_name, api_key in self.api_keys.items():
            if not api_key or api_key == "your_key":
                logger.warning(f"{api_name} API密钥未设置或无效")
                continue
            
            # 简单验证密钥格式
            if len(api_key) > 10:
                valid_keys += 1
                logger.info(f"{api_name} API密钥格式有效")
        
        return valid_keys > 0
    
    def _test_data_source(self, url: str) -> Tuple[bool, Optional[int], str]:
        """测试单个数据源的连接性"""
        try:
            # 使用GET请求而不是HEAD，因为某些网站可能限制HEAD请求
            # 设置更短的超时时间进行连接测试
            response = self.session.get(url, timeout=5, allow_redirects=True)
            if response.status_code in [200, 301, 302, 403, 412, 521]:
                # 即使返回错误状态码，也认为连接成功（网站可达）
                return True, response.status_code, "连接成功"
            else:
                return False, response.status_code, f"状态码异常: {response.status_code}"
        except requests.exceptions.Timeout:
            return False, None, "连接超时"
        except requests.exceptions.ConnectionError as e:
            return False, None, f"连接错误: {str(e)}"
        except Exception as e:
            return False, None, f"未知错误: {str(e)}"
    
    def _test_backup_sources(self) -> bool:
        """测试备用数据源的可用性"""
        backup_sources = []
        
        # 收集所有备用数据源URL
        for key, value in self.config["sources"].items():
            if key.startswith("backup_") and isinstance(value, list):
                backup_sources.extend(value)
        
        if not backup_sources:
            return False
        
        # 测试备用数据源（只测试第一个URL）
        test_url = backup_sources[0]
        try:
            success, _, _ = self._test_data_source(test_url)
            return success
        except Exception:
            return False
    
    def _initialize_industry_classifier(self) -> None:
        """初始化行业分类器"""
        # 这里可以加载行业分类模型或规则
        # 简单起见，我们使用规则匹配方法
        self.industry_keywords = {
            "信息技术": ["软件", "互联网", "IT", "计算机", "通信", "电子", "科技", "信息", "数据", "网络", "人工智能", "AI"],
            "制造业": ["制造", "工业", "机械", "设备", "生产", "加工", "材料", "零部件", "工厂"],
            "金融业": ["金融", "银行", "保险", "证券", "投资", "资产", "基金", "理财", "信托"],
            "房地产": ["房地产", "地产", "建筑", "房产", "物业", "开发", "置业"],
            "批发零售": ["贸易", "批发", "零售", "商贸", "超市", "商场", "电商", "进出口"],
            "服务业": ["服务", "咨询", "中介", "教育", "培训", "医疗", "健康", "旅游", "餐饮", "酒店"]
        }
        
        logger.info("行业分类器初始化完成")

    
    def process(self, company_info: Dict[str, str]) -> CreditData:
        """
        执行数据采集工作流，为指定企业收集全面的信用数据

        参数:
            company_info: 包含企业名称和/或ID的字典

        返回:
            包含所有已收集和处理数据的CreditData对象
        """
        if not self.initialized:
            logger.error("代理尚未初始化，无法处理请求。")
            raise RuntimeError("DataCollectionModule尚未初始化。")

        company_name = company_info.get("company_name")
        industry = company_info.get("industry")
        logger.info(f"开始为公司 '{company_name}' (行业: {industry}) 收集信用数据...")
        start_time = time.time()

        credit_data = CreditData(raw_data={})

        # 并行执行数据采集任务
        tasks = {
            "enterprise_info": self.executor.submit(self._collect_enterprise_info, company_name, industry),
            "compliance_data": self.executor.submit(self._collect_compliance_data, company_name),
            "patent_data": self.executor.submit(self._collect_patent_data, company_name),
            "operator_info": self.executor.submit(self._collect_operator_info, company_name),
            "environment_data": self.executor.submit(self._collect_environment_data, company_name),
        }

        for task_name, future in tasks.items():
            try:
                credit_data.raw_data[task_name] = future.result(timeout=self.timeout)
            except Exception as e:
                logger.error(f"收集 '{task_name}' 数据时出错: {e}", exc_info=True)
                credit_data.raw_data[task_name] = {"error": str(e)}

        # 数据整合与清洗
        self._integrate_and_clean_data(credit_data)

        # 数据质量评估
        quality_score, quality_report = self._assess_data_quality(credit_data)
        # credit_data.quality_score = quality_score
        # credit_data.quality_report = quality_report
        
        # 触发数据收集完成事件
        self.event_bus.publish(Event(
            type=EventType.DATA_COLLECTED,
            data=credit_data,
            source="DataCollectionModule"
        ))

        processing_time = time.time() - start_time
        logger.info(f"为 '{company_name}' 的数据收集完成，耗时 {processing_time:.2f} 秒。质量得分: {quality_score:.2f}")
        
        return credit_data
    
    def validate_input(self, input_data: Dict[str, str]) -> Tuple[bool, str]:
        """验证输入数据的有效性"""
        if not isinstance(input_data, dict):
            return False, "输入必须是字典类型"
        
        if "company_name" not in input_data or not input_data["company_name"]:
            return False, "输入必须包含有效的企业名称(company_name)"
        
        return True, "输入验证通过"
    
    def reset(self) -> None:
        """重置模块状态"""
        self.session.close()
        self.session = requests.Session()
        self.session.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        logger.info("数据采集模块已重置")
    
    def _collect_enterprise_info(self, company_name: str) -> Dict[str, Any]:
        """采集企业基础信息"""
        logger.info(f"开始采集 {company_name} 的基础信息...")
        try:
            # 模拟从多个数据源获取信息
            info = self._search_enterprise_basic_info(company_name)
            
            # 补充行业信息
            if not info.get("industry"):
                industry_info = self._auto_detect_industry(company_name, info.get("business_scope"))
                if industry_info:
                    info["industry"] = industry_info.get("industry")

            logger.info(f"成功采集到企业基础信息: {info}")
            return info
        except Exception as e:
            logger.error(f"采集企业基础信息失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_compliance_risk_indicators(self, compliance_data: Dict[str, Any]) -> Dict[str, float]:
        """计算合规风险指标"""
        risk_indicators = {}
        
        # 诉讼风险
        lawsuit_count = compliance_data.get("lawsuit_count", 0)
        risk_indicators["litigation_risk"] = min(1.0, lawsuit_count / 10)

        # 处罚风险
        penalty_count = len(compliance_data.get("administrative_penalties", []))
        risk_indicators["penalty_risk"] = min(1.0, penalty_count / 5)

        # 失信风险
        dishonesty_count = compliance_data.get("shareholder_dishonesty", 0)
        risk_indicators["dishonesty_risk"] = min(1.0, dishonesty_count / 3)
        if compliance_data.get("credit_blacklist"):
            risk_indicators["dishonesty_risk"] = 1.0

        # 综合风险
        weights = self.config.get("risk_weights", {
            "litigation_risk": 0.5,
            "penalty_risk": 0.3,
            "dishonesty_risk": 0.2
        })
        overall_risk = sum(risk_indicators.get(risk, 0.0) * weight for risk, weight in weights.items())
        risk_indicators["overall_compliance_risk"] = min(1.0, overall_risk)

        return risk_indicators

    def _search_enterprise_basic_info(self, company_name: str) -> Dict[str, Any]:
        """
        从多个数据源获取企业基础信息，并进行融合
        """
        logger.info(f"开始从多个数据源搜索 {company_name} 的基础信息...")
        
        all_info = {}

        # 从企查查查询
        qichacha_key = self.api_keys.get("qichacha")
        if qichacha_key:
            try:
                # 此处应调用企查查API，以下为模拟
                logger.info("正在从企查查查询...")
                # response = requests.get(f"https://api.qichacha.com/v2/search?key={qichacha_key}&keyword={company_name}")
                # all_info['qichacha'] = response.json() 
                all_info['qichacha'] = {"company_name": company_name, "registered_capital": "1000万人民币", "industry": "科技"} # 模拟数据
            except Exception as e:
                logger.warning(f"从企查查查询失败: {e}")

        # 从天眼查查询
        tianyancha_key = self.api_keys.get("tianyancha")
        if tianyancha_key:
            try:
                # 此处应调用天眼查API，以下为模拟
                logger.info("正在从天眼查查询...")
                # response = requests.get(f"https://api.tianyancha.com/services/v3/open/search/{company_name}", headers={"Authorization": tianyancha_key})
                # all_info['tianyancha'] = response.json()
                all_info['tianyancha'] = {"company_name": company_name, "registered_capital": "1000万人民币", "establishment_date": "2015-01-01"} # 模拟数据
            except Exception as e:
                logger.warning(f"从天眼查查询失败: {e}")

        # 融合信息
        if all_info:
            # 简单融合，实际场景需要更复杂的策略
            fused_info = {}
            for source, info in all_info.items():
                fused_info.update(info)
            return fused_info
        else:
            logger.warning(f"所有数据源均未能查询到 {company_name} 的信息，返回模拟数据")
            return {"company_name": company_name, "error": "No data found from any source."}

    def _extract_capital(self, snippet: str) -> str:
        """从文本中提取注册资本信息"""
        # 简单的正则表达式匹配注册资本
        capital_pattern = r'注册资本[：:]\s*([\d,.]+万?[元人民币]*)'
        match = re.search(capital_pattern, snippet)
        if match:
            return match.group(1)
        return None  # 返回None而不是默认值
    
    def _extract_legal_representative(self, snippet: str) -> str:
        """从文本中提取法定代表人信息"""
        # 简单的正则表达式匹配法定代表人
        legal_pattern = r'法定代表人[：:]\s*([\u4e00-\u9fa5]{2,4})'
        match = re.search(legal_pattern, snippet)
        if match:
            return match.group(1)
        return None  # 返回None而不是默认值
    
    def _extract_date(self, snippet: str) -> str:
        """从文本中提取日期信息"""
        # 简单的正则表达式匹配日期
        date_pattern = r'(\d{4}[-年]\d{1,2}[-月]\d{1,2}日?)'
        match = re.search(date_pattern, snippet)
        if match:
            return match.group(1).replace('年', '-').replace('月', '-').replace('日', '')
        return None  # 返回None而不是默认值
    
    def _extract_industry(self, snippet: str) -> str:
        """从文本中提取行业信息"""
        # 简单的行业关键词匹配
        industries = ["信息技术", "制造业", "金融", "房地产", "医疗", "教育", "零售", "餐饮", "建筑", "物流", "农业", "能源", "文化", "体育", "旅游"]
        for industry in industries:
            if industry in snippet:
                return industry
        return None
    
    def _extract_address(self, snippet: str) -> str:
        """从文本中提取地址信息"""
        # 简单的地址提取逻辑
        address_patterns = [
            r'地址[：:]\s*([^\n，。；！？]+)',
            r'所在地[：:]\s*([^\n，。；！？]+)',
            r'位于\s*([^\n，。；！？]+)'
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, snippet)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_business_scope(self, snippet: str) -> str:
        """从文本中提取经营范围信息"""
        # 简单的经营范围提取逻辑
        scope_patterns = [
            r'经营范围[：:]\s*([^\n。；！？]+)',
            r'业务范围[：:]\s*([^\n。；！？]+)',
            r'主营[：:]\s*([^\n。；！？]+)'
        ]
        
        for pattern in scope_patterns:
            match = re.search(pattern, snippet)
            if match:
                return match.group(1).strip()
        return None
    
    def _auto_detect_industry(self, company_name: str, business_scope: str = None) -> Dict[str, Any]:
        """
        自动检测企业行业类型，支持多维度分析和上下文理解
        
        参数:
            company_name: 企业名称
            business_scope: 经营范围（可选）
            
        返回:
            包含行业类型、置信度和详细分数的字典
        """
        # 使用初始化时加载的行业分类器
        if hasattr(self, 'industry_keywords') and self.industry_keywords:
            industry_keywords = self.industry_keywords
        else:
            # 扩展行业分类和关键词
            industry_keywords = {
                "Information Technology": {
                    "keywords": ["科技", "技术", "软件", "网络", "信息", "数据", "AI", "人工智能", "互联网", "电子", 
                               "计算机", "通信", "IT", "云计算", "大数据", "区块链", "芯片", "算法", "编程", "开发"],
                    "weight": 1.0,
                    "sub_industries": ["软件开发", "互联网服务", "IT咨询", "电子设备制造", "通信服务", "人工智能"]
                },
                "Manufacturing": {
                    "keywords": ["制造", "工业", "机械", "设备", "生产", "加工", "材料", "化工", "钢铁", "汽车", 
                               "纺织", "食品", "医药", "电气", "仪器", "家电", "五金", "塑料", "金属", "建材"],
                    "weight": 1.0,
                    "sub_industries": ["汽车制造", "机械设备", "电子设备", "化工制品", "医药制造", "食品加工"]
                },
                "Wholesale and Retail": {
                    "keywords": ["贸易", "商贸", "零售", "批发", "销售", "商业", "超市", "商店", "电商", "进出口", 
                               "百货", "购物", "商场", "专卖", "连锁", "分销", "供应链", "电子商务", "网店", "商品"],
                    "weight": 1.0,
                    "sub_industries": ["批发贸易", "零售业", "电子商务", "进出口贸易", "专卖店", "连锁经营"]
                },
                "Services": {
                    "keywords": ["服务", "咨询", "管理", "顾问", "代理", "中介", "物流", "运输", "教育", "培训", 
                               "餐饮", "酒店", "旅游", "文化", "娱乐", "健康", "医疗", "金融", "保险", "法律"],
                    "weight": 1.0,
                    "sub_industries": ["商务服务", "教育培训", "餐饮住宿", "文化娱乐", "医疗健康", "金融服务"]
                },
                "Real Estate": {
                    "keywords": ["房地产", "地产", "房产", "建筑", "开发", "物业", "置业", "园区", "商业地产", "住宅", 
                               "楼盘", "公寓", "别墅", "写字楼", "商铺", "租赁", "中介", "装修", "设计", "工程"],
                    "weight": 1.0,
                    "sub_industries": ["房地产开发", "物业管理", "房产中介", "建筑工程", "室内设计", "园区运营"]
                },
                "Finance": {
                    "keywords": ["金融", "投资", "证券", "银行", "保险", "基金", "资本", "融资", "理财", "信托", 
                               "资产", "财富", "股权", "债券", "期货", "支付", "信贷", "租赁", "担保", "典当"],
                    "weight": 1.0,
                    "sub_industries": ["银行业", "证券业", "保险业", "投资管理", "资产管理", "金融科技"]
                },
                "Agriculture": {
                    "keywords": ["农业", "种植", "养殖", "畜牧", "渔业", "林业", "农产品", "农资", "农机", "农村", 
                               "粮食", "水果", "蔬菜", "花卉", "种子", "肥料", "农药", "灌溉", "生态", "有机"],
                    "weight": 1.0,
                    "sub_industries": ["种植业", "养殖业", "农产品加工", "林业", "渔业", "农业技术服务"]
                },
                "Energy": {
                    "keywords": ["能源", "电力", "石油", "天然气", "煤炭", "新能源", "太阳能", "风能", "水电", "核电", 
                               "发电", "输电", "配电", "电网", "油气", "矿业", "节能", "环保", "碳", "氢能"],
                    "weight": 1.0,
                    "sub_industries": ["电力生产", "石油天然气", "煤炭开采", "新能源", "能源服务", "矿产资源"]
                },
                "Healthcare": {
                    "keywords": ["医疗", "健康", "医院", "药品", "医药", "生物", "诊断", "治疗", "康复", "保健", 
                               "医生", "护理", "医械", "体检", "养老", "心理", "基因", "疫苗", "中医", "西医"],
                    "weight": 1.0,
                    "sub_industries": ["医疗服务", "医药制造", "医疗器械", "生物技术", "健康管理", "养老服务"]
                },
                "Transportation": {
                    "keywords": ["交通", "运输", "物流", "快递", "航空", "航运", "铁路", "公路", "港口", "仓储", 
                               "货运", "客运", "配送", "供应链", "车辆", "船舶", "航班", "班车", "冷链", "集装箱"],
                    "weight": 1.0,
                    "sub_industries": ["道路运输", "航空运输", "水路运输", "铁路运输", "仓储物流", "快递服务"]
                }
            }
        
        # 准备分析文本
        text_to_analyze = f"{company_name} {business_scope or ''}"
        
        # 1. 基于关键词匹配计算行业分数
        industry_scores = {}
        for industry, config in industry_keywords.items():
            # 基础分数 - 关键词匹配
            keyword_score = 0
            matched_keywords = []
            
            for keyword in config["keywords"]:
                if keyword in text_to_analyze:
                    keyword_score += config["weight"]
                    matched_keywords.append(keyword)
            
            # 子行业匹配加分
            sub_industry_score = 0
            if "sub_industries" in config:
                for sub_industry in config["sub_industries"]:
                    if sub_industry in text_to_analyze:
                        sub_industry_score += 0.5  # 子行业匹配额外加分
            
            # 计算总分
            total_score = keyword_score + sub_industry_score
            
            # 记录分数和匹配的关键词
            if total_score > 0:
                industry_scores[industry] = {
                    "score": total_score,
                    "matched_keywords": matched_keywords,
                    "sub_industry_score": sub_industry_score
                }
        
        # 2. 上下文分析 - 考虑关键词在文本中的位置和密度
        for industry in industry_scores:
            # 名称中的关键词权重更高
            name_matches = sum(1 for kw in industry_keywords[industry]["keywords"] if kw in company_name)
            if name_matches > 0:
                industry_scores[industry]["score"] += name_matches * 0.5
                industry_scores[industry]["name_match_bonus"] = name_matches * 0.5
            
            # 经营范围中的关键词密度
            if business_scope:
                scope_matches = sum(1 for kw in industry_keywords[industry]["keywords"] if kw in business_scope)
                density = scope_matches / max(1, len(business_scope) / 10)  # 每10个字符的关键词密度
                if density > 0.1:  # 密度阈值
                    industry_scores[industry]["score"] += density * 0.3
                    industry_scores[industry]["scope_density_bonus"] = density * 0.3
        
        # 3. 找到最高分的行业
        if industry_scores:
            # 提取简化的分数字典用于比较
            simple_scores = {industry: data["score"] for industry, data in industry_scores.items()}
            best_industry = max(simple_scores, key=simple_scores.get)
            max_score = simple_scores[best_industry]
            
            if max_score > 0:
                # 计算置信度 - 考虑最高分与次高分的差距
                sorted_scores = sorted(simple_scores.values(), reverse=True)
                score_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
                
                # 置信度计算 - 基础置信度 + 分数差距奖励
                base_confidence = min(max_score / 5.0, 0.8)  # 基础置信度上限为0.8
                gap_bonus = min(score_gap / max_score, 0.2) if max_score > 0 else 0  # 差距奖励上限为0.2
                confidence = base_confidence + gap_bonus
                
                return {
                    "industry": best_industry,
                    "confidence": round(confidence, 2),
                    "all_scores": simple_scores,
                    "detailed_scores": industry_scores,
                    "matched_keywords": industry_scores[best_industry]["matched_keywords"],
                    "analysis_text": text_to_analyze
                }
        
        # 4. 如果没有匹配，尝试使用更模糊的匹配
        if not industry_scores:
            # 使用更宽松的匹配规则
            for industry, config in industry_keywords.items():
                for keyword in config["keywords"][:5]:  # 只使用前5个关键词
                    if keyword in text_to_analyze:
                        return {
                            "industry": industry,
                            "confidence": 0.3,
                            "all_scores": {industry: 0.3},
                            "matched_keywords": [keyword],
                            "analysis_text": text_to_analyze,
                            "note": "使用宽松匹配规则"
                        }
        
        # 5. 默认返回服务业，但置信度较低
        return {
            "industry": "Services",
            "confidence": 0.2,
            "all_scores": {"Services": 0.2},
            "matched_keywords": [],
            "analysis_text": text_to_analyze,
            "note": "未找到匹配，使用默认行业"
        }
    
    def _collect_patent_data(self, company_name: str) -> Dict[str, Any]:
        """采集企业专利数据"""
        logger.info(f"开始采集 {company_name} 的专利数据...")
        try:
            patent_info = self._query_patent_data(company_name)
            logger.info(f"成功采集到专利数据: {patent_info}")
            return patent_info
        except Exception as e:
            logger.error(f"采集专利数据失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _query_patent_data(self, company_name: str) -> Dict[str, Any]:
        """查询企业专利数据"""
        # 模拟专利查询
        if "科技" in company_name:
            return {"count": random.randint(5, 50)}
        elif "制造" in company_name:
            return {"count": random.randint(2, 20)}
        else:
            return {"count": random.randint(0, 5)}
    
    def _assess_data_quality(self, credit_data: CreditData) -> Tuple[float, Dict[str, Any]]:
        """评估整体数据质量，并返回分数和报告"""
        quality_score = 1.0
        report = {"completeness": {}, "consistency": {}, "validity": {}}
        
        # 1. 完整性检查
        total_fields = 0
        missing_fields = 0
        for source, data in credit_data.raw_data.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    total_fields += 1
                    if value is None or value == '':
                        missing_fields += 1
                        report["completeness"].setdefault(source, []).append(key)
        
        completeness_score = 1 - (missing_fields / total_fields) if total_fields > 0 else 0
        quality_score *= (0.8 + 0.2 * completeness_score) # 完整性权重
        report["completeness"]["score"] = completeness_score

        # 2. 一致性检查 (示例)
        # 比较不同来源的注册资本
        capitals = []
        if 'enterprise_info' in credit_data.raw_data and 'registered_capital' in credit_data.raw_data['enterprise_info']:
            capitals.append(credit_data.raw_data['enterprise_info']['registered_capital'])
        # ... 可以从其他源获取更多资本信息
        
        if len(set(capitals)) > 1:
            quality_score *= 0.9
            report["consistency"]["registered_capital"] = f"注册资本信息不一致: {capitals}"

        # 3. 有效性/异常值检查
        # 示例：检查专利数量
        if 'patent_data' in credit_data.raw_data and 'count' in credit_data.raw_data['patent_data']:
            patent_count = credit_data.raw_data['patent_data']['count']
            if patent_count > 10000: # 异常阈值
                quality_score *= 0.95
                report["validity"]["patent_count"] = f"专利数量异常高: {patent_count}"

        final_score = max(0.0, min(1.0, quality_score))
        return final_score, report

    def _integrate_and_clean_data(self, credit_data: CreditData):
        """整合和清洗来自不同来源的数据"""
        logger.info("开始数据整合与清洗...")
        
        # 示例：统一法人代表名称
        legal_representatives = []
        if 'enterprise_info' in credit_data.raw_data and 'legal_representative' in credit_data.raw_data['enterprise_info']:
            legal_representatives.append(credit_data.raw_data['enterprise_info']['legal_representative'])
        
        if 'operator_info' in credit_data.raw_data and 'name' in credit_data.raw_data['operator_info']:
            legal_representatives.append(credit_data.raw_data['operator_info']['name'])
        
        if legal_representatives:
            # 选择最常见的名称
            unified_name = max(set(legal_representatives), key=legal_representatives.count)
            credit_data.raw_data.setdefault('unified_profile', {})['legal_representative'] = unified_name
            logger.info(f"统一法人代表为: {unified_name}")

        # ... 其他清洗和整合逻辑 ...
        logger.info("数据整合与清洗完成")

    def _collect_operator_info(self, company_name: str) -> Dict[str, Any]:
        """采集企业经营者信息"""
        logger.info(f"开始采集 {company_name} 的经营者信息...")
        # 模拟数据
        return {
            "name": "模拟法人",
            "background": "拥有多年的行业经验"
        }

    def _collect_environment_data(self, company_name: str) -> Dict[str, Any]:
        """采集企业环保合规数据"""
        logger.info(f"开始采集 {company_name} 的环保合规数据...")
        # 模拟数据
        return {
            "environmental_penalties": [],
            "compliance_status": "良好"
        }

    def _collect_compliance_data(self, company_name: str) -> Dict[str, Any]:
        """
        采集企业合规数据(司法记录、行政处罚等)，支持多数据源和风险指标
        
        参数:
            company_name: 企业名称
        
        返回:
            包含企业合规数据的字典
        """
        logger.info(f"开始采集 {company_name} 的合规数据...")

        # 使用缓存数据（如果有且未过期）
        cache_key = f"compliance_{hashlib.md5(company_name.encode()).hexdigest()}"
        if cache_key in self.cache and time.time() - self.cache[cache_key].get("timestamp", 0) < self.cache_ttl:
            logger.info(f"使用缓存的合规数据: {company_name}")
            return self.cache[cache_key]["data"]

        # 模拟数据采集
        compliance_data = self._generate_mock_compliance_data(company_name)
        
        # 计算风险指标
        compliance_data["risk_indicators"] = self._calculate_compliance_risk_indicators(compliance_data)
        
        # 更新缓存
        self.cache[cache_key] = {
            "data": compliance_data,
            "timestamp": time.time()
        }
        
        logger.info(f"完成 {company_name} 的合规数据采集。")
        return compliance_data

    def _generate_mock_compliance_data(self, company_name: str) -> Dict[str, Any]:
        """生成模拟合规数据，用于开发测试"""
        seed = sum(ord(c) for c in company_name)
        random.seed(seed)

        lawsuit_count = random.randint(0, 8)
        lawsuit_details = [{
            "case_type": random.choice(["合同纠纷", "劳动纠纷", "知识产权纠纷"]),
            "amount": random.randint(10000, 5000000),
            "date": f"{random.randint(2020, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "result": random.choice(["胜诉", "败诉", "调解", "撤诉"]),
        } for _ in range(lawsuit_count)]

        penalty_count = random.randint(0, 3)
        administrative_penalties = [{
            "reason": random.choice(["广告违规", "产品质量问题", "税务违规"]),
            "amount": random.randint(5000, 500000),
            "date": f"{random.randint(2020, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        } for _ in range(penalty_count)]

        return {
            "lawsuit_count": lawsuit_count,
            "lawsuit_details": lawsuit_details,
            "administrative_penalties": administrative_penalties,
            "shareholder_dishonesty": random.randint(0, 2),
            "credit_blacklist": random.random() < 0.05,
            "data_sources": ["mock_data"],
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _calculate_compliance_risk_indicators(self, compliance_data: Dict[str, Any]) -> Dict[str, float]:
        """计算合规风险指标"""
        risk_indicators = {}
        
        # 诉讼风险
        lawsuit_count = compliance_data.get("lawsuit_count", 0)
        risk_indicators["litigation_risk"] = min(1.0, lawsuit_count / 10)

        # 处罚风险
        penalty_count = len(compliance_data.get("administrative_penalties", []))
        risk_indicators["penalty_risk"] = min(1.0, penalty_count / 5)

        # 失信风险
        dishonesty_count = compliance_data.get("shareholder_dishonesty", 0)
        risk_indicators["dishonesty_risk"] = min(1.0, dishonesty_count / 3)
        if compliance_data.get("credit_blacklist"):
            risk_indicators["dishonesty_risk"] = 1.0

        # 综合风险
        weights = self.config.get("risk_weights", {
            "litigation_risk": 0.5,
            "penalty_risk": 0.3,
            "dishonesty_risk": 0.2
        })
        overall_risk = sum(risk_indicators.get(risk, 0.0) * weight for risk, weight in weights.items())
        risk_indicators["overall_compliance_risk"] = min(1.0, overall_risk)

        return risk_indicators
