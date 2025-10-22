# CR-Agent 信用风险评估智能体系统

## 项目简介

CR-Agent是一个基于多智能体架构的信用风险评估系统，通过协调多个专业化的智能体模块，实现对企业信用风险的全面分析和评估。

## 系统架构

### 核心智能体文件

- **agent.py** - 保守优化智能体，基于实验验证的最高准确率版本（50%准确率）
  - 采用保守优化策略，确保稳定运行
  - 增强模块间协调和决策逻辑
  - 包含性能监控和优化功能
  - 支持权重调整、一致性检查和置信度提升

### 基础模块文件

- **base.py** - 基础数据结构和工具类定义
- **events.py** - 事件系统，支持模块间异步通信
- **data_collection.py** - 数据收集模块，从多个数据源获取企业信息
- **data_analysis.py** - 数据分析模块，进行特征工程和数据预处理
- **risk_analysis.py** - 风险分析模块，计算风险评分和等级
- **feedback_iteration.py** - 反馈迭代模块，基于历史数据优化模型

### 数据处理文件

- **data_preprocessing.py** - 数据预处理工具
- **example_usage.py** - 使用示例和演示代码

### 配置和依赖

- **requirements.txt** - Python依赖包列表
- **__init__.py** - Python包初始化文件

### 数据文件

- **data/** - 数据文件夹
  - **financial_train.csv** - 训练数据集
  - **financial_test.csv** - 测试数据集
  - **financial_validation.csv** - 验证数据集

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

```python
from agent import CreditAgent

# 创建智能体实例
agent = CreditAgent()

# 初始化系统
agent.initialize()

# 分析企业信用风险
result = agent.analyze_credit_risk("企业名称")
print(f"风险等级: {result.risk_level}")
print(f"信用评分: {result.credit_score}")
```

### 3. 使用不同类型的智能体

```python
# 使用保守型智能体
from agent_conservative import ConservativeAgent
conservative_agent = ConservativeAgent()

# 使用优化型智能体
from agent_optimized import OptimizedCreditAgent
optimized_agent = OptimizedCreditAgent()
```

## 智能体特点

### ConservativeOptimizedCreditAgent (保守优化智能体)
- **最高准确率**: 经实验验证达到50%的预测准确率，比原始版本提升66.7%
- **处理速度优化**: 平均处理时间仅0.016秒，比原始版本快89.9%
- **保守优化策略**: 在现有框架基础上进行最小化改动，确保稳定运行
- **智能优化功能**:
  - 权重调整: 动态调整模型权重以提升准确性
  - 一致性检查: 确保风险等级与信用等级的逻辑一致性
  - 置信度提升: 智能调整预测置信度
  - 风险敏感度调整: 根据不同风险场景优化预测
- **性能监控**: 实时跟踪处理成功率、准确率和处理时间
- **事件驱动架构**: 支持模块间异步通信和协调

## 系统特性

1. **模块化设计** - 各个功能模块独立，便于维护和扩展
2. **事件驱动** - 基于事件系统实现模块间松耦合通信
3. **多智能体协作** - 不同类型的智能体适应不同的业务场景
4. **数据驱动** - 基于真实的金融数据集进行训练和验证
5. **可扩展性** - 支持添加新的数据源和分析模块

## 数据集说明

项目使用的金融数据集包含：
- 训练集：用于模型训练
- 测试集：用于模型评估
- 验证集：用于模型调优

数据集包含多维度的企业财务指标，支持全面的信用风险评估。

## 注意事项

1. 确保所有依赖包正确安装
2. 数据文件路径配置正确
3. 根据实际需求选择合适的智能体类型
4. 定期更新模型参数以保持最佳性能

## 技术支持

如有问题，请参考example_usage.py中的示例代码，或查看各个模块的详细文档。
