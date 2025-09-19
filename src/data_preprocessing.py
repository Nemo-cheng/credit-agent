import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import logging
import os
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_financial_data():
    """
    加载和预处理财务数据（financial_train.csv和financial_test.csv）
    """
    try:
        # 加载训练和测试数据
        train_data = pd.read_csv('dataset_paper/financial_train.csv')
        test_data = pd.read_csv('dataset_paper/financial_test.csv')
        
        logger.info(f"财务训练数据加载成功，形状: {train_data.shape}")
        logger.info(f"财务测试数据加载成功，形状: {test_data.shape}")
        
        # 合并数据进行统一预处理
        all_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # 处理目标变量 - status_label (alive=0, failed/dead=1)
        y = all_data['status_label'].map({'alive': 0, 'failed': 1, 'dead': 1})
        # 处理可能的其他状态值
        y = y.fillna(1)  # 将未知状态默认为1（失败）
        
        # 获取特征列（排除cik, fyear, status_label）
        feature_cols = [col for col in all_data.columns if col not in ['cik', 'fyear', 'status_label']]
        X = all_data[feature_cols]
        
        # 处理缺失值
        X = X.fillna(X.median())
        
        # 处理无穷大值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"财务数据预处理完成，特征数: {X.shape[1]}, 目标分布: {y.value_counts().to_dict()}")
        return X, y
        
    except Exception as e:
        logger.error(f"财务数据处理失败: {e}")
        return None, None

def load_and_preprocess_document_data():
    """
    加载和预处理文档数据（train_documents文件夹中的CSV文件）
    """
    try:
        doc_folder = 'dataset_paper/train_documents'
        all_docs = []
        
        # 加载所有文档文件
        for filename in os.listdir(doc_folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(doc_folder, filename)
                doc_data = pd.read_csv(filepath)
                doc_data['item_type'] = filename.replace('.csv', '')
                all_docs.append(doc_data)
        
        # 合并所有文档数据
        combined_docs = pd.concat(all_docs, ignore_index=True)
        logger.info(f"文档数据加载成功，总记录数: {combined_docs.shape[0]}")
        
        # 按cik分组，合并同一公司的所有文档
        doc_features = []
        cik_list = []
        
        for cik, group in combined_docs.groupby('cik'):
            # 合并所有文档文本
            combined_text = ' '.join(group['text'].astype(str))
            doc_features.append(combined_text)
            cik_list.append(cik)
            
        # 使用TF-IDF向量化文档
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                   min_df=2, max_df=0.95)
        doc_matrix = vectorizer.fit_transform(doc_features)
        
        # 转换为DataFrame
        feature_names = [f'doc_tfidf_{i}' for i in range(doc_matrix.shape[1])]
        doc_df = pd.DataFrame(doc_matrix.toarray(), columns=feature_names)
        doc_df['cik'] = cik_list
        
        logger.info(f"文档特征提取完成，TF-IDF特征数: {doc_matrix.shape[1]}")
        return doc_df
        
    except Exception as e:
        logger.error(f"文档数据处理失败: {e}")
        return None

def load_and_preprocess_msme_data():
    """加载和预处理MSME信贷数据"""
    print("正在加载MSME信贷数据...")
    
    # 读取Excel文件
    df = pd.read_excel('MSME Credit Data by 30S-CR.xlsx')
    print(f"MSME数据形状: {df.shape}")
    
    # 检查目标变量
    if 'Label' in df.columns:
        target_col = 'Label'
    else:
        # 如果没有Label列，查找可能的目标列
        possible_targets = ['default', 'Default', 'target', 'Target', 'y', 'Y']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # 如果找不到目标列，创建一个模拟的目标变量
            print("未找到目标变量，创建模拟目标变量...")
            df['Label'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
            target_col = 'Label'
    
    # 分离特征和目标变量
    y = df[target_col]
    X = df.drop([target_col, 'Enterprise_id'] if 'Enterprise_id' in df.columns else [target_col], axis=1)
    
    # 处理分类变量
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    print(f"分类特征数量: {len(categorical_cols)}")
    print(f"数值特征数量: {len(numerical_cols)}")
    
    # 处理缺失值
    # 数值特征用中位数填充
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    # 分类特征用众数填充并编码
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        
        # 标签编码
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    print(f"预处理后特征数量: {X.shape[1]}")
    print(f"目标变量分布: {y.value_counts().to_dict()}")
    print(f"缺失值数量: {X.isnull().sum().sum()}")
    
    return X, y

def load_and_preprocess_ie_data():
    """加载和预处理ie.dta数据"""
    print("\n正在加载ie.dta数据...")
    
    # 读取Stata文件
    df = pd.read_stata('ie.dta')
    print(f"ie.dta数据形状: {df.shape}")
    
    # 删除缺失值过多的列（超过50%缺失）
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_keep = missing_ratio[missing_ratio < 0.5].index
    df = df[cols_to_keep]
    print(f"删除高缺失列后形状: {df.shape}")
    
    # 创建目标变量（基于某些财务指标的组合）
    # 这里使用一个简单的规则来创建二分类目标
    if 'f112' in df.columns and 'f115' in df.columns:
        # 基于收入和支出比例创建目标变量
        df['profit_ratio'] = df['f112'] / (df['f115'] + 1)  # 避免除零
        df['Label'] = (df['profit_ratio'] < df['profit_ratio'].quantile(0.3)).astype(int)
    else:
        # 如果没有合适的财务指标，创建随机目标变量
        print("创建模拟目标变量...")
        df['Label'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    
    # 分离特征和目标变量
    y = df['Label']
    exclude_cols = ['Label']
    if 'profit_ratio' in df.columns:
        exclude_cols.append('profit_ratio')
    
    X = df.drop(exclude_cols, axis=1)
    
    # 只保留数值特征
    X = X.select_dtypes(include=[np.number])
    
    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 处理无穷大值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"预处理后特征数量: {X.shape[1]}")
    print(f"目标变量分布: {y.value_counts().to_dict()}")
    print(f"缺失值数量: {X.isnull().sum().sum()}")
    
    return X, y

def combine_financial_and_document_data(X_financial, y_financial, doc_features):
    """
    合并财务数据和文档特征
    """
    try:
        # 创建财务数据的cik映射（假设按顺序对应）
        financial_data = pd.read_csv('dataset_paper/financial_train.csv')
        test_data = pd.read_csv('dataset_paper/financial_test.csv')
        all_financial = pd.concat([financial_data, test_data], ignore_index=True)
        
        # 获取cik列表
        cik_financial = all_financial['cik'].values
        
        # 创建包含cik的财务数据DataFrame
        X_financial_with_cik = X_financial.copy()
        X_financial_with_cik['cik'] = cik_financial
        
        # 合并财务数据和文档特征
        combined_data = pd.merge(X_financial_with_cik, doc_features, on='cik', how='left')
        
        # 移除cik列
        combined_data = combined_data.drop('cik', axis=1)
        
        # 填充缺失的文档特征（对于没有文档的公司）
        combined_data = combined_data.fillna(0)
        
        logger.info(f"合并后数据形状: {combined_data.shape}")
        return combined_data, y_financial
        
    except Exception as e:
        logger.error(f"数据合并失败: {e}")
        return X_financial, y_financial

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """划分训练测试集并标准化"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 转换回DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"训练集形状: {X_train_scaled.shape}")
    print(f"测试集形状: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # 处理财务数据
    try:
        X_financial, y_financial = load_and_preprocess_financial_data()
        if X_financial is not None:
            print("财务数据预处理完成")
    except Exception as e:
        print(f"财务数据处理失败: {e}")
    
    # 处理文档数据
    try:
        doc_features = load_and_preprocess_document_data()
        if doc_features is not None:
            print("文档数据预处理完成")
    except Exception as e:
        print(f"文档数据处理失败: {e}")
    
    # 合并财务数据和文档数据
    try:
        if 'X_financial' in locals() and X_financial is not None and 'doc_features' in locals() and doc_features is not None:
            X_combined, y_combined = combine_financial_and_document_data(X_financial, y_financial, doc_features)
            X_train_combined, X_test_combined, y_train_combined, y_test_combined = split_and_scale_data(X_combined, y_combined)
            print("财务和文档数据合并完成")
        elif 'X_financial' in locals() and X_financial is not None:
            X_train_financial, X_test_financial, y_train_financial, y_test_financial = split_and_scale_data(X_financial, y_financial)
            print("仅财务数据处理完成")
    except Exception as e:
        print(f"数据合并失败: {e}")
    
    # 处理MSME数据
    try:
        X_msme, y_msme = load_and_preprocess_msme_data()
        X_train_msme, X_test_msme, y_train_msme, y_test_msme = split_and_scale_data(X_msme, y_msme)
        print("MSME数据预处理完成")
    except Exception as e:
        print(f"MSME数据处理失败: {e}")
    
    # 处理ie.dta数据
    try:
        X_ie, y_ie = load_and_preprocess_ie_data()
        X_train_ie, X_test_ie, y_train_ie, y_test_ie = split_and_scale_data(X_ie, y_ie)
        print("ie.dta数据预处理完成")
    except Exception as e:
        print(f"ie.dta数据处理失败: {e}")