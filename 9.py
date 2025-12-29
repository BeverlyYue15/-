import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import openpyxl
import warnings
warnings.filterwarnings('ignore')

# 1. 读取数据 - 从Excel文件中读取Sheet1和Sheet2
file_path = r"附件1\回归预测.xlsx"
train_df = pd.read_excel(file_path, sheet_name=0, header=None)  # Sheet1作为训练集
test_df = pd.read_excel(file_path, sheet_name=1, header=None)   # Sheet2作为测试集

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print()

# 2. 数据预处理
# 前30列(索引0-29): 治疗前的30项指标分数（数值型）
# 第31列(索引30): 所服药物（分类型）
# 第32列(索引31): 治疗后的指标总分（数值型，目标变量）

# 为了保证独热编码在训练集和测试集上的一致性，先合并处理
train_df['dataset_type'] = 'train'
test_df['dataset_type'] = 'test'
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 分离特征和目标
feature_cols = list(range(30))  # 前30列数值特征
cat_col = 30                     # 第31列分类特征（药物）
target_col = 31                  # 第32列目标变量

# 对第31列（索引30）的药物进行独热编码
combined_encoded = pd.get_dummies(combined_df, columns=[cat_col], drop_first=True)

# 拆分回训练集和测试集
train_processed = combined_encoded[combined_encoded['dataset_type'] == 'train'].drop('dataset_type', axis=1)
test_processed = combined_encoded[combined_encoded['dataset_type'] == 'test'].drop('dataset_type', axis=1)

# 提取特征(X)和标签(y)
X_train = train_processed.drop(target_col, axis=1).values
y_train = train_processed[target_col].values
X_test = test_processed.drop(target_col, axis=1).values
y_test = test_processed[target_col].values

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"处理后的训练集特征维度: {X_train_scaled.shape}")
print(f"处理后的测试集特征维度: {X_test_scaled.shape}")
print()

# 3. 构建集成学习模型 - 使用多个基学习器
# 模型1: Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 模型2: Random Forest
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 模型3: AdaBoost
ada_model = AdaBoostRegressor(random_state=42)
ada_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'loss': ['linear', 'square', 'exponential']
}

print("=" * 60)
print("开始交叉验证调参...")
print("=" * 60)

# 对Gradient Boosting进行网格搜索
print("\n1. Gradient Boosting 调参中...")
gb_search = GridSearchCV(gb_model, gb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
gb_search.fit(X_train_scaled, y_train)
print(f"GB最优超参数: {gb_search.best_params_}")
print(f"GB最优CV得分 (MSE): {-gb_search.best_score_:.4f}")

# 对Random Forest进行网格搜索
print("\n2. Random Forest 调参中...")
rf_search = GridSearchCV(rf_model, rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_search.fit(X_train_scaled, y_train)
print(f"RF最优超参数: {rf_search.best_params_}")
print(f"RF最优CV得分 (MSE): {-rf_search.best_score_:.4f}")

# 对AdaBoost进行网格搜索
print("\n3. AdaBoost 调参中...")
ada_search = GridSearchCV(ada_model, ada_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
ada_search.fit(X_train_scaled, y_train)
print(f"Ada最优超参数: {ada_search.best_params_}")
print(f"Ada最优CV得分 (MSE): {-ada_search.best_score_:.4f}")

# 4. 集成多个最优模型 - 加权平均
best_gb = gb_search.best_estimator_
best_rf = rf_search.best_estimator_
best_ada = ada_search.best_estimator_

# 获取各模型在测试集上的预测
gb_pred = best_gb.predict(X_test_scaled)
rf_pred = best_rf.predict(X_test_scaled)
ada_pred = best_ada.predict(X_test_scaled)

# 使用加权平均组合多个模型的预测
y_pred = (gb_pred + rf_pred + ada_pred) / 3.0

# 5. 计算平方相对误差 (Squared Relative Error)
# 对于每个样本: ((预测值 - 真实值) / 真实值) ^ 2
relative_errors = (y_pred - y_test) / y_test
squared_relative_errors = relative_errors ** 2

# 6. 报告误差统计
mean_sre = np.mean(squared_relative_errors)
var_sre = np.var(squared_relative_errors)
std_sre = np.std(squared_relative_errors)

print("\n" + "=" * 60)
print("测试集结果 (平方相对误差统计):")
print("=" * 60)
print(f"均值 (Mean): {mean_sre:.6f}")
print(f"方差 (Variance): {var_sre:.6f}")
print(f"标准差 (Std): {std_sre:.6f}")
print(f"最小值: {np.min(squared_relative_errors):.6f}")
print(f"最大值: {np.max(squared_relative_errors):.6f}")
print(f"中位数: {np.median(squared_relative_errors):.6f}")
print("=" * 60)

# 7. 详细结果展示
print("\n测试样本预测结果详情 (前10个样本):")
print(f"{'样本号':<6} {'真实值':<12} {'预测值':<12} {'平方相对误差':<12}")
print("-" * 42)
for i in range(min(10, len(y_test))):
    print(f"{i+1:<6} {y_test[i]:<12.4f} {y_pred[i]:<12.4f} {squared_relative_errors[i]:<12.6f}")

if len(y_test) > 10:
    print(f"... (共 {len(y_test)} 个样本)")

# 8. 汇总报告
print("\n" + "=" * 60)
print("最终报告总结:")
print("=" * 60)
print(f"集成学习模型: Gradient Boosting + Random Forest + AdaBoost")
print(f"训练样本数: {len(X_train_scaled)}")
print(f"测试样本数: {len(X_test_scaled)}")
print(f"输入特征维度: {X_train_scaled.shape[1]}")
print(f"\n平方相对误差指标:")
print(f"  均值: {mean_sre:.6f}")
print(f"  方差: {var_sre:.6f}")
print("=" * 60)