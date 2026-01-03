import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 1. 加载全国数据
all_data = pd.read_excel("Cov2_data.xlsx")

# 2. 数据预处理
# 将日期转换为连续的数字
all_data["日(日期)"] = all_data["日(日期)"] - 43852


# 3. 为每个省份创建滞后特征
def create_lag_features(data):
    """为每个省份创建滞后特征"""
    provinces = data["省份"].unique()
    all_province_data = []

    for province in provinces:
        province_data = data[data["省份"] == province].copy()

        # 按日期排序
        province_data = province_data.sort_values("日(日期)")

        # 创建滞后特征
        province_data["新增确诊_lag1"] = province_data["新增确诊"].shift(1)
        province_data["新增死亡_lag1"] = province_data["新增死亡"].shift(1)
        province_data["新增治愈_lag1"] = province_data["新增治愈"].shift(1)
        province_data["累计确诊_lag1"] = province_data["累计确诊"].shift(1)
        province_data["累计死亡_lag1"] = province_data["累计死亡"].shift(1)
        province_data["累计治愈_lag1"] = province_data["累计治愈"].shift(1)
        province_data["现有确诊_lag1"] = province_data["现有确诊"].shift(1)

        # 创建移动平均特征
        province_data["新增确诊_ma3"] = province_data["新增确诊"].rolling(window=3, min_periods=1).mean()
        province_data["新增确诊_ma7"] = province_data["新增确诊"].rolling(window=7, min_periods=1).mean()

        # 创建目标变量：明天是否有新增确诊（二分类问题）
        # 这里我们预测下一天是否有新增确诊
        province_data["Outcome"] = np.where(province_data["新增确诊"].shift(-1) > 0, 1, 0)

        # 删除缺失值（第一行和最后一行）
        province_data = province_data.dropna()

        all_province_data.append(province_data)

    return pd.concat(all_province_data, ignore_index=True)


# 创建特征数据集
features_data = create_lag_features(all_data)

print(f"全国数据总样本数: {len(features_data)}")
print(f"省份数量: {features_data['省份'].nunique()}")
print("\n数据前几行:")
print(features_data.head())
print("\n数据信息:")
print(features_data.info())

# 4. 特征选择
# 选择数值型特征
feature_columns = [
    "日(日期)",
    "新增确诊_lag1", "新增死亡_lag1", "新增治愈_lag1",
    "累计确诊_lag1", "累计死亡_lag1", "累计治愈_lag1", "现有确诊_lag1",
    "新增确诊_ma3", "新增确诊_ma7"
]

# 添加省份的one-hot编码
province_dummies = pd.get_dummies(features_data["省份"], prefix="省份")
features_data = pd.concat([features_data, province_dummies], axis=1)

# 将省份的one-hot列添加到特征列表中
province_features = [col for col in province_dummies.columns]
feature_columns.extend(province_features)

# 5. 准备特征和目标变量
X = features_data[feature_columns]
y = features_data["Outcome"]

print(f"\n特征维度: {X.shape}")
print(f"正样本数（明天有新增确诊）: {sum(y == 1)}")
print(f"负样本数（明天无新增确诊）: {sum(y == 0)}")

# 6. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

# 7. 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(
    X_scaled_df, y,
    train_size=0.7,
    random_state=42,  # 固定随机种子以确保可重复性
    stratify=y  # 保持正负样本比例
)

print(f"\n训练集大小: {len(train_x)}")
print(f"测试集大小: {len(test_x)}")

# 8. 训练逻辑回归模型
log_mod = LogisticRegression(
    max_iter=1000,  # 增加迭代次数
    class_weight='balanced'  # 处理类别不平衡
)
log_mod.fit(train_x, train_y)

# 9. 预测和评估
predictions = log_mod.predict(test_x)
y_pred_prob = log_mod.predict_proba(test_x)[:, 1]

print("\n混淆矩阵:")
print(confusion_matrix(test_y, predictions))
print(f"\n准确率: {accuracy_score(test_y, predictions):.4f}")

# 10. 计算特征重要性
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': log_mod.coef_[0]
}).sort_values('coefficient', ascending=False)

print("\n特征重要性（前10个）:")
print(feature_importance.head(10))

# 11. ROC曲线
fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"\nAUC-ROC值: {roc_auc:.4f}")

# 12. 绘制ROC曲线
plt.rcParams['font.sans-serif']='SimSun'
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率', fontsize=12)
plt.ylabel('真阳性率', fontsize=12)
plt.title('全国新冠疫情新增确诊预测 - ROC曲线', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 13. 绘制特征重要性
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['coefficient'].abs())
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('特征系数绝对值', fontsize=12)
plt.title('逻辑回归模型特征重要性（前15个）', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 14. 按省份分析预测效果
print("\n按省份统计预测效果:")
province_results = []
for province in all_data["省份"].unique():
    province_mask = features_data["省份"] == province
    if sum(province_mask) > 0:
        province_x = X_scaled_df[province_mask]
        province_y = y[province_mask]

        if len(province_x) > 0:
            province_pred = log_mod.predict(province_x)
            province_accuracy = accuracy_score(province_y, province_pred)
            province_results.append({
                '省份': province,
                '样本数': len(province_x),
                '准确率': province_accuracy
            })

province_results_df = pd.DataFrame(province_results).sort_values('准确率', ascending=False)
print(province_results_df.head(10))

# 15. 预测示例
print("\n预测示例（测试集前5个样本）:")
sample_indices = test_x.index[:5]
for idx in sample_indices:
    actual = test_y.loc[idx]
    predicted = predictions[test_x.index.get_loc(idx)]
    prob = y_pred_prob[test_x.index.get_loc(idx)]
    province = features_data.loc[idx, "省份"]
    print(f"省份: {province:3s} | 实际: {actual} | 预测: {predicted} | 概率: {prob:.3f}")