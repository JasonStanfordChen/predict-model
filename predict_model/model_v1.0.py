## 尝试使用逻辑回归算法预测新发患者

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## 导入数据预处理模块中的标准化函数
from sklearn.preprocessing import StandardScaler
## 导入模型选择模块里用于切分测试集和训练集的函数
from sklearn.model_selection import train_test_split
## 导入线性模型中线性逻辑回归函数
from sklearn.linear_model import LogisticRegression
## 导入模型评估指标的混淆矩阵和准确率的函数
from sklearn.metrics import confusion_matrix,accuracy_score
## 导入AUC-ROC曲线模块评价模型性能
from sklearn.metrics import roc_curve,auc


pd.set_option('display.max_columns', None)## 不限最多显示的列数
pd.set_option('display.width', None)## 不限输出的总宽度
# 数据检查
anhui_data = pd.read_excel("Cov2_data.xlsx")
anhui_data = anhui_data.iloc[0:53, 0:9]##截取安徽省数据进行初代模型训练
anhui_data["日(日期)"] = anhui_data["日(日期)"] - 43852
anhui_data.set_index("日(日期)", drop=True, inplace=True)
print(anhui_data.head())
print(anhui_data.info(verbose=True))
print(anhui_data.describe())

## 数据预处理
anhui_data_copy = anhui_data.copy()
anhui_data_copy = anhui_data_copy.drop(columns = "省份")
anhui_data_copy["Outcome"] = np.where(anhui_data_copy["新增确诊"] == 0, 0, 1) ##新增确诊归一化 (本行由Deepseek实现debug）

##选择变量相关性
corr_anhui_data = anhui_data_copy.corr()
print(corr_anhui_data["Outcome"].sort_values())

# 数据标准化:数据标注化
sc_x= StandardScaler()
source_x = pd.DataFrame(sc_x.fit_transform(anhui_data_copy.drop(["Outcome"], axis=1),),
                        columns=["新增确诊","新增死亡","新增治愈", "累计确诊", "累计死亡",
                                 "累计治愈","现有确诊"])
print(source_x.head())
source_y=anhui_data_copy.Outcome

#构建训练集和测试集
train_x,test_x,train_y,test_y= train_test_split(source_x,source_y,train_size=0.7,random_state=0)

#逻辑回归训练模型
log_mod = LogisticRegression()
log_mod.fit(train_x,train_y)
predictions = log_mod.predict(test_x)
print(predictions)

#模型评价（准确性评分）
print(confusion_matrix(test_y,predictions))
print(accuracy_score(test_y,predictions))

# 模型在测试集曲线下面积绘图
plt.rcParams['font.sans-serif']='SimSun'  ##设置中文字体
plt.rcParams['axes.unicode_minus']=False
y_pred_prob =log_mod.predict_proba(test_x)[:,1]
fpr,tpr, thresholds = roc_curve(test_y,y_pred_prob) # 计算真阳性率(TPR)和假阳性率(FPR)
print(fpr, tpr, thresholds)
roc_auc=auc(fpr,tpr)#计算AUC-ROC值
plt.figure()#绘制ROC曲线
plt.plot(fpr,tpr,color='darkred', label='Roc 曲线(area = %.2f)'% roc_auc)
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('全国新冠疫情新增确诊预测 - ROC曲线')
plt.legend(loc="lower right")
plt.show()