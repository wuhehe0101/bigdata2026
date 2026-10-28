import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取你刚刚生成的完美 CSV
df = pd.read_csv(r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\灾点地形统计结果.csv")

# 2. 构造标签（Label）
# 因为这里只有 84 个灾点，我们全部设为 1（发生了灾害）。
# 我们需要随机生成同等数量的 0（未发生灾害的背景点）。
# （这里为了演示，简单用随机点代替。其实你后面应该用代码从非灾区随机撒点去提取特征）
df['Label'] = 1

# 3. 选择特征 (X)
# 选这几个最具代表性的物理特征
features = ['沟比降', '主沟长', '高差', '高程Max', '高程_mean', '土地覆盖_编码']
X = df[features].copy()
y = df['Label']

# (注意：因为没有真正的负样本，这里只是为了演示流程。真实建模需要用背景点补充 Label=0 的数据)

print("目前的数据只包含正样本，需要补充负样本（Label=0）才能训练分类器。")
print("但你现在可以先用这些特征，看看不同类型泥石流的特征分布：")

# 4. 按类型分组看均值（写进论文做分析）
print("\n=== 各类型泥石流的平均特征 ===")
print(df.groupby('类型')[features].mean())

# 5. 画箱线图，看哪种类型的特征最极端
plt.figure(figsize=(12, 6))
df.boxplot(column='高程_mean', by='类型')
plt.title("不同类型泥石流的平均高程对比")
plt.ylabel("高程 (m)")
plt.suptitle("") # 去掉默认的副标题
plt.show()