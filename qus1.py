import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 正确读取数据 =================
# 读取时强制把第一列解析为时间索引
df_all = pd.read_csv("clead.csv", index_col=0, parse_dates=True)

# ================= 2. 动态生成物理衍生特征 =================
print("正在计算衍生特征...")
# 因为咱们合并时加了前缀，你需要根据真实列名提取（下面用 try 防止找不到列报错）
try:
    # 假设降雨在 MET 表里
    precip_col = [c for c in df_all.columns if 'Precip' in c][0]
    df_all['Precip_24h'] = df_all[precip_col].rolling('24h').sum()
except:
    print("未找到降雨列")

try:
    # 温度衍生
    ta_col = [c for c in df_all.columns if 'Ta_2m' in c][0]
    df_all['PDD'] = df_all[ta_col].clip(lower=0)
    df_all['Ta_anomaly'] = df_all[ta_col] - df_all[ta_col].rolling('7D').mean()
except:
    pass

try:
    # 净辐射 (RADM表)
    rsd = [c for c in df_all.columns if 'Rsd' in c][0]
    rsu = [c for c in df_all.columns if 'Rsu' in c][0]
    rld = [c for c in df_all.columns if 'Rld' in c][0]
    rlu = [c for c in df_all.columns if 'Rlu' in c][0]
    df_all['Rn'] = (df_all[rsd] - df_all[rsu]) + (df_all[rld] - df_all[rlu])
except:
    print("未找到完整的辐射列，跳过 Rn 计算")

# ================= 3. 设定特征池 =================
# 这里我们直接用真实存在于 df_all 里的衍生特征名
features = ['Precip_24h', 'PDD', 'Rn', 'Ta_anomaly']
# 尝试把 FLUX 的 VPD 和 SOIL 的 VWC 加进去（假设有前缀）
vpd_col = [c for c in df_all.columns if 'VPD' in c]
vwc_col = [c for c in df_all.columns if 'VWC_0.1m' in c]
if vpd_col: features.append(vpd_col[0])
if vwc_col: features.append(vwc_col[0])

# 剔除特征池里实际不存在于表中的（防报错）
features = [f for f in features if f in df_all.columns]
print(f"当前用于打分的特征池: {features}")

# ================= 4. 构建正负样本 =================
# 清理特征表中的空值
df_clean = df_all[features].dropna()

# 事件窗口：8.26 (正样本)
event_data = df_clean.loc['2026-08-26 00:00':'2026-08-26 12:00']
# 背景窗口：2025年8月 (负样本)
background_data = df_clean.loc['2025-08-01':'2025-08-31']

X = pd.concat([event_data, background_data])
y = [1] * len(event_data) + [0] * len(background_data)

print(f"正样本量: {len(event_data)}, 负样本量: {len(background_data)}")

# ================= 5. 多方法计算 =================
# (1) 随机森林
# 修改随机森林那一行，加入 class_weight='balanced'
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X, y)
rf_imp = rf.feature_importances_

# (2) 互信息
mi_imp = mutual_info_classif(X, y, random_state=42)

# (3) 皮尔逊相关系数
pearson_imp = [abs(pearsonr(X[f], y)[0]) for f in features]
# ================= 7. 打印并可视化三个模型的原始得分 =================
print("\n" + "="*50)
print("各单项模型的原始权重得分（未归一化前）")
print("="*50)

# 构建一个包含所有原始得分的 DataFrame
raw_scores_df = pd.DataFrame({
    'Feature': features,
    'RandomForest_Imp': rf_imp,
    'MutualInfo_Imp': mi_imp,
    'Pearson_Corr': pearson_imp
})



# ================= 8. 画分组柱状图对比 =================
# 设置画布宽度
plt.figure(figsize=(14, 7))

# 设置柱子的宽度和位置
x = np.arange(len(features))
width = 0.25

# 画出三组柱子
plt.bar(x - width, raw_scores_df['RandomForest_Imp'], width, label='随机森林(RF)', color='#1f77b4')
plt.bar(x, raw_scores_df['MutualInfo_Imp'], width, label='互信息(MI)', color='#ff7f0e')
plt.bar(x + width, raw_scores_df['Pearson_Corr'], width, label='皮尔逊(Pearson)', color='#2ca02c')

# 设置标签和标题
plt.xticks(x, raw_scores_df['Feature'], rotation=15, fontsize=12)
plt.ylabel('原始得分 (Raw Score)', fontsize=12)
plt.title('多模型特征评估得分对比', fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱子上显示具体的数值
for i in range(len(features)):
    plt.text(i - width, raw_scores_df['RandomForest_Imp'][i] + 0.01, f"{raw_scores_df['RandomForest_Imp'][i]:.2f}", ha='center', fontsize=10)
    plt.text(i, raw_scores_df['MutualInfo_Imp'][i] + 0.01, f"{raw_scores_df['MutualInfo_Imp'][i]:.2f}", ha='center', fontsize=10)
    plt.text(i + width, raw_scores_df['Pearson_Corr'][i] + 0.01, f"{raw_scores_df['Pearson_Corr'][i]:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
# ================= 6. 归一化与综合打分 =================
scaler = MinMaxScaler()
rf_norm = scaler.fit_transform(rf_imp.reshape(-1, 1)).flatten()
mi_norm = scaler.fit_transform(np.array(mi_imp).reshape(-1, 1)).flatten()
pearson_norm = scaler.fit_transform(np.array(pearson_imp).reshape(-1, 1)).flatten()

final_scores = (rf_norm + mi_norm + pearson_norm) / 3

# 排序
result = pd.DataFrame({'Feature': features, 'Score': final_scores}).sort_values(by='Score', ascending=False)

print("\n=== 多方法融合排序 Top ===")
print(result)

plt.figure(figsize=(10, 6))
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
plt.barh(result['Feature'], result['Score'], color=colors[:len(result)])
plt.gca().invert_yaxis()
plt.title("基于多方法融合的泥石流特征关联度排序 (Top 5)", fontsize=15)
plt.xlabel("综合得分", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# 在柱子旁边加上具体的分数
for i, v in enumerate(result['Score']):
    plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=11)

plt.tight_layout()
plt.show()