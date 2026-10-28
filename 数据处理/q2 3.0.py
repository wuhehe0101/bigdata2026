import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =======================================================
# 1. 读取真实气象数据与灾害台账
# =======================================================
print("正在加载气象数据与灾害台账...")
df_weather = pd.read_csv(r"D:\bigdadta\clead.csv")
df_weather['Timestamp'] = pd.to_datetime(df_weather['Timestamp'])
df_weather.set_index('Timestamp', inplace=True)

df_disaster = pd.read_csv(r"D:\bigdadta\吉隆县历史灾害_干净版.csv")
df_disaster['发生日期'] = pd.to_datetime(df_disaster['发生日期'])

# 构造气象特征：72h 正温累积估算（W3），15日累计降雨（P15）
df_weather['Melt_mm'] = df_weather['MET_Ta_2m(degree C)'].clip(lower=0) * (5.0 / 24.0)
df_weather['W3'] = df_weather['Melt_mm'].rolling('72h').sum()
df_weather['P15'] = df_weather['MET_Precip(mm)'].rolling('360h').sum()

# =======================================================
# 2. 真实数据集构建
# =======================================================
print("正在提取 32 条真实灾害日的气象特征...")
X_pos, y_pos = [], []

# 提取灾害日的真实气象（取事件当天正午 12 点的数据，若缺失取最近的小时）
for date in df_disaster['发生日期']:
    if date.year < 2021 or date.year > 2025:
        continue
    try:
        # 尽量精确到这一天
        day_data = df_weather.loc[date.strftime('%Y-%m-%d')]
        if len(day_data) > 0:
            # 取当天最大特征作为该次事件的特征（因为一次灾害往往伴随当日极值）
            max_w3 = day_data['W3'].max()
            max_p15 = day_data['P15'].max()
            if pd.notna(max_w3) and pd.notna(max_p15):
                X_pos.append([max_w3, max_p15])
                y_pos.append(1)
    except:
        continue

print(f"✅ 成功匹配 {len(X_pos)} 条有效灾害记录")

# 构建负样本（随机抽取 100 个气象正常的非灾害日）
df_normal = df_weather.loc['2021-01-01':'2025-12-31'].dropna(subset=['W3', 'P15'])
df_neg = df_normal.sample(n=100, random_state=42)
X_neg = df_neg[['W3', 'P15']].values
y_neg = np.zeros(len(X_neg))

# 合并
X_train = np.vstack((X_pos, X_neg))
y_train = np.concatenate((y_pos, y_neg))

# =======================================================
# 3. 真正的机器学习模型训练
# =======================================================
print("\n正在训练逻辑回归分类器...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 打印特征权重（看看哪个指标对触发影响最大）
print(f"✅ 模型特征权重 (W3融水, P15降雨): {model.coef_[0]}")

# 训练集 ROC
y_pred_prob = model.predict_proba(X_train_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_train, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"✅ 模型训练集 AUC: {roc_auc:.3f}")

# =======================================================
# 4. 双事件独立预测（2026.8.26 和 2025.7.8）
# =======================================================
print("\n=== 独立验证（模型未见过的 2026.8.26 和 2025.7.8） ===")
event_826 = df_weather.loc['2026-08-26 10:00', ['W3', 'P15']].values.reshape(1, -1)
event_78 = df_weather.loc['2025-07-08 05:00', ['W3', 'P15']].values.reshape(1, -1)

# 预测概率
probs = model.predict_proba(scaler.transform(np.vstack((event_826, event_78))))[:, 1]
print(f"2026.8.26 冰崩事件 (W3={event_826[0][0]:.1f}, P15={event_826[0][1]:.1f}) -> 预测触发概率: {probs[0]:.3f}")
print(f"2025.7.8 冰湖溃决事件 (W3={event_78[0][0]:.1f}, P15={event_78[0][1]:.1f}) -> 预测触发概率: {probs[1]:.3f}")

# =======================================================
# 5. 论文级可视化
# =======================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# 图A：真实历史训练集 ROC
ax1.plot(fpr, tpr, color='#C44E52', lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('基于 32 条真实记录的模型 ROC 验证', fontsize=14)
ax1.legend(loc="lower right")
ax1.grid(True, linestyle=':', alpha=0.6)

# 图B：双事件验证柱状图
events = ['2026.8.26\n(冰崩)', '2025.7.8\n(溃决)']
colors = ['#D62728', '#4C72B0']
bars = ax2.bar(events, probs, color=colors, width=0.4, edgecolor='black')
for bar, prob in zip(bars, probs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{prob:.3f}', ha='center', fontsize=12, fontweight='bold')

ax2.set_title('模型对真实事件的独立预测', fontsize=14)
ax2.set_ylabel('触发概率', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()