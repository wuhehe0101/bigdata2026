import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# =======================================================
# 全局设置与中文字体 + 学术配色（Nature/ESCI期刊常用配色）
# =======================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# 学术配色（tab10，水文地质期刊常用）
COLORS = {
    '降雨型': '#1f77b4',
    '冰水混合型': '#2ca02c',
    '冰雪融水型': '#ff7f0e',
    '冰湖溃决型': '#9467bd',
    '触发信号': '#d62728',
    '基线气象': '#7f7f7f'
}

# =======================================================
# 1. 读取空间特征数据 (84个灾点)
# =======================================================
print("正在读取空间特征数据...")
df_spatial = pd.read_csv(r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\灾点地形统计结果.csv")

df_spatial['土地覆盖_5_比例'] = df_spatial.get('土地覆盖_5_比例', 0).fillna(0)
df_spatial['土地覆盖_2_比例'] = df_spatial.get('土地覆盖_2_比例', 0).fillna(0)

df_rain = df_spatial[df_spatial['类型'] == '降雨型']
df_ice = df_spatial[df_spatial['类型'] == '冰雪融水型']

print("\n=== 空间特征统计（用于论文） ===")
print(f"降雨型：平均沟比降 {df_rain['沟比降'].mean():.2f}，平均高差 {df_rain['高差'].mean():.1f}m")
print(f"冰雪型：平均沟比降 {df_ice['沟比降'].mean():.2f}，裸地占比 {df_ice['土地覆盖_5_比例'].mean()*100:.1f}%")

# =======================================================
# 2. 读取时间序列数据与特征构建
# =======================================================
print("\n正在读取气象大表并构建特征...")
df = pd.read_csv(r"D:\bigdadta\clead.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

df['PDD_7d'] = df['MET_Ta_2m(degree C)'].clip(lower=0).rolling('168h').sum()
df['Precip_24h'] = df['MET_Precip(mm)'].rolling('24h').sum()
df['Precip_3h'] = df['MET_Precip(mm)'].rolling('3h').sum()

# =======================================================
# 3. 建立气候基线 (2021-2023)
# =======================================================
print("正在建立历史气候基线 (2021-2023)...")
history = df.loc['2021-01-01':'2023-12-31']
stats = {
    'PDD_7d': {'mean': history['PDD_7d'].mean(), 'std': history['PDD_7d'].std()},
    'Precip_24h': {'mean': history['Precip_24h'].mean(), 'std': history['Precip_24h'].std()},
    'Precip_3h': {'mean': history['Precip_3h'].mean(), 'std': history['Precip_3h'].std()}
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_z(d, col):
    return ((d[col] - stats[col]['mean']) / stats[col]['std']).fillna(0)

def calc_base_prob(d):
    z_pdd = calc_z(d, 'PDD_7d').clip(-3, 5)
    z_p24 = calc_z(d, 'Precip_24h').clip(-3, 5)
    z_p3 = calc_z(d, 'Precip_3h').clip(-3, 5)
    raw_score = 0.4 * z_pdd + 0.4 * z_p24 + 0.2 * z_p3
    return sigmoid(raw_score)

# =======================================================
# 4. 事件回测与引信注入
# =======================================================
print("正在回测历史灾害事件...")
df_0928 = df.loc['2024-09-20':'2024-09-30'].copy()
df_0928['Risk'] = calc_base_prob(df_0928)

df_0708 = df.loc['2025-07-01':'2025-07-10'].copy()
z_pdd_0708 = calc_z(df_0708, 'PDD_7d')
z_p24_0708 = calc_z(df_0708, 'Precip_24h')
df_0708['Lake_Trigger'] = 0.0
df_0708.loc['2025-07-08 03:00':'2025-07-08 08:00', 'Lake_Trigger'] = 4.0
df_0708['Risk'] = sigmoid(0.3 * z_pdd_0708 + 0.3 * z_p24_0708 + 1.5 * df_0708['Lake_Trigger'])

df_0826 = df.loc['2026-08-20':'2026-08-28'].copy()
z_pdd_0826 = calc_z(df_0826, 'PDD_7d')
z_p24_0826 = calc_z(df_0826, 'Precip_24h')
df_0826['Base_Prob'] = calc_base_prob(df_0826)
df_0826['Ice_Trigger'] = 0.0
df_0826.loc['2026-08-26 06:00':'2026-08-26 12:00', 'Ice_Trigger'] = 4.0
df_0826['Total_Prob'] = sigmoid(0.4 * z_pdd_0826 + 0.2 * z_p24_0826 + 1.5 * df_0826['Ice_Trigger'])

# =======================================================
# 5. 极值扫描
# =======================================================
print("\n正在扫描 2024-2026 年气象极值日...")
df_scan = df.loc['2024-01-01':'2026-08-27'].copy()
for f in stats:
    df_scan[f'Z_{f}'] = (df_scan[f] - stats[f]['mean']) / stats[f]['std']
    df_scan[f'Z_{f}'] = df_scan[f'Z_{f}'].fillna(0)

raw_score = (
    0.4 * df_scan['Z_PDD_7d'].clip(-3, 5) +
    0.4 * df_scan['Z_Precip_24h'].clip(-3, 5) +
    0.2 * df_scan['Z_Precip_3h'].clip(-3, 5)
)
df_scan['Base_Prob'] = sigmoid(raw_score)

daily_data = df_scan.resample('D').agg({
    'Base_Prob': 'max',
    'MET_Precip(mm)': 'sum',
    'MET_Ta_2m(degree C)': 'max',
    'Z_Precip_24h': 'max',
    'Z_PDD_7d': 'max'
}).dropna()

top_10_days = daily_data.sort_values(by='Base_Prob', ascending=False).head(10)
print("\n=== 过去三年气象极值 Top 10（无物理引信情况下的最高风险） ===")
print(top_10_days[['Base_Prob', 'MET_Precip(mm)', 'MET_Ta_2m(degree C)']].rename(columns={
    'Base_Prob': '触发概率',
    'MET_Precip(mm)': '日降雨量',
    'MET_Ta_2m(degree C)': '最高温'
}))

# =======================================================
# 绘图：只保存，不弹窗
# =======================================================
# 图1：两次典型灾害事件风险时序对比
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5))
plt.subplots_adjust(hspace=0.4)

ax1.plot(df_0708.index, df_0708['Risk'], color=COLORS['冰湖溃决型'], linewidth=1.5, label='综合触发概率')
ax1.axvline(pd.Timestamp('2025-07-08 05:00'), color='#333333', linestyle='--', linewidth=1.2, label='冰湖溃决时刻')
ax1.fill_between(df_0708.index, 0, df_0708['Risk'], color=COLORS['冰湖溃决型'], alpha=0.15)
ax1.set_title('冰湖溃决型泥石流风险时序变化', fontsize=11, fontweight='bold', loc='left')
ax1.set_ylabel('触发概率', fontsize=9)
ax1.set_ylim(-0.05, 1.05)
ax1.tick_params(labelsize=8)
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(df_0826.index, df_0826['Base_Prob'], color=COLORS['基线气象'], linestyle='--', linewidth=1.2, label='纯气象驱动风险')
ax2.plot(df_0826.index, df_0826['Total_Prob'], color=COLORS['触发信号'], linewidth=1.5, label='耦合冰崩的综合风险')
ax2.axvline(pd.Timestamp('2026-08-26 10:30'), color='#333333', linestyle='--', linewidth=1.2, label='高位冰崩时刻')
ax2.fill_between(df_0826.index, df_0826['Base_Prob'], df_0826['Total_Prob'], color=COLORS['触发信号'], alpha=0.15)
ax2.set_title('高位冰崩型泥石流风险时序变化', fontsize=11, fontweight='bold', loc='left')
ax2.set_ylabel('触发概率', fontsize=9)
ax2.set_ylim(-0.05, 1.05)
ax2.tick_params(labelsize=8)
ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("图1_典型灾害事件风险时序对比.png")
plt.close(fig1)
print("✅ 已保存：图1_典型灾害事件风险时序对比.png")

# 图2：冰崩灾害多驱动因子解耦时序
fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
fig2.suptitle('冰崩灾害多驱动因子时序解耦', fontsize=12, fontweight='bold', y=0.98)
plt.subplots_adjust(hspace=0.2)

ax3.bar(df_0826.index, df_0826['MET_Precip(mm)'], color='#4477AA', width=0.04, alpha=0.8)
ax3.set_ylabel('小时降雨量(mm)', fontsize=9)
ax3.tick_params(labelsize=8)
ax3.grid(axis='y', linestyle=':', alpha=0.6)

ax4.plot(df_0826.index, df_0826['PDD_7d'], color='#EE9944', linewidth=1.5)
ax4.fill_between(df_0826.index, df_0826['PDD_7d'].min(), df_0826['PDD_7d'], color='#EE9944', alpha=0.2)
ax4.set_ylabel('7日积温(℃)', fontsize=9)
ax4.tick_params(labelsize=8)
ax4.grid(True, linestyle=':', alpha=0.6)

ax5.plot(df_0826.index, df_0826['Total_Prob'], color=COLORS['触发信号'], linewidth=1.5)
ax5.axvline(pd.Timestamp('2026-08-26 10:30'), color='#333333', linestyle='--', linewidth=1.2)
ax5.set_ylabel('触发概率', fontsize=9)
ax5.set_xlabel('北京时间', fontsize=10)
ax5.set_ylim(-0.05, 1.05)
ax5.tick_params(labelsize=8)
ax5.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("图2_冰崩灾害驱动因子时序解耦.png")
plt.close(fig2)
print("✅ 已保存：图2_冰崩灾害驱动因子时序解耦.png")

# 图3：历史灾点孕灾环境特征分布
fig3 = plt.figure(figsize=(7, 5))
ax_scatter = fig3.add_subplot(111)
for t in ['降雨型', '冰水混合型', '冰雪融水型', '冰湖溃决型']:
    subset = df_spatial[df_spatial['类型'] == t]
    if len(subset) > 0:
        ax_scatter.scatter(
            subset['土地覆盖_5_比例'] * 100,
            subset['沟比降'],
            s=subset['高差']/10,
            color=COLORS[t],
            edgecolor='black',
            linewidth=0.6,
            alpha=0.8,
            label=t
        )
ax_scatter.set_title('历史泥石流灾害孕灾环境特征分布', fontsize=11, fontweight='bold')
ax_scatter.set_xlabel('裸地占比(%)，物源条件', fontsize=9)
ax_scatter.set_ylabel('沟谷比降，势能条件', fontsize=9)
ax_scatter.legend(title='灾害成因类型', loc='upper left', fontsize=8, title_fontsize=9)
ax_scatter.tick_params(labelsize=8)
ax_scatter.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("图3_历史灾点孕灾环境特征分布.png")
plt.close(fig3)
print("✅ 已保存：图3_历史灾点孕灾环境特征分布.png")

# 图4：降雨-积温耦合风险散点分布
fig4 = plt.figure(figsize=(7, 5))
ax4_scatter = fig4.add_subplot(111)
cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=daily_data['Base_Prob'].min(), vmax=daily_data['Base_Prob'].max())
sc = ax4_scatter.scatter(
    daily_data['Z_Precip_24h'],
    daily_data['Z_PDD_7d'],
    c=daily_data['Base_Prob'],
    cmap=cmap,
    norm=norm,
    s=30,
    alpha=0.8,
    edgecolor='black',
    linewidth=0.4
)
ax4_scatter.axhline(0, color='#666666', linestyle='--', linewidth=1.0)
ax4_scatter.axvline(0, color='#666666', linestyle='--', linewidth=1.0)

def annotate_event(date_str, text, xytext):
    if pd.Timestamp(date_str) in daily_data.index:
        x = daily_data.loc[date_str]['Z_Precip_24h']
        y = daily_data.loc[date_str]['Z_PDD_7d']
        ax4_scatter.annotate(
            text, xy=(x, y), xytext=xytext,
            arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3,rad=.2"),
            fontsize=9, fontweight='bold', color='#333333',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )
annotate_event('2024-09-28', '2024-09-28', xytext=(2, -1.5))
annotate_event('2026-08-26', '2026-08-26', xytext=(-3, 1.5))

cbar = plt.colorbar(sc)
cbar.set_label('综合气象风险概率', fontsize=9)
cbar.ax.tick_params(labelsize=8)
ax4_scatter.set_title('降雨异常与积温异常耦合风险分布', fontsize=11, fontweight='bold')
ax4_scatter.set_xlabel('降雨异常度 Z-score', fontsize=9)
ax4_scatter.set_ylabel('积温异常度 Z-score', fontsize=9)
ax4_scatter.tick_params(labelsize=8)
ax4_scatter.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("图4_降雨积温耦合风险散点分布.png")
plt.close(fig4)
print("✅ 已保存：图4_降雨积温耦合风险散点分布.png")

print("\n✅ 全部绘图任务结束，图片保存在脚本同目录下！")
