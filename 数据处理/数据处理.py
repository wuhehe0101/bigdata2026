import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =======================================================
# 1. 路径与数据读取
# =======================================================
ROOT = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
PTS_OFFICIAL = os.path.join(ROOT, r"灾害点位置\灾害点位置.shp")
PTS_84 = os.path.join(ROOT, r"（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp")
GLACIER_SHP = os.path.join(ROOT, r"（3）日喀则市吉隆县特大洪水——200km周围内的冰川及冰川表面高程变化数据集 (2000-2020)\西藏日喀则市吉隆县特大洪水—200km周围内的冰川和冰川厚度变化数据集\RGI_V7_JilongFlood_Buffer_200km.shp")
DEM_PATH = os.path.join(ROOT, r"（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据\DEM.tif")
WEATHER_CSV = r"D:\bigdadta\clead.csv"

print("正在读取数据...")
gdf_4pts = gpd.read_file(PTS_OFFICIAL)
gdf_84pts = gpd.read_file(PTS_84)
gdf_glacier = gpd.read_file(GLACIER_SHP)

# 读取 DEM 并计算简单的坡度/高程提取
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(float)
    dem_crs = src.crs
    dem_transform = src.transform
    if src.nodata is not None:
        dem[dem == src.nodata] = np.nan
    dem[~np.isfinite(dem)] = np.nan

# 计算坡度 (简易梯度法)
dz_dx = np.gradient(dem, axis=1) / 30.0
dz_dy = np.gradient(dem, axis=0) / 30.0
slope_matrix = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

# =======================================================
# 2. 空间易发性：官方点与历史点的地形提取
# =======================================================
def extract_spatial_features(gdf):
    """严格按照 DEM 的 CRS 提取地形特征"""
    gdf_proj = gdf.to_crs(dem_crs)
    elevations, slopes = [], []
    for pt in gdf_proj.geometry.centroid:
        row, col = rowcol(dem_transform, pt.x, pt.y)
        if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
            elevations.append(dem[row, col])
            slopes.append(slope_matrix[row, col])
        else:
            elevations.append(np.nan)
            slopes.append(np.nan)
    return np.column_stack((elevations, slopes))

# 提取官方 4 点特征
X_official = extract_spatial_features(gdf_4pts)
# 提取 84 历史点特征 (去掉空值)
X_hist = extract_spatial_features(gdf_84pts)
X_hist = X_hist[~np.isnan(X_hist).any(axis=1)]

print("\n=== 空间特征提取 ===")
print(f"4 个官方受灾点 (高程, 坡度):\n{X_official}")
print(f"历史灾害点平均高程: {np.mean(X_hist[:,0]):.1f} m, 平均坡度: {np.mean(X_hist[:,1]):.1f}°")

# =======================================================
# 3. 动态触发：计算累计降水、正积温与历史分位数
# =======================================================
print("\n正在处理气象数据 (严格遵守保留缺测原则)...")
df = pd.read_csv(WEATHER_CSV)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# 计算累计指标 (如果是 NaN，累计结果就是 NaN，绝不填 0)
df['P_3h'] = df['MET_Precip(mm)'].rolling('3h').sum()
df['P_24h'] = df['MET_Precip(mm)'].rolling('24h').sum()
df['P_72h'] = df['MET_Precip(mm)'].rolling('72h').sum()
df['T_2m'] = df['MET_Ta_2m(degree C)']
# 计算 7 天累积正积温 (PDD_7d, 单位 ℃·d)
df['PDD_7d'] = df['T_2m'].clip(lower=0).rolling('168h').sum() / 24.0

# 历史参考期 (2021-2025)
history = df.loc['2021-01-01':'2025-12-31']
quantiles = [0.90, 0.95, 0.99]
thres_dict = {}
for col in ['P_3h', 'P_24h', 'P_72h', 'PDD_7d']:
    # 忽略 NaN 计算分位数
    thres_dict[col] = {q: history[col].quantile(q) for q in quantiles}

print("\n=== 历史气象参考分位数 (2021-2025) ===")
for col in thres_dict:
    print(f"{col}: 90%={thres_dict[col][0.90]:.2f}, 95%={thres_dict[col][0.95]:.2f}, 99%={thres_dict[col][0.99]:.2f}")

# =======================================================
# 4. 2026-08-26 事件前气象指标对照
# =======================================================
print("\n=== 2026-08-26 事件前气象指标对照 ===")
# 提取灾害发生前 12 小时的数据，检查是否符合异常
event_window = df.loc['2026-08-26 00:00':'2026-08-26 12:00'].copy()

if not event_window.empty:
    # 取峰值
    peak_p3 = event_window['P_3h'].max()
    peak_p24 = event_window['P_24h'].max()
    peak_pdd = event_window['PDD_7d'].max()

    print(f"事件前 3h 最大累计降水: {peak_p3 if pd.notna(peak_p3) else '缺测/无数据'} mm")
    print(f"事件前 24h 最大累计降水: {peak_p24 if pd.notna(peak_p24) else '缺测/无数据'} mm")
    print(f"事件前 7d 最大正积温: {peak_pdd if pd.notna(peak_pdd) else '缺测/无数据'} ℃·d")

    if pd.notna(peak_p3) and peak_p3 > thres_dict['P_3h'][0.99]:
        print(" -> 结论: 短时降雨异常显著 (超 99% 分位)。")
    if pd.notna(peak_pdd) and peak_pdd > thres_dict['PDD_7d'][0.99]:
        print(" -> 结论: 正积温显著异常 (超 99% 分位)，支持冰雪强烈融水假设。")
else:
    print("事件时段数据为空。")

# =======================================================
# 5. 自动生成论文专用数据表与图表
# =======================================================
print("\n正在生成论文参考图表...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# 图1：历史正积温分布与事件对比
axes[0].hist(history['PDD_7d'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(thres_dict['PDD_7d'][0.95], color='orange', linestyle='--', label='95% 分位')
axes[0].axvline(thres_dict['PDD_7d'][0.99], color='red', linestyle='--', label='99% 分位')
if pd.notna(peak_pdd):
    axes[0].axvline(peak_pdd, color='black', linewidth=2, label='8.26事件前')
axes[0].set_title('历史 7日正积温分布 (PDD_7d)', fontsize=14)
axes[0].set_xlabel('累积正积温 (℃·d)')
axes[0].set_ylabel('频数')
axes[0].legend()

# 图2：历史 24h 累计降水分布与事件对比
axes[1].hist(history['P_24h'].dropna(), bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].axvline(thres_dict['P_24h'][0.95], color='orange', linestyle='--', label='95% 分位')
axes[1].axvline(thres_dict['P_24h'][0.99], color='red', linestyle='--', label='99% 分位')
if pd.notna(peak_p24):
    axes[1].axvline(peak_p24, color='black', linewidth=2, label='8.26事件前')
axes[1].set_title('历史 24小时累积降水分布', fontsize=14)
axes[1].set_xlabel('24h 降水 (mm)')
axes[1].set_ylabel('频数')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n✅ 第二问基础分析完成。请将输出数据填入论文表格。")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =======================================================
# 假设你刚才已经跑出了以下真实变量（如果是同一次运行，直接跑即可）
# 1. X_official: 官方点的高程、坡度 (4行2列)
# 2. X_hist: 历史点的高程、坡度 (若干行2列)
# 3. thres_dict: 历史降雨和积温的分位数字典
# 4. peak_p3, peak_p24, peak_pdd: 8.26 事件前的极值
# =======================================================

print("====== 第二问：触发关键参数临界条件与评估模型 ======\n")

# 1. 提取地形临界条件 (历史数据均值)
slope_threshold = np.mean(X_hist[:, 1])
elev_threshold = np.mean(X_hist[:, 0])

print(f"1. 地形临界条件: 历史泥石流多发区平均坡度 = {slope_threshold:.2f}°, 平均高程 = {elev_threshold:.2f}m")

# 2. 提取气象临界条件 (99% 分位数)
rain_threshold = thres_dict['P_3h'][0.99]
pdd_threshold = thres_dict['PDD_7d'][0.99]

print(f"2. 降雨触发临界值 (3h, 99%分位) = {rain_threshold:.2f} mm")
print(f"3. 热力触发临界值 (7d PDD, 99%分位) = {pdd_threshold:.2f} ℃·d")


# =======================================================
# 构建综合触发评估模型 (规则判别)
# =======================================================
def evaluate_risk(slope, elev, rain, pdd):
    # 空间易发性 S(x)
    s_score = 0
    if slope >= slope_threshold: s_score += 1
    if elev >= elev_threshold: s_score += 1

    # 动态触发 T(t)
    t_score = 0
    # 触发了极端降雨，或者触发了极端融水
    if (pd.notna(rain) and rain >= rain_threshold) or (pd.notna(pdd) and pdd >= pdd_threshold):
        t_score = 1

    # 综合风险判定
    if s_score == 2 and t_score == 1:
        return "极高风险"
    elif s_score >= 1 and t_score == 1:
        return "高风险"
    elif s_score == 2 and t_score == 0:
        return "中风险"
    else:
        return "低风险"


# 评估 8.26 当天事件的官方 4 个点
print("\n=== 8.26 事件官方受灾点评估结果 ===")
for i in range(len(X_official)):
    elev = X_official[i, 0]
    slope = X_official[i, 1]

    # 如果某个点提取不到（NaN），就打印缺测
    if np.isnan(elev) or np.isnan(slope):
        print(f"官方点 {i + 1}: 地形数据缺测（可能不在DEM覆盖范围内），无法评估。")
        continue

    level = evaluate_risk(slope, elev, peak_p3, peak_pdd)
    print(
        f"官方点 {i + 1}: 高程={elev:.1f}m, 坡度={slope:.1f}°, 触发降雨={peak_p3}mm, 触发PDD={peak_pdd:.1f} -> 风险判定: {level}")

# =======================================================
# 画图：触发条件阈值判定图 (放论文里极其装X)
# =======================================================
plt.figure(figsize=(8, 5), dpi=300)

# 将历史数据画成散点 (表征正常状态)
plt.scatter(X_hist[:, 1], X_hist[:, 0], c='lightgray', edgecolors='k', label='历史灾害区地形', alpha=0.6, s=30)

# 将官方 4 个点画成红星
valid_official = X_official[~np.isnan(X_official).any(axis=1)]
plt.scatter(valid_official[:, 1], valid_official[:, 0], c='red', marker='*', s=200, edgecolors='k',
            label='8.26 官方受灾点')

# 画出地形阈值线 (均值线)
plt.axvline(slope_threshold, color='blue', linestyle='--', label=f'坡度临界线 ({slope_threshold:.1f}°)')
plt.axhline(elev_threshold, color='green', linestyle='--', label=f'高程临界线 ({elev_threshold:.1f}m)')

plt.xlabel("地形坡度 (°)", fontsize=12)
plt.ylabel("高程 (m)", fontsize=12)
plt.title("地形临界条件与本次灾害发生位置的关系", fontsize=14)
plt.legend(loc='lower left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n✅ 关键参数临界值与评估模型完成！可以直接写入论文。")