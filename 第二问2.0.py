import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import rowcol
from rasterio.merge import merge
import glob
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =======================================================
# 1. 路径与数据读取
# =======================================================
ROOT = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
pts_path = os.path.join(ROOT, r"灾害点位置\灾害点位置.shp")
glacier_path = os.path.join(ROOT,
                            r"（3）日喀则市吉隆县特大洪水——200km周围内的冰川及冰川表面高程变化数据集 (2000-2020)\西藏日喀则市吉隆县特大洪水—200km周围内的冰川和冰川厚度变化数据集\RGI_V7_JilongFlood_Buffer_200km.shp")
tif_folder = os.path.join(ROOT,
                          r"（3）日喀则市吉隆县特大洪水——200km周围内的冰川及冰川表面高程变化数据集 (2000-2020)\西藏日喀则市吉隆县特大洪水—200km周围内的冰川和冰川厚度变化数据集\13_14_15_rgi60_2015-01-01_2020-01-01\dhdt_UTM")
dem_path = os.path.join(ROOT, r"（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据\DEM.tif")

print("正在读取数据与栅格...")
gdf_4pts = gpd.read_file(pts_path).to_crs("EPSG:32645")
gdf_glacier = gpd.read_file(glacier_path).to_crs("EPSG:32645")

# 💡 核心修复：用 explode 把多边形拆散，彻底绕开 unary_union 的拓扑报错
gdf_glacier_exploded = gdf_glacier.explode(index_parts=False).reset_index(drop=True)

tif_files = glob.glob(os.path.join(tif_folder, "*.tif"))
mosaic, out_transform = merge(tif_files)
dhdt_data = mosaic[0]
dhdt_data[dhdt_data < -100] = np.nan

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(float)
    dem[dem < -100] = np.nan
    dem_crs = src.crs
    dem_transform = src.transform

gdf_glacier_exploded = gdf_glacier_exploded.to_crs(dem_crs)


# =======================================================
# 2. 真实的特征提取函数
# =======================================================
def extract_features(pts_gseries):
    distances = []
    for pt in pts_gseries:
        # 算点到所有独立冰川多边形的最短距离
        min_dist = gdf_glacier_exploded.geometry.distance(pt).min() / 1000.0
        distances.append(min_dist)

    thinning_rates = []
    elevations = []

    for pt in pts_gseries:
        # 变薄速率
        row, col = rowcol(out_transform, pt.x, pt.y)
        if 0 <= row < dhdt_data.shape[0] and 0 <= col < dhdt_data.shape[1]:
            val = dhdt_data[row, col]
            thinning_rates.append(val if not np.isnan(val) else 0.0)
        else:
            thinning_rates.append(0.0)

        # 高程
        pt_dem = gpd.GeoSeries([pt], crs="EPSG:32645").to_crs(dem_crs).iloc[0]
        row_d, col_d = rowcol(dem_transform, pt_dem.x, pt_dem.y)
        if 0 <= row_d < dem.shape[0] and 0 <= col_d < dem.shape[1]:
            elev = dem[row_d, col_d]
            elevations.append(elev if not np.isnan(elev) else 0.0)
        else:
            elevations.append(0.0)

    return np.column_stack((distances, thinning_rates, elevations))


print("正在提取真实特征...")
pos_pts = gdf_4pts.geometry.centroid
X_pos = extract_features(pos_pts)

# 负样本：随机生成
x_min, y_min, x_max, y_max = gdf_4pts.total_bounds
neg_pts = []
while len(neg_pts) < 40:
    pt = Point(np.random.uniform(x_min - 10000, x_max + 10000), np.random.uniform(y_min - 10000, y_max + 10000))
    # 限制负样本不要太靠近正样本
    if pt.distance(gdf_4pts.geometry.unary_union) > 3000:
        neg_pts.append(pt)
neg_series = gpd.GeoSeries(neg_pts, crs="EPSG:32645")
X_neg = extract_features(neg_series)

X = np.vstack((X_pos, X_neg))
y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))

# =======================================================
# 3. 训练模型与可视化
# =======================================================
print("正在训练多维空间物理模型...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(class_weight='balanced', random_state=42)
clf.fit(X_scaled, y)

y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\n✅ 真实三维特征模型验证 AUC: {roc_auc:.3f}")

plt.figure(figsize=(6, 5), dpi=300)
plt.plot(fpr, tpr, color='#4C72B0', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate (误报率)', fontsize=12)
plt.ylabel('True Positive Rate (命中率)', fontsize=12)
plt.title('多源空间物理模型 (静态环境约束) ROC 验证', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n=== 真实三维特征权重分析 ===")
print(f"距冰川距离权重: {clf.coef_[0][0]:.4f}")
print(f"冰川变薄速率权重: {clf.coef_[0][1]:.4f}")
print(f"地形高程权重: {clf.coef_[0][2]:.4f}")