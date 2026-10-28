import rasterio
from rasterio.merge import merge
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.transform import array_bounds

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 拼接 TIF =================
folder_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（3）日喀则市吉隆县特大洪水——200km周围内的冰川及冰川表面高程变化数据集 (2000-2020)\西藏日喀则市吉隆县特大洪水—200km周围内的冰川和冰川厚度变化数据集\13_14_15_rgi60_2000-01-01_2005-01-01\dhdt_UTM"
file_pattern = os.path.join(folder_path, "*.tif")
tif_files = glob.glob(file_pattern)

print(f"找到了 {len(tif_files)} 个 TIF 文件，准备开始拼接...")
mosaic, out_transform = merge(tif_files)
mosaic_data = mosaic[0]
mosaic_data = np.where((mosaic_data < -100) | (mosaic_data > 100), np.nan, mosaic_data)

# ================= 2. 获取地理边界 =================
height, width = mosaic_data.shape
bounds = array_bounds(height, width, out_transform)
extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

# ================= 3. 画图 =================
fig, ax = plt.subplots(figsize=(12, 12))

# 画冰川底图
img = ax.imshow(mosaic_data, cmap='RdBu', vmin=-2.0, vmax=1.0, extent=extent, alpha=0.8)
plt.colorbar(img, label='冰川高程变化率 (米/年)', fraction=0.046, pad=0.04)

# 读取 84 个历史灾点 (这里是 .shp 文件)
shp_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"
gdf = gpd.read_file(shp_path)

# 坐标系转换（这里假设冰川 TIF 是 EPSG:32645，如果是 32644 请自己改一下）
gdf_proj = gdf.to_crs("EPSG:32645")

# 按类型画点
colors = {'降雨型': 'blue', '冰水混合型': 'purple', '冰雪融水型': 'cyan', '冰湖溃决型': 'yellow'}
for t, color in colors.items():
    subset = gdf_proj[gdf_proj['类型'] == t]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, markersize=20, label=t, edgecolor='black', linewidth=0.5)

# 如果你想画 8.26 的受灾范围，请把下面代码的路径换成真实的多边形 shp 文件！
# gdf_826 = gpd.read_file(r"D:\...\8.26受灾区域.shp")
# gdf_826_proj = gdf_826.to_crs("EPSG:32645")
# gdf_826_proj.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, label='2026-08-26 受灾范围')

plt.title("2015-2020年冰川高程变化与历史灾点空间耦合关系", fontsize=18)
plt.xlabel("X Coordinate (UTM)")
plt.ylabel("Y Coordinate (UTM)")
plt.legend(loc='lower right')
plt.show()