import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LightSource

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dem_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据\DEM.tif"
shp_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(np.float32)
    dem[dem < 0] = np.nan
    res_x = src.transform[0] * 111000
    res_y = -src.transform[4] * 111000
    bounds = src.bounds
    crs = src.crs

# 计算坡度
dz_dx = np.gradient(dem, axis=1) / res_x
dz_dy = np.gradient(dem, axis=0) / res_y
slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
slope_deg = np.clip(slope_deg, 0, 60)

# 计算地形阴影（让山体有立体感）
ls = LightSource(azdeg=315, altdeg=45)
# 用填过Nan的dem计算阴影
dem_filled = np.nan_to_num(dem, nan=np.nanmin(dem))
hillshade = ls.hillshade(dem_filled, vert_exag=2, dx=res_x, dy=res_y)

gdf = gpd.read_file(shp_path)
gdf_proj = gdf.to_crs(crs)

fig, ax = plt.subplots(figsize=(14, 10))

# 1. 底层画阴影（灰度）
ax.imshow(hillshade, cmap='gray', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], alpha=0.6)

# 2. 上层画坡度图（设置 alpha=0.6，半透明透出底层的阴影）
img = ax.imshow(slope_deg, cmap='YlOrRd', vmin=0, vmax=60, alpha=0.6,
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

# 3. 顶层画受灾多边形（蓝色边框，加粗一点）
gdf_proj.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1.8)

plt.colorbar(img, label='坡度 (Degrees)', fraction=0.046, pad=0.04)
plt.title("吉隆县泥石流受灾区域与地形坡度叠加图", fontsize=18)
plt.xlabel("经度", fontsize=14)
plt.ylabel("纬度", fontsize=14)

# 自动缩放视图，只聚焦在蓝线覆盖的 DEM 区域
ax.set_xlim(bounds.left, bounds.right)
ax.set_ylim(bounds.bottom, bounds.top)

plt.show()

print(gdf.columns.tolist())
print(gdf['类型'].value_counts())

# 筛选出4个独立的 GeoDataFrame
gdf_rain = gdf[gdf['类型'] == '降雨型']
gdf_mix  = gdf[gdf['类型'] == '冰水混合型']
gdf_ice  = gdf[gdf['类型'] == '冰雪融水型']
gdf_lake = gdf[gdf['类型'] == '冰湖溃决型']

# print("---- 解译时间前 10 行 ----")
# print(gdf['解译Time'].head(10))
#
# print("\n---- 影像时间前 10 行 ----")
# print(gdf['影像Time'].head(10))

# 看看影像时间的分布范围
print("影像时间最早:", gdf['影像Time'].min())
print("影像时间最晚:", gdf['影像Time'].max())

# 看看不同年份有多少个灾害点
print(gdf['影像Time'].dt.year.value_counts().sort_index())

import matplotlib.pyplot as plt

# 1. 画个柱状图，看看年份分布
gdf['year'] = gdf['影像Time'].dt.year
gdf['year'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("吉隆县历史泥石流灾点年份分布 (2010-2017)")
plt.xlabel("年份")
plt.ylabel("灾害点数量")
plt.show()

# 2. 看看自带特征列（沟比降、高差、主沟长）的统计信息
print(gdf[['沟比降', '高差', '主沟长', '沟口H', '高程Max']].describe())