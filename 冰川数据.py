import geopandas as gpd
import pandas as pd

# 1. 路径设置
# 84个历史灾点
shp_pts = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"

# 冰川边界数据 (选V6或V7都可以，V7是最新的)
shp_glacier = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（3）日喀则市吉隆县特大洪水——200km周围内的冰川及冰川表面高程变化数据集 (2000-2020)\西藏日喀则市吉隆县特大洪水—200km周围内的冰川和冰川厚度变化数据集\RGI_V7_JilongFlood_Buffer_200km.shp"

print("正在读取数据...")
gdf_pts = gpd.read_file(shp_pts)
gdf_glacier = gpd.read_file(shp_glacier)

# 2. 统一投影坐标系 (转换为 UTM 45N，单位变成米，方便计算距离)
gdf_pts_utm = gdf_pts.to_crs("EPSG:32645")
gdf_glacier_utm = gdf_glacier.to_crs("EPSG:32645")

# 3. 核心：计算灾点到最近冰川的距离
print("正在计算灾害点到冰川的物理距离...")

# 将灾点提取为中心点
pts_centroids = gdf_pts_utm.geometry.centroid
# 将冰川合并为一个大几何体（这样算距离最快）
glacier_unary = gdf_glacier_utm.geometry.unary_union

# 计算距离 (单位：米)
# distance 函数会返回点到多边形边界的最短距离
gdf_pts_utm['距冰川距离_m'] = pts_centroids.distance(glacier_unary)

# 将距离转为公里
gdf_pts_utm['距冰川距离_km'] = gdf_pts_utm['距冰川距离_m'] / 1000

# 4. 统计看看不同类型灾点离冰川有多近
print("\n=== 各类泥石流灾点距离冰川的平均距离 ===")
print(gdf_pts_utm.groupby('类型')['距冰川距离_km'].mean())

# 5. 打印几个例子看看
print("\n=== 抽取几个灾点查看距离 ===")
print(gdf_pts_utm[['类型', '距冰川距离_km']].head(10))