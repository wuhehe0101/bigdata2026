import rasterio
import geopandas as gpd
import numpy as np
from rasterio.transform import rowcol
from rasterio.mask import mask

# 1. 读取土地覆盖数据
lucd_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（5）西藏日喀则市吉隆县泥石流灾害区土地覆盖数据（2020-2023）\jilong_LUCD_2020.tif" # 换成你的路径
with rasterio.open(lucd_path) as src:
    lucd = src.read(1)
    crs = src.crs

# 2. 读入灾点
shp_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"
gdf = gpd.read_file(shp_path).to_crs(crs)

# 3. 计算每个灾点周围 1km 缓冲区内，各类地物的占比
# （这里为了避免大范围计算，我们只取灾点中心点所在的像素值，或做小缓冲）
# 假设我们用中心点的地物类型作为主要特征
land_cover_types = []
for geom in gdf.geometry:
    centroid = geom.centroid
    row, col = rowcol(src.transform, centroid.x, centroid.y)
    if 0 <= row < lucd.shape[0] and 0 <= col < lucd.shape[1]:
        land_cover_types.append(lucd[row, col])
    else:
        land_cover_types.append(np.nan)

gdf['土地覆盖类型'] = land_cover_types

# 4. 统计一下，到底哪种类型的泥石流最喜欢待在哪种地物上？
print("=== 各类型泥石流所处的地物编号分布 ===")
print(gdf.groupby('类型')['土地覆盖类型'].value_counts())