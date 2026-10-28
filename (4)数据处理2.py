import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import rowcol

# 1. 文件路径设置
shp_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"
tif_paths = {
    '平均冰厚': r"D:\bigdadta\ice_thickness.tif",
    '表面流速': r"D:\bigdadta\surface_velocity.tif",
    '源区坡度': r"D:\bigdadta\glacier_slope.tif"
}

# 读取灾点
gdf = gpd.read_file(shp_path)
import xarray as xr
import numpy as np

# 1. 填入你的 .nc 路

print("正在读取蓝塘里壤峰冰崩源区数据...")
ds = xr.open_dataset(nc_path)

# 2. 提取你需要的变量 (去掉空值)
# 我们分别看冰厚、流速、坡度的最大值、平均值和90分位数
variables = ['thickness', 'v', 'slope']

print("\n=== 2026.8.26 冰崩源区核心动力学参数 ===")
for var in variables:
    if var in ds:
        # 把无效值（比如-9999或NaN）去掉
        data = ds[var].values.flatten()
        data = data[~np.isnan(data)]

        # 有些 TIF 会有 -9999 的 NoData 值
        if len(data) > 0:
            data = data[data > -1000]

        if len(data) > 0:
            print(f"\n【{var}】特征:")
            print(f"  最大值: {np.max(data):.2f}")
            print(f"  平均值: {np.mean(data):.2f}")
            print(f"  90分位数: {np.percentile(data, 90):.2f}")

# 3. 把这三个数字保存下来，直接写进论文
print("\n✅ 提取完毕！这些数字直接证明了冰崩的物理势能和触发条件。")