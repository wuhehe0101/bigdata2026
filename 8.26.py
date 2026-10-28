import rasterio
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dem_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据\DEM.tif"

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(np.float32)
    # ！！！关键第一步：把数据里的 -9999 或者低于 0 的无效值，全部变成 NaN
    dem[dem < 0] = np.nan

    # 获取分辨率（重要：如果是经纬度，要把度换算成米，大概 1度 = 111000米）
    # 这里的 transform 默认是度，为了算坡度，需要乘上 111000
    res_x = src.transform[0] * 111000
    res_y = -src.transform[4] * 111000

    # 计算梯度
    dz_dx = np.gradient(dem, axis=1) / res_x
    dz_dy = np.gradient(dem, axis=0) / res_y

    # 计算坡度
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

# ！！！关键第二步：截断异常值
# 把大于 60 度的（可能是边界效应）压平到 60 度，把 NaN 保持为 NaN
slope_deg = np.clip(slope_deg, 0, 60)

# 画图
plt.figure(figsize=(12, 8))
# 用 vmin=0, vmax=60 强制颜色条范围
img = plt.imshow(slope_deg, cmap='YlOrRd', vmin=0, vmax=60)
plt.colorbar(img, label='Slope (degrees)')
plt.title("吉隆县核心区 DEM 坡度图 (修复版)", fontsize=16)
plt.show()

# 打印真实统计信息看看
print("坡度最小值:", np.nanmin(slope_deg))
print("坡度最大值:", np.nanmax(slope_deg))
print("坡度平均值:", np.nanmean(slope_deg))