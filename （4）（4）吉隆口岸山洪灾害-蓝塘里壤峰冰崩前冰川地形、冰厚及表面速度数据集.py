import os
import xarray as xr
nc_path = r"D:\bigdadta\Glacier topographic and dynamic drivers (Glacier-collapse and debris-flow disasters at Langtang Lirung, Nepal).nc"
print(os.path.exists(nc_path))
ds = xr.open_dataset(nc_path)

# 加上 cache=False 规避xarray缓存报错
ds = xr.open_dataset(nc_path, cache=False)

# 打印数据集信息，确认读取成功
print(ds)

# 提取变量示例（后面你做分析绘图直接用）
dem = ds["dem"].values          # 高程
slope = ds["slope"].values      # 坡度
thickness = ds["thickness"].values # 冰厚
v = ds["v"].values              # 冰川合流速
vx = ds["vx"].values            # x方向流速分量
vy = ds["vy"].values            # y方向流速分量
glacier_mask = ds["glacier_mask"].values # 冰川掩膜
x = ds["x"].values
y = ds["y"].values

import xarray as xr
import rioxarray
import os



# 2. 读取数据集
ds = xr.open_dataset(nc_path)

# 3. 设置空间坐标系 (EPSG:32645，这是 UTM 45N)
# 从控制台打印的坐标 (x: 35万, y: 313万) 可以确认这是 UTM 投影坐标
ds = ds.rio.write_crs("EPSG:32645", inplace=True)

# 4. 定义我们要导出的三个核心变量和对应的中文文件名
variables_to_export = {
    'thickness': 'ice_thickness.tif',  # 冰厚
    'v': 'surface_velocity.tif',  # 表面速度
    'slope': 'glacier_slope.tif'  # 冰川坡度
}

# 5. 循环提取并保存
for var_name, file_name in variables_to_export.items():
    if var_name in ds:
        print(f"正在提取并保存 {var_name} -> {file_name} ...")

        # 提取变量
        data_array = ds[var_name]

        # 转换为 TIF 格式的 DataArray
        # 注意：rio.to_raster 要求维度是 (y, x)，你这数据已经是这个顺序了
        data_array.rio.to_raster(file_name)

        print(f"✅ {file_name} 保存成功！")
    else:
        print(f"⚠️ 未找到变量: {var_name}")

print("\n🎉 大功告成！现在你的文件夹里多了三张可以提取特征的 TIF 图。")

import xarray as xr
import numpy as np



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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 全局学术设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("正在绘制冰崩源区核心动力学参数可视化图...")

# 创建 1行3列 的画布，尺寸紧凑，适合放论文
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

# 提取坐标范围，用于设置 imshow 的 extent（不然坐标轴是像素）
# 这里的 x 和 y 是我们之前从 ds 里读出来的
extent = [x.min(), x.max(), y.min(), y.max()]

# ================= 子图 1：冰厚 (Thickness) =================
# 用 'YlOrRd' (黄-橙-红) 色系，越红说明冰越厚，势能越大
im1 = axes[0].imshow(thickness, cmap='YlOrRd', extent=extent, origin='upper')
axes[0].set_title("A: 冰厚 (Ice Thickness)", fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel("X坐标 (UTM)", fontsize=11)
axes[0].set_ylabel("Y坐标 (UTM)", fontsize=11)
cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.set_label("厚度 (m)", fontsize=11)
# 标记最大值点
max_thick = np.nanmax(thickness)
axes[0].text(0.05, 0.90, f"Max: {max_thick:.1f} m", transform=axes[0].transAxes,
             color='darkred', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# ================= 子图 2：表面流速 (Velocity) =================
# 用 'Blues' (浅蓝-深蓝) 色系，越深说明流动越快，越容易失稳
im2 = axes[1].imshow(v, cmap='Blues', extent=extent, origin='upper')
axes[1].set_title("B: 表面流速 (Surface Velocity)", fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel("X坐标 (UTM)", fontsize=11)
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.set_label("流速 (m/yr)", fontsize=11)
# 标记最大值点
max_v = np.nanmax(v)
axes[1].text(0.05, 0.90, f"Max: {max_v:.1f} m/yr", transform=axes[1].transAxes,
             color='navy', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# ================= 子图 3：源区坡度 (Slope) =================
# 用 'viridis' (绿-黄-紫) 或 'magma' 色系，展示陡峭程度
im3 = axes[2].imshow(slope, cmap='magma', extent=extent, origin='upper')
axes[2].set_title("C: 源区地形坡度 (Terrain Slope)", fontsize=14, fontweight='bold', pad=10)
axes[2].set_xlabel("X坐标 (UTM)", fontsize=11)
cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
cbar3.set_label("坡度 (度)", fontsize=11)
# 标记最大值点
max_slope = np.nanmax(slope)
axes[2].text(0.05, 0.90, f"Max: {max_slope:.1f}°", transform=axes[2].transAxes,
             color='gold', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

# 调整子图之间的间距
plt.tight_layout()
plt.show()

print("✅ 冰崩源区动力学参数可视化完成！这张图可以直接插入论文。")
