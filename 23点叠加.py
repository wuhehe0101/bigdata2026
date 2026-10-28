
import os
import glob
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================= 1. 中文显示与路径 =================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(
    r"D:\mju\2026-1\学习\竞赛\大数据挑战"
    r"\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
)

folder_path = (
    ROOT
    / "（3）日喀则市吉隆县特大洪水——200km周围内的冰川及冰川表面高程变化数据集 (2000-2020)"
    / "西藏日喀则市吉隆县特大洪水—200km周围内的冰川和冰川厚度变化数据集"
    / "13_14_15_rgi60_2015-01-01_2020-01-01"
    / "dhdt_UTM"
)

shp_path = (
    ROOT
    / "（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据"
    / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据"
    / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"
)

if not folder_path.exists():
    raise FileNotFoundError(f"冰川栅格目录不存在：{folder_path}")

if not shp_path.exists():
    raise FileNotFoundError(f"历史灾害文件不存在：{shp_path}")

# ================= 2. 读取并拼接冰川栅格 =================
tif_files = sorted(glob.glob(str(folder_path / "*.tif")))

if not tif_files:
    raise FileNotFoundError(f"目录中没有 TIF 文件：{folder_path}")

print(f"找到 {len(tif_files)} 个冰川 TIF 文件")

# 只拼接具有相同坐标系、波段类型等兼容条件的栅格
with rasterio.open(tif_files[0]) as src0:
    raster_crs = src0.crs
    raster_nodata = src0.nodata

if raster_crs is None:
    raise ValueError("冰川栅格缺少坐标系，无法安全叠加。")

compatible_files = []

for f in tif_files:
    with rasterio.open(f) as src:
        if src.crs != raster_crs:
            print("跳过坐标系不一致的文件：", Path(f).name)
            continue
        compatible_files.append(f)

if not compatible_files:
    raise ValueError("没有可拼接的同坐标系栅格。")

mosaic, out_transform = merge(
    compatible_files,
    nodata=raster_nodata
)

mosaic_data = mosaic[0].astype("float32")

# 屏蔽 NoData 和无穷值
if raster_nodata is not None:
    mosaic_data[mosaic_data == raster_nodata] = np.nan

mosaic_data[~np.isfinite(mosaic_data)] = np.nan

# 去除明显异常值；仅当数据单位确为米/年时适用
mosaic_data[(mosaic_data < -100) | (mosaic_data > 100)] = np.nan

# 小变化值不显示，突出显著变化区域
mosaic_data[
    (mosaic_data > -0.1) & (mosaic_data < 0.1)
] = np.nan

print("冰川栅格 CRS：", raster_crs)
print("拼接后数组尺寸：", mosaic_data.shape)

# ================= 3. 读取历史灾害数据 =================
gdf = gpd.read_file(shp_path)

if gdf.crs is None:
    raise ValueError("灾害矢量没有坐标系，请检查 .prj 文件。")

gdf = gdf[
    gdf.geometry.notna()
    & ~gdf.geometry.is_empty
].copy()

if "类型" not in gdf.columns:
    raise ValueError(
        f"找不到“类型”字段。现有字段为：{list(gdf.columns)}"
    )

# 投影到栅格坐标系，确保点和底图使用同一坐标系
gdf_raster = gdf.to_crs(raster_crs)

# 用代表点替代多边形质心，保证代表点位于多边形内部
gdf_points = gdf_raster.copy()
gdf_points["geometry"] = (
    gdf_points.geometry.representative_point()
)

# ================= 4. 计算栅格显示范围 =================
height, width = mosaic_data.shape
bounds = array_bounds(height, width, out_transform)

left, bottom, right, top = bounds
extent = [left, right, bottom, top]


from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D

# ================= 5. 优化绘图 =================

fig, ax = plt.subplots(figsize=(12, 10), dpi=160)

# 背景
ax.set_facecolor("#f4f4f4")

# 冰川高程变化栅格
img = ax.imshow(
    mosaic_data,
    cmap="RdBu_r",
    vmin=-2,
    vmax=2,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    alpha=1.0,
    zorder=1
)

# 色标放在右侧，避免挤压主图
cbar = fig.colorbar(
    img,
    ax=ax,
    fraction=0.045,
    pad=0.035,
    shrink=0.78
)
cbar.set_label("冰川表面高程变化率（米/年）", fontsize=11)
cbar.ax.tick_params(labelsize=9)

# 灾害类型配色
colors = {
    "降雨型": "#27c94b",
    "冰水混合型": "#e000e8",
    "冰雪融水型": "#00c8e8",
    "冰湖溃决型": "#ffc400"
}

# 先画代表点，再单独创建图例
for disaster_type, color in colors.items():
    subset = gdf_points[
        gdf_points["类型"].astype(str) == disaster_type
    ]

    if len(subset) > 0:
        subset.plot(
            ax=ax,
            color=color,
            marker="o",
            markersize=35,
            edgecolor="black",
            linewidth=0.65,
            alpha=0.95,
            zorder=5
        )

# 其他类型也显示，防止样本遗漏
other = gdf_points[
    ~gdf_points["类型"].astype(str).isin(colors.keys())
]

if len(other) > 0:
    other.plot(
        ax=ax,
        color="white",
        markersize=35,
        edgecolor="black",
        linewidth=0.65,
        zorder=5
    )

# ================= 6. 以灾害点为中心放大 =================

# 确保灾点坐标与栅格坐标系相同
# 这里的 gdf_points 应已转换到 raster_crs
minx, miny, maxx, maxy = gdf_points.total_bounds

# 放大范围：灾点外扩 15 公里
padding = 15000

ax.set_xlim(minx - padding, maxx + padding)
ax.set_ylim(miny - padding, maxy + padding)

# 坐标用公里显示，底层数据仍然是米
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x / 1000:.1f}")
)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f"{y / 1000:.1f}")
)

ax.set_xlabel("UTM 东坐标（公里）", fontsize=11)
ax.set_ylabel("UTM 北坐标（公里）", fontsize=11)

ax.set_title(
    "冰川表面高程变化与历史泥石流灾害分布",
    fontsize=15,
    pad=14
)

# 自定义图例
legend_items = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=8,
        label=label
    )
    for label, color in colors.items()
]

if len(other) > 0:
    legend_items.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="其他类型"
        )
    )

ax.legend(
    handles=legend_items,
    title="历史灾害类型",
    loc="best",
    fontsize=9,
    title_fontsize=10,
    framealpha=0.95
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.22
)


# ================= 7. 保存高清图片 =================

fig.subplots_adjust(
    left=0.12,
    right=0.87,
    bottom=0.11,
    top=0.91
)

out_png = ROOT / "冰川变化与历史灾害_局部放大图.png"

fig.savefig(
    out_png,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("图片已保存：", out_png)
print("当前地图 CRS：", raster_crs)
print("局部显示范围（米）：")
print("X:", minx - padding, "至", maxx + padding)
print("Y:", miny - padding, "至", maxy + padding)

