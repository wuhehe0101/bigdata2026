import os
import json
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ============================================================
# 1. 路径设置
# ============================================================
dem_path = (
    r"D:\mju\2026-1\学习\竞赛\大数据挑战"
    r"\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
    r"\（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据"
    r"\DEM.tif"
)

output_dir = os.path.join(os.path.dirname(dem_path), "DEM地形特征")
os.makedirs(output_dir, exist_ok=True)

target_crs = "EPSG:32645"
target_resolution = 30.0  # 米

# ============================================================
# 2. 读取原始 DEM，正确处理 NoData
# ============================================================
if not os.path.exists(dem_path):
    raise FileNotFoundError(f"找不到 DEM 文件：{dem_path}")

with rasterio.open(dem_path) as src:
    src_crs = src.crs
    src_nodata = src.nodata
    src_bounds = src.bounds
    dem_src = src.read(1).astype(np.float32)

    if src_crs is None:
        raise ValueError("原始 DEM 没有 CRS 信息，无法安全重投影。")

    invalid = ~np.isfinite(dem_src)
    if src_nodata is not None:
        if np.isnan(src_nodata):
            invalid |= np.isnan(dem_src)
        else:
            invalid |= dem_src == src_nodata
    dem_src[invalid] = np.nan

    # 保存源数据空间参数，后续关闭文件后仍可使用
    src_transform = src.transform
    src_width = src.width
    src_height = src.height

print("=" * 68)
print("【原始 DEM】")
print(f"CRS：{src_crs}")
print(f"尺寸：{src_height} 行 × {src_width} 列")
print(f"NoData：{src_nodata}")
print(f"目标 CRS：{target_crs}")
print(f"目标分辨率：{target_resolution:.1f} m")

# ============================================================
# 3. 重投影到米制坐标系
# ============================================================
dst_transform, dst_width, dst_height = calculate_default_transform(
    src_crs,
    target_crs,
    src_width,
    src_height,
    *src_bounds,
    resolution=target_resolution
)

dem_projected = np.full(
    (dst_height, dst_width), np.nan, dtype=np.float32
)

reproject(
    source=dem_src,
    destination=dem_projected,
    src_transform=src_transform,
    src_crs=src_crs,
    src_nodata=np.nan,
    dst_transform=dst_transform,
    dst_crs=target_crs,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear
)

valid = np.isfinite(dem_projected)
if not np.any(valid):
    raise ValueError("重投影后没有有效像元，请检查 CRS、NoData 或输入数据。")

# ============================================================
# 4. 计算坡度和坡向
# ============================================================
# 栅格行号由北向南增加：
# dz_drow 是向南方向的梯度，向北梯度 dz_dnorth = -dz_drow。
z = dem_projected.astype(np.float64)
valid = np.isfinite(z)
z_filled = np.where(valid, z, 0.0)

dz_drow, dz_dx = np.gradient(
    z_filled, target_resolution, target_resolution
)
dz_dnorth = -dz_drow

# 仅保留 3x3 邻域全部有效的像元，避免 NoData 边界产生伪坡度。
valid_neighbor = np.zeros_like(valid, dtype=bool)
if valid.shape[0] >= 3 and valid.shape[1] >= 3:
    neighborhood = np.ones(
        (valid.shape[0] - 2, valid.shape[1] - 2), dtype=bool
    )
    for dr in range(3):
        for dc in range(3):
            neighborhood &= valid[
                dr:dr + neighborhood.shape[0],
                dc:dc + neighborhood.shape[1]
            ]
    valid_neighbor[1:-1, 1:-1] = neighborhood

gradient_magnitude = np.sqrt(dz_dx**2 + dz_dnorth**2)

# 坡度：0–90 度
slope = np.degrees(np.arctan(gradient_magnitude)).astype(np.float32)

# 坡向：坡面最陡下降方向；0°北、90°东、180°南、270°西
# 东向下降由 -dz_dx 表示，北向下降由 -dz_dnorth 表示。
aspect = (
    np.degrees(np.arctan2(-dz_dx, -dz_dnorth)) + 360.0
) % 360.0
aspect = aspect.astype(np.float32)

# 消除浮点数造成的接近 360° 值，统一到 [0, 360)
aspect[np.isclose(aspect, 360.0, atol=1e-5)] = 0.0

# 平坦区域坡向无明确意义，设为 NoData
flat_mask = gradient_magnitude < 1e-6

slope[~valid_neighbor] = np.nan
aspect[~valid_neighbor | flat_mask] = np.nan
slope[~valid] = np.nan

# ============================================================
# 5. 保存 GeoTIFF
# ============================================================
nodata_out = -9999.0
profile = {
    "driver": "GTiff",
    "height": dem_projected.shape[0],
    "width": dem_projected.shape[1],
    "count": 1,
    "dtype": "float32",
    "crs": target_crs,
    "transform": dst_transform,
    "nodata": nodata_out,
    "compress": "deflate",
    "predictor": 3
}

def save_raster(path, array):
    out = np.where(
        np.isfinite(array), array, nodata_out
    ).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)

elevation_path = os.path.join(output_dir, "elevation_utm45n.tif")
slope_path = os.path.join(output_dir, "slope_degree.tif")
aspect_path = os.path.join(output_dir, "aspect_degree.tif")

save_raster(elevation_path, dem_projected)
save_raster(slope_path, slope)
save_raster(aspect_path, aspect)

# ============================================================
# 6. 统计并保存报告
# ============================================================
def summarize(name, arr):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {
            "name": name,
            "valid_pixels": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None
        }

    return {
        "name": name,
        "valid_pixels": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75))
    }

report = {
    "source_dem": dem_path,
    "source_crs": str(src_crs),
    "output_crs": target_crs,
    "resolution_m": target_resolution,
    "rows": int(dem_projected.shape[0]),
    "columns": int(dem_projected.shape[1]),
    "valid_elevation_pixels": int(np.count_nonzero(np.isfinite(dem_projected))),
    "elevation": summarize("elevation_m", dem_projected),
    "slope": summarize("slope_degree", slope),
    "aspect": summarize("aspect_degree", aspect),
    "notes": [
        "坡度单位为度。",
        "坡向表示坡面最陡下降方向：0度北、90度东、180度南、270度西。",
        "坡向值规范在0度（含）至360度（不含）之间。",
        "平坦区域的坡向设为NoData。",
        "坡度和坡向的边缘及无效像元邻域设为NoData。",
        "重投影使用双线性重采样，目标分辨率为30米。"
    ]
}

report_path = os.path.join(output_dir, "terrain_statistics.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=4)

# ============================================================
# 7. 输出结果
# ============================================================
print("\n" + "=" * 68)
print("【处理完成】")
print(f"输出 CRS：{target_crs}")
print(f"输出栅格尺寸：{dem_projected.shape[0]} 行 × {dem_projected.shape[1]} 列")
print(f"有效高程像元：{np.count_nonzero(np.isfinite(dem_projected)):,}")
print(f"高程文件：{elevation_path}")
print(f"坡度文件：{slope_path}")
print(f"坡向文件：{aspect_path}")
print(f"统计报告：{report_path}")

for arr_name, arr in [
    ("高程（m）", dem_projected),
    ("坡度（度）", slope),
    ("坡向（度）", aspect)
]:
    vals = arr[np.isfinite(arr)]
    if vals.size:
        print(
            f"{arr_name}：有效像元={vals.size:,}，"
            f"最小值={np.min(vals):.2f}，"
            f"最大值={np.max(vals):.2f}，"
            f"均值={np.mean(vals):.2f}"
        )

print("=" * 68)
print("提示：后续历史灾害面、灾害点、冰川和土地覆盖数据也要转换到同一目标 CRS，")
print("并在叠加/提取特征时核对空间范围与像元对齐。")
