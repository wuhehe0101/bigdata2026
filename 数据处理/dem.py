
import rasterio
import numpy as np
import os
import json

# =======================================================
# 1. 设置 DEM 路径和结果保存路径
# =======================================================
dem_path = (
    r"D:\mju\2026-1\学习\竞赛\大数据挑战"
    r"\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
    r"\（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据"
    r"\DEM.tif"
)

output_dir = os.path.join(os.path.dirname(dem_path), "DEM检查结果")
os.makedirs(output_dir, exist_ok=True)

print(f"正在读取 DEM 文件：\n{dem_path}\n")

if not os.path.exists(dem_path):
    raise FileNotFoundError(
        f"找不到 DEM 文件，请检查路径：{dem_path}"
    )

# =======================================================
# 2. 读取 DEM 并检查空间参考与数据质量
# =======================================================
with rasterio.open(dem_path) as src:

    # 基础空间信息
    crs = src.crs
    transform = src.transform
    bounds = src.bounds
    nodata = src.nodata

    height = src.height
    width = src.width
    band_count = src.count
    resolution = src.res

    # DEM 高程数据转为浮点型，便于处理 NaN
    dem_data = src.read(1).astype(np.float64)

    # 记录原始像元总数
    total_pixels = dem_data.size

    # ===================================================
    # 3. 构建有效像元掩膜
    # ===================================================
    # 首先排除 NaN、正无穷和负无穷
    valid_mask = np.isfinite(dem_data)

    # 排除文件元数据指定的 NoData
    if nodata is not None:
        if np.isnan(nodata):
            valid_mask &= ~np.isnan(dem_data)
        else:
            valid_mask &= dem_data != nodata

    # 统计低于 -1000 米的异常候选值。
    # 这里只统计，不自动删除，避免误删真实数据。
    low_value_mask = valid_mask & (dem_data < -1000)

    # 如果存在低于 -1000 米的值，需结合数据说明判断是否异常。
    # 本区域是高山地形，负海拔通常值得进一步检查。
    valid_values = dem_data[valid_mask]

    if valid_values.size == 0:
        raise ValueError(
            "DEM 中没有检测到有效像元，请检查 NoData 设置或数据文件。"
        )

    # ===================================================
    # 4. 输出基础空间信息
    # ===================================================
    print("=" * 60)
    print("【一、基础信息】")
    print(f"坐标系 CRS：{crs}")
    print(f"图像尺寸：{height} 行 × {width} 列")
    print(f"波段数：{band_count}")
    print(f"像元分辨率：{resolution[0]} × {resolution[1]}")
    print(f"NoData：{nodata}")
    print(f"仿射变换参数：{transform}")

    print("\n【二、地理边界】")
    print(f"最小经度 / X：{bounds.left:.6f}")
    print(f"最大经度 / X：{bounds.right:.6f}")
    print(f"最小纬度 / Y：{bounds.bottom:.6f}")
    print(f"最大纬度 / Y：{bounds.top:.6f}")

    print("\n【三、数据质量检查】")
    print(f"像元总数：{total_pixels:,}")
    print(f"有效像元数：{valid_values.size:,}")
    print(
        f"有效像元占比："
        f"{valid_values.size / total_pixels * 100:.4f}%"
    )
    print(f"无效像元数：{total_pixels - valid_values.size:,}")
    print(f"低于 -1000 米的像元数：{np.count_nonzero(low_value_mask):,}")

    # ===================================================
    # 5. 统计高程分布
    # ===================================================
    print("\n【四、高程统计】")
    print(f"最低高程：{np.min(valid_values):.2f} m")
    print(f"最高高程：{np.max(valid_values):.2f} m")
    print(f"平均高程：{np.mean(valid_values):.2f} m")
    print(f"高程标准差：{np.std(valid_values):.2f} m")
    print(f"高程中位数：{np.median(valid_values):.2f} m")
    print(f"25% 分位数：{np.percentile(valid_values, 25):.2f} m")
    print(f"75% 分位数：{np.percentile(valid_values, 75):.2f} m")

    # ===================================================
    # 6. 保存统计结果
    # ===================================================
    stats = {
        "crs": str(crs),
        "rows": int(height),
        "columns": int(width),
        "bands": int(band_count),
        "resolution_x": float(resolution[0]),
        "resolution_y": float(resolution[1]),
        "bounds_left": float(bounds.left),
        "bounds_right": float(bounds.right),
        "bounds_bottom": float(bounds.bottom),
        "bounds_top": float(bounds.top),
        "nodata": None if nodata is None else float(nodata),
        "total_pixels": int(total_pixels),
        "valid_pixels": int(valid_values.size),
        "invalid_pixels": int(total_pixels - valid_values.size),
        "valid_pixel_percent": float(
            valid_values.size / total_pixels * 100
        ),
        "pixels_below_minus_1000m": int(
            np.count_nonzero(low_value_mask)
        ),
        "elevation_min_m": float(np.min(valid_values)),
        "elevation_max_m": float(np.max(valid_values)),
        "elevation_mean_m": float(np.mean(valid_values)),
        "elevation_std_m": float(np.std(valid_values)),
        "elevation_median_m": float(np.median(valid_values)),
        "elevation_p25_m": float(np.percentile(valid_values, 25)),
        "elevation_p75_m": float(np.percentile(valid_values, 75)),
    }

    json_path = os.path.join(output_dir, "DEM_statistics.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print("\n【五、结果保存】")
    print(f"统计结果已保存至：{json_path}")
    print("=" * 60)

# =======================================================
# 7. 针对后续泥石流建模给出检查提示
# =======================================================
print("\n【六、后续建模提示】")

if crs is not None and crs.to_epsg() == 4326:
    print("DEM 使用经纬度坐标系 EPSG:4326。")
    print("提取 DEM 高程时，灾害点必须先转换到该坐标系。")
    print("计算坡度、距离、汇流长度等地形指标时，")
    print("建议将 DEM 重投影到适合当地的米制投影坐标系。")
else:
    print("请确认 DEM 坐标系是否适合后续距离和地形计算。")

if np.count_nonzero(low_value_mask) > 0:
    print("警告：发现低于 -1000 m 的像元，请检查其是否为异常值。")
else:
    print("未发现低于 -1000 m 的有效像元。")

print(
    "注意：有效像元比例接近 100% 不代表 DEM 一定没有问题；"
    "还需结合 NoData 元数据、空间范围和实际地形检查。"
)
