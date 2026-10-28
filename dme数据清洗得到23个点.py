
from pathlib import Path
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.mask import mask
from shapely.geometry import box
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 路径 =================
ROOT = Path(
    r"D:\mju\2026-1\学习\竞赛\大数据挑战"
    r"\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
)

dem_path = (
    ROOT / "（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据"
    / "DEM.tif"
)
land_path = (
    ROOT / "（5）西藏日喀则市吉隆县泥石流灾害区土地覆盖数据（2020-2023）"
    / "jilong_LUCD_2023.tif"
)
point_path = (
    ROOT / "（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据"
    / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据"
    / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"
)

for p in [dem_path, land_path, point_path]:
    if not p.exists():
        raise FileNotFoundError(f"找不到文件：{p}")

# ================= 2. 读取灾害多边形 =================
print("正在读取数据...")
gdf = gpd.read_file(point_path)

if gdf.crs is None:
    raise ValueError("灾害矢量数据没有 CRS，需先确认其原始坐标系。")

# 去除空几何、无效几何
gdf = gdf[
    gdf.geometry.notna()
    & ~gdf.geometry.is_empty
    & gdf.geometry.is_valid
].copy()

# 保存原始灾点编号，便于追踪
gdf["样本编号"] = gdf.index.astype(str)

# ================= 3. 高程分区统计 =================
elev_results = []

with rasterio.open(dem_path) as src:
    dem_crs = src.crs
    dem_bounds = src.bounds
    dem_box = box(
        dem_bounds.left, dem_bounds.bottom,
        dem_bounds.right, dem_bounds.top
    )

    gdf_dem = gdf.to_crs(dem_crs)

    for idx, geom in gdf_dem.geometry.items():
        result = {
            "高程_mean": np.nan,
            "高程_min": np.nan,
            "高程_max": np.nan,
            "高程有效像元数": 0
        }

        if not geom.intersects(dem_box):
            elev_results.append((idx, result))
            continue

        try:
            # filled=False 保留掩膜，避免将 NoData 当成真实高程
            arr, _ = mask(
                src,
                [geom],
                crop=True,
                all_touched=True,
                filled=False
            )

            band = arr[0]
            valid = ~np.ma.getmaskarray(band)
            values = np.asarray(band.data)[valid]
            values = values[np.isfinite(values)]

            if src.nodata is not None:
                values = values[values != src.nodata]

            if values.size > 0:
                result["高程_mean"] = float(values.mean())
                result["高程_min"] = float(values.min())
                result["高程_max"] = float(values.max())
                result["高程有效像元数"] = int(values.size)

        except ValueError:
            # 多边形与栅格没有重叠像元
            pass

        elev_results.append((idx, result))

elev_df = pd.DataFrame(
    {idx: value for idx, value in elev_results}
).T
elev_df.index = pd.Index(
    [idx for idx, _ in elev_results]
)

for col in elev_df.columns:
    gdf[col] = elev_df[col].reindex(gdf.index)

print("高程统计完成。")
print("有有效高程像元的灾点数：",
      int((gdf["高程有效像元数"] > 0).sum()))

# ================= 4. 土地覆盖类别比例 =================
# 注意：只对与土地覆盖栅格相交的样本统计
lc_results = []

with rasterio.open(land_path) as src:
    gdf_lc = gdf.to_crs(src.crs)

    for idx, geom in gdf_lc.geometry.items():
        result = {
            "土地覆盖有效像元数": 0,
            "土地覆盖主类别": np.nan
        }

        if not geom.intersects(
            box(*src.bounds)
        ):
            lc_results.append((idx, result))
            continue

        try:
            arr, _ = mask(
                src,
                [geom],
                crop=True,
                all_touched=True,
                filled=False
            )

            band = arr[0]
            valid = ~np.ma.getmaskarray(band)
            values = np.asarray(band.data)[valid]

            # 去掉 NoData；该数据的 NoData 若为 0，则 0 不计入类别
            if src.nodata is not None:
                values = values[values != src.nodata]

            result["土地覆盖有效像元数"] = int(values.size)

            if values.size > 0:
                classes, counts = np.unique(
                    values, return_counts=True
                )
                total = counts.sum()

                # 主类别为像元数最多的类别
                result["土地覆盖主类别"] = int(
                    classes[np.argmax(counts)]
                )

                # 输出每个类别的面积近似占比（按像元数）
                for cls, count in zip(classes, counts):
                    result[f"土地覆盖_{int(cls)}_比例"] = (
                        float(count / total)
                    )

        except ValueError:
            pass

        lc_results.append((idx, result))

lc_df = pd.DataFrame(
    {idx: value for idx, value in lc_results}
).T
lc_df.index = pd.Index(
    [idx for idx, _ in lc_results]
)

for col in lc_df.columns:
    gdf[col] = lc_df[col].reindex(gdf.index)

print("土地覆盖统计完成。")

# ================= 5. 整理赛方已有数值字段 =================
for col in ["沟比降", "高差", "主沟长", "高程Max", "Area"]:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(
            gdf[col], errors="coerce"
        )

# ================= 6. 保存结果 =================
out_path = ROOT / "灾点地形统计结果.csv"

gdf.drop(columns="geometry").to_csv(
    out_path,
    index=False,
    encoding="utf-8-sig"
)

show_cols = [
    "样本编号", "类型",
    "高程_mean", "高程_min", "高程_max",
    "高程有效像元数",
    "土地覆盖主类别", "土地覆盖有效像元数",
    "沟比降", "高差", "主沟长"
]
show_cols = [c for c in show_cols if c in gdf.columns]

print("\n=== 特征提取结果预览 ===")
print(gdf[show_cols].head(10).to_string(index=False))

print("\n=== 数据质量检查 ===")
print("原始有效几何样本数：", len(gdf))
print("有有效高程的样本数：",
      int((gdf["高程有效像元数"] > 0).sum()))
print("高程缺失样本数：",
      int(gdf["高程_mean"].isna().sum()))
print("土地覆盖缺失样本数：",
      int(gdf["土地覆盖主类别"].isna().sum()))

print(f"\n结果已保存至：{out_path}")
# 查看成功提取高程的灾点
valid_dem = gdf[gdf["高程有效像元数"] > 0]

print("\n=== 成功提取高程的全部灾点 ===")
print(valid_dem[[
    "样本编号", "类型",
    "高程_mean", "高程_min", "高程_max",
    "高程有效像元数"
]].to_string(index=False))

# 查看高程缺失的灾点
missing_dem = gdf[gdf["高程有效像元数"] == 0]

print("\n=== 高程缺失的灾点编号 ===")
print(missing_dem[["样本编号", "类型"]].to_string(index=False))

# 核对坐标系和栅格范围
with rasterio.open(dem_path) as src:
    print("\nDEM CRS：", src.crs)
    print("DEM bounds：", src.bounds)

print("灾点原始 CRS：", gpd.read_file(point_path).crs)
