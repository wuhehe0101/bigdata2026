import geopandas as gpd
import os

# =======================================================
# 1. 设置附件根目录
# =======================================================
ROOT = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件"

# 历史灾点路径 (84个)
pts_84_path = os.path.join(ROOT,
                           r"（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp")
# 官方4个受灾点路径
pts_4_path = os.path.join(ROOT, r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\灾害点位置\灾害点位置.shp")


def inspect_shp(path, name):
    print(f"\n{'=' * 50}")
    print(f"正在读取: {name}")
    print(f"路径: {path}")

    if not os.path.exists(path):
        print("❌ 错误：找不到该路径！")
        return

    gdf = gpd.read_file(path)

    print(f"坐标系 (CRS): {gdf.crs}")
    print(f"要素数量 (rows): {len(gdf)}")
    print(f"几何类型 (geometry type): {gdf.geom_type.unique()}")

    print("\n--- 包含的列名 (Columns) ---")
    for col in gdf.columns:
        print(f"  - {col}")

    print("\n--- 前 3 行数据预览 ---")
    print(gdf.head(3))


# =======================================================
# 2. 执行查看
# =======================================================
inspect_shp(pts_84_path, "84个历史灾点数据 (赛题附件6)")
inspect_shp(pts_4_path, "4个官方受灾点 (灾害点位置)")

# 如果你还要看 DEM，DEM 是栅格，不能用 read_file 读，需要用 rasterio，我刚才已经给你写过了。
# 这里就只查 SHP。

import geopandas as gpd
import pandas as pd

path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"

gdf = gpd.read_file(path)

# 1. 检查核心字段的缺失率、唯一值数量和数据类型
cols = [
    "Name", "flag", "height", "Area", "沟比降",
    "主沟长", "高差", "规模", "频率",
    "沟口H", "高程Max", "扇宽", "扇长",
    "有无_Fan", "解译Time", "影像Time"
]

cols = [c for c in cols if c in gdf.columns]

summary = pd.DataFrame({
    "数据类型": gdf[cols].dtypes.astype(str),
    "缺失数量": gdf[cols].isna().sum(),
    "缺失比例": gdf[cols].isna().mean().round(3),
    "唯一值数量": gdf[cols].nunique(dropna=True)
})

print("=== 核心字段质量检查 ===")
print(summary.to_string())

# 2. 查看重点数值字段的统计分布
numeric_cols = [
    "height", "Area", "沟比降", "主沟长",
    "高差", "沟口H", "高程Max", "扇宽", "扇长"
]
numeric_cols = [c for c in numeric_cols if c in gdf.columns]

print("\n=== 数值字段统计 ===")
print(gdf[numeric_cols].describe().T.to_string())

# 3. 查看灾害识别时间
for c in ["解译Time", "影像Time"]:
    if c in gdf.columns:
        print(f"\n=== {c} 前20条 ===")
        print(gdf[c].head(20).to_string(index=False))

# 4. 检查几何是否有效
print("\n=== 几何检查 ===")
print("总要素数：", len(gdf))
print("无效几何数：", (~gdf.geometry.is_valid).sum())
print("空几何数：", gdf.geometry.is_empty.sum())