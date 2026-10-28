
# -*- coding: utf-8 -*-
"""
吉隆县泥石流灾害数据预处理
功能：
1. 读取历史灾害面和官方灾害点
2. 检查属性、坐标系、空间范围、几何有效性
3. 统一投影到 EPSG:32645
4. 检查官方灾害点与历史灾害面的空间关系
5. 输出 Excel 检查表及转换后的矢量文件

注意：本程序不修改原始文件。
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd


# ============================================================
# 一、文件路径
# ============================================================

historical_path = Path(
    r"D:\mju\2026-1\学习\竞赛\大数据挑战"
    r"\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
    r"\（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据"
    r"\西藏日喀则市吉隆县泥石流灾害区历史灾点数据"
    r"\西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"
)

official_path = Path(
    r"D:\mju\2026-1\学习\竞赛\大数据挑战"
    r"\2026年MathorCup大数据挑战赛-赛道B初赛\附件"
    r"\灾害点位置\灾害点位置.shp"
)

output_dir = historical_path.parent / "灾害数据预处理结果"
output_dir.mkdir(parents=True, exist_ok=True)

target_crs = "EPSG:32645"


# ============================================================
# 二、读取与基础检查
# ============================================================

def read_vector(path, name):
    """读取矢量文件并进行基础检查。"""

    if not path.exists():
        raise FileNotFoundError(f"{name}文件不存在：\n{path}")

    print("\n" + "=" * 70)
    print(f"读取：{name}")
    print("=" * 70)

    gdf = gpd.read_file(path, encoding="GBK")

    print(f"文件：{path.name}")
    print(f"要素数量：{len(gdf)}")
    print(f"坐标系：{gdf.crs}")
    print(f"几何类型：")
    print(gdf.geom_type.value_counts(dropna=False).to_string())

    print("\n字段：")
    print(gdf.columns.tolist())

    print("\n前 5 行属性：")
    print(gdf.drop(columns="geometry").head().to_string())

    if gdf.crs is None:
        raise ValueError(
            f"{name}缺少坐标系信息，不能安全转换坐标。"
            "请先确认原始数据的真实坐标系。"
        )

    # 检查空几何、无效几何
    empty_count = int(gdf.geometry.isna().sum())
    invalid_count = int(
        (~gdf.geometry.is_valid & gdf.geometry.notna()).sum()
    )

    print(f"\n空几何数量：{empty_count}")
    print(f"无效几何数量：{invalid_count}")

    # 注意：这里只检查，不自动修复几何
    print(f"原始空间范围：{gdf.total_bounds}")

    return gdf


historical = read_vector(historical_path, "历史灾害面")
official = read_vector(official_path, "官方灾害点")


# ============================================================
# 三、检查原始数据空间范围
# ============================================================

def bounds_to_df(gdf, name):
    xmin, ymin, xmax, ymax = gdf.total_bounds

    return {
        "数据名称": name,
        "原始坐标系": str(gdf.crs),
        "要素数量": len(gdf),
        "最小X": xmin,
        "最小Y": ymin,
        "最大X": xmax,
        "最大Y": ymax,
        "空几何数": int(gdf.geometry.isna().sum()),
        "无效几何数": int(
            (~gdf.geometry.is_valid & gdf.geometry.notna()).sum()
        ),
    }


bounds_df = pd.DataFrame([
    bounds_to_df(historical, "历史灾害面"),
    bounds_to_df(official, "官方灾害点"),
])


# ============================================================
# 四、统一转换为 UTM 45N
# ============================================================

print("\n" + "=" * 70)
print("统一坐标系")
print("=" * 70)

historical_utm = historical.to_crs(target_crs).copy()
official_utm = official.to_crs(target_crs).copy()

print("历史灾害面转换后范围：", historical_utm.total_bounds)
print("官方灾害点转换后范围：", official_utm.total_bounds)
print("目标坐标系：", target_crs)


# ============================================================
# 五、检查官方灾害点与历史灾害面的空间关系
# ============================================================

print("\n" + "=" * 70)
print("检查官方灾害点与历史灾害面的空间关系")
print("=" * 70)

# 先过滤空几何，避免空间判断异常
historical_valid = historical_utm[
    historical_utm.geometry.notna()
    & ~historical_utm.geometry.is_empty
].copy()

official_valid = official_utm[
    official_utm.geometry.notna()
    & ~official_utm.geometry.is_empty
].copy()

relation_rows = []

for point_idx, point_row in official_valid.iterrows():
    point_geom = point_row.geometry

    # 找出包含该点或边界接触该点的历史灾害面
    matched = historical_valid[
        historical_valid.geometry.intersects(point_geom)
    ]

    if len(matched) == 0:
        relation_rows.append({
            "官方点索引": point_idx,
            "匹配历史灾害面数量": 0,
            "匹配历史灾害面原始索引": "",
            "匹配说明": "未与历史灾害面相交",
        })
    else:
        relation_rows.append({
            "官方点索引": point_idx,
            "匹配历史灾害面数量": len(matched),
            "匹配历史灾害面原始索引": ",".join(
                str(i) for i in matched.index.tolist()
            ),
            "匹配说明": "与历史灾害面相交",
        })

relation_df = pd.DataFrame(relation_rows)

print(relation_df.to_string(index=False))


# ============================================================
# 六、生成历史灾害面属性统计表
# ============================================================

historical_attr = historical_utm.drop(
    columns="geometry"
).copy()

historical_attr.insert(
    0, "原始索引", historical_utm.index.astype(str)
)

historical_attr["几何类型"] = historical_utm.geom_type.values
historical_attr["几何有效"] = historical_utm.geometry.is_valid.values

# 投影后面积单位为平方米
historical_attr["投影面积_m2"] = historical_utm.geometry.area.values
historical_attr["投影面积_km2"] = (
    historical_attr["投影面积_m2"] / 1_000_000
)


# ============================================================
# 七、生成官方灾害点属性统计表
# ============================================================

official_attr = official_utm.drop(
    columns="geometry"
).copy()

official_attr.insert(
    0, "原始索引", official_utm.index.astype(str)
)

official_attr["几何类型"] = official_utm.geom_type.values
official_attr["几何有效"] = official_utm.geometry.is_valid.values

# 输出投影坐标，方便后续与 DEM 进行空间提取
official_attr["投影X_m"] = official_utm.geometry.x.values
official_attr["投影Y_m"] = official_utm.geometry.y.values


# ============================================================
# 八、输出转换后的矢量文件
# ============================================================

historical_out = output_dir / "历史灾害面_UTM45N.shp"
official_out = output_dir / "官方灾害点_UTM45N.shp"

# Shapefile 对字段名长度有限制，GeoPandas 会发出相应提示
with warnings.catch_warnings():
    warnings.simplefilter("always")
    historical_utm.to_file(
        historical_out,
        driver="ESRI Shapefile",
        encoding="UTF-8",
    )
    official_utm.to_file(
        official_out,
        driver="ESRI Shapefile",
        encoding="UTF-8",
    )


# ============================================================
# 九、输出 Excel 检查表
# ============================================================

excel_out = output_dir / "灾害数据空间检查表.xlsx"

with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    bounds_df.to_excel(
        writer, sheet_name="数据范围与质量", index=False
    )

    historical_attr.to_excel(
        writer, sheet_name="历史灾害面属性", index=False
    )

    official_attr.to_excel(
        writer, sheet_name="官方灾害点属性", index=False
    )

    relation_df.to_excel(
        writer, sheet_name="点面空间关系", index=False
    )


# ============================================================
# 十、完成报告
# ============================================================

print("\n" + "=" * 70)
print("全部处理完成")
print("=" * 70)

print(f"结果目录：{output_dir}")
print(f"历史灾害面：{historical_out}")
print(f"官方灾害点：{official_out}")
print(f"Excel 检查表：{excel_out}")

print("\n请重点检查：")
print("1. 两份数据的坐标系是否符合预期。")
print("2. 官方灾害点是否与历史灾害面相交。")
print("3. 是否存在无效几何或空几何。")
print("4. 历史灾害面与官方灾害点是否覆盖研究区。")
print("5. 不要仅凭点面相交，就认定它们属于同一次灾害。")


import pandas as pd

# 计算每个官方点到最近历史灾害面的距离
rows = []

for point_idx, point_row in official_utm.iterrows():
    point_geom = point_row.geometry

    distances = historical_utm.geometry.distance(point_geom)
    nearest_idx = distances.idxmin()
    nearest_polygon = historical_utm.loc[nearest_idx]
    distance_m = float(distances.loc[nearest_idx])

    rows.append({
        "官方点索引": point_idx,
        "官方点经度": point_row.geometry.x,
        "官方点纬度": point_row.geometry.y,
        "最近历史灾害面索引": nearest_idx,
        "最近历史灾害面Code": nearest_polygon.get("Code", ""),
        "最近距离_m": round(distance_m, 2),
        "最近距离_km": round(distance_m / 1000, 3),
    })

distance_df = pd.DataFrame(rows)

print("\n官方灾害点到最近历史灾害面的距离：")
print(distance_df.to_string(index=False))

# 单独输出距离表
out_path = output_dir / "官方灾害点最近距离检查.xlsx"
distance_df.to_excel(out_path, index=False)

print(f"\n距离检查表已保存：{out_path}")


