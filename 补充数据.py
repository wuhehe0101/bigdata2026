import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']

# 1. 读取
shp_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\灾害点位置\灾害点位置.shp"
gdf = gpd.read_file(shp_path)

# 2. 看看 Extent_of_ 写了什么
print("=== 受灾点属性信息 ===")
print(gdf[['OBJECTID', 'lon', 'lat', 'Extent_of_']])

# 3. 画图展示
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 画出这4个红色受灾点
gdf.plot(ax=ax, color='red', marker='*', markersize=200, edgecolor='black', label='8.26 官方受灾点位', zorder=5)

# 添加经纬度标注
for idx, row in gdf.iterrows():
    ax.annotate(f"P{row['OBJECTID']}\n({row['lon']:.4f}, {row['lat']:.4f})",
                xy=(row['lon'], row['lat']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                arrowprops=dict(arrowstyle="-", color='gray'))

ax.set_title("2026-08-26 吉隆口岸泥石流核心受灾点位", fontsize=16)
ax.set_xlabel("经度", fontsize=12)
ax.set_ylabel("纬度", fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()