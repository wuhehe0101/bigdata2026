import streamlit as st
import rasterio
import numpy as np
import plotly.graph_objects as go
import geopandas as gpd

st.set_page_config(layout="wide", page_title="吉隆口岸泥石流动力仿真系统")

st.title("🏔️ 吉隆口岸高位冰崩-泥石流 3D 动力学仿真系统")
st.markdown("通过左侧面板调节物理参数，右侧实时查看三维仿真演化过程。")

# =======================================================
# 1. 侧边栏：物理参数调节面板
# =======================================================
st.sidebar.header("⚙️ 核心物理参数控制台")
friction_angle = st.sidebar.slider("内摩擦角 (度) - 越大流得越慢", min_value=5, max_value=45, value=22)
trigger_volume = st.sidebar.slider("冰崩注入体积 (相对值)", min_value=500, max_value=3000, value=1000, step=100)
time_step = st.sidebar.slider("仿真演化步数", min_value=10, max_value=100, value=50, step=10)


# =======================================================
# 2. 数据加载与预处理 (带缓存防卡顿)
# =======================================================
@st.cache_data
def load_data():
    dem_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据\DEM.tif"
    with rasterio.open(dem_path) as src:
        # 降采样以提升 3D 渲染速度 (步长4)
        dem = src.read(1)[::4, ::4].astype(float)
        bounds = src.bounds
    dem[dem < -100] = np.nan
    return dem, bounds


dem, bounds = load_data()
rows, cols = dem.shape

# 生成 X, Y 坐标网格
x_coords = np.linspace(bounds.left, bounds.right, cols)
y_coords = np.linspace(bounds.top, bounds.bottom, rows)
X, Y = np.meshgrid(x_coords, y_coords)

# 蓝塘里壤峰大致经纬度转换到 UTM 坐标 (这里用最高点近似替代源区)
min_row, min_col = np.unravel_index(np.nanargmax(dem), dem.shape)
source_x, source_y = X[min_row, min_col], Y[min_row, min_col]

# =======================================================
# 3. 物理仿真核心逻辑
# =======================================================
st.sidebar.markdown("---")
run_sim = st.sidebar.button("🚀 开始/更新 3D 仿真", type="primary")

if run_sim:
    st.toast("正在计算流体势能与摩擦阻力...", icon="⏳")

    # 初始化泥石流矩阵
    debris = np.zeros_like(dem)
    debris[max(0, min_row - 2):min_row + 3, max(0, min_col - 2):min_col + 3] = trigger_volume

    mu = np.tan(np.radians(friction_angle))
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    current_debris = debris.copy()

    # 执行元胞自动机演化
    for _ in range(time_step):
        surface = dem + current_debris
        new_debris = current_debris.copy()

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if current_debris[i, j] > 0.1:
                    h = surface[i, j]
                    flows = []
                    for di, dj in neighbors:
                        diff = h - surface[i + di, j + dj]
                        if diff > mu * 0.1:  # 克服屈服应力
                            flows.append((di, dj, diff))

                    if flows:
                        total_diff = sum(f[2] for f in flows)
                        for di, dj, diff in flows:
                            move = current_debris[i, j] * (diff / total_diff) * 0.4
                            new_debris[i + di, j + dj] += move
                            new_debris[i, j] -= move
        current_debris = np.clip(new_debris, 0, None)

    # =======================================================
    # 4. Plotly 3D 可视化绘制
    # =======================================================
    st.success("计算完成！请使用鼠标左键拖拽旋转，滚轮缩放。")

    # 将最终的泥石流厚度转为颜色，如果为0则是透明
    debris_plot = np.where(current_debris > 0.5, current_debris, np.nan)

    fig = go.Figure()

    # 图层1：3D 地形底座
    fig.add_trace(go.Surface(
        x=X, y=Y, z=dem,
        colorscale='Greys',
        opacity=0.8,
        name='地形DEM',
        showscale=False,
        hoverinfo='skip'
    ))

    # 图层2：3D 泥石流覆盖层
    fig.add_trace(go.Surface(
        x=X, y=Y, z=dem + debris_plot,  # 泥石流贴在表面
        colorscale='YlOrRd',  # 黄到红的警戒色
        opacity=0.9,
        name='泥石流',
        showscale=True,
        colorbar=dict(title="泥厚(m)")
    ))

    # 图层3：冰崩源区标记
    fig.add_trace(go.Scatter3d(
        x=[source_x], y=[source_y], z=[dem[min_row, min_col] + 500],  # 稍微抬高一点
        mode='markers+text',
        marker=dict(size=10, color='cyan', symbol='diamond'),
        text=["冰崩源区"],
        textposition="top center",
        name="冰崩点"
    ))

    # 布局设置
    fig.update_layout(
        scene=dict(
            xaxis_title='UTM X 坐标',
            yaxis_title='UTM Y 坐标',
            zaxis_title='高程 (m)',
            aspectratio=dict(x=1, y=1, z=0.5),  # 压扁一点更好看
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)  # 初始视角
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 请在左侧调节参数，然后点击【开始/更新 3D 仿真】按钮。")