# 吉隆“8·26”第四问：论文约束下的二维动力学与三维模拟

## 0. 先看这里：结果与可信度

本项目由用户原有 `Q4_文献约束三维灾害链仿真.zip` 修改而来，使用用户新提供的 `第四问.zip` 地形、灾害点、口岸和2023年土地覆盖数据。包括Python数值求解、Streamlit可调参数界面、Plotly可播放三维动画、离线综合网页、中文论文图、CSV与GeoTIFF。

**重要：这是文献约束的数值情景探索，不是2026年8月26日灾害已经通过实测验证的动力学复现。** 目前在全部已完成的600秒情景中，流动前缘均未进入吉隆口岸300米邻域。不得修改动画或结果来宣称已到达。Voellmy库仑摩擦μ、速度阻力ξ、侵蚀k没有本次灾害的独立实测或可靠反演值；旧程序给出的值继续作为待验证的基准假设。

## 1. 项目文件结构

```text
Q4_吉隆826_论文约束与参数筛选_升级版/
├─ app.py                    # Streamlit实时计算+三维动画
├─ q4_solver.py              # 2D深度平均守恒方程、Voellmy、侵蚀沉积
├─ q4_viz.py                 # Plotly动画、中文图、GeoTIFF与过程网页
├─ run_batch.py              # 一键数值运行及论文结果批量导出
├─ make_portal.py            # 离线结果门户index.html
├─ check_input_data.py       # 原始SHP与土地覆盖校核
├─ 一键启动交互网页.bat
├─ 一键重新计算和生成论文结果.bat
├─ index.html                # 已生成、免服务器、双击打开的综合结果网页
├─ data/
│  ├─ elevation_utm45n.tif  # EPSG:32645、30m DEM
│  ├─ jilong_LUCD_2023.tif  # 2023年土地覆盖（不用于摩擦估算）
│  ├─ locations.json        # 来源于实际SHP的口岸、灾害点坐标
│  └─ shapefiles/           # 用户原始两套点位SHP及配套文件
├─ results/
│  ├─ 三维运动动画_离线网页.html
│  ├─ 计算过程曲线_离线网页.html
│  ├─ 基准情景逐帧.npz      # 供Streamlit初次进入立即显示已算好动画
│  ├─ 模拟过程时序.csv
│  ├─ 参数情景敏感性.csv
│  ├─ 文献约束与计算对照.csv
│  ├─ 模型参数和结果.json
│  ├─ 数据集核查.json
│  ├─ 图01_...png 到 图07_...png
│  └─ *.tif                 # 流深/流速/侵蚀/沉积网格
├─ 模型计算过程与数学说明.md
├─ 论文结果简报.md
├─ requirements.txt
└─ tests/test_solver.py
```

## 2. 先看已生成的网页（不需Python）

解压整个ZIP文件，**保持文件夹结构不要更改**，双击项目根目录的 `index.html`：即可查看已经计算的3D地形动画、20秒一帧的运动过程、质量守恒曲线、参数情景对照和中文论文插图。离线网页不需要运行任何服务器，不提供再次计算按钮；如需调参计算，请打开Streamlit版。

注意：离线HTML里面的iframe相对路径指向 `results/三维运动动画_离线网页.html` 与 `results/计算过程曲线_离线网页.html`，所以**只拷走index.html会导致显示不全**。若浏览器安全策略不允许本地iframe，直接打开 `results` 目录下的这两个独立HTML文件即可。

## 3. Windows / PyCharm 启动Streamlit

将本文件夹解压至 `D:\bigdadta\Q4_吉隆826_论文约束与参数筛选_升级版`（也可以选择其他位置）。

1. 打开PyCharm底部 `Terminal`（终端）。
2. 若你使用 `D:\bigdadta\.venv`：

```powershell
& "D:\bigdadta\.venv\Scripts\python.exe" -m pip install -r "D:\bigdadta\Q4_吉隆826_论文约束与参数筛选_升级版\requirements.txt"
& "D:\bigdadta\.venv\Scripts\python.exe" -m streamlit run "D:\bigdadta\Q4_吉隆826_论文约束与参数筛选_升级版\app.py"
```

如果你使用conda环境 `math_model`：

```bat
conda activate math_model
cd /d "D:\bigdadta\Q4_吉隆826_论文约束与参数筛选_升级版"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

3. 浏览器打开终端打印的 `http://localhost:8501`。

**不要在PyCharm中直接点击绿色三角执行 `python app.py`**，否则会不断出现 `missing ScriptRunContext` 警告，且网页不会运行。建议直接双击 `一键启动交互网页.bat`，或者在PyCharm的Run Configuration选择 `Module name=streamlit`、参数 `run app.py`。

## 4. 重新计算并生成论文结果

先确保依赖安装完成，在项目目录执行：

```powershell
python -m unittest discover -s tests -v
python run_batch.py --sensitivity
```

批量运行会重新生成默认1.75亿m³基准、殷文1.1亿m³+约60万m²、刘文1.8亿m³+约400万m²，共3个文献规模情景。会重新覆盖 `results` 同名的论文成果，并更新网页。**计算时间由CPU、NUMBA首次编译、网格尺度和参数影响，不承诺每次相同。**

只算一次默认基准情景：

```powershell
python run_batch.py
```

修改命令行数值示例（此时源区几何默认为刘文大面积近似，若更换不同论文的体积，建议使用Streamlit相应成对情景，或在代码中同时更换释放区长度、宽度）：

```powershell
python run_batch.py --grid 90 --duration 600 --volume 175000000 --mu 0.20 --late-mu 0.05 --xi 1800 --erosion-k 0.003
```

Streamlit网页中的参数修改后必须点击「根据当前参数重新计算」，仅拖动滑块不会自动刷新原计算结果；可以通过计算过程页下载新的CSV与HTML。界面支持将当前结果保存至 `results/交互最新情景/`，不会覆盖预置的三组基准对照。

## 5. 已使用的论文证据与冲突

| 论文 | 可用于模型的证据 | 性质 |
|---|---|---|
| 殷跃平等（2026），DOI:10.16031/j.cghc.202610008 | 滑震初步定位85°33′00″E/28°20′09″N、失稳体积约1.1亿m³、1000m×600m尺度、两河段侵蚀约5860万m³ | 观测/反演，存在测量范围差异 |
| 杨云鹏等（2026），DOI:10.16089/j.cnki.1008-2786.000980 | 源区748m×1047m、75.68万m²，阶段持续约130s和320s，口岸到达约450s，前后阶段平均传播速度63.6/46m/s | 地震监测与现场/视频反演 |
| 刘昌军等（2026），DOI:10.3724/j.slxb.20260653 | 物源DoD约1.75亿m³、源区约400万m²、平均厚度约45m，数值计算口岸到达约505s、全程侵蚀约2.60亿m³ | 物源DoD+另一套数值复盘；使用**宾汉**阻力，不提供我们的Voellmy参数 |

**重要：文献中的源区面积与体积不是任意交叉配对的。** 默认基准取刘文DoD `1.75e8m³`与其约`4.0e6m²`，用一个**2040m×2040m超椭圆近似**，格网面积约4.026km²，初始流深约43.47m，属于与论文45m厚度匹配的输入构造。殷文情景则配对 `1.1e8m³`、`1000m×600m`近似，不把刘文体积灌入小源区。杨文提供的是不同的释放区轮廓估计，但没有本程序可直接采用的相同口径体积；Streamlit中额外提供“杨面积+殷体积测试”，**仅用于交叉假设试验**。

**坐标冲突：** 引用的滑震初步定位在用户30米DEM采样值约5430.6m，而杨文中描述的源区海拔约5152m，相差约279m；这不是一处精确释放点。不能仅凭高程差自动平移源区来“修正”。真实精确释放多边形需要灾前后高分辨率卫星资料。

## 6. 物理方程及模型中仍是假设的参数

- 状态变量 `h,hu,hv`，满足二维深度平均质量守恒/动量守恒；用静水重构的Rusanov有限体积法、CFL自适应时间步，重力与地形坡度在通量重构中体现。
- 阻力为Coulomb–Voellmy等效衰减；**初期μ=0.20、100s以后μ=0.05、ξ=1800 m/s²均为未标定的旧模型情景值**。
- 侵蚀：`E=k×max(v−v_c,0)×min(1,h/2)`，在格网内的最大侵蚀深度上限25m，初始`k=0.003`、`v_c=15m/s`。该k是**无量纲的本程序经验公式系数**，不是“Voellmy固有侵蚀常数”，不等于其他论文使用不同侵蚀公式中的k。
- 沉积：低速(<5m/s)时按`D=k_dep*h`，`k_dep=0.0003s^-1`，也是经验假设。
- 质量守恒检验：`流动体积+累计沉积体积−累计侵蚀体积−初始体积≈0`。此处是**体积收支**，假设密度恒定；未建模多相密度变化、冰水交换与冲击飞跃。
- 物理速度为深度平均二维速度，不等于论文测得的泥石流前缘传播平均速度；模拟最大直线扩展距离也不等于沿河道长度。

**不能声称 μ、ξ、k已自动反演完成。** 只有口岸位置单点、粗DEM和不同范围的侵蚀总量，没有精确交汇口坐标、源区边界和相同河段的侵蚀/堆积DEM，无法独立确定这三个参数。新项目提供了实算情景对照，后续可新增具有真实地理坐标的监测断面和侵蚀空间掩膜再标定。

## 7. 数据集实际核查

`check_input_data.py` 读取用户提供的SHP点位，输出 `data/locations.json` 和 `results/数据集核查.json`。确认：

- EPSG:32645原始UTM DEM，30米像元；基准计算重采样为90米，三维显示再独立降采样。
- 吉隆口岸UTM约 `(340897.896,3129236.913)`，灾害点约 `(340873.071,3129127.391)`。
- 2023土地覆盖EPSG:4326，NoData=0；**源区参考点、口岸和灾害点均采样为0（无数据）**。因此土地覆盖栅格可以用于展示覆盖缺口，但**不能用其分类编码自动给μ、ξ、k赋值**。
- 输入DEM是否为灾前地形尚待确认；若属于灾后DEM，不满足真实启动地形复现要求。

## 8. 已完成的600秒基准计算结果

由 `run_batch.py --sensitivity` 实算，默认为刘文1.75亿m³+约400万m²面积，90米网格：

- 初始体积 1.75亿m³；初始释放面积约4.026km²；平均初始厚度约43.47m。
- 600s最大直线扩展距离约14.61km；**距离口岸最近的有效运动单元仍约3.82km**；未满足口岸300m邻域流深≥0.5m的到达条件。
- 模拟全域历史最大格网速度约58.37m/s；累计经验侵蚀约5690万m³、累计沉积约1362万m³；覆盖累计约26.20km²。
- 最终流动层体积约2.183亿m³；体积收支相对误差约−4.23×10⁻⁹（无量纲）。

注意：两篇论文中的侵蚀约5860万m³与2.60亿m³**不是同一空间范围**，上面的5690万m³为我们整块裁剪计算域内的模拟值，不能说与5860万“验证吻合”。

## 9. 常见问题

1. **`missing ScriptRunContext` / 程序运行完自动退出**：启动方式不正确，请执行 `python -m streamlit run app.py`，不要 `python app.py`。
2. **`ModuleNotFoundError: streamlit/numba/rasterio`**：先在正确的解释器中安装 `requirements.txt`。
3. **模型到不了口岸**：当前参数/源区/地形/数值分辨率下的真实计算结论；不是网页故障。先核对DEM和源区、传播沟道，再进行μ、ξ、k反演，不要人为拉伸动画。
4. **30米格网无法跑**：现有求解器在灾害规模下尚未验证30m的稳定性；默认90m，提供45m选项并限制最大迭代次数，超过上限会报告失败。
5. **网页打不开**：若直接打开离线 `index.html` 无法显示子框架，试着双击 `results/三维运动动画_离线网页.html`；实时计算需要本机 `http://localhost:8501`。
6. **下载的是网页而不是可修改的网页源码？** 本项目包含 `app.py`、`q4_viz.py`、`index.html`、`make_portal.py` 全部源文件，均可编辑。

## 10. 验证状态

已经实测运行Python求解器、数值测试、600秒基准情景和论文体积对照，并生成CSV、PNG、GeoTIFF、Plotly HTML。**当前交付运行环境未安装Streamlit且没有网络安装权限，因此未能在这里启动Streamlit服务器作浏览器端到端测试；app.py通过Python语法检查，Windows上请安装依赖再启动**。


## 新增：文献约束条件拟合（2026-10-29）

本项目现提供`calibration/`下的实算筛选记录、条件拟合情景的独立动画和论文图、网格检验以及`run_literature_inversion.py`反演试算脚本。**默认拟合候选仅在90m网格的代理目标下相符，不是已验证的真实参数。**详见[`calibration/参数反演核查报告.md`](calibration/参数反演核查报告.md)。

在虚拟环境运行：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
python run_literature_inversion.py --mode quick
```


## 新增：文献约束条件拟合（2026-10-29）

本项目现提供`calibration/`下的实算筛选记录、条件拟合情景的独立动画和论文图、网格检验以及`run_literature_inversion.py`反演试算脚本。**默认拟合候选仅在90m网格的代理目标下相符，不是已验证的真实参数。**详见[`calibration/参数反演核查报告.md`](calibration/参数反演核查报告.md)。

在虚拟环境运行：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
python run_literature_inversion.py --mode quick
```
