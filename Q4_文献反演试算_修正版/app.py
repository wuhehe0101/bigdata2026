# -*- coding: utf-8 -*-
"""吉隆8·26第四问交互界面；运行: python -m streamlit run app.py"""
import io,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from q4_solver import Config,load_terrain,simulate,PAPER_REFERENCE,PORT,INCIDENT
from q4_viz import plotly_scene,save_all
ROOT=Path(__file__).resolve().parent
DEM=ROOT/'data'/'elevation_utm45n.tif'
RES=ROOT/'results'

st.set_page_config(page_title='吉隆8·26灾害链 | 第四问',page_icon='🏔️',layout='wide')
st.title('🏔️ 吉隆“8·26”冰岩崩—泥石流动力学仿真')
st.caption('文献约束 · 二维浅层质量/动量方程 · Voellmy阻力 · 侵蚀/沉积源汇 · UTM真实地形 · Plotly 3D时间动画')
st.warning('本项目给出90米计算网格下的**文献约束条件拟合候选**。μ、ξ、k不是论文直接实测值，60米网格下拟合结果发生明显变化，因此**不能称为真实灾害参数或已验证复现**。')

@st.cache_resource(show_spinner='载入真实DEM地形...')
def terrain_for(grid):
    return load_terrain(DEM,Config(grid_m=grid))

@st.cache_data(show_spinner='载入预先计算的基准情景...')
def precomputed_data(preset):
    folder=(ROOT/'calibration'/'拟合情景输出') if preset=='条件拟合候选（90米）' else RES
    datafile=folder/'基准情景逐帧.npz'
    reportfile=folder/'模型参数和结果.json'
    if not datafile.exists() or not reportfile.exists():return None
    report=json.loads(reportfile.read_text('utf-8'))
    with np.load(datafile,allow_pickle=False) as v:
        frames=[x.copy() for x in v['frames']]
        times=v['times'].tolist()
        hmax=v['hmax'].copy()
        vmax=v['vmax'].copy()
    records=pd.read_csv(folder/'模拟过程时序.csv').to_dict('records')
    return dict(cfg=Config(**report['模型参数假设']),frames=frames,times=times,
                hmax=hmax,vmax=vmax,records=records,precomputed=True)

st.sidebar.header('⚙️ 模型参数')
preset=st.sidebar.selectbox('参数方案',['条件拟合候选（90米）','上一版未经标定参数'],index=0)
is_cal=(preset=='条件拟合候选（90米）')
if st.session_state.get('prev_preset')!=preset:
    st.session_state.pop('simulation_result',None)
    st.session_state['prev_preset']=preset
st.sidebar.caption('条件拟合数值仅对指定DEM、90米网格、特定侵蚀公式和体积面积情景有效。')
st.sidebar.caption('源区定位是滑震反演点。不同论文的物源体积与释放面积配套使用；轮廓均为假设近似。')
vol_ref=st.sidebar.selectbox('文献物源规模情景',
    ['刘昌军DoD：1.75亿m³ / 约400万㎡','殷跃平：1.10亿m³ / 约60万㎡','刘昌军复盘：1.80亿m³ / 约400万㎡','杨云鹏面积＋殷体积：对照测试'],index=0)
volumes={
 '刘昌军DoD：1.75亿m³ / 约400万㎡':(1.75e8,2040,2040),
 '殷跃平：1.10亿m³ / 约60万㎡':(1.10e8,1000,600),
 '刘昌军复盘：1.80亿m³ / 约400万㎡':(1.80e8,2040,2040),
 '杨云鹏面积＋殷体积：对照测试':(1.10e8,748,1047)}
volume,source_length,source_width=volumes[vol_ref]
mu=st.sidebar.number_input('初期库仑摩擦系数 μ（非实测）',min_value=0.0,max_value=0.5,value=0.06 if is_cal else 0.20,step=0.005,format='%.3f',key='mu_'+preset)
xi=st.sidebar.number_input('速度阻力系数 ξ / m·s⁻²（非实测）',min_value=100.0,max_value=10000.0,value=5000.0 if is_cal else 1800.0,step=100.0,key='xi_'+preset)
late_on=st.sidebar.checkbox('采用后期摩擦参数变化情景（未验证）',value=True)
mu_late=st.sidebar.number_input('后期μ（非实测）',min_value=0.0,max_value=0.5,value=0.01 if is_cal else 0.05,step=0.005,format='%.3f',disabled=not late_on,key='mu_late_'+preset)
fluidize=st.sidebar.slider('切换参考时刻 / 秒',30,250,100,10,disabled=not late_on)
st.sidebar.markdown('**侵蚀与沉积（经验假设）**')
k=st.sidebar.number_input('经验侵蚀系数 k（条件拟合/非实测）',min_value=0.0,max_value=0.02,value=0.000678 if is_cal else 0.003,step=0.000010,format='%.6f',key='k_'+preset)
vc=st.sidebar.slider('临界侵蚀速度 / m/s',5,40,15,1)
cap=st.sidebar.slider('可侵蚀床层上限 / m',5,80,25,5)
dep=st.sidebar.slider('沉积系数 / s⁻¹',0.,.003,.0003,.0001,format='%.4f')
st.sidebar.markdown('**数值与显示参数**')
grid=st.sidebar.selectbox('计算网格 / m',[120,90,60,45],index=1)
duration=st.sidebar.slider('模拟时间 / s',120,1200,600,60)
snapshot=st.sidebar.selectbox('动画帧间隔 / s',[10,20,30,60],index=1)
stride=st.sidebar.selectbox('三维展示降采样步长',[1,2,3,4],index=1)
start=st.sidebar.button('🚀 根据当前参数重新计算',type='primary',width='stretch')

cfg=Config(grid_m=float(grid),duration_s=float(duration),snapshot_s=float(snapshot),
           initial_volume_m3=volume,source_length_m=source_length,source_width_m=source_width,
           mu=mu,xi_mps2=float(xi),fluidization_after_s=float(fluidize) if late_on else 0.,
           mu_late=mu_late,erosion_k=k,erosion_speed_threshold_mps=float(vc),
           erodible_depth_m=float(cap),deposit_k_per_s=dep)

if start:
    prog=st.progress(0,text='正在计算真实地形上的质量和动量变化……')
    try:
        T=terrain_for(grid)
        R=simulate(T,cfg,progress=lambda x:prog.progress(min(1.,x),text=f'数值求解：{x:.0%}'))
        st.session_state['simulation_result']=R
        st.session_state['simulation_grid']=grid
        st.success('模拟计算完成，参数和图表已更新。')
    except Exception as e:
        st.error(f'模拟失败：{type(e).__name__}: {e}')
    finally:prog.empty()

R=st.session_state.get('simulation_result')
if R is None:
    R=precomputed_data(preset)
    if R is not None:
        st.info('当前展示的是压缩包内已经实算的'+preset+'；改变参数后，请点击左侧“重新计算”。')
        T=terrain_for(R['cfg'].grid_m)
    else:
        T=None
else:
    T=terrain_for(st.session_state['simulation_grid'])
if R is None:
    st.error('尚无预计算结果。请运行 `python run_batch.py --sensitivity`，或点击“重新计算”。')
    st.stop()

r=R['records'][-1]
col=st.columns(5)
col[0].metric('最远直线扩展',f'{r["最大累计直线距离_km"]:.2f} km')
col[1].metric('全域峰值流速',f'{r["全程峰值速度_mps"]:.1f} m/s')
col[2].metric('距口岸最近',f'{r["距口岸最近有效流动单元_m"]/1000:.2f} km')
col[3].metric('累计侵蚀',f'{r["累计侵蚀体积_m3"]/1e6:.1f} 百万m³')
col[4].metric('口岸300米邻域', '已到达' if r['口岸300米邻域是否达到'] else '未到达')

t1,t2,t3,t4=st.tabs(['🌐 三维动态网页','📊 计算过程','🗺️ 逐帧分布','📚 论文与参数比较'])
with t1:
    st.plotly_chart(plotly_scene(T,R,stride),width='stretch',config={'displaylogo':False})
    st.caption('三维图层以散点表示计算网格上的流深，标高向上偏移10m是显示处理；地形与XY采用真实EPSG:32645坐标。')
    st.download_button('下载本次仿真三维动画HTML',
        data=plotly_scene(T,R,stride).to_html(include_plotlyjs=True,full_html=True).encode('utf-8'),
        file_name='三维运动动画.html',mime='text/html')
with t2:
    df=pd.DataFrame(R['records'])
    st.dataframe(df,width='stretch',hide_index=True)
    a,b=st.columns(2)
    with a:
        st.line_chart(df.set_index('时间_s')[['距口岸最近有效流动单元_m']],width='stretch')
        st.line_chart(df.set_index('时间_s')[['当前速度最大值_mps','当前流深最大值_m']],width='stretch')
    with b:
        st.line_chart(df.set_index('时间_s')[['累计侵蚀体积_m3','累计沉积体积_m3']],width='stretch')
        st.line_chart(df.set_index('时间_s')[['物质量平衡相对误差']],width='stretch')
    st.download_button('下载本次模拟逐时刻CSV',df.to_csv(index=False).encode('utf-8-sig'),file_name='模拟过程时序.csv',mime='text/csv')
    if R.get('precomputed'):
        st.caption('这是ZIP附带的预计算过程数据。')
    else:
        if st.button('保存本次互动计算结果到 results/交互最新情景',width='stretch'):
            folder=RES/'交互最新情景'
            save_all(T,R,folder,html=True)
            st.success(f'已保存：{folder}')
with t3:
    i=st.slider('选择模拟时刻',0,len(R['frames'])-1,len(R['frames'])//2)
    h=R['frames'][i]
    z=np.where(T.valid & (h>=R['cfg'].wet_threshold_m),h,np.nan)
    fg=go.Figure(go.Heatmap(z=z,x=T.x,y=T.y,colorscale='YlOrRd',colorbar=dict(title='模拟流深/m')))
    for name,xy,color in [('参考源区',PAPER_REFERENCE,'cyan'),('口岸',PORT,'blue'),('灾害参考点',INCIDENT,'red')]:
        fg.add_trace(go.Scatter(x=[xy[0]],y=[xy[1]],mode='markers+text',text=[name],
                               textposition='top center',marker=dict(size=10,color=color,line=dict(color='black',width=1)),name=name))
    fg.update_layout(title=f'{R["times"][i]:.0f}s 模拟流深（EPSG:32645）',height=650,
                     xaxis=dict(title='UTM 东坐标 / m'),yaxis=dict(title='UTM 北坐标 / m',scaleanchor='x'))
    st.plotly_chart(fg,width='stretch')
with t4:
    st.markdown('''**来源分类**：
    - 殷跃平等（2026）：源区滑震初步定位85°33′00″E、28°20′09″N，物源约1.1亿m³，尺寸约1000×600m，两河段侵蚀约5860万m³。
    - 杨云鹏等（2026）：源区约748×1047m、75.68万m²；约130秒到交汇口，再约320秒至口岸；平均传播速度约63.6m/s与46m/s。
    - 刘昌军等（2026）：DoD约1.75亿m³、另一情景约1.8亿m³；源区面积约400万㎡、平均厚度约45m；全程侵蚀2.60亿m³，口岸到达约505秒为该论文**数值模拟输出**。
    - **未验证假设**：Voellmy μ、ξ、侵蚀k、临界速度、可侵蚀厚度、分阶段切换时刻。不同论文侵蚀量范围不同，不能直接混为一个反演目标。
    ''')
    cfile=RES/'文献约束与计算对照.csv';sfile=RES/'参数情景敏感性.csv'
    if cfile.exists():
        st.subheader('文献参考量与本模型结果')
        st.dataframe(pd.read_csv(cfile),width='stretch',hide_index=True)
    if sfile.exists():
        st.subheader('参数情景对照')
        st.dataframe(pd.read_csv(sfile),width='stretch',hide_index=True)
    st.info('条件拟合候选仅匹配约450秒到达与约5860万m³侵蚀的两个代理量；60米网格到达时间比90米提早40秒。仍缺少河段掩膜、交汇口坐标和遥感真实边界，无法唯一辨识真实μ、ξ、k。')
    cdir=ROOT/'calibration'
    for f in ('参数试算结果.csv','追加试算.csv','联合参数试算.csv','细化与网格核查.csv'):
        if (cdir/f).exists():
            st.subheader('文献约束试算：'+f)
            st.dataframe(pd.read_csv(cdir/f),width='stretch',hide_index=True)
    if (cdir/'参数反演核查报告.md').exists():
        st.markdown((cdir/'参数反演核查报告.md').read_text('utf-8'))
