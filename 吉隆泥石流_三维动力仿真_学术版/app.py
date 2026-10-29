# -*- coding: utf-8 -*-
"""吉隆泥石流三维动力学仿真 | 启动：python -m streamlit run app.py"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from q4_solver import Config, load_terrain, simulate, PORT
from academic_visualization import make_animation_html
from q4_viz import save_all

ROOT=Path(__file__).resolve().parent
DEM=ROOT/'data'/'elevation_utm45n.tif'
RESULTS=ROOT/'results'

st.set_page_config(page_title='吉隆泥石流三维动力学仿真',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
html,body,[class*="css"]{font-family:"Microsoft YaHei","PingFang SC",sans-serif}
.block-container{padding-top:1.35rem;padding-bottom:2.8rem;max-width:1660px}
h1{font-weight:650;letter-spacing:.3px;font-size:2rem}h2,h3{letter-spacing:.1px}
[data-testid="stMetric"]{background:#f7f8f9;border:1px solid #e0e5e8;border-radius:7px;padding:13px 17px}
[data-testid="stMetricLabel"]{color:#556574}
[data-testid="stSidebar"]{border-right:1px solid #dfe4e8}
</style>''',unsafe_allow_html=True)

st.title('吉隆口岸冰岩崩—泥石流三维动力学仿真')
st.caption('二维深度平均动力学  /  Voellmy摩擦阻力  /  侵蚀与沉积  /  三维地形可视化')

@st.cache_resource(show_spinner=False)
def terrain_data(grid):
    return load_terrain(DEM,Config(grid_m=float(grid)))

@st.cache_data(show_spinner=False)
def load_precomputed():
    folder=RESULTS
    report=json.loads((folder/'模型参数和结果.json').read_text('utf-8'))
    with np.load(folder/'基准情景逐帧.npz',allow_pickle=False) as z:
        frames=[a.copy() for a in z['frames']]
        times=z['times'].tolist()
        hmax=z['hmax'].copy();vmax=z['vmax'].copy()
    df=pd.read_csv(folder/'模拟过程时序.csv')
    return dict(cfg=Config(**report['模型参数假设']),frames=frames,times=times,
                hmax=hmax,vmax=vmax,records=df.to_dict('records'),precomputed=True,
                first_port_frame_s=report.get('首个满足口岸300米邻域的保存帧时间_s'))

st.sidebar.subheader('数值模拟控制台')
st.sidebar.caption('物源与动力学参数')
initial_volume=st.sidebar.number_input('初始物源体积  m³',min_value=1.e7,max_value=5.e8,
                                       value=1.75e8,step=5.e6,format='%.0f')
c1,c2=st.sidebar.columns(2)
with c1:
    mu=st.number_input('初期摩擦  μ₀',min_value=0.,max_value=.50,value=.060,step=.005,format='%.3f')
with c2:
    mu_late=st.number_input('后期摩擦  μ₁',min_value=0.,max_value=.50,value=.010,step=.005,format='%.3f')
xi=st.sidebar.number_input('Voellmy阻力参数  ξ',min_value=100.,max_value=12000.,value=5000.,step=100.,format='%.0f')
k=st.sidebar.number_input('经验侵蚀系数  k',min_value=0.,max_value=.02,value=.000678,step=.00002,format='%.6f')
fluid=st.sidebar.number_input('阻力转换时间  s',min_value=0,max_value=300,value=100,step=10)
with st.sidebar.expander('源区与侵蚀设置'):
    length=st.number_input('释放区长度  m',min_value=400.,max_value=4000.,value=2040.,step=100.)
    width=st.number_input('释放区宽度  m',min_value=400.,max_value=4000.,value=2040.,step=100.)
    vcrit=st.number_input('侵蚀临界速度  m/s',min_value=0.,max_value=60.,value=15.,step=1.)
    erocap=st.number_input('可侵蚀厚度  m',min_value=0.,max_value=100.,value=25.,step=5.)
    depos=st.number_input('沉积系数  s⁻¹',min_value=0.,max_value=.005,value=.0003,step=.0001,format='%.4f')
st.sidebar.divider()
st.sidebar.caption('时间与网格')
grid=st.sidebar.selectbox('计算网格  m',[90,60,45,120],index=0)
duration=st.sidebar.slider('计算时长  s',120,1200,600,60)
snapshot=st.sidebar.selectbox('过程存储间隔  s',[10,20,30,60],index=0)
stride=st.sidebar.select_slider('三维地形显示密度',options=['标准','精细'],value='标准')
run=st.sidebar.button('开始计算',type='primary',width='stretch')
st.sidebar.caption('调整参数后点击“开始计算”更新图表。')

cfg=Config(initial_volume_m3=float(initial_volume),source_length_m=float(length),source_width_m=float(width),
           grid_m=float(grid),duration_s=float(duration),snapshot_s=float(snapshot),
           mu=float(mu),mu_late=float(mu_late),xi_mps2=float(xi),erosion_k=float(k),
           fluidization_after_s=float(fluid),erosion_speed_threshold_mps=float(vcrit),
           erodible_depth_m=float(erocap),deposit_k_per_s=float(depos))
if run:
    with st.status('正在执行二维有限体积计算',expanded=True) as status:
        prog=st.progress(0.0,text='网格与初始条件')
        try:
            T=terrain_data(grid)
            result=simulate(T,cfg,progress=lambda f:prog.progress(min(f,1.),text=f'求解进度  {f:.0%}'),max_steps=15000)
        except Exception as e:
            status.update(label='计算未完成',state='error')
            st.error(f'{type(e).__name__}：{e}')
        else:
            st.session_state['current_result']=result
            status.update(label='计算完成',state='complete',expanded=False)
            prog.empty()

R=st.session_state.get('current_result')
if R is None:
    try:
        R=load_precomputed()
    except Exception as exc:
        st.error('尚未发现预计算结果，请运行 python run_batch.py。')
        st.code(str(exc))
        st.stop()
T=terrain_data(R['cfg'].grid_m)
last=R['records'][-1]
arrival=R.get('first_port_frame_s')
if arrival is None:
    hits=[a['时间_s'] for a in R['records'] if a['口岸300米邻域是否达到']]
    arrival=min(hits) if hits else None
cols=st.columns(5)
cols[0].metric('累计扩展距离',f'{last["最大累计直线距离_km"]:.2f} km')
cols[1].metric('最大速度',f'{last["全程峰值速度_mps"]:.1f} m/s')
cols[2].metric('口岸邻域进入时刻',f'{arrival:.0f} s' if arrival is not None else '—')
cols[3].metric('累计侵蚀体积',f'{last["累计侵蚀体积_m3"]/1e6:.1f} 百万m³')
cols[4].metric('累计影响面积',f'{last["累计影响面积_km2"]:.2f} km²')

view,process,plan,export=st.tabs(['三维仿真','动力学过程','流深空间分布','数据与导出'])

with view:
    html=make_animation_html(T,R,stride=2 if stride=='标准' else 1)
    if hasattr(st,'iframe'):
        st.iframe(html,height=815,width='stretch',alt='泥石流三维地形动态仿真')
    else:
        import streamlit.components.v1 as components
        components.html(html,height=815,scrolling=False)
    st.download_button('下载三维动画网页',data=html.encode('utf-8'),
                       file_name='吉隆泥石流_三维动画.html',mime='text/html')

with process:
    df=pd.DataFrame(R['records'])
    left,right=st.columns(2)
    def line_chart(series,title,ytitle):
        fig=go.Figure()
        for label,y in series:
            fig.add_trace(go.Scatter(x=df['时间_s'],y=y,mode='lines',name=label,line=dict(width=2.2)))
        fig.update_layout(title=dict(text=title,font=dict(size=15)),
            template='plotly_white',height=320,margin=dict(l=30,r=15,b=40,t=47),
            xaxis_title='模拟时间 / s',yaxis_title=ytitle,legend=dict(orientation='h',y=-.28))
        return fig
    with left:
        st.plotly_chart(line_chart([('流速',df['当前速度最大值_mps'])],
                      '流动速度时序','速度 / m·s⁻¹'),width='stretch')
        st.plotly_chart(line_chart([('侵蚀',df['累计侵蚀体积_m3']/1e6),
                                   ('沉积',df['累计沉积体积_m3']/1e6)],
                      '累计侵蚀与沉积','体积 / 百万 m³'),width='stretch')
    with right:
        st.plotly_chart(line_chart([('最大流深',df['当前流深最大值_m'])],
                      '流深变化','流深 / m'),width='stretch')
        st.plotly_chart(line_chart([('距离口岸',df['距口岸最近有效流动单元_m']/1000)],
                      '前锋空间位置','距离 / km'),width='stretch')
    with st.expander('查看质量收支时序'):
        st.plotly_chart(line_chart([('质量收支误差',df['物质量平衡相对误差']*100)],
                      '物质收支误差','误差 / %'),width='stretch')
    st.dataframe(df,width='stretch',hide_index=True)
    st.download_button('下载全过程CSV',df.to_csv(index=False).encode('utf-8-sig'),
                       file_name='模拟过程时序.csv',mime='text/csv')

with plan:
    selected=st.slider('模拟时刻',min_value=0,max_value=len(R['times'])-1,value=len(R['times'])//2)
    H=R['frames'][selected]
    wet=H>=R['cfg'].wet_threshold_m
    z=np.where(wet,H,np.nan)
    Z=np.where(T.valid,T.z0,np.nan)
    from matplotlib.colors import LightSource
    from scipy.ndimage import distance_transform_edt
    zfill=Z.copy();valid=np.isfinite(zfill)
    if not valid.all():
        ids=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        zfill[~valid]=zfill[tuple(ids)][~valid]
    hill=LightSource(azdeg=315,altdeg=50).hillshade(zfill,vert_exag=.6,dx=T.dx,dy=T.dy)
    fig=go.Figure(go.Heatmap(x=T.x,y=T.y,z=hill,colorscale='Greys',showscale=False,hoverinfo='skip'))
    fig.add_trace(go.Heatmap(x=T.x,y=T.y,z=z,colorscale='YlOrRd',opacity=.89,
                            colorbar=dict(title='流深 / m'),zmin=0,hovertemplate='%{z:.1f} m<extra>流深</extra>'))
    marks=[('冰崩点',(R['cfg'].source_x,R['cfg'].source_y),'#e5bb75'),
           ('吉隆口岸',PORT,'#4db7d5')]
    for name,coord,color in marks:
        fig.add_trace(go.Scatter(x=[coord[0]],y=[coord[1]],mode='markers+text',
                     text=[name],textposition='top center',name=name,
                     marker=dict(size=10,color=color,line=dict(width=1,color='black'))))
    fig.update_layout(template='plotly_white',height=650,
                      title=f'流深空间分布   t = {R["times"][selected]:.0f} s',
                      xaxis=dict(title='UTM 东坐标 / m'),
                      yaxis=dict(title='UTM 北坐标 / m',scaleanchor='x'),margin=dict(l=50,r=30,t=65,b=50))
    st.plotly_chart(fig,width='stretch')

with export:
    conf=R['cfg']
    rows=[('初始物源体积',f'{conf.initial_volume_m3:,.0f}','m³'),
          ('释放区长度',f'{conf.source_length_m:.0f}','m'),
          ('释放区宽度',f'{conf.source_width_m:.0f}','m'),
          ('初期摩擦系数 μ₀',f'{conf.mu:.4f}','—'),
          ('后期摩擦系数 μ₁',f'{conf.mu_late:.4f}','—'),
          ('速度阻力参数 ξ',f'{conf.xi_mps2:.0f}','m/s²'),
          ('侵蚀系数 k',f'{conf.erosion_k:.6f}','—'),
          ('网格分辨率',f'{conf.grid_m:.0f}','m'),
          ('模拟时长',f'{conf.duration_s:.0f}','s')]
    st.dataframe(pd.DataFrame(rows,columns=['参数','数值','单位']),hide_index=True,width='stretch')
    st.download_button('下载当前参数 JSON',json.dumps(conf.__dict__,ensure_ascii=False,indent=2).encode('utf-8'),
                       file_name='模型参数.json',mime='application/json')
    if st.button('保存本次计算成果',type='secondary'):
        if R.get('precomputed'):
            st.success('预计算成果已位于项目 results 文件夹。')
        else:
            folder=RESULTS/'最新计算'
            with st.spinner('生成栅格、图表与三维网页'):
                save_all(T,R,folder,html=True)
            st.success(f'已保存至 {folder}')
    st.caption('文件输出包括时间序列、三维动画、流深与速度GeoTIFF、侵蚀和沉积栅格。')
