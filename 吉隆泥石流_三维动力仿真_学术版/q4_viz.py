# -*- coding: utf-8 -*-
"""输出中文论文图、可离线运行的Plotly网页、CSV和地理栅格。"""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import rasterio
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from q4_solver import PAPER_REFERENCE,PORT,OBSERVATIONS


def font_setup():
    from matplotlib import font_manager
    names={p.name for p in font_manager.fontManager.ttflist}
    for n in ('Noto Sans CJK SC','Noto Sans CJK JP','Microsoft YaHei','SimHei','WenQuanYi Micro Hei'):
        if n in names:
            plt.rcParams['font.sans-serif']=[n,'DejaVu Sans'];break
    plt.rcParams['axes.unicode_minus']=False


def height_at(T,xy):
    c=np.argmin(abs(T.x-xy[0]));r=np.argmin(abs(T.y-xy[1]))
    return float(T.z0[r,c])


def plotly_scene(T,R,stride=2):
    """用于兼容旧代码；交互播放使用academic_visualization模块。"""
    from academic_visualization import make_figure
    return make_figure(T,R,stride)[0]


def bounds(T):
    return (T.x[0]-T.dx/2,T.x[-1]+T.dx/2,T.y[-1]-T.dy/2,T.y[0]+T.dy/2)

def shade(T):
    from scipy.ndimage import distance_transform_edt
    Z=T.z0.copy()
    if not np.all(T.valid):
        ind=distance_transform_edt(~T.valid,return_distances=False,return_indices=True)
        Z[~T.valid]=Z[tuple(ind)][~T.valid]
    s=LightSource(azdeg=315,altdeg=45).hillshade(Z,vert_exag=0.4,dx=T.dx,dy=T.dy)
    return np.where(T.valid,s,np.nan)


def paper_figures(T,R,out,scenario_rows=None):
    font_setup();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    extent=bounds(T);sh=shade(T);cfg=R['cfg'];
    source=(cfg.source_x,cfg.source_y)
    def base(ax,label=False):
        ax.imshow(sh,cmap='gray',extent=extent,origin='upper',vmin=0,vmax=1)
        ax.scatter(*source,c='#00ffff',edgecolor='black',s=45,zorder=5,label='冰崩点')
        ax.scatter(*PORT,c='yellow',edgecolor='black',s=48,marker='^',zorder=5,label='吉隆口岸')
        ax.ticklabel_format(style='plain',useOffset=False)
        ax.set_xlabel('UTM东坐标 / m');ax.set_ylabel('UTM北坐标 / m')
        if label:ax.legend(fontsize=8,loc='upper left')
    fig,ax=plt.subplots(figsize=(11,7),constrained_layout=True)
    base(ax,True)
    ax.legend(fontsize=9);ax.set_title('图1 冰崩点及吉隆口岸地形分布')
    fig.savefig(out/'图01_真实地形与文献源区.png',dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,6),constrained_layout=True)
    base(ax)
    d=R['frames'][0]
    img=ax.imshow(np.ma.masked_where(d<cfg.wet_threshold_m,d),cmap='YlOrRd',extent=extent,origin='upper')
    ax.set_xlim(source[0]-1400,source[0]+1400);ax.set_ylim(source[1]-1400,source[1]+1400)
    ax.set_title('图2 初始释放区域与流深')
    fig.colorbar(img,ax=ax,label='流深 / m',shrink=.80)
    fig.savefig(out/'图02_初始源区与释放厚度.png',dpi=165);plt.close(fig)
    idx=np.unique(np.round(np.linspace(0,len(R['times'])-1,6)).astype(int))
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    maxval=float(np.percentile(R['hmax'][R['hmax']>cfg.wet_threshold_m],99)) if np.any(R['hmax']>cfg.wet_threshold_m) else 1.
    for ax,i in zip(axes.flat,idx):
        base(ax)
        H=R['frames'][i]
        img=ax.imshow(np.ma.masked_where(H<cfg.wet_threshold_m,H),extent=extent,origin='upper',
                     cmap='YlOrRd',vmin=0,vmax=maxval)
        ax.set_title(f't = {R["times"][i]:.0f} s');ax.tick_params(axis='both',labelsize=7)
    fig.colorbar(img,ax=axes.ravel().tolist(),fraction=.015,pad=.005,label='流深 / m')
    fig.suptitle('图3 六时刻流深演化',fontsize=16)
    fig.savefig(out/'图03_六时刻流深演化.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    for ax in axes:base(ax)
    img=axes[0].imshow(np.ma.masked_where(R['hmax']<cfg.wet_threshold_m,R['hmax']),extent=extent,origin='upper',cmap='YlOrRd')
    fig.colorbar(img,ax=axes[0],shrink=.7,label='最大流深 / m')
    axes[0].set_title('图4(a) 最大流深')
    img=axes[1].imshow(np.ma.masked_where(R['hmax']<cfg.wet_threshold_m,R['vmax']),extent=extent,origin='upper',cmap='viridis')
    fig.colorbar(img,ax=axes[1],shrink=.7,label='最大速度 / (m/s)')
    axes[1].set_title('图4(b) 最大流速')
    fig.savefig(out/'图04_峰值流深与流速.png',dpi=155);plt.close(fig)
    df=pd.DataFrame(R['records'])
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axs[0,0].plot(df['时间_s'],df['最大累计直线距离_km']);axs[0,0].set_title('直线扩展距离（非沿沟道距离）')
    axs[0,0].set_ylabel('最远直线距离 / km')
    axs[0,1].plot(df['时间_s'],df['累计影响面积_km2']);axs[0,1].set_ylabel('累计流动覆盖面积 / km²')
    axs[1,0].plot(df['时间_s'],df['累计侵蚀体积_m3']/1e6,label='模拟全域累计侵蚀')
    axs[1,0].set_ylabel('累计侵蚀 / 百万m³');axs[1,0].legend(fontsize=8)
    axs[1,1].plot(df['时间_s'],df['物质量平衡相对误差']*100)
    axs[1,1].set_ylabel('(流动+沉积−侵蚀−初始)/初始 / %')
    for a in axs.flat:a.set_xlabel('模拟时间 / s');a.grid(alpha=.25)
    fig.suptitle('图5 动态指标与含侵蚀沉积的质量平衡检查',fontsize=16)
    fig.savefig(out/'图05_时序与质量守恒.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    for a in axes:base(a)
    E=np.where(R['eroded']>0.05,R['eroded'],np.nan)
    D=np.where(R['deposited']>0.05,R['deposited'],np.nan)
    e=axes[0].imshow(E,extent=extent,origin='upper',cmap='Reds');axes[0].set_title('图6(a) 侵蚀深度');fig.colorbar(e,ax=axes[0],shrink=.7,label='m')
    d=axes[1].imshow(D,extent=extent,origin='upper',cmap='Blues');axes[1].set_title('图6(b) 沉积深度');fig.colorbar(d,ax=axes[1],shrink=.7,label='m')
    fig.savefig(out/'图06_侵蚀与沉积分布.png',dpi=155);plt.close(fig)
    if scenario_rows:
        s=pd.DataFrame(scenario_rows)
        fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        axs[0].bar(s['情景名称'],s['最大直线距离_km']);axs[0].set_ylabel('最大直线距离/km')
        axs[1].bar(s['情景名称'],s['累计侵蚀_百万m3'])
        axs[1].set_ylabel('累计侵蚀/百万m³')
        for a in axs:a.tick_params(axis='x',rotation=24);a.grid(axis='y',alpha=.2)
        fig.suptitle('图7 参数敏感性分析')
        fig.savefig(out/'图07_参数敏感性.png',dpi=160);plt.close(fig)


def write_tif(T,array,path,nodata=-9999):
    Z=np.where(T.valid,array,nodata).astype('float32')
    with rasterio.open(path,'w',driver='GTiff',height=Z.shape[0],width=Z.shape[1],count=1,
        dtype='float32',crs='EPSG:32645',transform=T.transform,compress='deflate',nodata=nodata) as ds:
        ds.write(Z,1)


def save_all(T,R,out,scenarios=None,html=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=R['cfg'];last=R['records'][-1]
    pd.DataFrame(R['records']).to_csv(out/'模拟过程时序.csv',index=False,encoding='utf-8-sig')
    if scenarios:
        pd.DataFrame(scenarios).to_csv(out/'参数情景敏感性.csv',index=False,encoding='utf-8-sig')
    for name,data in [('最大流深_m.tif',R['hmax']),('最大流速_mps.tif',R['vmax']),
        ('最终流深_m.tif',R['h']),('侵蚀深度_m.tif',R['eroded']),('沉积深度_m.tif',R['deposited'])]:
        write_tif(T,data,out/name)
    center_idx=(int(np.argmin(abs(T.y-cfg.source_y))),int(np.argmin(abs(T.x-cfg.source_x))))
    source_dem=float(T.z0[center_idx])
    summary={'情景性质':'文献数据约束的未验证数值情景，并非2026灾害真实重建',
      '引用论文':['殷跃平等(2026), DOI:10.16031/j.cghc.202610008','刘昌军等(2026), DOI:10.3724/j.slxb.20260653','杨云鹏等(2026), DOI:10.16089/j.cnki.1008-2786.000980'],
      '参考观测':OBSERVATIONS,'模型参数假设':asdict(cfg),
      'DEM源区参考点高程_m':source_dem,
      '杨云鹏论文源区文字海拔_m':5152,
      '初始格网平均厚度_m':float(R['frames'][0].max()),
      '初始释放区面积_km2':float(np.sum(R['source_mask'])*T.dx*T.dy/1e6),
      '边界受影响像元数':R['edge_wet_cells'],
      '首个满足口岸300米邻域的保存帧时间_s':R['first_port_frame_s'],
      '首个满足灾害点300米邻域的保存帧时间_s':R['first_incident_frame_s'],
      '数值时间步数':R['steps'],'最终结果':last,
      '主要限制':['源区超椭圆依据论文尺寸构造，非遥感解译边界','DEM日期未验证是灾前','摩擦和侵蚀参数为情景假设，不是论文测量值','流动层计算未描述冰岩崩块体飞行、冲击爬高及冰水相变','直线扩展距离不等于文献弯曲沟道23km','域外边界为封闭边界，若边缘有流动则相应距离不可信','到达时间依保存帧识别，精度受snapshot_s约束']}
    (out/'模型参数和结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    paper_figures(T,R,out,scenarios)
    if html:
        from academic_visualization import save_animation_html
        save_animation_html(T,R,out/'三维运动动画_离线网页.html',stride=2)
    np.savez_compressed(out/'基准情景逐帧.npz',
        times=np.asarray(R['times'],dtype=np.float64),
        frames=np.stack(R['frames']).astype(np.float32),
        hmax=R['hmax'].astype(np.float32),vmax=R['vmax'].astype(np.float32))
    return summary


def process_dashboard(R,out,scenarios):
    """生成独立的可交互计算进度网页，包含全过程时序和质量收支。"""
    from plotly.subplots import make_subplots
    out=Path(out);df=pd.DataFrame(R['records'])
    fig=make_subplots(rows=2,cols=2,subplot_titles=(
       '至吉隆口岸最近有效流动单元距离','累计侵蚀与沉积体积',
       '当前最大流速与最大流深','守恒误差（%）'))
    t=df['时间_s']
    fig.add_trace(go.Scatter(x=t,y=df['距口岸最近有效流动单元_m']/1000,mode='lines+markers',name='距口岸最近距离 / km'),row=1,col=1)
    fig.add_trace(go.Scatter(x=t,y=df['累计侵蚀体积_m3']/1e6,name='累计侵蚀 / 百万m³'),row=1,col=2)
    fig.add_trace(go.Scatter(x=t,y=df['累计沉积体积_m3']/1e6,name='累计沉积 / 百万m³'),row=1,col=2)
    fig.add_trace(go.Scatter(x=t,y=df['当前速度最大值_mps'],name='当前最大速度 m/s'),row=2,col=1)
    fig.add_trace(go.Scatter(x=t,y=df['当前流深最大值_m'],name='当前最大流深 m'),row=2,col=1)
    fig.add_trace(go.Scatter(x=t,y=df['物质量平衡相对误差']*100,name='物质守恒相对误差 %'),row=2,col=2)
    fig.update_layout(height=710,title='泥石流动力学计算全过程',
                      template='plotly_white',hovermode='x unified',legend=dict(orientation='h',y=-.13),
                      margin=dict(l=60,r=30,t=75,b=65))
    fig.update_xaxes(title_text='模拟时间 / s')
    fig.write_html(out/'计算过程曲线_离线网页.html',include_plotlyjs=True,full_html=True,auto_play=False)
