# -*- coding: utf-8 -*-
"""三维地形与泥石流运动层；前端只更新流动表面，保持相机交互状态。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from q4_solver import PORT

TERRAIN_COLORS = [
    [0.00, '#22483d'], [0.11, '#3f5944'], [0.22, '#657155'],
    [0.34, '#827f66'], [0.46, '#9a8a70'], [0.58, '#a79a86'],
    [0.70, '#9b948d'], [0.80, '#b4b0a8'], [0.90, '#d1d0c9'],
    [0.97, '#f0f0e8'], [1.00, '#fdfdf9'],
]
FLOW_COLORS = [
    [0.00, '#F6C568'],
    [0.15, '#ED9B42'],
    [0.35, '#D96C29'],
    [0.60, '#B94A20'],
    [0.80, '#8F2F20'],
    [1.00, '#641B1B'],
]

def _surface_data(T, R, stride=2):
    step=max(1,int(stride))
    xx=T.x[::step].astype(float)
    yy=T.y[::step].astype(float)
    zz=np.where(T.valid[::step,::step],T.z0[::step,::step],np.nan)
    valid= np.isfinite(zz)
    hx=np.nanpercentile(zz, [1,99.7])
    # 经纬区域已在米制UTM坐标，使用轻度坡度光照调制增强谷地阴影
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    zfix=zz.copy()
    if np.any(~valid):
        ids=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        zfix[~valid]=zz[tuple(ids)][~valid]
    sm=gaussian_filter(zfix, 0.60)
    grad_y,grad_x=np.gradient(sm,abs(T.dy)*step,T.dx*step)
    azimuth=np.deg2rad(320.0)
    slope=np.arctan(np.hypot(grad_x,grad_y))
    aspect=np.arctan2(grad_y,grad_x)
    light=np.clip(0.75*np.cos(slope)+0.65*np.sin(slope)*np.cos(aspect-azimuth),0.15,1.)
    surfacecolor = np.where(valid, zz + (light-.62)*350, np.nan)
    flow_frames=[h[::step,::step] for h in R['frames']]
    visible=np.concatenate([h[(h>=R['cfg'].wet_threshold_m)] for h in flow_frames])
    vhigh=float(np.percentile(visible,99.0)) if visible.size else 1.
    return xx,yy,zz,surfacecolor,flow_frames,max(3.,vhigh)


def _terrain_at(T,pt):
    i=int(np.argmin(abs(T.y-pt[1])));j=int(np.argmin(abs(T.x-pt[0])))
    return float(T.z0[i,j])


def make_figure(T,R,stride=2):
    x,y,z,shade,frames,limit=_surface_data(T,R,stride)
    xspan=float(x.max()-x.min());yspan=float(y.max()-y.min());zspan=float(np.nanmax(z)-np.nanmin(z))
    X,Y=np.meshgrid(x,y)
    wet0=np.where(frames[0]>=R['cfg'].wet_threshold_m,frames[0],np.nan)
    flow_z=z+wet0+1.2
    fig=go.Figure()
    fig.add_trace(go.Surface(x=X,y=Y,z=z,surfacecolor=shade,
        colorscale=TERRAIN_COLORS,cmin=float(np.nanpercentile(z,1)),
        cmax=float(np.nanpercentile(z,99.7)),showscale=False,
        hovertemplate='东坐标 %{x:.0f} m<br>北坐标 %{y:.0f} m<br>高程 %{z:.1f} m<extra>地形</extra>',
        lighting=dict(ambient=.62,diffuse=.73,specular=.14,roughness=.85,fresnel=.02),
        lightposition=dict(x=-17000,y=21000,z=19000),connectgaps=False,
        name='地形',uid='terrain'))
    fig.add_trace(go.Surface(x=X,y=Y,z=flow_z,surfacecolor=wet0,
        colorscale=FLOW_COLORS,cmin=0,cmax=limit,opacity=.96,connectgaps=False,
        showscale=True,colorbar=dict(title=dict(text='流深 / m',side='top'),
                     tickfont=dict(size=11,color='#46505b'),
                     len=.50,thickness=12,x=1.015,xpad=0,y=.50),
        hovertemplate='东坐标 %{x:.0f} m<br>北坐标 %{y:.0f} m<br>运动层高程 %{z:.1f} m<extra>泥石流</extra>',
        lighting=dict(ambient=.83,diffuse=.38,specular=.12,roughness=.9),
        name='泥石流',uid='flow'))
    cfg=R['cfg']
    marks=[('冰崩点',(cfg.source_x,cfg.source_y),'#cba565','circle'),
           ('吉隆口岸',PORT,'#58bfd4','diamond')]
    for name,loc,col,shape in marks:
        zz_loc=_terrain_at(T,loc)
        fig.add_trace(go.Scatter3d(x=[loc[0]],y=[loc[1]],z=[zz_loc+110],
            mode='markers+text',text=[name],textposition='top center',
            textfont=dict(color='#17232d',size=13,family='Microsoft YaHei, Noto Sans CJK SC, sans-serif'),
            marker=dict(size=7,color=col,symbol=shape,line=dict(color='white',width=1.8)),
            hovertemplate=f'{name}<br>UTM X: %{{x:.0f}} m<br>UTM Y: %{{y:.0f}} m<extra></extra>',
            name=name,showlegend=False,uid=name))
    fig.update_layout(
        title=None,paper_bgcolor='#fafaf9',plot_bgcolor='#fafaf9',
        font=dict(family='Microsoft YaHei, Noto Sans CJK SC, sans-serif',color='#263442'),
        margin=dict(l=0,r=6,t=5,b=5),height=670,
        showlegend=False,dragmode='turntable',uirevision='fixed-camera-v1',
        scene=dict(
           xaxis=dict(title='东坐标 / m',showbackground=False,showgrid=False,zeroline=False,
                      showspikes=False,tickfont=dict(size=10)),
           yaxis=dict(title='北坐标 / m',showbackground=False,showgrid=False,zeroline=False,
                      showspikes=False,tickfont=dict(size=10)),
           zaxis=dict(title='高程 / m',showbackground=False,showgrid=False,zeroline=False,
                      showspikes=False,tickfont=dict(size=10)),
           aspectmode='manual',
            aspectratio=dict(
                x=1.70,
                y=1.70 * yspan / xspan,
                z=max(.34, min(.67, 1.6 * zspan / xspan))
            ),
           bgcolor='#fafaf9',uirevision='persist-terrain-camera',
           camera=dict(eye=dict(x=1.45,y=-1.70,z=1.35),up=dict(x=0,y=0,z=1)),
           hovermode='closest'))
    return fig,z,frames


def make_animation_html(T,R,stride=2,include_plotlyjs=True):
    fig,z,frames=make_figure(T,R,stride)
    fig_div=fig.to_html(full_html=False,include_plotlyjs=include_plotlyjs,
                        div_id='q4-terrain-3d',
                        config=dict(displaylogo=False,responsive=True,
                                    modeBarButtonsToRemove=['lasso3d','select3d','toImage'],
                                    scrollZoom=True,displayModeBar=True))
    m,n=z.shape
    flat=np.nan_to_num(z.ravel(),nan=-99999).astype('float32')
    base_z=flat.round(2).tolist()
    thresh=R['cfg'].wet_threshold_m
    sparse=[]
    for H in frames:
        vals=H.ravel()
        keep=np.flatnonzero((vals>=thresh)&np.isfinite(z.ravel()))
        sparse.append([[int(a),round(float(vals[a]),2)] for a in keep])
    scene=dict(nrows=m,ncols=n,base_z=base_z,
               times=[round(float(t),1) for t in R['times']],
               frames=sparse)
    json_data=json.dumps(scene,ensure_ascii=False,separators=(',',':'))
    # 模拟只调用Plotly.restyle更新流动层；三维地形和相机实例常驻。
    page='''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box} html,body{margin:0;padding:0;background:#fafaf9;color:#233445;font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",system-ui,sans-serif}
.wrapper{width:100%;background:#fafaf9} .controls{display:flex;align-items:center;justify-content:space-between;padding:12px 15px 5px;gap:12px;flex-wrap:wrap}
.group{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.label{font-weight:650;font-size:13px;letter-spacing:.2px}.small{font-size:12px;color:#647280}
button,select{background:#fff;color:#293845;border:1px solid #cfd7dc;border-radius:6px;padding:8px 12px;font-size:12px;font-family:inherit;cursor:pointer}
button:hover{background:#eff4f5}#play{color:#fff;background:#28485c;border-color:#28485c}#play:hover{background:#1c394b}
#timeline{width:100%;accent-color:#28485c;cursor:pointer;margin:3px 0 6px}
.bar{display:flex;gap:12px;align-items:center;padding:2px 16px 12px}#clock{font-variant-numeric:tabular-nums;white-space:nowrap;min-width:84px;font-size:14px;font-weight:700;color:#243748}
#q4-terrain-3d{width:100%;height:650px}.guide{border-top:1px solid #e0e5e6;padding:9px 15px;color:#6b7681;font-size:11px;display:flex;justify-content:space-between;gap:8px}
@media(max-width:700px){.controls{padding:8px 10px}.bar{padding:0 10px 10px}#q4-terrain-3d{height:520px}}
</style></head><body><div class="wrapper">
<div class="controls"><div class="group"><span class="label">三维流动演化</span><span class="small">DEM 地形 · 动态流深</span></div>
<div class="group"><button id="play">播放</button><button id="prev">上一帧</button><button id="next">下一帧</button>
<select id="speed" aria-label="播放速度"><option value="950">0.5 倍速</option><option value="430" selected>1 倍速</option><option value="230">2 倍速</option></select>
<button id="top">俯视</button><button id="reset">复位视角</button></div></div>
__FIGURE__
<div class="bar"><span id="clock">t = 0 s</span><input id="timeline" type="range" min="0" max="0" step="1" value="0" aria-label="仿真时间轴"></div>
<div class="guide"><span>鼠标拖动旋转 · 滚轮缩放 · 播放过程中可调整视角</span><span>EPSG:32645</span></div>
</div><script>
(function(){'use strict';
 const dat=__DATA__;
 const gd=document.getElementById('q4-terrain-3d');
 let index=0,playing=false,busy=false,dragging=false,timer=null,camera=null;
 const range=document.getElementById('timeline'),clock=document.getElementById('clock');
 const play=document.getElementById('play');const spd=document.getElementById('speed');
 const initialCamera={eye:{x:1.45,y:-1.70,z:1.35},up:{x:0,y:0,z:1}};
 function buildGrid(pairs){
   let flatZ=Array(dat.base_z.length).fill(null),flatColor=Array(dat.base_z.length).fill(null);
   pairs.forEach(function(p){const ix=p[0],h=p[1],zz=dat.base_z[ix];if(zz>-90000){flatZ[ix]=zz+h*4.0+8.0;
flatColor[ix]=h;;}});
   let rows=[],colors=[];
   for(let j=0;j<dat.nrows;j++){
       rows.push(flatZ.slice(j*dat.ncols,(j+1)*dat.ncols));
       colors.push(flatColor.slice(j*dat.ncols,(j+1)*dat.ncols));
   }
   return [rows,colors];
 }
 function setFrame(k){
   if(busy)return Promise.resolve();
   busy=true;index=Math.max(0,Math.min(k,dat.frames.length-1));
   range.value=index;clock.textContent='t = '+dat.times[index]+' s';
   let f=buildGrid(dat.frames[index]);
   return Plotly.restyle(gd,{z:[f[0]],surfacecolor:[f[1]]},[1]).then(function(){
     busy=false;
     // 相机旋转由Plotly自身的uirevision保持；交互过程中不覆盖视角。
   }).catch(function(err){busy=false;console.error(err);});
 }
 function stop(){playing=false;play.textContent='播放';if(timer){clearTimeout(timer);timer=null;}}
 function tick(){
   if(!playing)return;
   if(dragging||busy){timer=setTimeout(tick,60);return;}
   if(index>=dat.frames.length-1){stop();return;}
   setFrame(index+1).then(function(){if(playing)timer=setTimeout(tick,Number(spd.value));});
 }
 function begin(){if(playing){stop();return;}if(index>=dat.frames.length-1)index=-1;
   playing=true;play.textContent='暂停';tick();}
 function setup(){
   if(!gd._fullLayout || !gd._fullLayout.scene){setTimeout(setup,150);return;}
   range.max=dat.frames.length-1;play.onclick=begin;
   document.getElementById('next').onclick=()=>{stop();setFrame(index+1);};
   document.getElementById('prev').onclick=()=>{stop();setFrame(index-1);};
   range.oninput=()=>{stop();setFrame(Number(range.value));};
   document.getElementById('reset').onclick=()=>Plotly.relayout(gd,{'scene.camera':initialCamera});
   document.getElementById('top').onclick=()=>Plotly.relayout(gd,{'scene.camera':{eye:{x:0,y:0,z:2.3},up:{x:0,y:1,z:0}}});
   gd.on('plotly_relayout',function(evt){if(evt['scene.camera'])camera=evt['scene.camera'];});
   gd.addEventListener('pointerdown',()=>{dragging=true;});
   window.addEventListener('pointerup',()=>{dragging=false;});
   gd.addEventListener('mouseleave',()=>{dragging=false;});
   // 仅初始化一次，避免重绘3D地形。
   setFrame(0);
   window.__q4_control={setFrame:setFrame,start:begin,stop:stop,
     getFrame:()=>index,getCamera:()=> gd._fullLayout.scene.camera};
 }
 setup();
})();
</script></body></html>'''
    return page.replace('__FIGURE__',fig_div).replace('__DATA__',json_data)


def save_animation_html(T,R,filename,stride=2):
    p=Path(filename);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(make_animation_html(T,R,stride),encoding='utf-8')
    return p
