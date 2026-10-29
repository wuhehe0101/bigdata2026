# -*- coding: utf-8 -*-
"""生成可脱机访问的学术风格结果网站，不包含模型文献讨论。"""
from pathlib import Path
import json
from html import escape
import pandas as pd


def build_portal(root):
    root=Path(root); folder=root/'results'
    report=json.loads((folder/'模型参数和结果.json').read_text('utf-8'))
    r=report['最终结果'];cfg=report['模型参数假设']
    df=pd.read_csv(folder/'模拟过程时序.csv')
    entered=report.get('首个满足口岸300米邻域的保存帧时间_s')
    if entered is None:
        subset=df.loc[df['口岸300米邻域是否达到']==True,'时间_s']
        entered=None if subset.empty else float(subset.iloc[0])
    initial=cfg['initial_volume_m3']/1e8
    first_time=f'{entered:.0f} s' if entered is not None else '—'
    figures=[('图01_真实地形与文献源区.png','地形与空间位置'),
      ('图02_初始源区与释放厚度.png','初始释放场'),
      ('图03_六时刻流深演化.png','流深时序演化'),
      ('图04_峰值流深与流速.png','峰值流深与速度'),
      ('图05_时序与质量守恒.png','数值过程与守恒'),
      ('图06_侵蚀与沉积分布.png','侵蚀与沉积')]
    gallery=''.join(f'<figure><a href="results/{escape(file)}" target="_blank"><img loading="lazy" src="results/{escape(file)}" alt="{escape(name)}"></a><figcaption>{escape(name)}</figcaption></figure>'
       for file,name in figures if (folder/file).exists())
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>吉隆口岸冰岩崩—泥石流三维动力学仿真</title>
<style>
:root{--ink:#1d2a35;--quiet:#667683;--line:#dce3e6;--background:#f6f7f7;--paper:#fff;--navy:#294759}
*{box-sizing:border-box}body{margin:0;background:var(--background);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",system-ui,sans-serif}
header{background:linear-gradient(124deg,#192e3d,#344d5a);padding:31px max(26px,calc((100vw - 1510px)/2));color:white}
header .kicker{font-size:11px;letter-spacing:2.4px;color:#b1c5cf}header h1{font-size:clamp(25px,3vw,38px);font-weight:570;margin:13px 0 9px;letter-spacing:.6px}
header p{font-size:13px;color:#cdd9df;margin:0;letter-spacing:.2px} main{max-width:1550px;margin:auto;padding:23px 24px 85px}
nav{display:flex;gap:22px;flex-wrap:wrap;padding:7px 0 19px;border-bottom:1px solid var(--line)}nav a{color:#415769;text-decoration:none;font-size:13px}nav a:hover{color:#172e3f;text-decoration:underline}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:19px 0}
.metric,.section{background:var(--paper);border:1px solid var(--line);border-radius:7px}.metric{padding:19px 18px}
.metric .value{font-variant-numeric:tabular-nums;font-size:clamp(20px,2.2vw,29px);font-weight:640;color:#1e3d50;white-space:nowrap}.metric .label{font-size:12px;color:var(--quiet);margin-top:7px}
.section{padding:18px 18px 21px;margin:14px 0 21px}.heading{display:flex;justify-content:space-between;align-items:end;gap:10px;border-bottom:1px solid #e7ebee;margin-bottom:15px;padding-bottom:14px}
h2{font-size:19px;font-weight:620;margin:0}.hint{color:var(--quiet);font-size:11px}iframe{width:100%;height:835px;border:1px solid #eff2f3;border-radius:4px;background:#fafaf9;display:block}
iframe.chart{height:730px}.tools{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}
.tools a{font-size:12px;padding:9px 14px;background:#f2f5f6;color:#31516a;border:1px solid var(--line);border-radius:5px;text-decoration:none}.tools a:hover{background:#e4ecef}
.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{padding:0;margin:0;border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#fff}figure img{width:100%;display:block;object-fit:contain}figcaption{font-size:12px;color:#5b6a76;padding:10px 13px;border-top:1px solid #eef0f1}
.footer{display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;font-size:11px;color:#7a8995;margin:28px 0 5px;padding-top:18px;border-top:1px solid var(--line)}
@media(max-width:1100px){.metrics{grid-template-columns:repeat(3,1fr)}}
@media(max-width:700px){main{padding:15px 12px 48px}header{padding:22px 16px}.metrics{grid-template-columns:repeat(2,1fr)}.gallery{grid-template-columns:1fr}iframe{height:690px}iframe.chart{height:670px}}
</style></head><body>
<header><div class="kicker">GEOSPATIAL DYNAMICS / NUMERICAL SIMULATION</div>
<h1>吉隆口岸冰岩崩—泥石流三维动力学仿真</h1>
<p>真实DEM地形 · 二维深度平均模型 · Voellmy摩擦阻力 · 侵蚀沉积耦合</p></header>
<main><nav><a href="#scene">三维场景</a><a href="#process">计算过程</a><a href="#figures">空间结果</a><a href="#files">结果文件</a></nav>
<section class="metrics">
<div class="metric"><div class="value">__VOLUME__ 亿 m³</div><div class="label">初始物源体积</div></div>
<div class="metric"><div class="value">__DIST__ km</div><div class="label">累计最大直线扩展</div></div>
<div class="metric"><div class="value">__SPEED__ m/s</div><div class="label">模拟最大流速</div></div>
<div class="metric"><div class="value">__ARRIVAL__</div><div class="label">口岸邻域首次进入</div></div>
<div class="metric"><div class="value">__EROSION__ 百万 m³</div><div class="label">累计侵蚀体积</div></div>
</section>
<section class="section" id="scene"><div class="heading"><h2>01　三维地形与运动过程</h2><span class="hint">流深随时间更新 · 鼠标控制视角</span></div>
<iframe src="results/三维运动动画_离线网页.html" title="泥石流三维地形动画" loading="eager"></iframe>
<div class="tools"><a href="results/三维运动动画_离线网页.html" target="_blank">独立打开三维场景</a><a href="results/基准情景逐帧.npz" download>逐帧数值阵列</a></div></section>
<section class="section" id="process"><div class="heading"><h2>02　动力学计算过程</h2><span class="hint">速度 · 流深 · 侵蚀 · 沉积 · 质量收支</span></div>
<iframe class="chart" src="results/计算过程曲线_离线网页.html" title="计算过程曲线" loading="lazy"></iframe>
<div class="tools"><a href="results/计算过程曲线_离线网页.html" target="_blank">独立打开过程曲线</a><a href="results/模拟过程时序.csv" download>下载逐时刻CSV</a></div></section>
<section class="section" id="figures"><div class="heading"><h2>03　空间分布与数值图件</h2><span class="hint">平面流深 · 速度 · 侵蚀沉积</span></div><div class="gallery">__GALLERY__</div></section>
<section class="section" id="files"><div class="heading"><h2>04　模型参数与成果输出</h2></div>
<div class="tools"><a href="results/模型参数和结果.json" download>参数与结果 JSON</a>
<a href="results/最大流深_m.tif" download>最大流深 GeoTIFF</a>
<a href="results/最大流速_mps.tif" download>最大流速 GeoTIFF</a>
<a href="results/侵蚀深度_m.tif" download>侵蚀深度 GeoTIFF</a>
<a href="results/沉积深度_m.tif" download>沉积深度 GeoTIFF</a>
<a href="README_使用说明.md" target="_blank">项目使用说明</a></div></section>
<div class="footer"><span>吉隆口岸冰岩崩—泥石流三维动力学仿真系统</span><span>坐标系：EPSG:32645　｜　计算网格：__GRID__ m　｜　时长：__DURATION__ s</span></div>
</main></body></html>'''
    repl={'__VOLUME__':f'{initial:.2f}','__DIST__':f'{r["最大累计直线距离_km"]:.2f}',
          '__SPEED__':f'{r["全程峰值速度_mps"]:.1f}',
          '__ARRIVAL__':first_time,'__EROSION__':f'{r["累计侵蚀体积_m3"]/1e6:.1f}',
          '__GRID__':str(int(cfg['grid_m'])),'__DURATION__':str(int(cfg['duration_s'])),
          '__GALLERY__':gallery}
    for key,val in repl.items():html=html.replace(key,val)
    dest=root/'index.html';dest.write_text(html,encoding='utf-8')
    return dest

if __name__=='__main__':
    build_portal(Path(__file__).resolve().parent)
