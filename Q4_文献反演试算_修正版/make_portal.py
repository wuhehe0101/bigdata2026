# -*- coding: utf-8 -*-
"""生成免服务器的离线门户，和Streamlit的实时计算页面是两个独立交付物。"""
from pathlib import Path
from html import escape
import json,pandas as pd

def build_portal(root):
    root=Path(root);out=root/'results'
    report=json.loads((out/'模型参数和结果.json').read_text(encoding='utf-8'))
    r=report['最终结果'];p=report['模型参数假设']
    runs=pd.read_csv(out/'参数情景敏感性.csv')
    compared=pd.read_csv(out/'文献约束与计算对照.csv')
    fmt=lambda x:'' if pd.isna(x) else escape(str(x))
    headings=['情景名称','源区释放体积_m3','近似释放区面积_m2','平均初始流深_m','初期mu_假设','后期mu_假设','xi_mps2_假设','侵蚀k_假设',
              't450s流动到口岸最近距离_m','t600s流动到口岸最近距离_m','最大直线传播距离_km','累计侵蚀体积_m3','模拟600s到达口岸300m']
    rows=''.join('<tr>'+''.join('<td>'+fmt(a.get(k))+'</td>' for k in headings)+'</tr>' for a in runs.to_dict('records'))
    comp_rows=''.join('<tr>'+''.join('<td>'+fmt(a[k])+'</td>' for k in compared.columns)+'</tr>' for a in compared.to_dict('records'))
    imgs=[('图01_真实地形与文献源区.png','真实地形和源区'),('图02_初始源区与释放厚度.png','文献尺寸约束的释放区'),('图03_六时刻流深演化.png','六时刻流深'),('图04_峰值流深与流速.png','历史峰值流深与速度'),('图05_时序与质量守恒.png','数值过程及守恒'),('图06_侵蚀与沉积分布.png','经验侵蚀与沉积'),('图07_参数敏感性.png','物源体积与释放区情景对照')]
    gallery=''.join(f'<figure><img loading="lazy" src="results/{f}" alt="{n}"><figcaption>{n}</figcaption></figure>' for f,n in imgs if (out/f).exists())
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>吉隆8·26第四问·参数情景动力学与三维动画</title>
<style>:root{--bg:#0b1220;--card:#162134;--ink:#e8f0fa;--muted:#adbed1;--accent:#62d6d7}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:'Microsoft YaHei','PingFang SC',system-ui,sans-serif}main{max-width:1420px;margin:auto;padding:28px 24px 70px}h1{font-size:clamp(24px,3vw,38px)}h2{font-size:20px}p{line-height:1.8;color:var(--muted)}.tag{color:var(--accent);font-weight:bold;letter-spacing:1px;font-size:13px}.alert{background:#372a23;border-left:4px solid #ffae65;border-radius:8px;padding:16px;line-height:1.8;margin:16px 0}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.card,.section{border-radius:13px;border:1px solid #314158;background:var(--card);padding:17px;margin:16px 0}.card b{font-size:25px;display:block}.card small{color:var(--muted)}a{color:#a9e6ed}.tools{display:flex;gap:9px;flex-wrap:wrap}.tools a{display:inline-block;padding:9px 13px;background:#273c53;border-radius:8px;text-decoration:none}.viewer{background:white;border-radius:9px;overflow:hidden}.viewer iframe{width:100%;height:740px;border:0;display:block}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{margin:0;background:white;border-radius:8px;overflow:hidden;color:#25354a}figure img{max-width:100%;display:block}figcaption{padding:10px 14px}.table-wrap{overflow:auto;max-height:520px}table{border-collapse:collapse;min-width:1040px;width:100%}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #304359;font-size:13px;white-space:nowrap}th{color:#76d6db}td{color:#e0eafa}.hint{font-size:12px;color:var(--muted)}@media(max-width:1050px){.cards{grid-template-columns:repeat(3,1fr)}}@media(max-width:680px){.cards{grid-template-columns:repeat(2,1fr)}.gallery{grid-template-columns:1fr}.viewer iframe{height:580px}}</style></head><body><main>
<div class="tag">2026 MATHORCUP · QUESTION 4 · OFFLINE RESEARCH VIEWER</div>
<h1>吉隆“8·26”冰岩崩—泥石流动力学推演</h1>
<p>基于用户提供的EPSG:32645真实30米DEM、二维深度平均守恒方程、Voellmy阻力与经验侵蚀沉积。数值计算使用90米格网（或命令行指定），三维显示单独降采样。可在网页中旋转地形、播放帧、查看计算曲线及情景对比。</p>
<div class="alert"><b>研究边界：</b>本项目是文献约束的数值情景，不是被验证的真实灾害复现；源区边界由文献尺度近似，不是遥感精确解译；μ、ξ及侵蚀k未经独立反演。不能根据直线传播距离判断沿沟道23km传播，也不能将不同论文、不同河段的侵蚀量直接作为同一目标。</div>
<div class="cards">
<div class="card"><b>__VOLUME__ 亿m³</b><small>基准初始体积（输入）</small></div>
<div class="card"><b>__DIST__ km</b><small>最大直线传播距离</small></div>
<div class="card"><b>__SPEED__ m/s</b><small>全域历史最大速度</small></div>
<div class="card"><b>__EROSION__ 百万m³</b><small>全计算域累计侵蚀</small></div>
<div class="card"><b>__PORT__</b><small>模拟是否到达口岸300米邻域</small></div></div>
<div class="section"><h2>① 交互式三维时间动画</h2><div class="viewer"><iframe src="results/三维运动动画_离线网页.html" title="三维流深动画"></iframe></div><p class="hint">上面的播放条按模型保存帧逐时刻切换；流动图层是可视化处理，非实际三维颗粒运动。</p><div class="tools"><a href="results/三维运动动画_离线网页.html" target="_blank">独立打开3D动画</a><a href="results/模拟过程时序.csv">下载逐时刻CSV</a><a href="results/参数情景敏感性.csv">下载全部情景参数与结果</a><a href="results/文献约束与计算对照.csv">下载论文对照表</a><a href="results/模型参数和结果.json">下载模型参数JSON</a><a href="README_使用说明.md">查看运行说明</a></div></div>
<div class="section"><h2>② 动态计算过程与质量收支</h2><div class="viewer"><iframe src="results/计算过程曲线_离线网页.html" title="计算指标变化"></iframe></div><p>守恒检查应使用：流动物体积 + 累计沉积 − 累计侵蚀 ≈ 初始体积。到达时间采用保存帧的首次阈值触发，误差受保存步长限制。</p></div>
<div class="section"><h2>③ 文献物源场景比较（非已标定真值）</h2><div class="table-wrap"><table><thead><tr>__SCENARIOHEAD__</tr></thead><tbody>__SCENARIOROWS__</tbody></table></div><p class="hint">仅在同一数据、公式、边界条件下进行比较；没有真实地质侵蚀分区，不能宣称k已经物理标定。</p></div>
<div class="section"><h2>④ 论文约束与模型计算对照</h2><div class="table-wrap"><table><thead><tr>__COMPHEAD__</tr></thead><tbody>__COMPROWS__</tbody></table></div></div>
<div class="section"><h2>⑤ 中文论文插图</h2><div class="gallery">__GALLERY__</div></div>
<div class="section"><h2>⑥ 模型及论文来源说明</h2>
<p>源区初步定位（殷跃平等）：85°33′00″E，28°20′09″N；殷文物源约1.1亿m³，源区约1000×600m；刘昌军等DoD初始1.75亿m³，初始面积约400万m²、厚度约45m；杨云鹏等源区约75.68万m²，前段约130秒、后段约320秒，到口岸约450秒。当前默认使用刘文体积与面积；各文献对物源几何有显著差异，不能混用。</p>
<p>三篇论文数据并非同一统计口径；文献没有直接给出本求解器适用的Voellmy μ、ξ及经验侵蚀k。刘昌军等采用宾汉阻力，不可将其参数直接代入Voellmy模型。</p>
<p>可复现：<code>python -m unittest discover -s tests -v</code>、<code>python run_batch.py --sensitivity</code>，网页实时计算用 <code>python -m streamlit run app.py</code>。下载包中包含README和脚本。</p></div></main></body></html>'''
    replacements={'__VOLUME__':f'{p["initial_volume_m3"]/1e8:.2f}',
       '__DIST__':f'{r["最大累计直线距离_km"]:.2f}',
       '__SPEED__':f'{r["全程峰值速度_mps"]:.1f}',
       '__EROSION__':f'{r["累计侵蚀体积_m3"]/1e6:.1f}',
       '__PORT__':'是' if r['口岸300米邻域是否达到'] else '否',
       '__SCENARIOHEAD__':''.join('<th>'+escape(x)+'</th>' for x in headings),
       '__SCENARIOROWS__':rows,
       '__COMPHEAD__':''.join('<th>'+escape(str(x))+'</th>' for x in compared.columns),
       '__COMPROWS__':comp_rows,
       '__GALLERY__':gallery}
    for k,v in replacements.items():html=html.replace(k,v)
    (root/'index.html').write_text(html,encoding='utf-8')
    return root/'index.html'

if __name__=='__main__':build_portal(Path(__file__).resolve().parent)
