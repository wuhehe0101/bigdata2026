# -*- coding: utf-8 -*-
"""Q4 一键批量：真实DEM + 论文参数情景 + 可追溯数值表 + 网页动画。
此脚本不声称反演出了真实阻力参数；全部μ、ξ、k仍为试验情景。
"""
import argparse
import json
from dataclasses import asdict,replace
from pathlib import Path
import time
import numpy as np
import pandas as pd
from q4_solver import Config,load_terrain,simulate,PORT
from q4_viz import save_all
from make_portal import build_portal
ROOT=Path(__file__).resolve().parent

def cfg_from_cli():
    a=argparse.ArgumentParser()
    a.add_argument('--grid',type=float,default=90)
    a.add_argument('--duration',type=float,default=600)
    a.add_argument('--snapshot',type=float,default=20)
    a.add_argument('--volume',type=float,default=1.75e8,help='刘昌军等DoD 1.75e8；参考范围1.1e8~1.8e8')
    a.add_argument('--mu',type=float,default=.20)
    a.add_argument('--late-mu',type=float,default=.05)
    a.add_argument('--xi',type=float,default=1800)
    a.add_argument('--erosion-k',type=float,default=.003)
    a.add_argument('--deposit-k',type=float,default=.0003)
    a.add_argument('--fluidize-after',type=float,default=100)
    a.add_argument('--sensitivity',action='store_true',help='加跑2组文献规模与厚度配套情景')
    a.add_argument('--output',type=Path,default=ROOT/'results')
    a.add_argument('--dem',type=Path,default=ROOT/'data'/'elevation_utm45n.tif')
    return a.parse_args()

def run():
    args=cfg_from_cli()
    p=Config(grid_m=args.grid,duration_s=args.duration,snapshot_s=args.snapshot,
       initial_volume_m3=args.volume,mu=args.mu,mu_late=args.late_mu,
       xi_mps2=args.xi,erosion_k=args.erosion_k,deposit_k_per_s=args.deposit_k,
       fluidization_after_s=args.fluidize_after)
    args.output.mkdir(parents=True,exist_ok=True)
    T=load_terrain(args.dem,p)
    print('DEM:',T.z0.shape,'分辨率:',T.dx,T.dy,'源区参考:',(p.source_x,p.source_y),flush=True)
    cases=[('当前文献约束基准',p)]
    if args.sensitivity:
        cases += [
         ('殷论文1.1亿与约60万㎡',replace(p,initial_volume_m3=1.1e8,source_length_m=1000,source_width_m=600)),
         ('刘论文1.8亿与约400万㎡',replace(p,initial_volume_m3=1.8e8))]
    records=[];results=[]
    start=time.monotonic()
    for n,(name,c) in enumerate(cases,1):
        print(f'计算 {n}/{len(cases)}：{name}',flush=True)
        try:
            r=simulate(T,c,max_steps=5200)
            last=r['records'][-1]
            nearest450=min(r['records'],key=lambda x:abs(x['时间_s']-450.))
            row={'情景名称':name,'计算状态':'完成','源区释放体积_m3':c.initial_volume_m3,
                '近似释放区面积_m2':r['source_mask'].sum()*T.dx*T.dy,
                '平均初始流深_m':c.initial_volume_m3/(r['source_mask'].sum()*T.dx*T.dy),
                '初期mu_假设':c.mu,'后期mu_假设':c.mu_late if c.fluidization_after_s>0 else c.mu,
                'xi_mps2_假设':c.xi_mps2,'侵蚀k_假设':c.erosion_k,
                '临界侵蚀速度_mps_假设':c.erosion_speed_threshold_mps,
                '可侵蚀上限_m_假设':c.erodible_depth_m,
                '模拟600s到达口岸300m':bool(last['口岸300米邻域是否达到']),
                '首次到达口岸记录时刻_s':r['first_port_frame_s'],
                't450s流动到口岸最近距离_m':nearest450['距口岸最近有效流动单元_m'],
                't600s流动到口岸最近距离_m':last['距口岸最近有效流动单元_m'],
                '最大直线传播距离_km':last['最大累计直线距离_km'],
                '累计影响面积_km2':last['累计影响面积_km2'],
                '全域峰值速度_mps':last['全程峰值速度_mps'],
                '累计侵蚀体积_m3':last['累计侵蚀体积_m3'],
                '累计沉积体积_m3':last['累计沉积体积_m3'],
                '最终流动物体积_m3':last['运动层体积_m3'],
                '最大质量收支误差_percent':max(abs(x['物质量平衡相对误差']) for x in r['records'])*100,
                '边界受影响网格数':r['edge_wet_cells'],'求解步数':r['steps']}
            records.append(row);results.append((name,r))
            print('   最邻近口岸:',round(row['t600s流动到口岸最近距离_m']),
                  'm; t=450:',round(row['t450s流动到口岸最近距离_m']),
                  'm; 侵蚀:',round(row['累计侵蚀体积_m3']/1e6,2),'百万m³',flush=True)
        except Exception as e:
            records.append({'情景名称':name,'计算状态':f'失败：{type(e).__name__} {e}'})
            print('   失败：',type(e).__name__,e,flush=True)
            if n==1:raise
    df=pd.DataFrame(records)
    df.to_csv(args.output/'参数情景敏感性.csv',index=False,encoding='utf-8-sig')
    baseline=results[0][1]
    # 图表兼容源模板字段
    plot_rows=[]
    for a in records:
        if a['计算状态']!='完成':continue
        plot_rows.append({'情景名称':a['情景名称'],'最大直线距离_km':a['最大直线传播距离_km'],
                          '累计侵蚀_百万m3':a['累计侵蚀体积_m3']/1e6})
    save_all(T,baseline,args.output,scenarios=plot_rows,html=True)
    # 保存插图用简表后恢复完整的情景参数与计算输出；避免被绘图模块覆盖。
    df.to_csv(args.output/'参数情景敏感性.csv',index=False,encoding='utf-8-sig')
    # 对照：注意多篇论文不一致及不同统计口径
    last=baseline['records'][-1]
    comparison=[
      ('初始失稳体积','刘昌军等DoD','1.75亿m³',f'{p.initial_volume_m3/1e8:.3f}亿m³','该值是输入，不是预测'),
      ('初始失稳体积','殷跃平等','1.1亿m³',f'{p.initial_volume_m3/1e8:.3f}亿m³','不同团队估算结果不一致，需要做敏感性'),
      ('初始释放区面积','刘昌军等','约400万m²',f'{baseline["source_mask"].sum()*T.dx*T.dy/1e4:.2f}万m²','由超椭圆近似与网格离散形成，非真实多边形'),
      ('初始平均厚度','刘昌军等','约45m',f'{p.initial_volume_m3/(baseline["source_mask"].sum()*T.dx*T.dy):.2f}m','体积除以近似源区面积；不能代表逐网格真实厚度'),
      ('源区高程','杨云鹏等','5152m',f'{T.z0[np.argmin(abs(T.y-p.source_y)),np.argmin(abs(T.x-p.source_x))]:.1f}m','初步滑震定位点和论文平均源区高程存在差异'),
      ('冰岩崩至交汇口时间','杨云鹏等','约130s','未计算','缺少有坐标的交汇口控制断面，禁止虚构'),
      ('交汇口至口岸时间','杨云鹏等','约320s','未计算','需交汇口断面位置后才可分段计算'),
      ('启动至口岸到达时间','杨云鹏等','约450s',str(baseline['first_port_frame_s']),'口岸300m流深阈值判定；未到达为None'),
      ('启动至口岸到达时间','刘昌军等数值结果','约505s',str(baseline['first_port_frame_s']),'论文为独立数值模拟输出，非直接实测'),
      ('早期平均传播速度','杨云鹏等反演','63.6m/s','未计算','需分段空间线路及穿越时刻，不能用网格峰速代替'),
      ('下游平均传播速度','杨云鹏等反演','46m/s','未计算','需对应路段测量，不能用网格峰速代替'),
      ('口岸上游泥痕高度','杨云鹏等','45m','未计算','泥痕包含冲高，非单纯数值流深'),
      ('两河段侵蚀总体积','殷跃平等','5860万m³',f'{last["累计侵蚀体积_m3"]/1e4:.1f}万m³','模型为全裁剪域，与两河段范围不一致，不能据此直接反演k'),
      ('全程侵蚀总体积','刘昌军等','2.60亿m³',f'{last["累计侵蚀体积_m3"]/1e8:.3f}亿m³','需统一全程范围及DEM差分才能反演k')]
    pd.DataFrame(comparison,columns=['对照指标','来源','论文参考','本模型输出','可比性与说明']).to_csv(args.output/'文献约束与计算对照.csv',index=False,encoding='utf-8-sig')
    from q4_viz import process_dashboard
    process_dashboard(baseline,args.output,df)
    # Source/data and run info
    summary_file=args.output/'模型参数和结果.json'
    summary=json.loads(summary_file.read_text('utf-8'))
    summary['文献范围选择']='基准刘昌军等1.75亿m³与约400万㎡；对照殷跃平等1.1亿m³/约60万㎡、刘昌军等1.8亿m³/约400万㎡；杨文面积为另一个独立约束'
    summary['基准选取依据']='参数基准来自旧程序；μ、ξ、k不是文献实测值，也未完成独立标定'
    summary['参数筛选性质']='使用文献指标比较的敏感性探索，不能仅用最接近口岸的情景声称真实参数'
    summary['资料局限']='仅有单点灾害位置与口岸坐标；缺少交汇口精确位置、源区多边形、灾前高精度DEM与沿河床侵蚀空间分布'
    summary['计算耗时_s']=round(time.monotonic()-start,2)
    summary['计算情景个数']=len(results)
    summary_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    build_portal(ROOT)
    print('成功生成网页：',ROOT/'index.html',flush=True)
    print('计算耗时:',round(time.monotonic()-start,1),'秒',flush=True)

if __name__=='__main__':run()
