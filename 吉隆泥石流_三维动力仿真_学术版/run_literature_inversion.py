# -*- coding: utf-8 -*-
"""文献约束的条件拟合试算。不是对真实灾害的唯一物理反演。
使用：python run_literature_inversion.py --mode quick
      python run_literature_inversion.py --mode full --grid-check
定义 E=k max(v-vcrit,0) (且有可侵蚀厚度上限)，k无量纲。
校准目标的5860万m³是殷跃平等对特定河段的报告；本程序当前
统计的是裁剪DEM范围内的总侵蚀。二者统计空间不一致，**因此
只能作演示性的条件拟合，并不是侵蚀系数的真实物理反演**。
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,time
import pandas as pd
from q4_solver import Config,load_terrain,simulate
from q4_viz import save_all,process_dashboard

ROOT=Path(__file__).resolve().parent
DEM=ROOT/'data'/'elevation_utm45n.tif'
OUT=ROOT/'calibration'
TARGET_T=450.0   # 杨云鹏等：总传播历时约450s
TARGET_E=5.86e7   # 殷跃平等：两河段侵蚀量, 与本程序全域不同, 只能用作代理目标


def run_case(T,cfg,label):
    start=time.monotonic()
    out={'情景':label,'网格_m':cfg.grid_m,'mu0':cfg.mu,'mu1':cfg.mu_late,
         'xi_mps2':cfg.xi_mps2,'k':cfg.erosion_k}
    try:
        result=simulate(T,cfg,max_steps=15000)
        end=result['records'][-1]
        arrived=result['first_port_frame_s']
        out.update({'计算状态':'完成','到达口岸300m_s':arrived,
                    '距目标450秒误差_s':None if arrived is None else arrived-TARGET_T,
                    '模拟全域累计侵蚀_m3':end['累计侵蚀体积_m3'],
                    '相对代理侵蚀目标误差_percent':100*(end['累计侵蚀体积_m3']/TARGET_E-1),
                    '全程最大单元速度_mps':end['全程峰值速度_mps'],
                    '600秒口岸最近距离_m':end['距口岸最近有效流动单元_m'],
                    '守恒相对误差':end['物质量平衡相对误差'],
                    '求解步数':result['steps']})
    except Exception as e:
        result=None;out.update({'计算状态':type(e).__name__+': '+str(e)})
    out['运行耗时_s']=round(time.monotonic()-start,2)
    print(out,flush=True)
    return out,result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['quick','full'],default='quick')
    parser.add_argument('--grid-check',action='store_true')
    a=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    base=Config(duration_s=600.,snapshot_s=10.,grid_m=90.,initial_volume_m3=1.75e8,
                mu=.06,mu_late=.01,xi_mps2=5000.,erosion_k=.000678,
                deposit_k_per_s=.0003,fluidization_after_s=100.)
    cases=[('条件拟合候选',base)]
    if a.mode=='full':
        cases += [
          ('原假设基准',replace(base,mu=.20,mu_late=.05,xi_mps2=1800,erosion_k=.003)),
          ('拟合附近较低k',replace(base,erosion_k=.00065)),
          ('拟合附近较高k',replace(base,erosion_k=.00090)),
          ('另一摩擦方案',replace(base,mu=.10,mu_late=.015,xi_mps2=4500,erosion_k=.00080)),
        ]
    if a.grid_check:cases.append(('相同参数60米检验',replace(base,grid_m=60.)))
    items=[]
    for label,c in cases:
        terrain=load_terrain(DEM,c)
        row,R=run_case(terrain,c,label)
        items.append(row)
        pd.DataFrame(items).to_csv(OUT/'可重复参数反演运行结果.csv',index=False,encoding='utf-8-sig')
        if label=='条件拟合候选' and R is not None:
            folder=OUT/'拟合情景输出'
            save_all(terrain,R,folder,html=True)
            process_dashboard(R,folder,pd.DataFrame(items))
            summary=json.loads((folder/'模型参数和结果.json').read_text('utf-8'))
            summary['参数来源']='仅为90米网格下匹配代理到达时间与代理侵蚀总量的条件拟合候选，不是论文直接测得的mu、xi、k'
            summary['校准口径限制']='模型为全计算域累计侵蚀；殷文5860万m³为两河段，不能视作严格相同量'
            (folder/'模型参数和结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    print('结果：',OUT/'可重复参数反演运行结果.csv')

if __name__=='__main__':main()
