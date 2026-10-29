# -*- coding: utf-8 -*-
"""运行基准动力学仿真，并输出论文插图、数据栅格与完整网页。"""
import argparse
import json
import time
from pathlib import Path
from q4_solver import Config, load_terrain, simulate
from q4_viz import save_all, process_dashboard
from make_portal import build_portal

ROOT=Path(__file__).resolve().parent

def main():
    cli=argparse.ArgumentParser(description='吉隆泥石流二维动力学批量计算')
    cli.add_argument('--grid',type=float,default=90)
    cli.add_argument('--duration',type=float,default=600)
    cli.add_argument('--snapshot',type=float,default=10)
    cli.add_argument('--volume',type=float,default=1.75e8)
    cli.add_argument('--mu',type=float,default=.060)
    cli.add_argument('--late-mu',type=float,default=.010)
    cli.add_argument('--xi',type=float,default=5000)
    cli.add_argument('--erosion-k',type=float,default=.000678)
    cli.add_argument('--deposit-k',type=float,default=.0003)
    cli.add_argument('--fluidize-after',type=float,default=100)
    cli.add_argument('--output',type=Path,default=ROOT/'results')
    cli.add_argument('--dem',type=Path,default=ROOT/'data'/'elevation_utm45n.tif')
    args=cli.parse_args()
    cfg=Config(grid_m=args.grid,duration_s=args.duration,snapshot_s=args.snapshot,
        initial_volume_m3=args.volume,mu=args.mu,mu_late=args.late_mu,
        xi_mps2=args.xi,erosion_k=args.erosion_k,
        deposit_k_per_s=args.deposit_k,fluidization_after_s=args.fluidize_after)
    t0=time.monotonic()
    print('载入DEM...',flush=True)
    T=load_terrain(args.dem,cfg)
    print('计算网格',T.z0.shape,flush=True)
    R=simulate(T,cfg,progress=lambda f:print(f'计算进度 {f:.0%}',flush=True),max_steps=15000)
    args.output.mkdir(parents=True,exist_ok=True)
    print('生成三维网页与地理数据...',flush=True)
    summary=save_all(T,R,args.output,html=True)
    process_dashboard(R,args.output,scenarios=None)
    if args.output.resolve()==(ROOT/'results').resolve():
        build_portal(ROOT)
    print('结果文件夹：',args.output,flush=True)
    print('首个口岸300米邻域进入时刻：',R['first_port_frame_s'],flush=True)
    print('累计侵蚀：',R['records'][-1]['累计侵蚀体积_m3'],flush=True)
    print('耗时秒：',round(time.monotonic()-t0,2),flush=True)

if __name__=='__main__':main()
