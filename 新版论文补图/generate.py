"""针对新版论文的补图：python revision_figures/generate.py。输入来自项目运行产物。"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from sklearn.metrics import roc_curve,confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent;D=O/'底表';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
blue='#276782';orange='#d27743';green='#397b67';entries=[]
def save(fig,name,where,caption,priority='建议正文'):
 fig.savefig(O/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig);entries.append(dict(file=name+'.png',where=where,caption=caption,priority=priority))
def main():
 raw=pd.read_csv(R/'weather_raw.csv',index_col=0,parse_dates=True).loc[:'2026-08-27 23:00'];q=pd.read_csv(R/'weather_q1.csv',index_col=0,parse_dates=True).loc[:'2026-08-27 23:00']
 # Missingness measured only in the manuscript's declared study period.
 group=pd.DataFrame({g:raw[[c for c in raw if c.startswith(g+'_')]].notna().mean(axis=1) for g in ['MET','RADM','SOIL','FLUX']}).resample('MS').mean()
 group.to_csv(D/'01_月度非缺失覆盖.csv');fig,ax=plt.subplots(figsize=(12,3),layout='constrained');im=ax.imshow(group.T,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu');ticks=[i for i,t in enumerate(group.index) if t.month in [1,7]];ax.set_xticks(ticks,group.index[ticks].strftime('%Y-%m'),rotation=30);ax.set_yticks(range(4),group.columns);fig.colorbar(im,ax=ax,label='组内变量非缺失比例');ax.set_title('多源数据月度完整性（哨兵值置缺失后、插值前）');save(fig,'补图01_多源数据月度完整性','4.1.2末尾，放在“经过上述处理……”之后','各组变量逐小时非缺失比例的月均值，2026年8月仅统计至27日；非仪器QC通过率。')
 x=pd.DataFrame(index=q.index);x['正积温24h']=q['MET_Ta_2m(degree C)'].clip(lower=0).rolling(24,min_periods=24).sum()/24
 for name,col in [('VPD','FLUX_VPD(hPa)'),('感热通量','FLUX_H(W/m2)'),('土壤热通量','SOIL_G_0.1m(W/m2)')]:x[name]=q[col]
 x['净辐射']=q['RADM_Rsd(W/m2)']-q['RADM_Rsu(W/m2)']+q['RADM_Rld(W/m2)']-q['RADM_Rlu(W/m2)']
 units={'正积温24h':'℃·日','VPD':'hPa','净辐射':'W/m²','感热通量':'W/m²','土壤热通量':'W/m²'};end=pd.Timestamp('2026-08-26 10:00');rows=[]
 fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
 for ax,c in zip(axs.flat,x):
  vals=[]
  for y in range(2021,2026):vals.extend(x.loc[f'{y}-06-01':f'{y}-09-30',c].rolling(72,min_periods=58).mean().dropna())
  vals=np.sort(vals);event=x.loc[end-pd.Timedelta(hours=71):end,c].mean();pct=np.mean(vals<event)
  ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),color=blue);ax.axvline(event,color=orange,ls='--');ax.scatter([event],[pct],color=orange,zorder=4);ax.text(.04,.92,f'事件窗口：{event:.3g}\n经验百分位：{pct:.1%}',transform=ax.transAxes,va='top');ax.set(title=c,xlabel=units[c],ylabel='暖季经验累积分布');ax.set_ylim(0,1.03);ax.grid(alpha=.15);rows.append(dict(variable=c,event=event,percentile=pct,windows=len(vals)))
 axs.flat[-1].axis('off');axs.flat[-1].text(.03,.85,'历史背景：2021—2025年6—9月\n统计尺度：72小时均值\n窗口步长：1小时\n有效观测：至少58/72小时\n\n百分位只描述背景中的位置，\n不是灾害概率或显著性。',va='top',linespacing=1.8)
 pd.DataFrame(rows).to_csv(D/'02_暖季经验分布事件位置.csv',index=False);save(fig,'补图02_关键因素暖季经验分布','4.3.1表4-1之后，放在“温度与大气水汽条件分析”之前','本次数据处理后重算的五类指标72h均值与暖季经验分布。积温单位已转为℃·日；若与表4-1旧值不同，应更新表文，不能强改图值。')
 events=pd.read_csv(R/'吉隆县历史灾害_干净版.csv');counts=pd.to_datetime(events['发生日期']).value_counts().sort_index();counts.rename('记录条数').to_csv(D/'03_事件日聚合.csv')
 fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained',gridspec_kw={'width_ratios':[3,1]});axs[0].bar(np.arange(len(counts)),counts,color=blue);axs[0].set_xticks(range(len(counts)),counts.index.strftime('%Y-%m-%d'),rotation=65,ha='right',fontsize=8);axs[0].set(ylabel='同日记录条数',title='按真实记录日期聚合，不扩增独立事件数');years=sorted(set(counts.index.year));nrec=[int(counts[counts.index.year==y].sum()) for y in years];ndays=[int((counts.index.year==y).sum()) for y in years];idx=np.arange(len(years));axs[1].bar(idx-.18,nrec,.36,label='记录数',color=blue);axs[1].bar(idx+.18,ndays,.36,label='独立日期数',color=orange);axs[1].set_xticks(idx,years);axs[1].legend(fontsize=9);save(fig,'补图03_历史记录与独立事件日','5.1.2最后一段之后','34条记录合并为19个日期；2021年32条记录对应17个日期。日期聚合避免共享气象过程被重复计样。','可选正文或附录')
 samples=pd.read_csv(R/'question2/model_samples.csv',index_col=0,parse_dates=True);train=samples.loc[:'2021-07-31'];test=samples.loc['2021-08-16':];cols=['P72','D72'];model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=42));model.fit(train[cols],train.y);scores=model.predict_proba(train[cols])[:,1];fpr,tpr,th=roc_curve(train.y,scores);finite=np.isfinite(th);th,fpr,tpr=th[finite],fpr[finite],tpr[finite];best=np.argmax(tpr-fpr);m=json.loads((R/'question2/results_modified.json').read_text())['comparisons']['P72_D72'];assert abs(th[best]-m['threshold_from_training'])<1e-9
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');order=np.argsort(th)
 for v,label,col in [(tpr,'训练事件日检出比例',blue),(fpr,'训练背景日报警比例',orange),(tpr-fpr,'Youden 指数',green)]:ax.step(th[order],v[order],where='post',label=label,color=col)
 ax.axvline(th[best],color='#555',ls='--',label=f'所选阈值={th[best]:.6f}');ax.set(xlabel='候选评分阈值',ylabel='比例 / 指数',title='仅由训练期确定评分阈值');ax.legend(fontsize=9);pd.DataFrame(dict(threshold=th,tpr=tpr,fpr=fpr,youden=tpr-fpr)).to_csv(D/'04_训练期阈值.csv',index=False);save(fig,'补图04_训练期报警阈值选择','5.2.5最后一句“避免测试信息参与模型训练”之后','按原项目设置重新拟合训练样本，Youden最大值确定阈值；测试标签不参与选阈值。')
 fig,ax=plt.subplots(figsize=(8,5),layout='constrained');px=np.linspace(0,samples.P72.max()*1.05,160);dy=np.linspace(samples.D72.min()-1,samples.D72.max()+1,160);X,Y=np.meshgrid(px,dy);eta=m['intercept_standardized']+m['coefficients_standardized'][0]*(X-m['scaler_mean'][0])/m['scaler_scale'][0]+m['coefficients_standardized'][1]*(Y-m['scaler_mean'][1])/m['scaler_scale'][1];z=1/(1+np.exp(-eta));im=ax.contourf(X,Y,z,levels=np.linspace(0,1,21),cmap='YlOrRd');ax.contour(X,Y,z,levels=[th[best]],colors='#303030',linewidths=1.5)
 for y,marker in [(0,'o'),(1,'*')]:
  for frame,label,edge in [(train,'训练','#333'),(test,'测试','#2470ad')]:
   sel=frame[frame.y==y];ax.scatter(sel.P72,sel.D72,marker=marker,s=30 if y==0 else 110,facecolors='white' if y==0 else edge,edgecolors=edge,label=label+('事件日' if y else '背景日'))
 ax.set(xlabel='72h累计降雨 P72 / mm',ylabel='72h正积温 D72 / ℃·日',title='降雨—正积温联合经验判别边界');ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='模型评分（非校准概率）');save(fig,'补图05_联合经验判别边界','5.4.3末尾，“该判别边界仅适用于……”之后','黑线为训练期评分阈值等值线；分别标出训练/测试及事件/背景样本，仅解释经验判别关系。')
 hold=pd.read_csv(R/'question2/temporal_holdout_P72_D72.csv');cm=confusion_matrix(hold.y,hold.alert,labels=[0,1]);assert cm.tolist()==[[29,14],[1,2]];fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,a,title in [(axs[0],cm,'测试集计数（46日）'),(axs[1],cm/cm.sum(axis=1)[:,None],'按真实类别归一化')]:
  im=ax.imshow(a,cmap='Blues');ax.set_xticks([0,1],['未报警','报警']);ax.set_yticks([0,1],['无记录背景日','记录事件日']);ax.set(xlabel='模型判定',ylabel='记录标签',title=title)
  for i in range(2):
   for j in range(2):ax.text(j,i,f'{a[i,j]:.1%}' if a.dtype.kind=='f' else str(a[i,j]),ha='center',va='center',fontsize=20,color='white' if a[i,j]>a.max()/2 else '#183344')
 save(fig,'补图06_测试集混淆矩阵','5.4.4表2-4之后；可与原表二选一','测试事件日检出2/3，背景日报警14/43。右图按行归一化，避免类别不平衡掩盖漏报。')
 cal=json.loads((R/'q3/Q3_stage2_results/stage2_summary.json').read_text())['calibration'];hourly=pd.read_csv(R/'q3/Q3_stage1_results/hourly_q2_score.csv',index_col=0,parse_dates=True);s=hourly.loc['2026-08-25 10:00','score'];p=1/(1+np.exp(-(cal['intercept']+cal['coef']*np.log(s/(1-s)))));lam=-np.log1p(-p)/24;h=np.linspace(0,48,1000);cdf=(1-np.exp(-lam*h))/(1-np.exp(-lam*48));pdf=lam*np.exp(-lam*h)/(1-np.exp(-lam*48));quant=-np.log(1-np.array([.025,.975])*(1-np.exp(-lam*48)))/lam
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');axs[0].plot(h,pdf,color=blue);mask=(h>=quant[0])&(h<=quant[1]);axs[0].fill_between(h,pdf,where=mask,color=blue,alpha=.25,label='连续公式95%等尾区间');axs[0].set(xlabel='距发布时间 / h',ylabel='条件概率密度 / 1/h');axs[0].legend(fontsize=8);axs[1].plot(h,cdf,color=blue)
 for a in [quant[0],quant[1]]:axs[1].axvline(a,color=orange,ls='--')
 for a in [.025,.975]:axs[1].axhline(a,color='grey',ls=':')
 axs[1].set(xlabel='距发布时间 / h',ylabel='条件累计概率',title='条件：事件在未来48h内发生');fig.suptitle('2026案例的条件时间分布（常数危险率假设）');pd.DataFrame(dict(hours=h,pdf=pdf,cdf=cdf)).to_csv(D/'07_条件时间分布.csv',index=False);save(fig,'补图07_条件时间概率分布','6.2.1式（6-11）及小时离散说明之后','参数来自训练期校准与2026-08-25 10:00评分。连续公式端点用于解释分布，与程序整小时区间存在离散差异；不是实测起动时刻分布。')
 fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
 for i,(a,b,label) in enumerate([(-2,26,'完全包含：确定覆盖'),(3,21,'部分重叠：小时未知'),(27,42,'完全分离：确定未覆盖')]):
  ax.broken_barh([(0,24)],(i-.27,.54),facecolors='#bfc8cf',alpha=.5);ax.plot([a,b],[i,i],lw=5,color=blue);ax.text(44,i,label,va='center')
 ax.set(xlim=(-5,68),ylim=(-.6,2.6),xlabel='相对记录日期00:00 / h',title='日期级标签下的三类区间关系（方法示意，非观测数据）');ax.set_yticks([]);ax.axvline(0,color='grey',ls=':');ax.axvline(24,color='grey',ls=':');save(fig,'补图08_区间删失评价示意','6.3.1覆盖率上下界公式之后','灰带为已知的发生日期范围，蓝线为条件预测区间。示意时间段不代表真实事件，部分重叠不能判定真实小时已覆盖。')
 ea=pd.read_csv(R/'q3/Q3_final_validation_results/test_distinct_event_audit.csv');fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
 for i,row in ea.iterrows():
  if row.any_prior_warning:ax.plot([row.lead_lower_hours,row.lead_upper_exclusive_hours],[i,i],lw=7,color=blue);ax.scatter([row.lead_lower_hours],[i],color=blue);ax.scatter([row.lead_upper_exclusive_hours],[i],facecolors='white',edgecolors=blue,zorder=4);ax.text(51,i,'发布 '+str(row.earliest_prior_warning)[:16],va='center',fontsize=9)
  else:ax.text(5,i,'无满足条件的事前48h报警',color=orange,va='center')
 ax.set_yticks(range(len(ea)),ea.event_date);ax.set(xlim=(0,91),ylim=(-.5,len(ea)-.5),xlabel='距未知真实起动时刻的提前量 / h',title='测试事件的提前量范围（右端点不包含）');save(fig,'补图09_测试事件提前量区间','6.4.2最后一段之后','按覆盖该事件日期的48h预报发布计算提前量为[24,48)h；与正文“持续预警片段从8月16日开始”是不同口径，不能混用。')
 # Initial condition and real simulation output, no rerun needed to render retained snapshot.
 import rasterio
 sys.path.insert(0,str(R/'q4'));from q4_solver import Config,load_terrain,initial_state,PORT
 from q4_viz import shade,bounds
 cfg=Config(grid_m=90);T=load_terrain(R/'q4/data/elevation_utm45n.tif',cfg);depth,_,_,mask=initial_state(T,cfg);extent=np.array(bounds(T))/1000;fig,ax=plt.subplots(figsize=(7,6),layout='constrained');ax.imshow(shade(T),extent=extent,cmap='gray');im=ax.imshow(np.ma.masked_where(depth<=0,depth),extent=extent,cmap='YlOrRd',vmin=0,vmax=50);sx,sy=cfg.source_x/1000,cfg.source_y/1000;ax.set(xlim=(sx-1.6,sx+1.6),ylim=(sy-1.6,sy+1.6),xlabel='UTM45N东坐标 / km',ylabel='UTM45N北坐标 / km',title='90m计算网格的初始释放区');fig.colorbar(im,ax=ax,label='初始等效流深 / m');ax.text(.02,.02,f'体积 {depth.sum()*T.dx*T.dy/1e8:.3f}×10^8 m³\n栅格面积 {mask.sum()*T.dx*T.dy/1e6:.3f} km²\n初始流深 {depth.max():.2f} m',transform=ax.transAxes,bbox=dict(fc='white',alpha=.85));save(fig,'补图10_初始释放区域','7.1.2末尾，紧接“图7-2给出……”','补齐正文已引用但未嵌入的图7-2。释放区为数值近似，不是实测源区边界。')
 arr=np.load(R/'q4/results_90/基准情景逐帧.npz');fig,axs=plt.subplots(2,3,figsize=(12,5.8),layout='constrained');ticks=[0,120,240,360,480,600];limit=80
 for ax,t in zip(axs.flat,ticks):
  k=np.argmin(abs(arr['times']-t));a=arr['frames'][k];ax.imshow(shade(T),extent=extent,cmap='gray');im=ax.imshow(np.ma.masked_where(a<.1,a),extent=extent,cmap='YlOrRd',vmin=0,vmax=limit);ax.scatter(PORT[0]/1000,PORT[1]/1000,marker='^',c='#12a6ac',s=20);ax.set(title=f'{arr["times"][k]:.0f} s',xlabel='UTM东坐标 / km',ylabel='UTM北坐标 / km')
 fig.colorbar(im,ax=axs.ravel().tolist(),label='流深 / m（超过80m取色标上限）',shrink=.7,extend='max');save(fig,'补图11_二维流深时空演化','7.5.1式（7-20）之后、7.5.2之前','90m主情景六个时刻流深，统一色标；蓝色三角为口岸参考点。模型输出不构成实测路径验证。')
 times=pd.read_csv(R/'q4/results_90/模拟过程时序.csv');times.to_csv(D/'12_体积收支时序.csv',index=False);fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');t=times['时间_s'];v=times['运动层体积_m3'];e=times['累计侵蚀体积_m3'];dep=times['累计沉积体积_m3'];budget=1.75e8+e-dep
 axs[0].plot(t,v/1e8,label='数值流动层体积',color=blue);axs[0].plot(t,budget/1e8,'--',label='初始体积＋侵蚀－沉积',color=orange);axs[0].set(xlabel='时间 / s',ylabel='体积 / 10^8 m³');axs[0].legend(fontsize=9);axs[1].plot(t,times['物质量平衡相对误差'],color=green);axs[1].set(xlabel='时间 / s',ylabel='相对体积收支残差');axs[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));save(fig,'补图12_体积收支与数值残差','第八章第一段之后，紧接体积残差−2.97×10⁻⁹的说明','两条体积曲线近乎重合；残差小仅说明数值记账一致，不能替代动力学实测验证。')
 grid=pd.read_csv(R/'q4/grid_sensitivity.csv');grid.to_csv(D/'13_网格敏感性.csv',index=False);fig,axs=plt.subplots(1,3,figsize=(11,4),layout='constrained')
 for ax,col,unit,factor in zip(axs,['arrival_s','累计侵蚀体积_m3','全程峰值速度_mps'],['到达保存时刻 / s','累计侵蚀 / 10^7 m³','全域峰值速度 / (m/s)'],[1,1e7,1]):
  vals=grid[col]/factor;ax.bar(grid.grid_m.astype(str),vals,color=[blue,orange],width=.55);ax.set(xlabel='计算网格 / m',ylabel=unit,ylim=(0,vals.max()*1.2))
  for i,value in enumerate(vals):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
 save(fig,'补图13_多指标网格敏感性','第八章第二段“60m与90m……”之后','两个网格采用相同参数和600s模拟时长。除到达时刻外，峰值速度和侵蚀量也对网格敏感；两种分辨率不足以证明数值收敛。')
 pd.DataFrame(entries).to_csv(O/'插图位置清单.csv',index=False,encoding='utf-8-sig')
 (O/'插图位置与图注.md').write_text('# 新版论文补图位置与图注\n\n共13张PNG，300dpi。标题编号为补图工作编号，插入正文后统一重排图号。原DOCX未修改。\n\n'+ '\n\n'.join(f'## {v["file"]}\n\n插入位置：{v["where"]}（{v["priority"]}）\n\n图注：{v["caption"]}\n\n![{v["file"]}]({v["file"]})' for v in entries),encoding='utf-8')
 print('generated',len(entries),'figures')
if __name__=='__main__':main()
