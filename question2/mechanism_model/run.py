"""Reproducible two-layer logistic baseline; outputs are scores, not calibrated probabilities.
Run: /workspace/macap-env/bin/python question2/mechanism_model/run.py
"""
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import xarray as xr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.model_selection import GroupKFold
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'outputs'; OUT.mkdir(exist_ok=True)
SEED=42
INPUTS=[]
def source(p):
 p=Path(p); INPUTS.append(p); return p

from estimators import MonotoneLogistic

def logistic(C=.1):
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=C,max_iter=3000,random_state=SEED))

def metrics(y,p,t=.5):
 y=np.asarray(y);p=np.asarray(p)
 tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel()
 return dict(n=len(y),positive=int(y.sum()),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,ap=float(average_precision_score(y,p)) if y.sum() else None,prevalence=float(y.mean()),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),recall=float(tp/(tp+fn)) if tp+fn else None,precision=float(tp/(tp+fp)) if tp+fp else None,threshold=float(t))

def block_ids(g):
 p=g.geometry.representative_point()
 return (np.floor(p.x/20000).astype(int).astype(str)+'_'+np.floor(p.y/20000).astype(int).astype(str)).values

def weights(frame):
 # Each positive event date has equal total weight; each background date likewise.
 w=1/frame.groupby(['date','y']).y.transform('size').to_numpy(dtype=float)
 n=frame.date.nunique()
 for k in [0,1]:
  ix=frame.y.to_numpy()==k;w[ix]*=n/(2*w[ix].sum())
 assert np.isclose(w.sum(),n)
 return w

print('Extracting inventory and glacier features...',flush=True)
inv=gpd.read_file(source(next(ROOT.glob('（6）*/*/*.shp')))).to_crs(32645)
inv['ice_label']=(inv['类型']!='降雨型').astype(int)
inv['block']=block_ids(inv)
events=pd.read_csv(source(ROOT/'吉隆县历史灾害_干净版.csv'));events['date']=pd.to_datetime(events['发生日期'])
events['site']=events['经度'].round(6).astype(str)+'_'+events['纬度'].round(6).astype(str)
sites=gpd.GeoDataFrame(events.drop_duplicates('site').copy(),geometry=gpd.points_from_xy(events.drop_duplicates('site')['经度'],events.drop_duplicates('site')['纬度']),crs=4326).to_crs(32645).reset_index(drop=True)
sites['block']=block_ids(sites)
glaciers=gpd.read_file(source(next(ROOT.rglob('RGI_V7*.shp')))).to_crs(32645)
# Use the provided polygons consistently; nearest polygon is a proximity proxy, not a hydrological link.
thinpath=source(ROOT/'GIS成果/dhdt_2015_2020_mosaic.tif')
cache={}
def spatial_features(g):
 query=g[['geometry']].copy();query.geometry=query.geometry.representative_point()
 near=gpd.sjoin_nearest(query,glaciers[['geometry']],how='left',distance_col='distance_m')
 # Intersections may tie. Deterministic lowest glacier index, recorded explicitly.
 near=near.reset_index().sort_values(['index','index_right']).drop_duplicates('index').set_index('index').reindex(g.index)
 vals=[]
 with rasterio.open(thinpath) as src:
  for idx,row in near.iterrows():
   gi=int(row.index_right);gl=glaciers.loc[gi]
   if gi not in cache:
    try:
     a,_=mask(src,[gl.geometry],crop=True,filled=False)
     v=a.compressed();v=v[np.isfinite(v)&(np.abs(v)<100)]
     cache[gi]=(float(np.mean(v)) if len(v) else np.nan,int(len(v)))
    except ValueError:cache[gi]=(np.nan,0)
   dh,n=cache[gi]
   vals.append(dict(glacier_id=gl.rgi_id,distance_m=row.distance_m,log_distance=np.log1p(row.distance_m/1000),log_area=np.log1p(float(gl.area_km2)),glacier_slope=float(gl.slope_deg),dhdt=dh,thinning=max(-dh,0) if np.isfinite(dh) else np.nan,dhdt_pixels=n))
 return pd.DataFrame(vals,index=g.index)
S=spatial_features(inv); E=spatial_features(sites)
SC=['log_distance','log_area','glacier_slope','thinning']
# Spatial blocked validation of mechanism gate (NOT occurrence validation).
gate_oof=np.full(len(inv),np.nan);gate_folds=[]
for fold,(a,b) in enumerate(GroupKFold(n_splits=min(4,inv.block.nunique())).split(S,inv.ice_label,groups=inv.block)):
 gate=logistic();gate.fit(S.iloc[a][SC],inv.ice_label.iloc[a]);gate_oof[b]=gate.predict_proba(S.iloc[b][SC])[:,1]
 gate_folds.append({'fold':fold,**metrics(inv.ice_label.iloc[b],gate_oof[b]),'test_blocks':sorted(inv.block.iloc[b].unique().tolist())})
gate=logistic().fit(S[SC],inv.ice_label)
E['gate']=gate.predict_proba(E[SC])[:,1]
S['gate_oof']=gate_oof;S['type']=inv['类型'];S['block']=inv.block
S.to_csv(OUT/'inventory_features.csv',index=False)
E['site']=sites.site;E['block']=sites.block;E.to_csv(OUT/'site_features.csv',index=False)
# Match audit: nearest does not establish identity.
match=gpd.sjoin_nearest(sites[['site','geometry']],inv[['类型','geometry']],distance_col='distance_m')
match.drop(columns='geometry').to_csv(OUT/'catalog_match_audit.csv',index=False)

print('Building strictly past-only weather windows...',flush=True)
m=pd.concat([pd.read_csv(source(p)) for p in sorted((ROOT/'data').glob('MET_*.csv'))],ignore_index=True)
m.index=pd.to_datetime(m.pop('Timestamp'),format='mixed');m=m.sort_index();assert m.index.is_unique
p=pd.to_numeric(m['Precip(mm)'],errors='coerce');t=pd.to_numeric(m['Ta_2m(degree C)'],errors='coerce')
h=pd.DataFrame({'P':p.where((p>=0)&(p<999)),'T':t.where((t>-80)&(t<60))}).asfreq('h')
def features(h):
 f=pd.DataFrame(index=h.index)
 f['P24']=h.P.rolling(24,min_periods=24).sum().shift(1)
 f['A48']=h.P.shift(24).rolling(48,min_periods=48).sum().shift(1)
 f['D72']=h['T'].clip(lower=0).rolling(72,min_periods=72).sum().shift(1)/24
 return f.loc[f.index.hour==0]
d=features(h); d.index.name='date';d.to_csv(OUT/'daily_weather.csv')
# Causality test: modifying observations at and after a date must not alter that date's predictors.
probe=pd.Timestamp('2021-08-18');altered=h.loc[probe-pd.Timedelta(days=5):probe+pd.Timedelta(days=1)].copy();altered.loc[probe:]=123
assert np.allclose(features(altered).loc[probe],d.loc[probe],equal_nan=True)
period=d.loc['2021-04-01':'2021-09-30'];usable=period.dropna()
active=sites[sites.site.isin(events.loc[events.date.dt.year==2021,'site'])].copy()
panel=active[['site','block']].merge(usable.reset_index(),how='cross')
pos=set(zip(events.site,events.date));panel['y']=[int((s,t) in pos) for s,t in zip(panel.site,panel.date)]
panel=panel.merge(E[['site','gate']],on='site',validate='many_to_one')
panel['logP24']=np.log1p(panel.P24);panel['logA48']=np.log1p(panel.A48);panel['gD72']=panel.gate*panel.D72
COLS=['logP24','logA48','D72','gD72'];BASE=COLS[:-1]
train=panel[panel.date<='2021-07-31'].copy();test=panel[panel.date>='2021-08-04'].copy() # 72h embargo
assert train.date.max()+pd.Timedelta(hours=72)<test.date.min()
assert not set(train.date)&set(test.date)
assert train.y.sum()>0 and test.y.sum()>0
panel.to_csv(OUT/'site_day_panel.csv',index=False)

def daily_prediction(frame,p):
 q=frame[['date','y']].copy();q['score']=p
 # Regional aggregation is max over the fixed set of historical sites, not a calibrated union probability.
 return q.groupby('date').agg(y=('y','max'),score=('score','max')).reset_index()

def threshold(y,p):
 fpr,tpr,ts=roc_curve(y,p);valid=np.isfinite(ts);idx=np.argmax((tpr-fpr)[valid]);return float(ts[valid][idx])

def fit(frame,cols,tree=False,monotone=False):
 model=(make_pipeline(SimpleImputer(strategy='median'),DecisionTreeClassifier(max_depth=2,min_samples_leaf=100,random_state=SEED)) if tree else logistic())
 if monotone:model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MonotoneLogistic())
 model.fit(frame[cols],frame.y,**{model.steps[-1][0]+'__sample_weight':weights(frame)})
 return model

results={};predictions={};fitted={}
for name,cols,tree in [('plain_logistic',BASE,False),('gated_logistic',COLS,False),('monotone_logistic',COLS,False),('shallow_tree',COLS,True)]:
 z=fit(train,cols,tree,monotone=name=='monotone_logistic');fitted[name]=z
 qtrain=daily_prediction(train,z.predict_proba(train[cols])[:,1]);cut=threshold(qtrain.y,qtrain.score)
 ptest=z.predict_proba(test[cols])[:,1];q=daily_prediction(test,ptest);predictions[name]=q
 results[name]={'regional_daily':metrics(q.y,q.score,cut),'site_day':metrics(test.y,ptest,cut)}
 q.to_csv(OUT/(name+'_time_holdout.csv'),index=False)

# Fixed 7-day block resampling of held-out days; acknowledges dependence and reports valid replicates.
rng=np.random.default_rng(SEED);q=predictions['monotone_logistic'];groups=((q.date-q.date.min()).dt.days//7).to_numpy();blocks=np.unique(groups);aucs=[]
for _ in range(1000):
 chosen=rng.choice(blocks,len(blocks),replace=True);ix=np.concatenate([np.flatnonzero(groups==b) for b in chosen]);qq=q.iloc[ix]
 if qq.y.nunique()==2:aucs.append(roc_auc_score(qq.y,qq.score))
results['block_bootstrap']={'valid_replicates':len(aucs),'requested':1000,'auc_95_percentile':np.quantile(aucs,[.025,.975]).tolist() if aucs else None}

# Stronger transfer check: hold out entire 20-km blocks AND future dates.
# Also remove auxiliary inventory blocks used by held-out sites from fitting the gate.
space=[]
for block in sorted(active.block.unique()):
 a=train[train.block!=block].copy();b=test[test.block==block].copy();ig=inv.block!=block
 if b.y.sum()==0 or a.y.nunique()!=2 or inv.loc[ig,'ice_label'].nunique()!=2:continue
 sg=logistic().fit(S.loc[ig,SC],inv.loc[ig,'ice_label'])
 eg=sg.predict_proba(E[SC])[:,1];mapping=dict(zip(E.site,eg))
 for f in [a,b]:f['gate']=f.site.map(mapping);f['gD72']=f.gate*f.D72
 z=fit(a,COLS,monotone=True);pa=z.predict_proba(a[COLS])[:,1];qa=daily_prediction(a,pa);cut=threshold(qa.y,qa.score)
 qb=daily_prediction(b,z.predict_proba(b[COLS])[:,1]);space.append({'held_block':block,**metrics(qb.y,qb.score,cut),'train_sites':a.site.nunique(),'test_sites':b.site.nunique(),'removed_inventory':int((~ig).sum())})

# Save actual parameters and artifact, including imputer/scaler for exact reproduction.
model=fitted['monotone_logistic'];scaler=model[1];lr=model[2]
coef=pd.DataFrame({'feature':COLS,'mean':scaler.mean_,'scale':scaler.scale_,'standardized_beta':lr.coef_[0],'raw_beta':lr.coef_[0]/scaler.scale_})
coef.to_csv(OUT/'trigger_coefficients.csv',index=False)
gatecoef=pd.DataFrame({'feature':SC,'mean':gate[1].mean_,'scale':gate[1].scale_,'standardized_beta':gate[2].coef_[0]})
gatecoef.to_csv(OUT/'gate_coefficients.csv',index=False)
# Verify monotonicity numerically in all four model inputs.
x=test[COLS].iloc[:20].copy();p0=model.predict_proba(x)[:,1]
for c in COLS:
 xx=x.copy();xx[c]+=1
 assert np.all(model.predict_proba(xx)[:,1]>=p0-1e-10)

b0=float(lr.intercept_[0]-np.sum(lr.coef_[0]*scaler.mean_/scaler.scale_));betas=dict(zip(COLS,coef.raw_beta));cut=results['monotone_logistic']['regional_daily']['threshold']
# Critical rain as a CONDITIONAL model boundary. Negative beta => upper boundary, not physical threshold.
crit=[]
for g in [0,.5,1]:
 for D in [20,30,40]:
  P=1.;den=betas['logA48']
  z=(np.log(cut/(1-cut))-b0-betas['logP24']*np.log1p(P)-(betas['D72']+g*betas['gD72'])*D)/den if den>1e-10 else np.nan
  rain=float(np.expm1(z)) if np.isfinite(z) and -700<z<700 else None
  crit.append({'gate':g,'D72':D,'P24':P,'algebraic_A48_boundary':rain,'direction':'lower' if den>1e-10 else 'unidentified','within_training_range':bool(rain is not None and train.A48.min()<=rain<=train.A48.max())})
pd.DataFrame(crit).to_csv(OUT/'conditional_thresholds.csv',index=False)
joblib.dump(fitted['gated_logistic'],OUT/'unconstrained_model.joblib')
joblib.dump({'gate':gate,'trigger':model,'gate_features':SC,'trigger_features':COLS,'threshold':cut,'training_cutoff':'2021-07-31','seed':SEED},OUT/'model.joblib')
loaded=joblib.load(OUT/'model.joblib');assert np.allclose(model.predict_proba(test[COLS]),loaded['trigger'].predict_proba(test[COLS]))

# Independent case reports: unknown event location/type labels cannot be invented.
cases=[]
for day,mechanism in [('2025-07-08','glacial-lake outburst (user supplied)'),('2026-08-26','ice collapse (problem statement)')]:
 row=d.loc[day]
 for g in [0,.5,1]:
  x=pd.DataFrame([dict(logP24=np.log1p(row.P24),logA48=np.log1p(row.A48),D72=row.D72,gD72=g*row.D72)])
  score=float(model.predict_proba(x[COLS])[:,1][0]) if row.notna().all() else None
  cases.append({'date':day,'mechanism':mechanism,'assumed_gate':g,**row.to_dict(),'score':score,'above_cutoff':bool(score>=cut) if score is not None else None})
pd.DataFrame(cases).to_csv(OUT/'independent_cases.csv',index=False)
with xr.open_dataset(source(next((ROOT/'(4)冰川数据').glob('*.nc')))) as ds:
 glacier={k:{'max':float(ds[k].where(ds[k]>-999).max()),'unit':ds[k].attrs.get('units','not declared')} for k in ['thickness','v','slope']}
 glacier['scope']='One glacier only; no universal collapse threshold or volume/energy inferred.'

# Training-only 7-day moving-block resampling: stability of critical surface, not independent validation.
bootstrap_rng=np.random.default_rng(SEED)
train_blocks=((train.date-train.date.min()).dt.days//7)
unique_blocks=np.sort(train_blocks.unique());bootstrap=[]
for repeat in range(200):
 pieces=[]
 for draw,block in enumerate(bootstrap_rng.choice(unique_blocks,len(unique_blocks),replace=True)):
  piece=train.loc[train_blocks==block].copy()
  # Distinct replicated clusters retain bootstrap multiplicity in the event-date weights.
  piece['date']=piece['date']+pd.Timedelta(days=draw*400)
  pieces.append(piece)
 bt=pd.concat(pieces,ignore_index=True)
 if bt.y.nunique()<2:continue
 bm=fit(bt,COLS,monotone=True);bp=bm.predict_proba(bt[COLS])[:,1];bq=daily_prediction(bt,bp);bc=threshold(bq.y,bq.score)
 rb=bm[2].coef_[0]/bm[1].scale_;ri=float(bm[2].intercept_[0]-np.dot(rb,bm[1].mean_))
 boundary=None
 if rb[1]>1e-8:
  val=(np.log(bc/(1-bc))-ri-rb[0]*np.log(2)-(rb[2]+.5*rb[3])*30)/rb[1]
  if -700<val<700:boundary=float(np.expm1(val))
 bootstrap.append({'replicate':repeat,'threshold':bc,'beta_P24':rb[0],'beta_A48':rb[1],'beta_D72':rb[2],'beta_gD72':rb[3],'A48_at_D30_g05_P1':boundary})
bdf=pd.DataFrame(bootstrap);bdf.to_csv(OUT/'training_block_bootstrap.csv',index=False)
stability={'replicates':len(bdf),'ice_interaction_zero_fraction':float((bdf.beta_gD72<1e-8).mean()),'rain24_zero_fraction':float((bdf.beta_P24<1e-8).mean()),'A48_boundary_finite_count':int(bdf.A48_at_D30_g05_P1.notna().sum()),'A48_boundary_percentiles':bdf.A48_at_D30_g05_P1.dropna().quantile([.025,.5,.975]).to_dict()}

summary={'training_threshold_stability':stability,'inventory_types':inv['类型'].value_counts().to_dict(),'gate_spatial_oof':metrics(inv.ice_label,gate_oof),'gate_folds':gate_folds,'events_by_year':events.date.dt.year.value_counts().sort_index().to_dict(),'unique_days_2021':int(events.loc[events.date.dt.year==2021,'date'].nunique()),'missing_event_days':[str(t.date()) for t in events.loc[events.date.dt.year==2021,'date'].unique() if t not in usable.index],'panel_sites':active.site.nunique(),'train_days':train.date.nunique(),'train_event_days':train.loc[train.y==1,'date'].nunique(),'test_days':test.date.nunique(),'test_event_days':test.loc[test.y==1,'date'].nunique(),'results':results,'space_time_holdouts':space,'raw_intercept':b0,'raw_coefficients':betas,'threshold':cut,'glacier_diagnostics':glacier,'thin_missing_inventory':int(S.thinning.isna().sum()),'thin_missing_sites':int(E.thinning.isna().sum()),'validation_assertions':'past-only features, embargo, disjoint dates, effective date weights, model serialization passed'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# Include all sidecars of vector inputs in integrity manifest.
allinputs=set(INPUTS)
for p in INPUTS:
 if p.suffix=='.shp':allinputs.update(x for x in p.parent.glob(p.stem+'.*') if x.suffix in ['.shp','.shx','.dbf','.prj','.cpg'])
manifest=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(allinputs)]
(OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
# Compact validation plots, using ASCII labels for portable rendering.
fig,axs=plt.subplots(1,3,figsize=(15,4))
for name,q in predictions.items():
 fpr,tpr,_=roc_curve(q.y,q.score);axs[0].plot(fpr,tpr,label=name)
axs[0].plot([0,1],[0,1],'k--');axs[0].set(xlabel='False-positive rate',ylabel='Recall',title='Future-day holdout ROC');axs[0].legend(fontsize=7)
q=predictions['monotone_logistic'];axs[1].plot(q.date,q.score);axs[1].scatter(q.loc[q.y==1,'date'],q.loc[q.y==1,'score'],c='r',label='Recorded event day');axs[1].axhline(cut,ls='--',c='gray');axs[1].set(title='Monotone logistic: unseen dates',ylabel='Relative score');axs[1].tick_params(axis='x',rotation=45);axs[1].legend(fontsize=7)
axs[2].barh(COLS,coef.standardized_beta);axs[2].axvline(0,c='gray');axs[2].set(title='Standardized coefficients')
fig.tight_layout();fig.savefig(OUT/'validation.png',dpi=180);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
