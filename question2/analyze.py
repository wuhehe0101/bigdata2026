from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
R=Path(__file__).resolve().parents[1]; O=R/'question2'
e=pd.read_csv(R/'吉隆县历史灾害_干净版.csv');e['date']=pd.to_datetime(e['发生日期'])
m=pd.concat([pd.read_csv(p) for p in sorted((R/'data').glob('MET_*.csv'))],ignore_index=True)
m.index=pd.to_datetime(m.pop('Timestamp'),format='mixed');m=m.sort_index();assert not m.index.duplicated().any()
p=pd.to_numeric(m['Precip(mm)'],errors='coerce');t=pd.to_numeric(m['Ta_2m(degree C)'],errors='coerce')
p=p.where((p>=0)&(p<999));t=t.where((t>-80)&(t<60))
# No interpolation: preserve gaps and avoid using future observations.
h=pd.DataFrame({'P':p,'T':t}).asfreq('h')
f=pd.DataFrame(index=h.index)
for n in [24,72,360]:f[f'P{n}']=h.P.rolling(n,min_periods=n).sum().shift(1)
f['D72']=h['T'].clip(lower=0).rolling(72,min_periods=72).sum().shift(1)/24
# At midnight, all windows end before the event day.
d=f.loc[f.index.hour==0].copy();d['y']=d.index.normalize().isin(e.date.unique()).astype(int)
d.to_csv(O/'daily_features.csv')
audit={'records':len(e),'records_by_year':e.date.dt.year.value_counts().sort_index().to_dict(),'unique_event_days':int(e.date.nunique()),'events_2021':int(e.loc[e.date.dt.year==2021,'date'].nunique()),'missing_event_features':[]}
cols=['P24','P360','D72'];v=d.loc['2021'].copy()
audit['missing_event_features']=[str(x.date()) for x in v.index[(v.y==1)&v[cols].isna().any(axis=1)]]
v=v.dropna(subset=cols)
# Restrict background comparison to the recorded event season, not unseen years.
v=v.loc['2021-04-01':'2021-09-30']; v.to_csv(O/'model_samples.csv')
train=v.loc[:'2021-07-31']; test=v.loc['2021-08-16':] # 15-day embargo
model=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,class_weight='balanced',max_iter=2000,random_state=42))
model.fit(train[cols],train.y)
tr=model.predict_proba(train[cols])[:,1]; fpr,tpr,threshold=roc_curve(train.y,tr); cut=float(threshold[np.argmax(tpr-fpr)])
pr=model.predict_proba(test[cols])[:,1]
audit.update({'features':cols,'train_n':len(train),'train_positive_days':int(train.y.sum()),'test_n':len(test),'test_positive_days':int(test.y.sum()),'threshold_from_training':cut,'test_auc':float(roc_auc_score(test.y,pr)),'test_ap':float(average_precision_score(test.y,pr)),'test_prevalence':float(test.y.mean()),'test_confusion_tn_fp_fn_tp':confusion_matrix(test.y,pr>=cut).ravel().tolist(),'scaler_mean':model[0].mean_.tolist(),'scaler_scale':model[0].scale_.tolist(),'coefficients':model[1].coef_[0].tolist(),'intercept':float(model[1].intercept_[0])})
pd.DataFrame({'y':test.y,'score':pr,'alert':pr>=cut}).to_csv(O/'temporal_holdout.csv')
# Threshold sensitivity to rain accumulation windows, fixed train/test split.
sens=[]
for rain in ['P24','P72','P360']:
 c=[rain,'D72'];a=train.dropna(subset=c);b=test.dropna(subset=c)
 z=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000)).fit(a[c],a.y)
 sens.append({'rain':rain,'test_auc':roc_auc_score(b.y,z.predict_proba(b[c])[:,1])})
audit['window_sensitivity']=sens
v.loc[v.y==1,cols].describe(percentiles=[.1,.5,.9]).to_csv(O/'event_feature_distribution.csv')
# 2026 is a case study, not a new positive-label training sample.
case=d.loc['2026-08-26',cols];audit['case_2026_features']=case.to_dict()
if case.notna().all():audit['case_2026_score']=float(model.predict_proba(case.to_frame().T)[:,1][0])
import xarray as xr
with xr.open_dataset(R/'glacier.nc') as ds:
 audit['glacier_maxima']={k:float(ds[k].where(ds[k]>-999).max()) for k in ['thickness','v','slope'] if k in ds}
(O/'results.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
