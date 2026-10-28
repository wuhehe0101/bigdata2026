"""Predict a relative trigger score using past-only inputs, not calibrated risk."""
from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from estimators import MonotoneLogistic
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rain24',type=float,required=True,help='Rain over preceding 0-24 hours, mm')
p.add_argument('--antecedent48',type=float,required=True,help='Rain over preceding 24-72 hours, mm')
p.add_argument('--degree72',type=float,required=True,help='Positive temperature sum over previous 72h /24, degree-days')
g=p.add_mutually_exclusive_group(required=True);g.add_argument('--gate',type=float);g.add_argument('--site',help='Site ID from outputs/site_features.csv')
a=p.parse_args()
if min(a.rain24,a.antecedent48,a.degree72)<0:p.error('Inputs must be nonnegative')
out=Path(__file__).resolve().parent/'outputs'
if a.site:
 sites=pd.read_csv(out/'site_features.csv').set_index('site')
 if a.site not in sites.index:p.error('Unknown site')
 gate=float(sites.loc[a.site,'gate'])
else:gate=a.gate
if not 0<=gate<=1:p.error('Gate must be within [0,1]')
artifact=joblib.load(out/'model.joblib')
x=pd.DataFrame([dict(logP24=np.log1p(a.rain24),logA48=np.log1p(a.antecedent48),D72=a.degree72,gD72=gate*a.degree72)])
score=float(artifact['trigger'].predict_proba(x[artifact['trigger_features']])[:,1][0])
print(json.dumps({'relative_score':score,'empirical_threshold':artifact['threshold'],'above_threshold':score>=artifact['threshold'],'gate':gate,'interpretation':'Uncalibrated retrospective score; not a probability or validated operational warning. Ice-collapse/lake failure thresholds remain unidentified.'},ensure_ascii=False,indent=2))
