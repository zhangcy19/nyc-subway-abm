"""Daily, independent normal/event tasks. Run primary, repeat, transfer, then analysis."""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from api import request
from model import profiles,rule_prob,fit_population,predict
from prepare import GROUPS

PEOPLE=profiles()
WORKERS=int(os.environ.get('ABM_WORKERS','6'))
SHRINK=[0.,.1,1.,10.]
CALENDAR_PRIOR=[0.,1.,3.]
RAIN=[0.,.25,.5,1.,2.]
WEATHER=['weekday','school_term','holiday','rain_mm','peak_mm','six_mm','temp_c','wind_ms']

def environment(row):
    result={k:round(float(row[k]),2) if k not in ['weekday','school_term','holiday'] else int(row[k]) for k in WEATHER}
    result['weekday']=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][int(row['weekday'])]
    return result

def examples(pool,target,n):
    """All exemplars precede target and belong to this task's training split only."""
    p=pool[pool.date<target['date']].copy()
    p['distance']=abs(np.log1p(p.rain_mm)-np.log1p(target['rain_mm']))+1.*(p.weekday!=target['weekday'])+.5*(p.school_term!=target['school_term'])
    return p.sort_values(['distance','date'],ascending=[True,False]).head(n).sort_values('date')

def group_history(train,group,target):
    complete=train[train.observed_hours==24]
    # Pool station-normalized group demand: the index is not a person's observed choice probability.
    past=complete[complete.date<target['date']].copy()
    if past.empty:return []
    means=past.groupby('station_complex_id')[group].transform('mean')
    past['entry_index']=past[group]/means.replace(0,np.nan)
    daily=past.groupby('date').agg(**{k:(k,'first') for k in WEATHER},entry_index=('entry_index','mean')).reset_index().dropna()
    return [{'calendar_weather':environment(r),'observed_group_entry_index':round(float(r.entry_index),3)} for _,r in examples(daily,target,4).iterrows()]

def role_view(person,row,train,few):
    view={'person':{k:v for k,v in person.items() if k not in ['agent_id','group','kind']},'target':environment(row),
        'scenario':{'cost_usd':{'subway':3,'bus':3,'taxi':20,'walk':0},'minutes':{'subway':25+person['access_walk_minutes'],'bus':45,'taxi':25,'walk':65}},
        'scope':'hypothetical profile and mode attributes; no observed personal history'}
    if few:
        view['group_history']=group_history(train,person['group'],row)
        view['history_definition']='Earlier TRAINING observations of your fare group, averaged across anonymous stations; index 1 is its earlier-training mean. Not an action probability or your own past decision.'
    return view

def role_policy(target_dates,train,few,replicate=0,offline=False):
    jobs=[(r,p) for _,r in target_dates.iterrows() for p in PEOPLE]
    def run(job):
        r,p=job
        answer=request('role',role_view(p,r,train,few),p,replicate,offline)
        return (r.date,p['agent_id']),answer
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:return dict(pool.map(run,jobs))

def rule_policy(dates,scale):
    return {(r.date,p['agent_id']):dict(probabilities=rule_prob(p,r,scale).tolist(),reason='Multinomial utility rule with daily participation and mode choice.',hash='') for _,r in dates.iterrows() for p in PEOPLE}

def direct_predictions(targets,train,replicate=0,offline=False):
    def run(item):
        _,r=item;past=train[(train.station_complex_id==r.station_complex_id)&train.total_ridership.notna()]
        selected=examples(past,r,8)
        view={'target':environment(r),'observed_training_examples':[{'input':environment(e),'entries':round(float(e.total_ridership),1)} for _,e in selected.iterrows()]}
        answer=request('direct',view,replicate=replicate,offline=offline)
        return dict(station_complex_id=r.station_complex_id,date=r.date,actual=r.total_ridership,predicted=answer['entries'],request_hash=answer['hash'],reason=answer['reason'],example_count=len(selected))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:return list(pool.map(run,targets.iterrows()))

def wape(rows):
    x=pd.DataFrame(rows).dropna(subset=['actual'])
    return float(abs(x.predicted-x.actual).sum()/x.actual.sum())

def feature(r):
    return [int(r.weekday==i) for i in range(7)]+[r.school_term,r.holiday,np.log1p(r.rain_mm),np.log1p(r.peak_mm),r.temp_c/20,r.wind_ms/10]

def numerical(train,valid,test):
    candidates=[];fits={}
    for alpha in [.1,1,10,100]:
        models={};rows=[]
        for station,t in train.groupby('station_complex_id'):
            t=t[t.total_ridership.notna()];model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(np.array([feature(r) for _,r in t.iterrows()]),np.log1p(t.total_ridership));models[station]=model
            for _,r in valid[valid.station_complex_id==station].iterrows():rows.append(dict(actual=r.total_ridership,predicted=max(0,float(np.expm1(model.predict([feature(r)])[0])))))
        candidates.append(dict(method='ridge',alpha=alpha,validation_wape=wape(rows)));fits[alpha]=models
    best=min(candidates,key=lambda r:r['validation_wape']);models=fits[best['alpha']]
    rows=[]
    for _,r in test.iterrows():rows.append(dict(station_complex_id=r.station_complex_id,date=r.date,actual=r.total_ridership,predicted=max(0,float(np.expm1(models[r.station_complex_id].predict([feature(r)])[0])))))
    return rows,candidates,models

def weekday_mean(train,test):
    rows=[]
    for _,r in test.iterrows():
        t=train[(train.station_complex_id==r.station_complex_id)&train.total_ridership.notna()];same=t[t.weekday==r.weekday]
        rows.append(dict(station_complex_id=r.station_complex_id,date=r.date,actual=r.total_ridership,predicted=float((same if len(same) else t).total_ridership.mean())))
    return rows

def choose_population(train,valid,policies,method):
    candidates=[];fits={}
    for scale,policy in policies.items():
        for shrink in SHRINK:
            for calendar_prior in CALENDAR_PRIOR:
                weights=fit_population(train,policy,PEOPLE,shrink,calendar_prior)
                predictions,_=predict(valid,weights,policy,PEOPLE)
                score=wape(predictions)
                candidates.append(dict(method=method,rain_scale=scale,shrink=shrink,calendar_prior=calendar_prior,validation_wape=score))
                fits[(scale,shrink,calendar_prior)]=weights
    best=min(candidates,key=lambda r:r['validation_wape'])
    return fits[(best['rain_scale'],best['shrink'],best['calendar_prior'])],policies[best['rain_scale']],candidates,best
