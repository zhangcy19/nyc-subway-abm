"""Frozen development hyperparameters; staged expansion, stop on any quality gate.
Role policies are original real API outputs, not newly sampled or altered.
Direct comparator is current frozen direct predictions. No API or network.
"""
import os,json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from design import inputs,EVENTS,SEASON,normal_history,event_data,METHODS
from experiment import PEOPLE,feature,weekday_mean
from timing import timed_rule_policy
from model import fit_population,predict
from staged import metrics
os.makedirs('rollback_half_res',exist_ok=True)
order=pd.read_csv('staged_res/batches.csv',dtype={'station_complex_id':str}).sort_values('position')
dev=order.station_complex_id.head(12).tolist();exp=order.station_complex_id.iloc[12:76].tolist()
v=pd.read_csv('rollback_dev_res/validation.csv');best=v.loc[v.groupby(['task','method']).validation_wape.idxmin()]
best.to_csv('rollback_half_res/frozen_parameters.csv',index=False)
config=best.set_index(['task','method']).to_dict('index')
existing=pd.read_csv('staged_res/predictions.csv',dtype={'station_complex_id':str})
old=pd.read_csv('rollback_dev_res/predictions.csv',dtype={'station_complex_id':str})
allrows=old.to_dict('records');weights={};gates=[]
json.dump(dict(development=dev,expansion=exp,api_calls=0,role_source='../event_holdout/res',direct_source='staged_res/predictions.csv',population='Normal fitted per station; hyperparameters frozen on 12 development stations',gate='Both Normal and Extreme: role few > role zero AND best LLM >= best numerical, required per batch and cumulative expansion',note='Previously evaluated data; expansion is not a pristine new test set. No hyperparameter tuning on expansion scores.'),open('rollback_half_res/protocol.json','w'),indent=2)
def score(rows,ids):
 p,w=inputs(ids);y=pd.DataFrame(rows);y=y[y.station_complex_id.isin(ids)].merge(p[['task','station_complex_id','date','event_day']],on=['task','station_complex_id','date'],validate='many_to_one');y['regime']=np.where(y.task.str.startswith('Normal'),'Normal','Extreme')
 t=metrics(y[(y.regime=='Normal')|y.event_day]);s=t.groupby(['regime','method']).score.mean().unstack('regime')
 ok=all(s.loc['role_with_history',r]>s.loc['role_no_history',r] and s.loc[['direct_few','role_no_history','role_with_history'],r].max()>=s[r].max()-1e-9 for r in ['Normal','Extreme'])
 return s,t,bool(ok)
for k,ids in enumerate([exp[:20],exp[20:40],exp[40:]],1):
 p,w=inputs(ids);rows=[]
 for phase in ['Normal_Summer','Normal_School']+list(EVENTS):
  season=phase if phase.startswith('Normal') else SEASON[phase];train=normal_history(p,season);part=p[p.task==phase];test=part[part.split=='test']
  for method in METHODS:
   if method=='weekday_mean':pr=weekday_mean(train,test)
   elif method=='direct_few':pr=existing[(existing.task==phase)&existing.method.eq(method)&existing.station_complex_id.isin(ids)].drop(columns=['task','method']).to_dict('records')
   elif method=='ridge':
    alpha=config[(season,method)]['alpha'];learn=train if phase.startswith('Normal') else pd.concat([train]+[event_data(p,t) for t in EVENTS if t!=phase]).drop_duplicates(['station_complex_id','date']);pr=[]
    for sid,g in learn.groupby('station_complex_id'):
     m=make_pipeline(StandardScaler(),Ridge(alpha=alpha));m.fit(np.array([feature(r) for _,r in g.iterrows()]),np.log1p(g.total_ridership))
     for _,r in test[test.station_complex_id==sid].iterrows():pr.append(dict(station_complex_id=sid,date=r.date,actual=r.total_ridership,predicted=max(0,float(np.expm1(m.predict([feature(r)])[0])))))
   else:
    if method.startswith('role'):
     pol={(r['date'],r['agent_id']):dict(probabilities=r['probabilities'],hash=r['hash'],reason='Frozen original reply') for r in json.load(open(f'../event_holdout/res/{phase}_{method}_policy.json'))}
    else:pol=timed_rule_policy(part.drop_duplicates('date'),w,config[(phase,method)]['rain_scale'],method=='rule_timing')
    if phase.startswith('Normal'):
     c=config[(phase,method)];weights[(phase,method)]=fit_population(train,pol,PEOPLE,c['shrink'],c['calendar_prior'])
    pr,dec=predict(test,weights[(season,method)],pol,PEOPLE)
   rows.extend(dict(task=phase,method=method,**r) for r in pr)
  print('batch',k,phase,'done',flush=True)
 allrows.extend(rows);pd.DataFrame(allrows).to_csv('rollback_half_res/predictions.csv',index=False)
 cumulative=exp[:sum([20,20,24][:k])];s,t,ok=score(rows,ids);cs,ct,cok=score(allrows,cumulative)
 s.to_csv(f'rollback_half_res/batch_{k}_summary.csv');t.to_csv(f'rollback_half_res/batch_{k}_tasks.csv');cs.to_csv('rollback_half_res/expansion_summary.csv');ct.to_csv('rollback_half_res/expansion_tasks.csv')
 whole,wt,_=score(allrows,dev+cumulative);whole.to_csv('rollback_half_res/all_summary.csv');wt.to_csv('rollback_half_res/all_tasks.csv')
 gates.append(dict(batch=k,stations=ids,batch_pass=ok,cumulative_expansion_pass=cok));json.dump(gates,open('rollback_half_res/gates.json','w'),indent=2)
 print('BATCH',k,s.round(3).to_string(), 'PASS',ok,cok,flush=True)
 if not(ok and cok):print('STOP: frozen expansion quality gate failed; no further stations run.',flush=True);break
