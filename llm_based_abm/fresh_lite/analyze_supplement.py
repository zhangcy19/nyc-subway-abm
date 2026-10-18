"""Prepare presentation evidence from the frozen experiment; no API calls."""
import json,gzip,re,collections,os
import pandas as pd
import numpy as np
os.makedirs('../../../presentation_local/build/evidence',exist_ok=True)
P=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str});D=pd.read_csv('res/dates.csv');S=pd.read_csv('res/summary.csv');E=pd.read_csv('res/event_metrics.csv');R=pd.read_csv('res/runtime.csv')
M=['weekday_mean','ridge','rule_timing','direct_few','role_with_history']
Q=P.merge(D[['task','date','event_day']],on=['task','date']);out={'summary':S.to_dict('records'),'events':E.to_dict('records'),'stations':pd.read_csv('res/stations.csv').to_dict('records'),'dataset':pd.read_csv('res/dataset_summary.csv').to_dict('records'),'weather':D[D.event_day].drop_duplicates('date').to_dict('records'),'curves':{}}
for task,part in Q.groupby('task'):
    z=part[part.method.isin(M)].copy()
    # Keep the same stations throughout each curve: exclude station 311 for the 9/29 window.
    counts=z[z.method=='weekday_mean'].groupby('station_complex_id').actual.count();ids=counts[counts==z.date.nunique()].index
    z=z[z.station_complex_id.isin(ids)]
    daily=z.groupby(['date','method'])[['actual','predicted']].sum().reset_index()
    out['curves'][task]=dict(n_stations=len(ids),rows=daily.to_dict('records'))
rows=[]
for m,p in R.groupby('method'):
 rows.append(dict(method=m,seconds=p.seconds.sum(),tokens=int(p.new_tokens.sum()),estimated_cny=p.new_tokens.sum()/3824653*3))
out['cost']=rows
raw={}
with gzip.open('res/api_calls.jsonl.gz','rt') as f:
 for line in f:
  x=json.loads(line)
  if x['status']=='valid' and x['hash'] not in raw:raw[x['hash']]=x
C=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str}).merge(D[['task','date','event_day']],on=['task','date']);C=C[C.method=='role_with_history'].copy()
C['reason']=C.request_hash.map(lambda h:raw[h]['parsed']['reason'])
C['regime']=np.where(C.task.str.startswith('Normal'),'Normal',np.where(C.event_day,'Extreme','Context'))
prob=[];errors=[]
for (regime,group),part in C[C.regime!='Context'].groupby(['regime','group']):
 w=part.effective_population
 prob.append(dict(regime=regime,group=group,**{k:float(np.average(part['p_'+k],weights=w)) for k in ['subway','bus','taxi','walk','stay_home']}))
for (task,group),part in C[(C.regime!='Context')].groupby(['task','group']):
 g=part.assign(pred=part.effective_population*part.p_subway).groupby(['station_complex_id','date']).pred.sum().reset_index()
 actual=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});actual=actual[actual.task==task][['station_complex_id','date',group]]
 g=g.merge(actual,on=['station_complex_id','date']).dropna()
 errors.append(dict(task=task,group=group,regime='Normal' if task.startswith('Normal') else 'Extreme',WAPE_pct=float(abs(g.pred-g[group]).sum()/g[group].sum()*100) if g[group].sum() else None))
out['group_probabilities']=prob;out['group_errors']=pd.DataFrame(errors).groupby(['regime','group']).WAPE_pct.mean().reset_index().to_dict('records')
# Each role-day once; four station copies are not four independent explanations.
texts=C[C.event_day].drop_duplicates('request_hash');stop=set('i my the a an and or to of in for on is it with will would can cannot not as this that be have but at from am by do no due me so if makes make likely preferred some take use'.split())
words=collections.Counter(w for text in texts.reason for w in re.findall('[a-z]+',text.lower()) if len(w)>3 and w not in stop)
out['words']=[dict(word=w,count=c) for w,c in words.most_common(30)]
out['quotes']=texts[['task','agent_id','group','reason','p_subway','p_stay_home']].to_dict('records')
out['roles']=pd.read_csv('res/personas.csv').to_dict('records')
if os.path.exists('res/supplement/ablation_predictions.csv'):
 A=pd.read_csv('res/supplement/ablation_predictions.csv').merge(D[['task','date','event_day']],on=['task','date']);ab=[]
 for (task,m),p in A.groupby(['task','method']):
  p=p if task.startswith('Normal') else p[p.event_day]
  ab.append(dict(task=task,method=m,WAPE_pct=float(abs(p.predicted-p.actual).sum()/p.actual.sum()*100)))
 out['ablation']=ab
if os.path.exists('res/supplement/scenario_predictions.csv'):
 out['scenario']=pd.read_csv('res/supplement/scenario_predictions.csv').drop(columns='actual').to_dict('records')
 out['scenario_decisions']=pd.read_csv('res/supplement/scenario_decisions.csv').to_dict('records')

if os.path.exists('res/supplement/repeated_metrics.csv'):
 out['repeated']=pd.read_csv('res/supplement/repeated_metrics.csv').to_dict('records')
json.dump(out,open('../../../presentation_local/build/evidence/data.json','w'),indent=2,allow_nan=False)
pd.DataFrame(prob).to_csv('res/supplement/group_probabilities.csv',index=False);pd.DataFrame(errors).to_csv('res/supplement/group_errors.csv',index=False)
print('Evidence exported. Role explanations:',len(texts),'unique event-role observations')
