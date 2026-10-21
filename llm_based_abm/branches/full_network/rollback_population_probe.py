"""Replay earliest frozen role strategies on 12 development stations.
Compare event-specific calibration with seasonal Normal transfer; no paid calls.
"""
import os,json,glob
import pandas as pd
import numpy as np
from prepare import GROUPS,CHANNELS
from experiment import PEOPLE,choose_population
from model import predict
os.makedirs('rollback_population_res',exist_ok=True)
ids=pd.read_csv('staged_res/batches.csv',dtype={'station_complex_id':str}).sort_values('position').station_complex_id.head(12).tolist()
dates=pd.read_csv('../fresh_lite/res/dates.csv')
if not os.path.exists('rollback_population_res/panel.csv'):
 parts=[]
 cols=['station_complex_id','timestamp','total_ridership']+sum(CHANNELS.values(),[])
 for f in ['../../../data/station_hourly.csv.gz']+sorted(glob.glob('../../../data/extensions/*.csv.gz')):
  for x in pd.read_csv(f,usecols=cols,dtype={'station_complex_id':str},chunksize=100000):
   x=x[x.station_complex_id.isin(ids)].copy();x['date']=x.timestamp.str[:10]
   for g,cs in CHANNELS.items():x[g]=x[cs].sum(axis=1)
   parts.append(x[['station_complex_id','date','total_ridership']+GROUPS])
 x=pd.concat(parts);g=x.groupby(['station_complex_id','date']);daily=g[['total_ridership']+GROUPS].sum(min_count=24).reset_index();daily['observed_hours']=g.total_ridership.count().values
 p=dates.merge(daily,on='date');p.loc[p.observed_hours.ne(24),['total_ridership']+GROUPS]=np.nan;p.to_csv('rollback_population_res/panel.csv',index=False)
p=pd.read_csv('rollback_population_res/panel.csv',dtype={'station_complex_id':str})
records=[];fits={};decisions=[]
for task in ['Normal_Summer','Normal_School','Extreme_0729','Extreme_0813','Extreme_0929','Extreme_1007']:
 part=p[p.task==task];train=part[part.split=='train'];valid=part[part.split=='validation'];test=part[part.split=='test']
 for method in ['role_no_history','role_with_history']:
  saved=json.load(open(f'../fresh_lite/res/{task}_{method}_policy.json'));pol={(r['date'],r['agent_id']):dict(probabilities=r['probabilities'],hash=r.get('hash',''),reason='Original fresh_lite response') for r in saved}
  fitted,policy,candidates,best=choose_population(train,valid,{0.:pol},method);fits[(task,method)]=fitted
  variants={'own_window':fitted}
  if task.startswith('Extreme'):
   season='Normal_Summer' if task in ['Extreme_0729','Extreme_0813'] else 'Normal_School';variants['normal_transfer']=fits[(season,method)]
  for variant,weights in variants.items():
   rows,dec=predict(test,weights,policy,PEOPLE)
   records.extend(dict(task=task,method=method,variant=variant,**r) for r in rows)
  print('done',task,method,flush=True)
y=pd.DataFrame(records).merge(dates[['task','date','event_day']],on=['task','date']);y.to_csv('rollback_population_res/predictions.csv',index=False)
y['regime']=np.where(y.task.str.startswith('Normal'),'Normal','Extreme');z=y[(y.regime=='Normal')|y.event_day];z=z.copy();z['ae']=abs(z.predicted-z.actual)
t=z.groupby(['regime','task','method','variant'])[['ae','actual']].sum();t['score']=100*(1-t.ae/t.actual);t.reset_index().to_csv('rollback_population_res/task_scores.csv',index=False)
print(t.groupby(['regime','method','variant']).score.mean().round(3).to_string())
json.dump(dict(stations=ids,api_calls=0,policy_source='../fresh_lite/res',note='Development replay; original four-station historical group context retained. Own-window uses event-specific chronological train/validation; transfer uses Normal fitted weights. Different historical protocol from event-holdout, do not mix headline scores.'),open('rollback_population_res/protocol.json','w'),indent=2)
