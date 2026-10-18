"""Two additional paired event-day samples; fixed weights, no model selection."""
import json,time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import supplement_api as api
from experiment import PEOPLE,examples
from timing import hourly_weather,person_view,weather_view
from model import predict
p=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});w=hourly_weather();jobs=[]
for task,part in p[p.task.str.startswith('Extreme')].groupby('task'):
 train=part[part.split=='train'];targets=part[part.event_day&part.total_ridership.notna()];r=targets.iloc[0]
 for replicate in [1,2]:
  for timed in [False,True]:
   for person in PEOPLE:jobs.append(dict(task=task,replicate=replicate,timed=timed,route='role',person=person,date=r.date,view=person_view(person,r,train,w,True,timed)))
   for _,r in targets.iterrows():
    past=train[(train.station_complex_id==r.station_complex_id)&train.total_ridership.notna()]
    view={'target':weather_view(r,w,timed),'observed_training_examples':[{'input':weather_view(e,w,timed),'entries':round(float(e.total_ridership),1)} for _,e in examples(past,r,8).iterrows()]}
    jobs.append(dict(task=task,replicate=replicate,timed=timed,route='direct',date=r.date,station_complex_id=r.station_complex_id,actual=r.total_ridership,view=view))
print('Additional paired event calls',len(jobs),'estimated tokens',round(sum(len(json.dumps(j['view']).encode())/3+250 for j in jobs)),flush=True)
def call(j):return j,api.request(j['route'],j['view'],j.get('person'),replicate=j['replicate'])
start=time.perf_counter()
with ThreadPoolExecutor(max_workers=6) as pool:responses=list(pool.map(call,jobs))
rows=[]
for task,part in p[p.task.str.startswith('Extreme')].groupby('task'):
 targets=part[part.event_day&part.total_ridership.notna()];fit=weights[(weights.task==task)&(weights.method=='role_with_history')].to_dict('records')
 for rep in [1,2]:
  for timed in [False,True]:
   policy={(j['date'],j['person']['agent_id']):a for j,a in responses if j['task']==task and j['replicate']==rep and j['timed']==timed and j['route']=='role'}
   predicted,_=predict(targets,fit,policy,PEOPLE);rows.extend(dict(task=task,replicate=rep,timed=timed,method='role',**r) for r in predicted)
for j,a in responses:
 if j['route']=='direct':rows.append(dict(task=j['task'],replicate=j['replicate'],timed=j['timed'],method='direct',date=j['date'],station_complex_id=j['station_complex_id'],actual=j['actual'],predicted=a['entries']))
pd.DataFrame(rows).to_csv('res/supplement/repeated_ablation.csv',index=False)
json.dump(dict(seconds=time.perf_counter()-start,charged_tokens=sum(api.charge(r) for r in api.ledger()),cap=api.CAP),open('res/supplement/budget.json','w'),indent=2)
print('Repeated ablation complete; cumulative charge',sum(api.charge(r) for r in api.ledger()))
