"""Check matched input ablation, scoring samples, budget and synthetic OD conservation."""
import json,gzip
import numpy as np
import pandas as pd
import supplement_api as api
from supplement import jobs
from timing import hourly_weather,person_view,weather_view
from experiment import PEOPLE,examples
panel,work=jobs();w=hourly_weather();checked=0
for j in work:
 part=panel[panel.task==j['task']];train=part[part.split=='train'];r=part[part.date==j['date']].iloc[0]
 if j['route']=='role':
  v=person_view(j['person'],r,train,w,True,True)
 else:
  r=part[(part.date==j['date'])&(part.station_complex_id==j['station_complex_id'])].iloc[0]
  past=train[(train.station_complex_id==r.station_complex_id)&train.total_ridership.notna()]
  v={'target':weather_view(r,w,True),'observed_training_examples':[{'input':weather_view(e,w,True),'entries':round(float(e.total_ridership),1)} for _,e in examples(past,r,8).iterrows()]}
 def strip(x):
  if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if k!='hourly_rain_mm_00_to_23'}
  if isinstance(x,list):return [strip(v) for v in x]
  return x
 if strip(v)!=j['view']:raise ValueError('Ablation changed information other than hourly rain')
 checked+=1
A=pd.read_csv('res/supplement/ablation_predictions.csv',dtype={'station_complex_id':str});P=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str})
for new,old in [('role_few_daily_input','role_with_history'),('direct_few_daily_input','direct_few')]:
 a=A[A.method==new];b=P[(P.method==old)&P.actual.notna()]
 keys=['task','date','station_complex_id'];m=a.merge(b,on=keys)
 if len(m)!=len(a) or len(m)!=len(b) or not np.allclose(m.actual_x,m.actual_y):raise ValueError('Scoring samples differ')
weights=pd.read_csv('web_data/network_population_weights.csv',dtype={'station_complex_id':str});old=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});old=old[(old.task=='Extreme_0929')&(old.method=='role_with_history')]
m=weights.merge(old,on=['station_complex_id','agent_id','group'],suffixes=('_new','_old'))
error=max(float(abs(m[k+'_new']-m[k+'_old']).max()) for k in ['effective_population']+[f'weekday_{i}' for i in range(7)])
if error>1e-6:raise ValueError('Network fitting changed the four station weights')
flow=pd.read_csv('web_data/station_flows.csv',dtype={'station_complex_id':str});od=pd.read_csv('web_data/od_flows.csv',dtype={'origin':str,'destination':str})
m=od.groupby(['scenario','origin']).expected_trips.sum().reset_index().merge(flow,left_on=['scenario','origin'],right_on=['scenario','station_complex_id']);od_error=float(abs(m.expected_trips-m.predicted).max())
if od_error>1e-6 or flow.loc[~flow.fit_available,'predicted'].notna().any():raise ValueError('OD conservation or missing-data handling error')
rows=api.ledger()
if not rows:raise RuntimeError('Local API ledger required for provenance and budget audit; use saved numerical results for public replay')
charge=sum(api.charge(r) for r in rows)
if charge>api.CAP:raise ValueError('Budget exceeded')
result=dict(matched_ablation_inputs=checked,scoring_samples_equal=True,network_four_station_refit_max_error=error,od_conservation_max_error=od_error,supplement_api_attempts=len(rows),charged_tokens=charge,cap=api.CAP,api_failures=sum(r['status']!='valid' for r in rows),raw_logs_public=False)
json.dump(result,open('res/supplement/audit.json','w'),indent=2);print(result)
