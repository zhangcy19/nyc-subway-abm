"""Check completeness, aggregation, population bounds and API budget after the run."""
import json
import numpy as np
import pandas as pd
p=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str})
y=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str})
d=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str})
w=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str})
r=pd.read_csv('res/runtime.csv',dtype={'station_complex_id':str})
keys=['station_complex_id','task','method','date']
if y.duplicated(keys).any():raise ValueError('Duplicate predictions')
if len(r)!=84:raise ValueError('Expected 2 stations × 6 tasks × 7 methods')
expected=set(map(tuple,p[(p.split=='test')&p.total_ridership.notna()][['station_complex_id','task','date']].to_numpy()))
for method,z in y.groupby('method'):
 actual=set(map(tuple,z[z.actual.notna()][['station_complex_id','task','date']].to_numpy()))
 if actual!=expected:raise ValueError('Evaluation mismatch '+method)
 if not np.isfinite(z.predicted).all() or (z.predicted<0).any():raise ValueError('Invalid predictions')
cols=['p_subway','p_bus','p_taxi','p_walk','p_stay_home']
if not np.allclose(d[cols].sum(axis=1),1):raise ValueError('Probabilities do not sum to 1')
if ((d[cols]<0)|(d[cols]>1)).any().any():raise ValueError('Invalid probabilities')
d['entries']=d.effective_population*d.p_subway
agg=d.groupby(keys).entries.sum().reset_index().merge(y,on=keys,validate='one_to_one')
if not np.allclose(agg.entries,agg.predicted):raise ValueError('Mesa aggregation mismatch')
shares=w.effective_population/w.groupby(['station_complex_id','task','method','group']).effective_population.transform('sum')
if ((shares.dropna()<.1-1e-6)|(shares.dropna()>.6+1e-6)).any():raise ValueError('Population concentration bound')
rows=[json.loads(line) for line in open('res/api_calls.jsonl')]
charged=sum(x.get('charged_tokens',x.get('usage',{}).get('total_tokens',0)) for x in rows)
if charged>7500000:raise ValueError('Budget exceeded')
report=dict(status='passed',completed_station_task_methods=len(r),predictions=len(y),decisions=len(d),api_attempts=len(rows),valid_api_responses=sum(x['status']=='valid' for x in rows),charged_tokens=charged,input_tokens=sum(x.get('usage',{}).get('input_tokens',0) for x in rows),output_tokens=sum(x.get('usage',{}).get('output_tokens',0) for x in rows),checks=['Identical observed evaluation keys across methods','Finite nonnegative predictions','Probability simplex','Mesa weighted sum equals prediction','Within-group population shares 10–60%','7500000 token cap'])
json.dump(report,open('res/result_audit.json','w'),indent=2);print(json.dumps(report,indent=2))

cost=[]
for x in rows:
 v=x['request']['view'];method='direct_few' if x['request']['route']=='direct' else 'role_with_history' if 'group_history' in v else 'role_no_history'
 cost.append(dict(method=method,attempts=1,charged_tokens=x.get('charged_tokens',x.get('usage',{}).get('total_tokens',0)),input_tokens=x.get('usage',{}).get('input_tokens',0),output_tokens=x.get('usage',{}).get('output_tokens',0),summed_request_seconds=x['elapsed_seconds']))
pd.DataFrame(cost).groupby('method').sum().reset_index().to_csv('res/api_cost.csv',index=False)
