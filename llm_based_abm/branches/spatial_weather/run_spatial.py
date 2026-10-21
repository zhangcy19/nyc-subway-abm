"""Two-station rerun with an isolated ledger; invoked from spatial_weather/."""
import os,sys,json,argparse,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
sys.path.insert(0,'../fresh_lite')
import api
api.CAP=7500000
from experiment import PEOPLE,RAIN,examples,choose_population,weekday_mean,numerical
from timing import person_view,weather_view,timed_rule_policy
from model import predict
METHODS=['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few']

def inputs():
 p=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});h=pd.read_csv('res/history_pool.csv',dtype={'station_complex_id':str});w=pd.read_csv('res/spatial_hourly.csv',dtype={'station_complex_id':str},parse_dates=['timestamp']);w['date']=w.timestamp.dt.strftime('%Y-%m-%d');w['hour']=w.timestamp.dt.hour
 weather={s:g.pivot(index='date',columns='hour',values='rain_mm').to_dict('index') for s,g in w.groupby('station_complex_id')}
 weather={s:{d:np.array([r[i] for i in range(24)],float) for d,r in ds.items()} for s,ds in weather.items()}
 return p,h,weather

def views(part,pool,w,method):
 train=part[part.split=='train'];td=part.drop_duplicates('date');hist=pool[(pool.task==part.task.iloc[0])&(pool.split=='train')].copy()
 for col in ['rain_mm','peak_mm','six_mm','temp_c','wind_ms']:
  hist[col]=hist.date.map(td.set_index('date')[col])
 if method=='direct_few':
  return [(r,None,{'target':weather_view(r,w,True),'observed_training_examples':[{'input':weather_view(e,w,True),'entries':round(float(e.total_ridership),1)} for _,e in examples(train[train.total_ridership.notna()],r,8).iterrows()]}) for _,r in part[(part.split=='test')&part.total_ridership.notna()].iterrows()]
 return [(r,a,person_view(a,r,hist,w,method=='role_with_history',True)) for _,r in td.iterrows() for a in PEOPLE]

def preflight():
 p,h,weather=inputs();unique={};scope=[]
 for (sid,task),part in p.groupby(['station_complex_id','task']):
  for m in METHODS[-3:]:
   js=views(part,h,weather[sid],m);scope.append(dict(station=sid,task=task,method=m,requests=len(js)))
   for r,a,v in js:
    route='direct' if a is None else 'role';system=api.DIRECT_SYSTEM if route=='direct' else api.ROLE_SYSTEM;payload=system+json.dumps(v,separators=(',',':'));unique[hashlib.sha256(payload.encode()).hexdigest()]=(route,len(payload.encode()))
 estimate=sum(n*(.584 if route=='direct' else .380)+55 for route,n in unique.values())
 report=dict(unique_requests=len(unique),estimated_tokens=round(estimate),estimate_basis='Previous same-model observed input token/byte ratios plus 55 output tokens; failures extra',cap=7500000,scope=scope)
 json.dump(report,open('res/preflight.json','w'),indent=2);print({k:v for k,v in report.items() if k!='scope'})

def run(offline=False):
 p,h,weather=inputs();names=['predictions','decisions','populations','validation','runtime'];out={}
 for n in names:
  f='res/'+n+'.csv'
  out[n]=pd.read_csv(f,dtype={'station_complex_id':str}).to_dict('records') if os.path.exists(f) else []
 done={(r['station_complex_id'],r['task'],r['method']) for r in out['runtime']}
 for (sid,task),part in p.groupby(['station_complex_id','task']):
  train=part[part.split=='train'];valid=part[part.split=='validation'];test=part[part.split=='test'];td=part.drop_duplicates('date');w=weather[sid]
  for method in METHODS:
   if (sid,task,method) in done:continue
   print('Starting',sid,task,method,flush=True);start=time.perf_counter();before=sum(api.charge(r) for r in api.ledger());meta=dict(task=task,method=method)
   if method=='weekday_mean':rows=weekday_mean(train,test)
   elif method=='ridge':
    rows,candidates,_=numerical(train,valid,test);out['validation'].extend(dict(station_complex_id=sid,task=task,**r) for r in candidates)
   elif method=='direct_few':
    def call(job):
     r,person,view=job;answer=api.request('direct',view,offline=offline)
     return dict(station_complex_id=sid,date=r.date,actual=r.total_ridership,predicted=answer['entries'],request_hash=answer['hash'])
    with ThreadPoolExecutor(max_workers=32) as executor:rows=list(executor.map(call,views(part,h,w,method)))
   else:
    if method.startswith('rule_'):policies={scale:timed_rule_policy(td,w,scale,method=='rule_timing') for scale in RAIN}
    else:
     def call(job):
      r,person,view=job
      return (r.date,person['agent_id']),api.request('role',view,person,offline=offline)
     with ThreadPoolExecutor(max_workers=32) as executor:policies={0.:dict(executor.map(call,views(part,h,w,method)))}
    weights,policy,candidates,best=choose_population(train,valid,policies,method)
    rows,dec=predict(test,weights,policy,PEOPLE)
    # Reasons remain in the private ledger; published decisions are numerical only.
    out['decisions'].extend(dict(**meta,**{k:v for k,v in r.items() if k!='reason'}) for r in dec)
    out['populations'].extend(dict(**meta,**r) for r in weights)
    out['validation'].extend(dict(station_complex_id=sid,task=task,**r) for r in candidates)
   out['predictions'].extend(dict(**meta,**r) for r in rows)
   out['runtime'].append(dict(station_complex_id=sid,**meta,seconds=time.perf_counter()-start,new_tokens=sum(api.charge(r) for r in api.ledger())-before))
   for n in names:
    if out[n]:pd.DataFrame(out[n]).to_csv('res/'+n+'.csv',index=False)
   print('Completed',sid,task,method,'tokens',sum(api.charge(r) for r in api.ledger()),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');a.add_argument('--offline',action='store_true');args=a.parse_args()
 if args.preflight:preflight()
 else:run(args.offline)
