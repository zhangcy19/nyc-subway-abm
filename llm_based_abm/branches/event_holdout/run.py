"""Normal fitting once; frozen populations across four event holdouts."""
import os,json,time,argparse,hashlib
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from design import *
from experiment import choose_population,weekday_mean,numerical,feature,RAIN,wape
from timing import timed_rule_policy
from model import predict

NAMES=['predictions','decisions','populations','validation','runtime']
def save(out):
 for n in NAMES:
  if out[n]:pd.DataFrame(out[n]).to_csv('res/'+n+'.csv',index=False)

def policy_calls(p,w,phase,method,offline):
 def call(job):
  row,person,view=job
  return (row.date,person['agent_id']),api.request('role',view,person,offline=offline)
 with ThreadPoolExecutor(max_workers=24) as pool:policy=dict(pool.map(call,jobs(p,w,phase,method)))
 json.dump([dict(date=d,agent_id=a,probabilities=v['probabilities'],hash=v['hash'],reason='Private response retained locally') for (d,a),v in policy.items()],open(f'res/{phase}_{method}_policy.json','w'))
 return policy

def direct_calls(p,w,phase,offline):
 def call(job):
  row,person,view=job;answer=api.request('direct',view,offline=offline)
  return dict(station_complex_id=row.station_complex_id,date=row.date,actual=row.total_ridership,predicted=answer['entries'],request_hash=answer['hash'])
 with ThreadPoolExecutor(max_workers=24) as pool:return list(pool.map(call,jobs(p,w,phase,'direct_few')))

def run(offline=False):
 p,w=inputs();out={n:pd.read_csv('res/'+n+'.csv',dtype={'station_complex_id':str}).to_dict('records') if os.path.exists('res/'+n+'.csv') else [] for n in NAMES}
 done={(r['task'],r['method']) for r in out['runtime']}
 for phase in ['Normal_Summer','Normal_School']+list(EVENTS):
  normal=phase.startswith('Normal');season=phase if normal else SEASON[phase]
  part=p[p.task==phase];train=normal_history(p,season);valid=p[(p.task==season)&(p.split=='validation')];test=part[part.split=='test'];td=part.drop_duplicates('date')
  for method in METHODS:
   if (phase,method) in done:continue
   print('Starting',phase,method,flush=True);start=time.perf_counter();before=sum(api.charge(r) for r in api.ledger());meta=dict(task=phase,method=method)
   if method=='weekday_mean':rows=weekday_mean(train,test)
   elif method=='direct_few':rows=direct_calls(p,w,phase,offline)
   elif method=='ridge':
    if normal:
     rows,candidates,_=numerical(train,valid,test);out['validation'].extend(dict(task=phase,**r) for r in candidates)
    else:
     choices=[r for r in out['validation'] if r['task']==season and r['method']=='ridge'];alpha=min(choices,key=lambda r:r['validation_wape'])['alpha']
     learn=pd.concat([train]+[event_data(p,t) for t in EVENTS if t!=phase]).drop_duplicates(['station_complex_id','date']);rows=[]
     for sid,g in learn.groupby('station_complex_id'):
      g=g[g.total_ridership.notna()];model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(np.array([feature(r) for _,r in g.iterrows()]),np.log1p(g.total_ridership))
      for _,r in test[test.station_complex_id==sid].iterrows():rows.append(dict(station_complex_id=sid,date=r.date,actual=r.total_ridership,predicted=max(0,float(np.expm1(model.predict([feature(r)])[0])))))
   else:
    if normal:
     policies={s:timed_rule_policy(td,w,s,method=='rule_timing') for s in RAIN} if method.startswith('rule_') else {0.:policy_calls(p,w,phase,method,offline)}
     fitted,policy,candidates,best=choose_population(train,valid,policies,method)
     out['populations'].extend(dict(task=phase,method=method,**r) for r in fitted);out['validation'].extend(dict(task=phase,**r) for r in candidates)
    else:
     fitted=[r for r in out['populations'] if r['task']==season and r['method']==method]
     if method.startswith('rule_'):
      candidates=[]
      for scale in RAIN:
       scores=[]
       for other in EVENTS:
        if other==phase:continue
        block=event_data(p,other);target=block[block.event_day];policy=timed_rule_policy(block.drop_duplicates('date'),w,scale,method=='rule_timing');other_fit=[r for r in out['populations'] if r['task']==SEASON[other] and r['method']==method]
        pr,_=predict(target,other_fit,policy,PEOPLE);scores.append(wape(pr))
       candidates.append(dict(task=phase,method=method,rain_scale=scale,validation_wape=float(np.mean(scores)),selection_source='Three OTHER event centers; frozen Normal populations'))
      best=min(candidates,key=lambda r:r['validation_wape']);out['validation'].extend(candidates);policy=timed_rule_policy(test.drop_duplicates('date'),w,best['rain_scale'],method=='rule_timing')
     else:policy=policy_calls(p,w,phase,method,offline)
    rows,dec=predict(test,fitted,policy,PEOPLE)
    out['decisions'].extend(dict(**meta,population_source=season,**{k:v for k,v in r.items() if k!='reason'}) for r in dec)
   out['predictions'].extend(dict(**meta,**r) for r in rows)
   out['runtime'].append(dict(**meta,seconds=time.perf_counter()-start,new_tokens=sum(api.charge(r) for r in api.ledger())-before));save(out)
   print('Completed',phase,method,'tokens',sum(api.charge(r) for r in api.ledger()),flush=True)

def preflight():
 p,w=inputs();unique={};scope=[]
 for phase in ['Normal_Summer','Normal_School']+list(EVENTS):
  for method in METHODS[-3:]:
   j=jobs(p,w,phase,method);scope.append(dict(task=phase,method=method,requests=len(j)))
   for row,person,view in j:
    route='direct' if person is None else 'role';system=api.DIRECT_SYSTEM if route=='direct' else api.ROLE_SYSTEM;payload=system+json.dumps(view,separators=(',',':'),allow_nan=False)
    unique[hashlib.sha256(payload.encode()).hexdigest()]=(route,len(payload.encode()))
 estimate=sum(size*(.584 if route=='direct' else .40)+60 for route,size in unique.values())
 report=dict(unique_requests=len(unique),estimated_tokens=round(estimate),token_cap=api.CAP,model=api.MODEL,scope=scope)
 json.dump(report,open('res/preflight.json','w'),indent=2);print({k:v for k,v in report.items() if k!='scope'})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');a.add_argument('--offline',action='store_true');args=a.parse_args()
 if args.preflight:preflight()
 else:run(args.offline)
