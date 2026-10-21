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
from timing import timed_rule_policy,revised_rule_policy
from model import predict

OUT='res'
CONFIG={}
NAMES=['predictions','decisions','populations','validation','runtime']
def save(out):
 # One atomic checkpoint is authoritative; CSV exports can be regenerated.
 path=OUT+'/checkpoint.json'
 with open(path+'.tmp','w') as f:json.dump(out,f,allow_nan=False)
 os.replace(path+'.tmp',path)
 for n in NAMES:
  if out[n]:
   pd.DataFrame(out[n]).to_csv(OUT+'/'+n+'.csv.tmp',index=False)
   os.replace(OUT+'/'+n+'.csv.tmp',OUT+'/'+n+'.csv')


def policy_calls(p,w,phase,method,offline):
 def call(job):
  row,person,view=job
  return (row.date,person['agent_id']),api.request('role',view,person,offline=offline)
 with ThreadPoolExecutor(max_workers=CONFIG.get("workers",12)) as pool:policy=dict(pool.map(call,jobs(p,w,phase,method)))
 json.dump([dict(date=d,agent_id=a,probabilities=v['probabilities'],hash=v['hash'],reason='Private response retained locally') for (d,a),v in policy.items()],open(f'{OUT}/{phase}_{method}_policy.json','w'))
 return policy

def direct_calls(p,w,phase,offline,station_ids=None):
 if phase in EVENTS and CONFIG.get('batch_events',True):
  def batch(item):
   sid,g=item;answer=api.request('batch',batch_view(p,w,g,phase),offline=offline)
   result={r['date']:r['entries'] for r in answer['predictions']}
   return [dict(station_complex_id=sid,date=r.date,actual=r.total_ridership,predicted=result[r.date],request_hash=answer['hash']) for _,r in g.iterrows()]
  with ThreadPoolExecutor(max_workers=CONFIG.get('workers',12)) as pool:
   targets=event_data(p,phase)
   if station_ids is not None:targets=targets[targets.station_complex_id.isin(station_ids)]
   return [r for rows in pool.map(batch,targets.groupby('station_complex_id')) for r in rows]

 def call(job):
  row,person,view=job;answer=api.request('direct',view,offline=offline)
  return dict(station_complex_id=row.station_complex_id,date=row.date,actual=row.total_ridership,predicted=answer['entries'],request_hash=answer['hash'])
 scheduled=jobs(p,w,phase,'direct_few')
 if station_ids is not None:scheduled=[j for j in scheduled if j[0].station_complex_id in station_ids]
 with ThreadPoolExecutor(max_workers=CONFIG.get("workers",12)) as pool:return list(pool.map(call,scheduled))

def run(offline=False):
 p,w=inputs(CONFIG.get('station_ids'));out=json.load(open(OUT+'/checkpoint.json')) if os.path.exists(OUT+'/checkpoint.json') else {n:[] for n in NAMES}
 done={(r['task'],r['method']) for r in out['runtime']}
 for phase,method in done:
  expected=set(map(tuple,p[(p.task==phase)&(p.split=='test')][['station_complex_id','date']].to_numpy()))
  saved=[(r['station_complex_id'],r['date']) for r in out['predictions'] if r['task']==phase and r['method']==method]
  if len(saved)!=len(expected) or set(saved)!=expected:raise ValueError('Checkpoint coverage mismatch')

 for phase in CONFIG.get('tasks',['Normal_Summer','Normal_School']+list(EVENTS)):
  normal=phase.startswith('Normal');season=phase if normal else SEASON[phase]
  part=p[p.task==phase];train=normal_history(p,season);valid=p[(p.task==season)&(p.split=='validation')];test=part[part.split=='test'];td=part.drop_duplicates('date')
  for method in CONFIG.get('methods',METHODS):
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
     policies={s:rule_policy(td,w,s,method=='rule_timing') for s in RAIN} if method.startswith('rule_') else {0.:policy_calls(p,w,phase,method,offline)}
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
        block=event_data(p,other);target=block[block.event_day];policy=rule_policy(block.drop_duplicates('date'),w,scale,method=='rule_timing');other_fit=[r for r in out['populations'] if r['task']==SEASON[other] and r['method']==method]
        pr,_=predict(target,other_fit,policy,PEOPLE);scores.append(wape(pr))
       candidates.append(dict(task=phase,method=method,rain_scale=scale,validation_wape=float(np.mean(scores)),selection_source='Three OTHER event centers; frozen Normal populations'))
      best=min(candidates,key=lambda r:r['validation_wape']);out['validation'].extend(candidates);policy=rule_policy(test.drop_duplicates('date'),w,best['rain_scale'],method=='rule_timing')
     else:policy=policy_calls(p,w,phase,method,offline)
    rows,dec=predict(test,fitted,policy,PEOPLE)
    out['decisions'].extend(dict(**meta,population_source=season,**{k:v for k,v in r.items() if k!='reason'}) for r in dec)
   out['predictions'].extend(dict(**meta,**r) for r in rows)
   out['runtime'].append(dict(**meta,seconds=time.perf_counter()-start,new_tokens=sum(api.charge(r) for r in api.ledger())-before));save(out)
   print('Completed',phase,method,'tokens',sum(api.charge(r) for r in api.ledger()),flush=True)

def rule_policy(dates,weather,scale,timed):
 return (revised_rule_policy if CONFIG.get('revised_rule',True) else timed_rule_policy)(dates,weather,scale,timed)

def jobs(p,w,phase,method):
 part=p[p.task==phase]
 if method=='direct_few':return [(r,None,direct_view_for(p,w,r,phase)) for _,r in part[part.split=='test'].iterrows()]
 return [(r,person,role_view_for(p,w,r,person,phase,method=='role_with_history')) for _,r in part.drop_duplicates('date').iterrows() for person in PEOPLE]

def configure(filename):
 global CONFIG,OUT
 CONFIG=json.load(open(filename));OUT=CONFIG.get('output','res')
 os.makedirs(OUT,exist_ok=True);api.LOGDIR=OUT;api.CAP=CONFIG['token_cap']
 # Whole experiment provenance, not just a task/method completion marker.
 files=['api.py','design.py','experiment.py','model.py','prepare.py','run.py','timing.py','staged.py','prepared/daily.csv','prepared/dates.csv','prepared/station_ids.json']
 digest={f:hashlib.sha256(open(f,'rb').read()).hexdigest() for f in files}
 scientific_config={k:v for k,v in CONFIG.items() if k not in ['token_cap','workers','output']}
 signature=hashlib.sha256(json.dumps(dict(config=scientific_config,files=digest),sort_keys=True).encode()).hexdigest()
 path=OUT+'/manifest.json'
 if os.path.exists(path) and json.load(open(path))['signature']!=signature:raise ValueError('Configuration/data/code changed: select a new output directory')
 json.dump(dict(signature=signature,config=CONFIG,files=digest),open(path,'w'),indent=2)

def preflight():
 p,w=inputs(CONFIG.get('station_ids'));scope=[];byte_count=0;calls=0;seen=set()
 for phase in CONFIG.get('tasks',['Normal_Summer','Normal_School']+list(EVENTS)):
  for method in CONFIG.get('methods',METHODS):
   if method not in METHODS[-3:]:continue
   requests=[]
   if method=='direct_few' and phase in EVENTS and CONFIG.get('batch_events',True):
    requests=[('batch',batch_view(p,w,g,phase)) for _,g in event_data(p,phase).groupby('station_complex_id')]
   else:requests=[('direct' if person is None else 'role',view) for _,person,view in jobs(p,w,phase,method)]
   scope.append(dict(task=phase,method=method,requests=len(requests)))
   for route,view in requests:
    system=api.BATCH_SYSTEM if route=='batch' else api.DIRECT_SYSTEM if route=='direct' else api.ROLE_SYSTEM
    text=system+json.dumps(view,separators=(',',':'),allow_nan=False);key=(route,text)
    if key in seen:continue
    seen.add(key);calls+=1;byte_count+=len(text.encode())
 report=dict(stations=p.station_complex_id.nunique(),requests=calls,input_bytes=byte_count,approx_tokens_low=int(byte_count*.35+calls*60),approx_tokens_high=int(byte_count*.55+calls*150),token_cap=api.CAP,scope=scope)
 json.dump(report,open(OUT+'/preflight.json','w'),indent=2);print({k:v for k,v in report.items() if k!='scope'},flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--config',default='config.json');a.add_argument('--execute',action='store_true');a.add_argument('--offline',action='store_true');args=a.parse_args()
 configure(args.config)
 if args.execute:run(args.offline)
 else:preflight()
