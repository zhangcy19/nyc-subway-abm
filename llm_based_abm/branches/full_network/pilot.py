"""Small changed-method pilot. No all-network run. Separate 1M token hard cap."""
import os,json,time,argparse
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import api
from design import *
from experiment import choose_population,wape,RAIN
from model import predict
from timing import timed_rule_policy,revised_rule_policy

IDS=['256','385'];OUT='pilot_res';os.makedirs(OUT,exist_ok=True)
def metric(rows):
 t=pd.DataFrame(rows);return 100*(1-abs(t.predicted-t.actual).sum()/t.actual.sum())

def rules():
 p,w=inputs(IDS);out=[];fits={};choices=[]
 for version,fn in [('old',timed_rule_policy),('new',revised_rule_policy)]:
  for phase in ['Normal_Summer','Normal_School']:
   part=p[p.task==phase];policies={s:fn(part.drop_duplicates('date'),w,s,True) for s in RAIN}
   fitted,policy,candidates,best=choose_population(part[part.split=='train'],part[part.split=='validation'],policies,'rule_timing')
   fits[(version,phase)]=fitted;pred,_=predict(part[part.split=='test'],fitted,policy,PEOPLE)
   out.extend(dict(version=version,task=phase,**r) for r in pred);choices.append(dict(version=version,task=phase,**best))
  for phase in EVENTS:
   scores=[]
   for scale in RAIN:
    s=[]
    for other in EVENTS:
     if other==phase:continue
     target=event_data(p,other);target=target[target.event_day]
     pred,_=predict(target,fits[(version,SEASON[other])],fn(target.drop_duplicates('date'),w,scale,True),PEOPLE);s.append(wape(pred))
    scores.append((float(np.mean(s)),scale))
   best,scale=min(scores);target=event_data(p,phase)
   pred,_=predict(target,fits[(version,SEASON[phase])],fn(target.drop_duplicates('date'),w,scale,True),PEOPLE)
   out.extend(dict(version=version,task=phase,**r) for r in pred);choices.append(dict(version=version,task=phase,rain_scale=scale,validation_wape=best))
 pd.DataFrame(out).to_csv(OUT+'/rule_predictions.csv',index=False);pd.DataFrame(choices).to_csv(OUT+'/rule_selection.csv',index=False)
 print('Rule comparison finished',flush=True)

def role_calls(p,w,rows,phase):
 def call(job):
  r,person=job
  return (r.date,person['agent_id']),api.request('role',role_view_for(p,w,r,person,phase,True),person)
 with ThreadPoolExecutor(max_workers=8) as pool:return dict(pool.map(call,[(r,a) for _,r in rows.drop_duplicates('date').iterrows() for a in PEOPLE]))

def llm():
 api.LOGDIR=OUT;api.CAP=1000000
 p,w=inputs(IDS);part=p[p.task=='Normal_School'];train=part[part.split=='train'];valid=part[part.split=='validation']
 before=sum(api.charge(r) for r in api.ledger());started=time.perf_counter()
 # Changed role histories need Normal recalibration, but no Normal test sweep.
 policy=role_calls(p,w,part[part.split!='test'],'Normal_School')
 fitted,_,candidates,best=choose_population(train,valid,{0.:policy},'role_with_history')
 pd.DataFrame(fitted).to_csv(OUT+'/role_populations.csv',index=False)
 pd.DataFrame(candidates).to_csv(OUT+'/role_validation.csv',index=False)
 target=event_data(p,'Extreme_1007');pol=role_calls(p,w,target,'Extreme_1007')
 pred,dec=predict(target,fitted,pol,PEOPLE)
 pd.DataFrame(pred).to_csv(OUT+'/role_predictions.csv',index=False)
 pd.DataFrame([{k:v for k,v in r.items() if k!='reason'} for r in dec]).to_csv(OUT+'/role_decisions.csv',index=False)
 # A dry validation day, never a Normal test day. No tuning on probe results.
 probe=valid[valid.rain_mm<1];probe=probe[probe.date==probe.date.min()]
 event_policy=role_calls(p,w,probe,'Extreme_1007');probe_rows=[]
 for context,pp in [('normal',policy),('event',event_policy)]:
  pred,_=predict(probe,fitted,pp,PEOPLE);probe_rows.extend(dict(context=context,**r) for r in pred)
 pd.DataFrame(probe_rows).to_csv(OUT+'/context_probe.csv',index=False)
 direct=[]
 # Compare old individual predictions, compact individual, full-history batch,
 # and compact-history batch. Only one five-day event and two stations.
 for sid,g in target.groupby('station_complex_id'):
  for compact in [False,True]:
   answer=api.request('batch',batch_view(p,w,g,'Extreme_1007',compact))
   lookup={r['date']:r['entries'] for r in answer['predictions']}
   direct.extend(dict(arm='batch_compact' if compact else 'batch_full',station_complex_id=sid,date=r.date,actual=r.total_ridership,predicted=lookup[r.date],request_hash=answer['hash']) for _,r in g.iterrows())
  for _,r in g.iterrows():
   answer=api.request('direct',direct_view_for(p,w,r,'Extreme_1007',True))
   direct.append(dict(arm='single_compact',station_complex_id=sid,date=r.date,actual=r.total_ridership,predicted=answer['entries'],request_hash=answer['hash']))
 pd.DataFrame(direct).to_csv(OUT+'/direct_predictions.csv',index=False)
 json.dump(dict(token_cap=api.CAP,charged_tokens=sum(api.charge(r) for r in api.ledger()),new_tokens=sum(api.charge(r) for r in api.ledger())-before,seconds=time.perf_counter()-started,role_normal_selection=best),open(OUT+'/usage.json','w'),indent=2)
 print('Pilot API complete',json.load(open(OUT+'/usage.json')),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['rules','llm']);args=a.parse_args()
 if args.phase=='rules':rules()
 else:llm()
