"""Dry Normal test-day probes: change only few-shot history context, never refit."""
import json,time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from design import *
from model import predict

p,w=inputs();weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});out=[];probabilities=[];before=sum(api.charge(r) for r in api.ledger())
for phase,season in SEASON.items():
 normal=p[(p.task==season)&(p.split=='test')];days=normal.drop_duplicates('date');dry=days[days.rain_mm<1]
 selected=[]
 for weekend in [False,True]:
  candidates=dry[dry.weekday.ge(5).eq(weekend)].sort_values('date')
  if candidates.empty:raise ValueError('No dry probe day')
  selected.append(candidates.iloc[0].date)
 target=normal[normal.date.isin(selected)]
 jobs=[(r,a,role_view_for(p,w,r,a,phase,True)) for _,r in target.drop_duplicates('date').iterrows() for a in PEOPLE]
 def call(job):
  row,person,view=job;return (row.date,person['agent_id']),api.request('role',view,person)
 with ThreadPoolExecutor(max_workers=24) as executor:policy=dict(executor.map(call,jobs))
 old={(r['date'],r['agent_id']):r for r in json.load(open(f'res/{season}_role_with_history_policy.json'))}
 fit=weights[(weights.task==season)&weights.method.eq('role_with_history')].to_dict('records')
 for context,pol in [('normal_history',old),('other_events_history',policy)]:
  pred,dec=predict(target,fit,pol,PEOPLE)
  out.extend(dict(held_event=EVENTS[phase],context=context,**r) for r in pred)
  probabilities.extend(dict(held_event=EVENTS[phase],context=context,**{k:v for k,v in r.items() if k!='reason'}) for r in dec)
 pd.DataFrame(out).to_csv('res/context_probe_predictions.csv',index=False)
 pd.DataFrame(probabilities).to_csv('res/context_probe_decisions.csv',index=False)
print('Probe new tokens:',sum(api.charge(r) for r in api.ledger())-before)
