"""No API: export event/day/station/group scores from the completed checkpoint."""
import argparse,json,os
import numpy as np
import pandas as pd
from design import inputs,EVENTS

def analyze(config):
 c=json.load(open(config));out=c.get('output','res');saved=json.load(open(out+'/checkpoint.json'));p,w=inputs(c.get('station_ids'))
 y=pd.DataFrame(saved['predictions']);tasks=c.get('tasks',['Normal_Summer','Normal_School']+list(EVENTS));methods=c.get('methods',['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few'])
 if len(saved['runtime'])!=len(tasks)*len(methods):raise ValueError('Run is incomplete')
 for task in tasks:
  expected=set(map(tuple,p[(p.task==task)&p.split.eq('test')][['station_complex_id','date']].to_numpy()))
  for method in methods:
   g=y[(y.task==task)&(y.method==method)]
   if len(g)!=len(expected) or set(map(tuple,g[['station_complex_id','date']].to_numpy()))!=expected:raise ValueError('Mismatched evaluation coverage')
 y=y.merge(p[['task','station_complex_id','date','event_day']],on=['task','station_complex_id','date'],validate='many_to_one')
 if not np.isfinite(y.predicted).all() or (y.predicted<0).any():raise ValueError('Invalid predictions')
 y['regime']=np.where(y.task.str.startswith('Normal'),'Normal','Extreme');y['error']=y.predicted-y.actual
 y.to_csv(out+'/prediction_details.csv',index=False)
 rows=[]
 def stats(g):
  return dict(score=100*(1-g.error.abs().sum()/g.actual.sum()),wape=100*g.error.abs().sum()/g.actual.sum(),bias=100*g.error.sum()/g.actual.sum(),mae=g.error.abs().mean(),n=len(g))
 for scope in ['primary','five_day_windows']:
  z=y if scope=='five_day_windows' else y[(y.regime=='Normal')|y.event_day]
  for (task,method,regime),g in z.groupby(['task','method','regime']):rows.append(dict(scope=scope,task=task,method=method,regime=regime,**stats(g)))
 metrics=pd.DataFrame(rows);metrics.to_csv(out+'/task_metrics.csv',index=False)
 metrics.groupby(['scope','regime','method'])[['score','wape','bias','mae']].mean().reset_index().to_csv(out+'/summary.csv',index=False)
 pd.DataFrame([dict(task=t,method=m,station_complex_id=s,**stats(g)) for (t,m,s),g in y.groupby(['task','method','station_complex_id'])]).to_csv(out+'/station_metrics.csv',index=False)
 # Keep decisions and actual fare groups for later group/event figures.
 if saved['decisions']:
  d=pd.DataFrame(saved['decisions']);pop=pd.DataFrame(saved['populations'])
  if not pop.task.str.startswith('Normal').all():raise ValueError('Event population refit detected')
  lookup=pop.set_index(['task','method','station_complex_id','agent_id'])
  for _,r in d.iterrows():
   fit=lookup.loc[(r.population_source,r.method,r.station_complex_id,r.agent_id)]
   if not np.isclose(r.base_population,fit.effective_population) or not np.isclose(r.weekday_multiplier,fit['weekday_'+str(pd.Timestamp(r.date).weekday())]):raise ValueError('Population transfer mismatch')
  if not np.allclose(d[['p_subway','p_bus','p_taxi','p_walk','p_stay_home']].sum(axis=1),1):raise ValueError('Probability sum')
  d['predicted_group_entries']=d.effective_population*d.p_subway
  group=d.groupby(['task','method','station_complex_id','date','group']).predicted_group_entries.sum().reset_index()
  actual=p.melt(id_vars=['task','station_complex_id','date'],value_vars=['students','fair_fare','seniors_disability','general'],var_name='group',value_name='actual_group_entries')
  group.merge(actual,on=['task','station_complex_id','date','group'],validate='many_to_one').to_csv(out+'/group_predictions.csv',index=False)
  check=d.groupby(['task','method','station_complex_id','date']).predicted_group_entries.sum().reset_index().merge(y,on=['task','method','station_complex_id','date'])
  if not np.allclose(check.predicted_group_entries,check.predicted):raise ValueError('Mesa sum mismatch')
 ledger=[json.loads(s) for s in open(out+'/api_calls.jsonl')] if os.path.exists(out+'/api_calls.jsonl') else []
 json.dump(dict(status='passed',stations=p.station_complex_id.nunique(),prediction_rows=len(y),tokens_reported=sum(r.get('usage',{}).get('total_tokens',0) for r in ledger),tokens_conservative=sum(r.get('charged_tokens',r.get('usage',{}).get('total_tokens',0)) for r in ledger)),open(out+'/audit.json','w'),indent=2)
 print(metrics[metrics.scope=='primary'].to_string(index=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--config',default='config.json');analyze(a.parse_args().config)
