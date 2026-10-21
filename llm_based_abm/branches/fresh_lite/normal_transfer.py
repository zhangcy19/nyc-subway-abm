"""Offline population transfer: same JFK weather, same cached decisions, zero API calls.
Run from llm_based_abm/fresh_lite: python normal_transfer.py
"""
import os,json
import numpy as np
import pandas as pd
from model import fit_population,predict,profiles

os.makedirs('res/normal_transfer',exist_ok=True)
people=profiles()
panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str})
weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str})
validation=pd.read_csv('res/validation.csv')
saved=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str})
methods=['rule_daily','rule_timing','role_no_history','role_with_history']
pairs=[('Extreme_0729','Normal_Summer','2023-07-29'),('Extreme_1007','Normal_School','2023-10-07')]
predictions=[];populations=[];decisions=[];datasets=[];config=[];audits=[]
for task,normal,event in pairs:
 part=panel[panel.task==task];target=part[(part.split=='test')&part.event_day].copy()
 clean=panel[(panel.task==normal)&(panel.split=='train')&(panel.date<event)].copy()
 start,end=clean.date.min(),clean.date.max()
 matched=part[(part.date>=start)&(part.date<=end)].copy()
 if clean.date.max()>=event or part[part.split=='validation'].date.max()>=event:raise ValueError('Post-event calibration')
 for arm,frame in [('normal_pre_event',clean),('matched_mixed',matched)]:
  datasets.extend(dict(arm=arm,target_task=task,**r) for r in frame.to_dict('records'))
 for method in methods:
  candidates=validation[(validation.task==task)&(validation.method==method)]
  best=candidates.loc[candidates.validation_wape.idxmin()]
  config.append(dict(task=task,method=method,shrink=best.shrink,calendar_prior=best.calendar_prior,rain_scale=best.rain_scale,hyperparameter_source='Original Extreme validation, before target',normal_start=start,normal_end=end,normal_days=clean.date.nunique(),matched_days=matched.date.nunique()))
  policy={(r['date'],r['agent_id']):r for r in json.load(open(f'res/{task}_{method}_policy.json'))}
  fitted={
   'original_extreme':weights[(weights.task==task)&(weights.method==method)].to_dict('records'),
   'normal_pre_event':fit_population(clean,policy,people,float(best.shrink),float(best.calendar_prior)),
   'matched_mixed':fit_population(matched,policy,people,float(best.shrink),float(best.calendar_prior)),
   'saved_normal_diagnostic':weights[(weights.task==normal)&(weights.method==method)].to_dict('records')}
  reference=saved[(saved.task==task)&(saved.method==method)&(saved.date==event)]
  for arm,fit in fitted.items():
   pred,dec=predict(target,fit,policy,people)
   if arm=='original_extreme':
    check=pd.DataFrame(pred).merge(reference,on=['station_complex_id','date'],suffixes=('_replay','_saved'),validate='one_to_one')
    np.testing.assert_allclose(check.predicted_replay,check.predicted_saved,rtol=1e-9,atol=1e-6)
    audits.append(dict(task=task,method=method,max_replay_error=float(abs(check.predicted_replay-check.predicted_saved).max())))
   predictions.extend(dict(task=task,method=method,arm=arm,**r) for r in pred)
   populations.extend(dict(**{k:v for k,v in r.items() if k not in ['task','method']},task=task,method=method,arm=arm) for r in fit)
   decisions.extend(dict(task=task,method=method,arm=arm,**{k:v for k,v in r.items() if k!='reason'}) for r in dec)
results=pd.DataFrame(predictions);decs=pd.DataFrame(decisions);pops=pd.DataFrame(populations)
# Every arm must use the same travelers' probabilities for each event and method.
keys=['task','method','station_complex_id','date','agent_id']
for c in ['p_subway','p_bus','p_taxi','p_walk','p_stay_home']:
 if decs.groupby(keys)[c].nunique().max()!=1:raise ValueError('Decisions changed between arms')
decs['entries']=decs.effective_population*decs.p_subway
keys=['task','method','arm','station_complex_id','date']
check=decs.groupby(keys).entries.sum().reset_index().merge(results,on=keys,validate='one_to_one')
np.testing.assert_allclose(check.entries,check.predicted,atol=1e-6)
share=pops.effective_population/pops.groupby(['task','method','arm','station_complex_id','group']).effective_population.transform('sum')
if ((share.dropna()<.1-1e-6)|(share.dropna()>.6+1e-6)).any():raise ValueError('Population concentration')
if len(results)!=128 or results.duplicated(keys).any():raise ValueError('Incomplete comparison')
results['ape_percent']=100*abs(results.predicted-results.actual)/results.actual
metrics=[]
for (task,method,arm),g in results.groupby(['task','method','arm']):
 error=g.predicted-g.actual
 metrics.append(dict(task=task,method=method,arm=arm,n=len(g),wape_percent=100*error.abs().sum()/g.actual.sum(),accuracy_percent=100-100*error.abs().sum()/g.actual.sum(),bias_percent=100*error.sum()/g.actual.sum()))
metrics=pd.DataFrame(metrics)
summary=metrics.groupby(['method','arm'])[['accuracy_percent','wape_percent','bias_percent']].mean().reset_index()
for name,frame in [('predictions',results),('event_metrics',metrics),('summary',summary),('populations',pops),('decisions',decs.drop(columns='entries')),('calibration_days',pd.DataFrame(datasets)),('config',pd.DataFrame(config))]:frame.to_csv('res/normal_transfer/'+name+'.csv',index=False)
json.dump(dict(status='passed',new_api_calls=0,new_tokens=0,prediction_rows=len(results),stations=sorted(results.station_complex_id.unique()),replay_checks=audits,checks=['Same action probabilities across all arms','Pre-event Normal calibration labels','Pre-event hyperparameter selection','Population shares within 10–60%','Mesa aggregation reconciled'],caveats=['saved_normal_diagnostic uses post-event Normal validation; not prospective','normal_pre_event fixes Extreme hyperparameters and policies; it is a population-calibration ablation, not a fully Normal-trained model','Previously inspected events; not a pristine holdout']),open('res/normal_transfer/audit.json','w'),indent=2)
print(metrics.pivot(index=['task','method'],columns='arm',values='accuracy_percent').round(2).to_string())
print('\nEqual-event average:');print(summary.pivot(index='method',columns='arm',values='accuracy_percent').round(2).to_string())
