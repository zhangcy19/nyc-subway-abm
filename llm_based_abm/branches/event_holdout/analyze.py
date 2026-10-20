"""Validate frozen populations and report event-holdout scores."""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from design import EVENTS,SEASON,METHODS

p=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});y=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str});d=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str});pop=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});runtime=pd.read_csv('res/runtime.csv')
keys=['task','method','station_complex_id','date']
if len(runtime)!=42 or y.duplicated(keys).any():raise ValueError('Incomplete/duplicate result')
expected=set(map(tuple,p[(p.split=='test')&p.total_ridership.notna()][['task','station_complex_id','date']].to_numpy()))
for method,g in y.groupby('method'):
 if set(map(tuple,g[g.actual.notna()][['task','station_complex_id','date']].to_numpy()))!=expected:raise ValueError('Mismatched tests '+method)
if not np.isfinite(y.predicted).all() or (y.predicted<0).any():raise ValueError('Invalid prediction')
if not pop.task.str.startswith('Normal').all():raise ValueError('Population fitted on an Extreme task')
lookup=pop.set_index(['task','method','station_complex_id','agent_id'])
for _,r in d.iterrows():
 source=r.population_source;fit=lookup.loc[(source,r.method,r.station_complex_id,r.agent_id)];day=pd.Timestamp(r.date).dayofweek
 if not np.isclose(r.base_population,fit.effective_population) or not np.isclose(r.weekday_multiplier,fit[f'weekday_{day}']):raise ValueError('Frozen population changed')
cols=['p_subway','p_bus','p_taxi','p_walk','p_stay_home']
if not np.allclose(d[cols].sum(axis=1),1) or ((d[cols]<0)|(d[cols]>1)).any().any():raise ValueError('Invalid probability')
d['expected_entries']=d.effective_population*d.p_subway
check=d.groupby(keys).expected_entries.sum().reset_index().merge(y,on=keys,validate='one_to_one');np.testing.assert_allclose(check.expected_entries,check.predicted,rtol=1e-9,atol=1e-6)
share=pop.effective_population/pop.groupby(['task','method','station_complex_id','group']).effective_population.transform('sum');share=share.dropna()
if ((share<.1-1e-6)|(share>.6+1e-6)).any():raise ValueError('Population concentration')
y=y.merge(p[['task','station_complex_id','date','event_day']],on=['task','station_complex_id','date'],validate='many_to_one');y['regime']=np.where(y.task.str.startswith('Normal'),'Normal','Extreme');y['ape_percent']=100*abs(y.predicted-y.actual)/y.actual;y.to_csv('res/prediction_details.csv',index=False)
rows=[]
for scope in ['primary','all_test_days']:
 z=y if scope=='all_test_days' else y[(y.regime=='Normal')|y.event_day]
 for (task,method),g in z.groupby(['task','method']):
  g=g.dropna(subset=['actual']);e=g.predicted-g.actual
  rows.append(dict(scope=scope,task=task,regime=g.regime.iloc[0],method=method,n=len(g),accuracy_percent=100-100*e.abs().sum()/g.actual.sum(),wape_percent=100*e.abs().sum()/g.actual.sum(),bias_percent=100*e.sum()/g.actual.sum(),mae=e.abs().mean()))
metrics=pd.DataFrame(rows);metrics.to_csv('res/task_metrics.csv',index=False);summary=metrics.groupby(['scope','regime','method'])[['accuracy_percent','wape_percent','bias_percent','mae']].mean().reset_index();summary.to_csv('res/summary.csv',index=False)
labels=['Weekday mean','Ridge','Rule: daily','Rule: timing','Role: zero-shot','Role: few-shot','Direct: few-shot'];colors=['#9baab1','#b8aa82','#b38b69','#8b705c','#92b5bf','#237c87','#d97765']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
for ax,regime in zip(axes,['Normal','Extreme']):
 t=summary[(summary.scope=='primary')&(summary.regime==regime)].set_index('method').reindex(METHODS);bars=ax.bar(range(7),t.accuracy_percent,color=colors)
 for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.1f}',ha='center',fontsize=9)
 ax.set_xticks(range(7),labels,rotation=35,ha='right');ax.set_title(regime);ax.set_ylabel('100 × (1 − WAPE) (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Two stations • frozen Normal populations • event holdouts');fig.tight_layout();fig.savefig('res/performance.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax,(sid,g) in zip(axes,y[y.event_day].groupby('station_complex_id')):
 dates=sorted(g.date.unique());actual=g.drop_duplicates('date').set_index('date').actual.reindex(dates);ax.plot(range(4),actual,'ko-',lw=2.5,label='Observed')
 for method,label,color in zip(METHODS,labels,colors):ax.plot(range(4),g[g.method==method].set_index('date').predicted.reindex(dates),'o-',label=label,color=color)
 ax.set_xticks(range(4),[s[5:] for s in dates]);ax.set_title('Parsons Blvd' if sid=='256' else '176 St');ax.set_ylabel('Daily entries');ax.grid(alpha=.15)
axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig('res/events.png',dpi=180);plt.close(fig)
ledger=[json.loads(s) for s in open('res/api_calls.jsonl')];tokens=sum(r.get('charged_tokens',r.get('usage',{}).get('total_tokens',0)) for r in ledger)
if tokens>5000000:raise ValueError('Budget cap exceeded')
json.dump(dict(status='passed',predictions=len(y),normal_population_rows=len(pop),decisions=len(d),task_method_combinations=len(runtime),api_attempts=len(ledger),valid_responses=sum(r['status']=='valid' for r in ledger),charged_tokens=tokens,checks=['Identical test keys across seven models','No Extreme population fitting','Every decision uses exact Normal base population and weekday coefficient','Probability and Mesa aggregation checks','Within-group representative shares 10–60%','5M token cap']),open('res/result_audit.json','w'),indent=2)
print(summary[summary.scope=='primary'].pivot(index='method',columns='regime',values='accuracy_percent').round(2).to_string());print('API tokens',tokens)
