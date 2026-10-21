"""Compare identical station/test-day keys; report equal-task mean WAPE."""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str})
new=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str})
old=pd.read_csv('../fresh_lite/res/predictions.csv',dtype={'station_complex_id':str})
keys=['station_complex_id','task','date','method']
if new.duplicated(keys).any():raise ValueError('Duplicate predictions')
paired=new.merge(old[keys+['predicted']],on=keys,validate='one_to_one',suffixes=('_new','_old'))
if len(paired)!=len(new):raise ValueError('Unmatched predictions')
paired=paired.merge(p[['station_complex_id','task','date','event_day','rain_mm','peak_mm']],on=keys[:3],validate='many_to_one')
paired['regime']=np.where(paired.task.str.startswith('Normal'),'Normal','Extreme')
paired.to_csv('res/paired_predictions.csv',index=False)
evaldata=paired[(paired.regime=='Normal')|paired.event_day].dropna(subset=['actual'])

# Secondary predeclared local-weather subset; no refitting or additional API use.
local=evaldata[((evaldata.regime=='Normal')&(evaldata.rain_mm<10)&(evaldata.peak_mm<10))|((evaldata.regime=='Extreme')&((evaldata.rain_mm>=25)|(evaldata.peak_mm>=10)))]
sensitivity=[]
for (regime,method,task),g in local.groupby(['regime','method','task']):
 for version in ['new','old']:
  sensitivity.append(dict(regime=regime,method=method,task=task,weather=version,n=len(g),accuracy_percent=100-100*(g['predicted_'+version]-g.actual).abs().sum()/g.actual.sum()))
pd.DataFrame(sensitivity).to_csv('res/local_weather_subset_metrics.csv',index=False)
p[['station_complex_id','task','date','split','rain_mm','peak_mm','event_day']].to_csv('res/dataset_days.csv',index=False)

rows=[]
for (task,method),g in evaldata.groupby(['task','method']):
 for version in ['new','old']:
  delta=g['predicted_'+version]-g.actual
  rows.append(dict(task=task,regime=g.regime.iloc[0],method=method,weather=version,n=len(g),wape_percent=100*delta.abs().sum()/g.actual.sum(),accuracy_percent=100-100*delta.abs().sum()/g.actual.sum(),bias_percent=100*delta.sum()/g.actual.sum(),mae=delta.abs().mean()))
detail=pd.DataFrame(rows);detail.to_csv('res/task_metrics.csv',index=False)
summary=detail.groupby(['regime','method','weather'])[['accuracy_percent','wape_percent','bias_percent','mae']].mean().reset_index();summary.to_csv('res/summary.csv',index=False)
st=[]
for (sid,task,method),g in evaldata.groupby(keys[:2]+['method']):
 for version in ['new','old']:
  st.append(dict(station_complex_id=sid,task=task,regime=g.regime.iloc[0],method=method,weather=version,accuracy_percent=100-100*(g['predicted_'+version]-g.actual).abs().sum()/g.actual.sum()))
pd.DataFrame(st).to_csv('res/station_task_metrics.csv',index=False)
order=['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few'];labels=['Weekday mean','Ridge','Rule: daily','Rule: timing','Role: zero-shot','Role: few-shot','Direct: few-shot']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(14,5.8),sharey=True)
for ax,regime in zip(axs,['Normal','Extreme']):
 tab=summary[summary.regime==regime].pivot(index='method',columns='weather',values='accuracy_percent').reindex(order)
 x=np.arange(len(order));ax.bar(x-.19,tab.old,.36,color='#bcc8d2',label='Previous JFK weather');bars=ax.bar(x+.19,tab.new,.36,color='#168b91',label='Station weather')
 for bar in bars:
  if np.isfinite(bar.get_height()):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.1f}',ha='center',fontsize=9)
 ax.set_xticks(x,labels,rotation=40,ha='right');ax.set_title(regime);ax.set_ylabel('100 − WAPE (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axs[0].legend(loc='lower left',fontsize=9);fig.suptitle('Two-station rerun • Doubao Lite • matched evaluation days',fontsize=16);fig.tight_layout();fig.savefig('res/accuracy_comparison.png',dpi=180);plt.close(fig)
events=evaldata[evaldata.regime=='Extreme'];fig,axs=plt.subplots(1,2,figsize=(14,5),sharey=True)
colors=['#89969c','#bead73','#d8ac92','#b77f4d','#8f8ac1','#137b80','#da5e54']
for ax,(sid,g) in zip(axs,events.groupby('station_complex_id')):
 dates=sorted(g.date.unique());actual=g.drop_duplicates('date').set_index('date').actual.reindex(dates);ax.plot(range(len(dates)),actual,'ko-',lw=2.7,label='Observed')
 for m,col,label in zip(order,colors,labels):
  z=g[g.method==m].set_index('date').predicted_new.reindex(dates);ax.plot(range(len(dates)),z,'o-',color=col,label=label,alpha=.85)
 ax.set_xticks(range(len(dates)),[d[5:] for d in dates]);ax.set_title('Parsons Blvd' if sid=='256' else '176 St');ax.set_ylabel('Daily entries');ax.grid(alpha=.15)
axs[1].legend(fontsize=8);fig.suptitle('Event-day predictions using station weather');fig.tight_layout();fig.savefig('res/event_predictions.png',dpi=180);plt.close(fig)
print(summary.pivot(index='method',columns=['regime','weather'],values='accuracy_percent').round(2).to_string())

event_details=paired[paired.event_day].copy()
event_details['absolute_error_new']=(event_details.predicted_new-event_details.actual).abs()
event_details['absolute_error_old']=(event_details.predicted_old-event_details.actual).abs()
event_details.to_csv('res/event_details.csv',index=False)
