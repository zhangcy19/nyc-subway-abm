"""Offline metrics and figures. Never invokes the API."""
import collections
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from api import ledger, charge
from model import ACTIONS

NAMES={'weekday_mean':'Weekday mean','ridge':'Ridge','rule_abm':'Rule ABM','direct_few':'Direct few-shot','role_no_history':'Role zero-shot','role_with_history':'Role few-shot'}
ORDER=list(NAMES);COLORS=['#89939e','#bc8d42','#cb6a53','#527fb3','#378d78','#855da8']
plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})

def metrics(x):
    x=x.dropna(subset=['actual','predicted']);e=x.predicted-x.actual;den=x.actual.sum()
    return dict(n_station_days=len(x),actual_entries=den,WAPE_pct=100*abs(e).sum()/den,MAE=abs(e).mean(),RMSE=np.sqrt((e**2).mean()),bias_pct=100*e.sum()/den,within_20pct=100*(abs(e)<=.2*x.actual).mean())

def table(p,keys):
    return pd.DataFrame([{**dict(zip(keys,k if isinstance(k,tuple) else [k])),**metrics(x)} for k,x in p.groupby(keys,dropna=False)])

def savefig(fig,name):
    fig.tight_layout();fig.savefig('res/figures/'+name+'.png');plt.close(fig)

def analyze():
    os.makedirs('res/figures',exist_ok=True)
    dates=pd.read_csv('res/dates.csv');panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});primary=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str})
    pred=primary.merge(dates[['task','date','event_day','event']],on=['task','date'])
    tasks=sorted(dates.task.unique());normal=[t for t in tasks if t.startswith('Normal')];ext=[t for t in tasks if t.startswith('Extreme')]
    result=table(pred,['task','method']);result.to_csv('res/metrics.csv',index=False)
    extremes=pred[pred.task.str.startswith('Extreme')].copy();extremes['condition']=np.where(extremes.event_day,'event','surrounding')
    table(extremes,['condition','method']).to_csv('res/extreme_metrics.csv',index=False)
    table(pred,['task','station_complex_id','method']).to_csv('res/station_metrics.csv',index=False)
    summary=[];ranges=[]
    for (task,split),r in panel.groupby(['task','split']):
        summary.append(dict(task=task,split=split,days=r.date.nunique(),complete_station_days=r.total_ridership.notna().sum(),mean_entries=r.total_ridership.mean(),min_entries=r.total_ridership.min(),max_entries=r.total_ridership.max(),rain_min=r.rain_mm.min(),rain_max=r.rain_mm.max(),weekdays=','.join(map(str,sorted(r.weekday.unique())))))
    pd.DataFrame(summary).to_csv('res/dataset_summary.csv',index=False)
    for t in ext:
        d=dates[dates.task==t];train=d[d.split=='train'];test=d[(d.split=='test')&d.event_day].iloc[0]
        ranges.append(dict(task=t,event=test.date,train_max_rain=train.rain_mm.max(),train_max_peak=train.peak_mm.max(),event_rain=test.rain_mm,event_peak=test.peak_mm,outside_daily_range=bool(test.rain_mm>train.rain_mm.max()),outside_hourly_range=bool(test.peak_mm>train.peak_mm.max())))
    pd.DataFrame(ranges).to_csv('res/weather_range.csv',index=False)
    # Test repetitions concern event days only; never mix them with whole windows.
    repeat=pd.concat([primary]+[pd.read_csv(f,dtype={'station_complex_id':str}) for f in sorted(glob.glob('res/repeat*_predictions.csv'))],ignore_index=True)
    ev=repeat.merge(dates[['task','date','event_day']],on=['task','date']);ev=ev[ev.event_day]
    em=table(ev,['task','method','replicate']);em.to_csv('res/event_repeat_metrics.csv',index=False)
    es=em.groupby(['task','method']).agg(WAPE_mean=('WAPE_pct','mean'),WAPE_std=('WAPE_pct','std'),WAPE_min=('WAPE_pct','min'),WAPE_max=('WAPE_pct','max'),bias_mean=('bias_pct','mean'),runs=('replicate','size')).reset_index();es.to_csv('res/event_summary.csv',index=False)
    pooled=table(ev,['method','replicate']);pooled.to_csv('res/pooled_event_metrics.csv',index=False)
    pooled.groupby('method').agg(WAPE_mean=('WAPE_pct','mean'),WAPE_std=('WAPE_pct','std'),bias_mean=('bias_pct','mean'),runs=('replicate','size')).to_csv('res/pooled_event_summary.csv')
    ev['weather_scope']=np.where(ev.task=='Extreme_0929','outside_training_range','within_training_range')
    table(ev,['weather_scope','method','replicate']).to_csv('res/range_metrics.csv',index=False)
    # The old/new comparison uses exactly the same dates and station observations.
    old=pd.read_csv('res/previous_reference.csv',dtype={'station_complex_id':str});comparison=[]
    for label,p in [('previous',old),('current',repeat[repeat.task.isin(['Extreme_0929','Extreme_1007'])])]:
        q=p[p.date.isin(['2023-09-29','2023-10-07'])]
        for (day,method,rep),r in q.groupby(['date','method','replicate']):comparison.append(dict(version=label,date=day,method=method,replicate=rep,**metrics(r)))
    cmp=pd.DataFrame(comparison);cmp.to_csv('res/previous_comparison.csv',index=False)
    # Population capacity is a structural upper bound, not true headcount.
    weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});caps=[]
    for (task,method,group),r in weights.groupby(['task','method','group']):
        upper=r.groupby('station_complex_id').effective_population.sum().rename('base_capacity')
        calendar=r.groupby('station_complex_id')[[f'weekday_{d}' for d in range(7)]].first()
        truth=panel[(panel.task==task)&(panel.split=='test')][['station_complex_id','weekday',group]].rename(columns={group:'actual'}).dropna().merge(upper,on='station_complex_id').merge(calendar,on='station_complex_id')
        truth['capacity']=truth.base_capacity*np.array([row[f'weekday_{int(row.weekday)}'] for _,row in truth.iterrows()])
        caps.append(dict(task=task,method=method,group=group,effective_scale=upper.sum(),unavoidable_underprediction_pct=100*(truth.actual-truth.capacity).clip(lower=0).sum()/truth.actual.sum()))
    pd.DataFrame(caps).to_csv('res/capacity_diagnostic.csv',index=False)
    dec=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str}).merge(dates[['task','date','event_day']],on=['task','date']);dec['predicted']=dec.effective_population*dec.p_subway
    gp=dec.groupby(['task','method','date','station_complex_id','group'],as_index=False).predicted.sum()
    truth=panel.melt(id_vars=['task','date','station_complex_id'],value_vars=['students','fair_fare','seniors_disability','general'],var_name='group',value_name='actual')
    gp=gp.merge(truth,on=['task','date','station_complex_id','group']);gp.to_csv('res/group_predictions.csv',index=False);table(gp,['task','method','group']).to_csv('res/group_metrics.csv',index=False)
    actions=[]
    for (task,method,event,group),r in dec.groupby(['task','method','event_day','group']):
        for action in ACTIONS:actions.append(dict(task=task,method=method,event_day=event,group=group,action=action,probability=np.average(r['p_'+action],weights=r.effective_population) if r.effective_population.sum()>0 else np.nan))
    actions=pd.DataFrame(actions);actions.to_csv('res/group_actions.csv',index=False)
    reasons=dec[dec.method.str.startswith('role')].drop_duplicates(['task','date','agent_id','request_hash']);reasons[['task','date','method','group','agent_id','p_subway','p_stay_home','reason']].to_csv('res/reason_examples.csv',index=False)
    # Figures use English labels for portable fonts; Chinese explanations are in RESULTS.
    weather=pd.read_csv('res/weather_daily.csv');palette={'train':'#78a38b','validation':'#dfb75e','test':'#8d79b7'}
    fig,axes=plt.subplots(6,1,figsize=(13,12),sharex=True)
    for ax,t in zip(axes,tasks):
        ax.bar(pd.to_datetime(weather.date),weather.rain_mm,color='#e4e6e9')
        for split,c in palette.items():
            d=dates[(dates.task==t)&(dates.split==split)];ax.bar(pd.to_datetime(d.date),d.rain_mm,color=c,label=split);ax.scatter(pd.to_datetime(d.date),np.zeros(len(d)),marker='|',s=60,color=c)
        ax.set(title=t,ylabel='Rain mm',yscale='symlog');ax.legend(ncol=3,fontsize=8,loc='upper left')
    fig.autofmt_xdate();savefig(fig,'01_calendar')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,t in zip(axes.flat,tasks):
        values=[panel[(panel.task==t)&(panel.split==s)].total_ridership.dropna() for s in ['train','validation','test']]
        ax.boxplot(values,tick_labels=['Train','Validation','Test']);ax.set(title=t,ylabel='Entries / station-day')
    savefig(fig,'02_distribution')
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,t in zip(axes.flat,tasks):
        r=result[result.task==t].set_index('method').reindex(ORDER);bars=ax.barh([NAMES[m] for m in ORDER],r.WAPE_pct,color=COLORS)
        for b,v in zip(bars,r.WAPE_pct):ax.text(v+.4,b.get_y()+b.get_height()/2,f'{v:.1f}',va='center')
        ax.invert_yaxis();ax.set(title=t+' | whole test window',xlabel='WAPE % (lower better)',xlim=(0,r.WAPE_pct.max()*1.22))
    savefig(fig,'03_accuracy')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,t in zip(axes.flat,ext):
        r=es[es.task==t].set_index('method').reindex(ORDER);bars=ax.barh([NAMES[m] for m in ORDER],r.WAPE_mean,xerr=r.WAPE_std.fillna(0),color=COLORS,capsize=3)
        for b,v,err in zip(bars,r.WAPE_mean,r.WAPE_std.fillna(0)):ax.text(v+err+.5,b.get_y()+b.get_height()/2,f'{v:.1f}',va='center',fontsize=8)
        ax.set_xlim(0,float((r.WAPE_mean+r.WAPE_std.fillna(0)).max())*1.14)
        ax.invert_yaxis();ax.set(title=t+' | event day only',xlabel='WAPE %; LLM mean ± sample SD')
    savefig(fig,'04_event_repeats')
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for ax,t in zip(axes.flat,ext):
        r=extremes[extremes.task==t].dropna(subset=['actual']).groupby(['date','method'])[['actual','predicted']].sum().reset_index();a=r[r.method=='weekday_mean']
        ax.plot(a.date.str[5:],a.actual,'k-o',label='Actual',linewidth=2.5)
        for m,c in zip(ORDER,COLORS):
            a=r[r.method==m];ax.plot(a.date.str[5:],a.predicted,'--o',color=c,label=NAMES[m],alpha=.8)
        ax.set(title=t+' | primary inference',ylabel='Entries across observed stations');ax.legend(fontsize=7,ncol=2)
    savefig(fig,'05_event_trajectories')
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,day in zip(axes,['2023-09-29','2023-10-07']):
        r=cmp[cmp.date==day].groupby(['version','method']).WAPE_pct.mean();y=np.arange(6)
        ax.barh(y-.18,[r.loc['previous',m] for m in ORDER],height=.35,label='Previous split',color='#aeb9c1')
        ax.barh(y+.18,[r.loc['current',m] for m in ORDER],height=.35,label='Same-regime rolling split',color='#527fb3')
        ax.set(yticks=y,yticklabels=[NAMES[m] for m in ORDER],title=day,xlabel='Same event-day WAPE %; LLM mean of runs');ax.legend(fontsize=8)
    axes[0].invert_yaxis();savefig(fig,'06_same_event_comparison')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,m in zip(axes,['rule_abm','role_no_history','role_with_history']):
        r=actions[(actions.task=='Extreme_1007')&actions.event_day&(actions.method==m)].pivot(index='group',columns='action',values='probability').reindex(columns=ACTIONS)
        r.plot.bar(stacked=True,ax=ax,colormap='Set2',legend=False);ax.set(title=NAMES[m]+' | Oct 7',ylim=(0,1),ylabel='Model choice probability');ax.tick_params(axis='x',rotation=30)
    axes[-1].legend(bbox_to_anchor=(1,1),fontsize=8);savefig(fig,'07_group_choices')
    stop=set('i a an the to of and or in on is are be my me it so with for at as that this will would can likely not no more from have do but rather than due day'.split());words=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,m in zip(axes,['role_no_history','role_with_history']):
        r=reasons[reasons.event_day&(reasons.method==m)];counts=collections.Counter(w for text in r.reason for w in re.findall('[a-z]+',str(text).lower()) if w not in stop and len(w)>2);top=counts.most_common(12)[::-1]
        ax.barh([x[0] for x in top],[x[1] for x in top],color='#527fb3');ax.set(title=NAMES[m]+' | four event days',xlabel='Word count in generated short summaries');words.extend(dict(method=m,word=w,count=n) for w,n in counts.items())
    savefig(fig,'08_reason_words');pd.DataFrame(words).to_csv('res/reason_words.csv',index=False)
    calls=ledger();cost=[]
    for r in calls:
        q=r['request'];method='direct_few' if q['route']=='direct' else 'role_with_history' if 'group_history' in q['view'] else 'role_no_history'
        cost.append(dict(method=method,tokens=r.get('usage',{}).get('total_tokens',0),charged_tokens=charge(r),api_seconds=r.get('elapsed_seconds',0)))
    costs=pd.DataFrame(cost).groupby('method').agg(new_calls=('method','size'),tokens=('tokens','sum'),charged_tokens=('charged_tokens','sum'),api_seconds_sum=('api_seconds','sum'),mean_latency_s=('api_seconds','mean')).reset_index();costs.to_csv('res/api_cost.csv',index=False)
    hits=set(open('res/reused_hashes.txt').read().splitlines())
    budget=dict(new_calls=len(calls),reused_unique_requests=len(hits),reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in calls),charged_tokens=sum(charge(r) for r in calls),cap=2000000)
    json.dump(budget,open('res/budget.json','w'),indent=2)
    runtime=pd.read_csv('res/runtime.csv').groupby('method').seconds.sum().reindex(ORDER)
    fig,axes=plt.subplots(1,2,figsize=(12,5));axes[0].barh([NAMES[m] for m in ORDER],runtime/60,color=COLORS);axes[0].set(title='Offline recalibration elapsed time',xlabel='Minutes')
    axes[1].barh([NAMES[m] for m in ORDER],costs.set_index('method').charged_tokens.reindex(ORDER).fillna(0)/1000,color=COLORS);axes[1].set(title='New calls only, including event repeats/errors',xlabel='Thousand tokens');savefig(fig,'09_cost')
    if os.path.exists('res/context_probe_predictions.csv'):
        probe=pd.read_csv('res/context_probe_predictions.csv')
        baseline=ev[(ev.task=='Extreme_1007')&(ev.method=='role_with_history')]
        cm=table(pd.concat([probe,baseline],ignore_index=True),['method','replicate']);cm.to_csv('res/context_probe_metrics.csv',index=False)
        cs=cm.groupby('method').agg(WAPE_mean=('WAPE_pct','mean'),WAPE_std=('WAPE_pct','std'),bias_mean=('bias_pct','mean')).reset_index();cs.to_csv('res/context_probe_summary.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,4))
        for x,(m,r) in enumerate(cm.groupby('method')):
            label='Include Sep 29' if m=='role_storm_context' else 'Nearest 4 examples'
            ax.bar([x],[r.WAPE_pct.mean()],yerr=[r.WAPE_pct.std()],capsize=5,color='#855da8' if x==0 else '#527fb3')
            ax.scatter(np.full(len(r),x),r.WAPE_pct,color='black',s=20,zorder=3)
            ax.text(x,-.1,label,transform=ax.get_xaxis_transform(),ha='center')
        ax.set(xticks=[],ylabel='Oct 7 WAPE %; mean ± sample SD',title='History-context probe | fixed population weights')
        savefig(fig,'10_context_probe')
    print(result[['task','method','WAPE_pct','bias_pct']].round(2).to_string(index=False));print(budget)

if __name__=='__main__':analyze()
