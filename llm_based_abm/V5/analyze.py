"""Offline scoring and scientific figures; no API requests."""
import collections
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from api import ledger,charge
from model import ACTIONS

NAMES={'weekday_mean':'Weekday mean','ridge':'Ridge','rule_abm':'Rule ABM','direct_few':'Direct few-shot',
       'role_no_history':'Role: no history','role_with_history':'Role: group history'}
ORDER=list(NAMES)
COLORS=['#89939e','#bc8d42','#cb6a53','#527fb3','#378d78','#855da8']
plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})

def metrics(x):
    x=x.dropna(subset=['actual','predicted']);e=x.predicted-x.actual
    return {'n_station_days':len(x),'actual_entries':x.actual.sum(),'WAPE_pct':100*abs(e).sum()/x.actual.sum(),
        'MAE':abs(e).mean(),'RMSE':np.sqrt((e**2).mean()),'bias_pct':100*e.sum()/x.actual.sum(),
        'within_10pct':100*(abs(e)<=.1*x.actual).mean(),'within_20pct':100*(abs(e)<=.2*x.actual).mean()}

def annotate(pred):
    dates=pd.read_csv('res/dates.csv')
    return pred.merge(dates[['task','date','event','event_day','rain_mm']],on=['task','date'],how='left')

def table(pred,keys):
    return pd.DataFrame([{**dict(zip(keys,k if isinstance(k,tuple) else [k])),**metrics(x)} for k,x in pred.groupby(keys,dropna=False)])

def analyze():
    primary=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str});pred=annotate(primary)
    result=table(pred,['task','method']);result.to_csv('res/metrics.csv',index=False)
    extreme=pred[pred.task=='Extreme'].copy();extreme['condition']=np.where(extreme.event_day,'event day','surrounding days')
    em=table(extreme,['condition','method']);em.to_csv('res/extreme_metrics.csv',index=False)
    table(extreme[extreme.event_day],['event','method']).to_csv('res/event_metrics.csv',index=False)
    table(pred,['task','station_complex_id','method']).to_csv('res/station_metrics.csv',index=False)
    dates=pd.read_csv('res/dates.csv');panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str})
    summary=[]
    for (task,split),r in panel.groupby(['task','split']):
        summary.append(dict(task=task,split=split,days=r.date.nunique(),station_days=len(r),complete_station_days=r.total_ridership.notna().sum(),
            mean_daily_entries=r.total_ridership.mean(),min_daily_entries=r.total_ridership.min(),max_daily_entries=r.total_ridership.max(),rain_min=r.rain_mm.min(),rain_max=r.rain_mm.max()))
    pd.DataFrame(summary).to_csv('res/dataset_summary.csv',index=False)
    # Structural diagnostic: each representative can contribute at most one entry opportunity.
    populations=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});capacity=[]
    for (task,method,group),part in populations.groupby(['task','method','group']):
        upper=part.groupby('station_complex_id').effective_population.sum().rename('capacity')
        r=panel[(panel.task==task)&(panel.split=='test')][['station_complex_id','date',group]].rename(columns={group:'actual'}).dropna().merge(upper,on='station_complex_id')
        capacity.append(dict(task=task,method=method,group=group,potential_scale_all_stations=upper.sum(),
            unavoidable_underprediction_pct=100*(r.actual-r.capacity).clip(lower=0).sum()/r.actual.sum(),days_over_capacity=int((r.actual>r.capacity).sum())))
    pd.DataFrame(capacity).to_csv('res/capacity_diagnostic.csv',index=False)

    fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True)
    weather=pd.read_csv('res/weather_daily.csv');palette={'train':'#a8c7b0','validation':'#e4bc67','test':'#8d79b7'}
    for a,task in zip(ax,['Normal_A','Normal_B','Extreme']):
        a.bar(pd.to_datetime(weather.date),weather.rain_mm,color='#e3e5e9',width=.9)
        for split,color in palette.items():
            d=dates[(dates.task==task)&(dates.split==split)]
            a.bar(pd.to_datetime(d.date),d.rain_mm,color=color,label=split,width=.9)
            a.scatter(pd.to_datetime(d.date),np.zeros(len(d)),marker='|',s=70,color=color,linewidths=2,zorder=3)
        a.set(title=task,ylabel='Rain (mm/day)');a.legend(ncol=3,loc='upper left');a.set_yscale('symlog',linthresh=5)
    fig.suptitle('All selected days, chronological splits, and rainfall\nColoured ticks at zero mark selected days; bars show actual rain; grey days excluded',fontsize=12)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig('res/figures/01_dataset_calendar.png');plt.close(fig)

    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for a,task in zip(ax,['Normal_A','Normal_B','Extreme']):
        for j,split in enumerate(['train','validation','test']):
            values=panel[(panel.task==task)&(panel.split==split)].total_ridership.dropna()
            a.boxplot(values,positions=[j],widths=.55,patch_artist=True,boxprops={'facecolor':palette[split]})
        a.set(xticks=[0,1,2],xticklabels=['Train','Validation','Test'],ylabel='Entries / station-day',title=task)
    fig.tight_layout();fig.savefig('res/figures/02_actual_distribution.png');plt.close(fig)

    fig,ax=plt.subplots(1,4,figsize=(17,5),sharey=True)
    panels=[('Normal_A',result[result.task=='Normal_A']),('Normal_B',result[result.task=='Normal_B']),('Extreme window',result[result.task=='Extreme']),('Extreme event days',em[em.condition=='event day'])]
    for a,(title,r) in zip(ax,panels):
        r=r.set_index('method').reindex(ORDER);bars=a.barh(np.arange(6),r.WAPE_pct,color=COLORS)
        a.set(yticks=np.arange(6),yticklabels=[NAMES[m] for m in ORDER],xlabel='Daily WAPE (%) — lower is better',title=title)
        for b,v in zip(bars,r.WAPE_pct):a.text(v+.3,b.get_y()+b.get_height()/2,f'{v:.1f}',va='center',fontsize=9)
        a.set_xlim(0,max(r.WAPE_pct)*1.2)
    ax[0].invert_yaxis();fig.tight_layout();fig.savefig('res/figures/03_accuracy.png');plt.close(fig)

    fig,ax=plt.subplots(2,1,figsize=(12,7))
    for a,(event,rows) in zip(ax,extreme.groupby('event')):
        daily=rows.dropna(subset=['actual']).groupby(['date','method'])[['actual','predicted']].sum(min_count=1).reset_index()
        actual=daily[daily.method=='weekday_mean'];a.plot(actual.date,actual.actual,'k-o',label='Actual',linewidth=2.5)
        for m,c in zip(ORDER,COLORS):
            r=daily[daily.method==m];a.plot(r.date,r.predicted,'o--',color=c,label=NAMES[m],alpha=.85)
        a.set(title=f'Event {event} — sum of observed station-days',ylabel='Daily entries');a.legend(ncol=4,fontsize=8)
    fig.tight_layout();fig.savefig('res/figures/04_event_predictions.png');plt.close(fig)

    dec=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str}).merge(dates[['task','date','event_day']],on=['task','date'])
    dec['condition']=np.where(dec.task=='Extreme',np.where(dec.event_day,'Extreme peak','Extreme surrounding'),dec.task)
    actions=[];group_prediction=[]
    for (task,method,condition,group),r in dec.groupby(['task','method','condition','group']):
        for action in ACTIONS:actions.append(dict(task=task,method=method,condition=condition,group=group,action=action,
            probability=float(np.average(r['p_'+action],weights=r.effective_population)) if r.effective_population.sum()>0 else np.nan))
    actions=pd.DataFrame(actions);actions.to_csv('res/group_actions.csv',index=False)
    for (task,method,date,station,group),r in dec.groupby(['task','method','date','station_complex_id','group']):
        truth=panel[(panel.task==task)&(panel.date==date)&(panel.station_complex_id==station)][group].iloc[0]
        group_prediction.append(dict(task=task,method=method,date=date,station_complex_id=station,group=group,actual=truth,predicted=(r.effective_population*r.p_subway).sum()))
    gp=pd.DataFrame(group_prediction)
    table(gp,['task','method','group']).to_csv('res/group_metrics.csv',index=False)
    fig,ax=plt.subplots(1,4,figsize=(14,4))
    peak_dates=set(dates[(dates.task=='Extreme')&dates.event_day&(dates.split=='test')].date)
    for a,(group,r) in zip(ax,gp[(gp.task=='Extreme')&gp.date.isin(peak_dates)].groupby('group')):
        totals=r.groupby('method')[['actual','predicted']].sum()
        methods=['rule_abm','role_no_history','role_with_history']
        a.bar(['Actual','Rule','Role no hist.','Role hist.'],[totals.actual.iloc[0]]+[totals.loc[m,'predicted'] for m in methods],color=['#333333',COLORS[2],COLORS[4],COLORS[5]])
        a.set(title=group,ylabel='Entries across two event days');a.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig('res/figures/09_group_entry_errors.png');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    for a,method in zip(ax,['rule_abm','role_no_history','role_with_history']):
        r=actions[(actions.condition=='Extreme peak')&(actions.method==method)].pivot(index='group',columns='action',values='probability').reindex(columns=ACTIONS)
        r.plot.bar(stacked=True,ax=a,colormap='Set2',legend=False);a.set(title=NAMES[method],ylabel='Model daily choice probability',ylim=(0,1));a.tick_params(axis='x',rotation=30)
    ax[-1].legend(loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig('res/figures/05_group_choices.png');plt.close(fig)

    # Transparent word frequencies, rather than treating generated text as measured psychology.
    stop=set('i a an the to of and or in on is are be my me it so with for at as that this will would can likely not no more from have do but rather than due day'.split())
    fig,ax=plt.subplots(1,2,figsize=(12,5));keywords=[]
    for a,condition in zip(ax,['Normal','Extreme']):
        r=dec[(dec.method=='role_no_history')&(dec.task.str.startswith(condition))].drop_duplicates(['task','date','agent_id','request_hash'])
        counts=collections.Counter(w for reason in r.reason for w in re.findall('[a-z]+',str(reason).lower()) if w not in stop and len(w)>2)
        top=counts.most_common(12);a.barh([w for w,n in top][::-1],[n for w,n in top][::-1],color='#527fb3');a.set(title=condition+' — short decision-summary words',xlabel='Occurrences, deduplicated shared policies')
        keywords.extend(dict(condition=condition,word=w,count=n) for w,n in counts.items())
    fig.tight_layout();fig.savefig('res/figures/06_reason_words.png');plt.close(fig);pd.DataFrame(keywords).to_csv('res/reason_words.csv',index=False)
    dec[(dec.method.str.startswith('role'))].drop_duplicates(['task','date','agent_id','request_hash'])[['task','date','method','group','agent_id','p_subway','p_stay_home','reason']].to_csv('res/reason_examples.csv',index=False)

    repeats=[primary]+[pd.read_csv(f,dtype={'station_complex_id':str}) for f in sorted(glob.glob('res/repeat*_predictions.csv'))]
    rm=table(pd.concat(repeats,ignore_index=True),['task','method','replicate']);rm.to_csv('res/repeat_metrics.csv',index=False)
    rms=rm.groupby(['task','method']).WAPE_pct.agg(['mean','std','min','max','count']).reset_index();rms.to_csv('res/repeat_summary.csv',index=False)
    if len(repeats)>1:
        combined=pd.concat(repeats,ignore_index=True)
        repeated_events=annotate(combined)
        event_repeats=table(repeated_events[(repeated_events.task=='Extreme')&repeated_events.event_day],['method','replicate'])
        event_repeats.groupby('method').WAPE_pct.agg(['mean','std','min','max','count']).to_csv('res/repeat_event_summary.csv')
        ensemble=combined[combined.method.isin(['direct_few','role_no_history','role_with_history'])].groupby(['task','method','date','station_complex_id']).agg(actual=('actual','first'),predicted=('predicted','mean')).reset_index()
        table(ensemble,['task','method']).to_csv('res/ensemble_metrics.csv',index=False)
        fig,ax=plt.subplots(1,4,figsize=(17,4))
        for a,task in zip(ax,['Normal_A','Normal_B','Extreme']):
            r=rms[(rms.task==task)&(rms['count']>1)];a.bar(r.method.map(NAMES),r['mean'],yerr=r['std'],color=['#527fb3','#378d78','#855da8'],capsize=4)
            a.set(title=task,ylabel='WAPE mean ± sample SD (pp)');a.tick_params(axis='x',rotation=20)
        event_summary=event_repeats.groupby('method').WAPE_pct.agg(['mean','std','count']).reset_index()
        r=event_summary[event_summary['count']>1]
        ax[-1].bar(r.method.map(NAMES),r['mean'],yerr=r['std'],color=['#527fb3','#378d78','#855da8'],capsize=4)
        ax[-1].set(title='Extreme event days ONLY',ylabel='WAPE mean ± sample SD (pp)');ax[-1].tick_params(axis='x',rotation=20)
        fig.tight_layout();fig.savefig('res/figures/07_repeats.png');plt.close(fig)
    if glob.glob('res/transfer_predictions.csv'):
        transfer=annotate(pd.read_csv('res/transfer_predictions.csv',dtype={'station_complex_id':str}));tm=table(transfer,['source_task','task','method'])
        tm=tm.merge(result[['task','method','WAPE_pct']].rename(columns={'WAPE_pct':'in_domain_WAPE_pct'}),on=['task','method']);tm['loss_pp']=tm.WAPE_pct-tm.in_domain_WAPE_pct;tm.to_csv('res/transfer_metrics.csv',index=False)
        fig,ax=plt.subplots(1,2,figsize=(12,5),sharey=True)
        for a,((source,target),r) in zip(ax,tm.groupby(['source_task','task'])):
            r=r.set_index('method').reindex(ORDER);y=np.arange(len(ORDER))
            a.barh(y-.18,r.in_domain_WAPE_pct,height=.35,color='#9eafa7',label='Trained on target task')
            a.barh(y+.18,r.WAPE_pct,height=.35,color='#bc8d42',label='Transferred without refitting')
            a.set(yticks=y,yticklabels=[NAMES[m] for m in ORDER],title=f'{source} -> {target}',xlabel='Daily WAPE (%)');a.legend(fontsize=8)
        fig.tight_layout();fig.savefig('res/figures/10_transfer.png');plt.close(fig)

    calls=ledger();cost=[]
    for r in calls:
        route=r['request']['route'];method='direct_few' if route=='direct' else 'role_with_history' if 'group_history' in r['request']['view'] else 'role_no_history'
        cost.append(dict(method=method,replicate=r['request']['replicate'],status=r['status'],tokens=r.get('usage',{}).get('total_tokens',0),charged_tokens=charge(r),api_seconds=r.get('elapsed_seconds',0)))
    costs=pd.DataFrame(cost).groupby('method').agg(calls=('method','size'),tokens=('tokens','sum'),charged_tokens=('charged_tokens','sum'),api_seconds_sum=('api_seconds','sum'),mean_latency_s=('api_seconds','mean')).reset_index()
    costs.to_csv('res/api_cost.csv',index=False)
    runtime=pd.read_csv('res/runtime.csv').groupby('method').seconds.sum().reindex(ORDER)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].barh([NAMES[m] for m in ORDER],runtime/60,color=COLORS);ax[0].set(xlabel='Minutes, main experiment stages',title='Observed wall-clock runtime')
    used=costs.set_index('method').charged_tokens.reindex(ORDER).fillna(0)
    ax[1].barh([NAMES[m] for m in ORDER],used/1000,color=COLORS);ax[1].set(xlabel='Thousand tokens, conservative charged',title='All calls, including repeats, transfer and errors')
    fig.tight_layout();fig.savefig('res/figures/08_cost.png');plt.close(fig)
    json.dump(dict(real_calls=len(calls),reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in calls),conservative_tokens=sum(charge(r) for r in calls),cap=4000000),open('res/budget.json','w'),indent=2)
    print(result[['task','method','WAPE_pct','within_20pct']].round(2).to_string(index=False));print(costs.round(2).to_string(index=False))

if __name__=='__main__':analyze()
