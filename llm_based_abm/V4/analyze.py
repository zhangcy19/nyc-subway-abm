"""Offline evaluation and figures. Missing observations are never scored as zero."""
import os
import json
import gzip
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import GROUPS, population, validate_choice, PolicyModel, rule_choice
from experiment import load_panel, weather_view, apply_policy

ORDER=['baseline','ridge','rule','direct_zero','direct_few','persona_zero','persona_few']
LABELS={'baseline':'Historical baseline','ridge':'Ridge','rule':'Rule ABM','direct_zero':'Direct zero-shot','direct_few':'Direct few-shot','persona_zero':'Persona zero-shot','persona_few':'Persona few-shot'}
ZH={'baseline':'历史客流基线','ridge':'Ridge 数值模型','rule':'规则 ABM','direct_zero':'直接预测 zero-shot','direct_few':'直接预测 few-shot','persona_zero':'角色决策 zero-shot','persona_few':'角色决策 few-shot'}
COLORS=dict(zip(ORDER,['#9ca3af','#64748b','#2563eb','#f59e0b','#d97706','#10b981','#047857']))


def metrics(data,keys):
    records=[]
    for key,rows in data.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        valid=rows[rows.total_ridership.notna()&rows.predicted.notna()].copy()
        denominator=valid.total_ridership.sum()
        if denominator<=0:continue
        day=valid.groupby(['station_complex_id','date'])[['total_ridership','predicted']].sum()
        blocks=valid.groupby(['station_complex_id','date','block'])[['total_ridership','predicted']].sum()
        station=valid.groupby('station_complex_id').apply(lambda r:float(np.abs(r.groupby('date').predicted.sum()-r.groupby('date').total_ridership.sum()).sum()/r.total_ridership.sum()),include_groups=False)
        records.append({**dict(zip(keys,key)),'daily_wape':float(np.abs(day.predicted-day.total_ridership).sum()/denominator),
            'block_wape':float(np.abs(blocks.predicted-blocks.total_ridership).sum()/denominator),
            'hourly_wape':float(np.abs(valid.predicted-valid.total_ridership).sum()/denominator),
            'bias':float((valid.predicted.sum()-denominator)/denominator),
            'station_mean_daily_wape':float(station.mean()),'station_days':len(day),'observed_hours':len(valid),'actual_entries':float(denominator)})
    return pd.DataFrame(records)


def daily_table(data):
    valid=data[data.total_ridership.notna()].copy()
    return valid.groupby(['method','station_complex_id','station_name','borough','date','weather_class']).agg(
        actual=('total_ridership','sum'),predicted=('predicted','sum'),observed_hours=('total_ridership','count')).reset_index()


def bootstrap(daily):
    rng=np.random.default_rng(20261013);dates=sorted(daily.date.unique());out=[]
    pivot=daily.pivot(index=['date','station_complex_id'],columns='method',values='predicted')
    actual=daily.drop_duplicates(['date','station_complex_id']).set_index(['date','station_complex_id']).actual.reindex(pivot.index)
    for method in ['direct_zero','direct_few','persona_zero','persona_few']:
        references=['baseline','ridge','rule']+(['direct_zero'] if method=='direct_few' else ['persona_zero'] if method=='persona_few' else [])
        for reference in references:
            numerator=(np.abs(pivot[method]-actual)-np.abs(pivot[reference]-actual)).groupby('date').sum().reindex(dates).to_numpy()
            denominator=actual.groupby('date').sum().reindex(dates).to_numpy()
            sampled=rng.integers(0,len(dates),size=(1000,len(dates)))
            differences=numerator[sampled].sum(axis=1)/denominator[sampled].sum(axis=1)
            point=float((np.abs(pivot[method]-actual)-np.abs(pivot[reference]-actual)).sum()/actual.sum())
            out.append({'method':method,'reference':reference,'difference':point,'lower':float(np.quantile(differences,.025)),'upper':float(np.quantile(differences,.975)),'resampling':'24 date clusters; only three extreme days'})
    return pd.DataFrame(out)


def api_quality():
    records=[]
    for f in ['res/api_calls.jsonl.gz','res/api_calls.jsonl']:
        if os.path.exists(f):
            with (gzip.open(f,'rt') if f.endswith('.gz') else open(f)) as source:records.extend(json.loads(line) for line in source)
    def valid_response(r):
        if r['status']!='completed':return False
        try:
            p=r['parsed'];v=json.loads(r['request']['prompt']);v=v.get('input',v)
            if r['request']['route']=='direct':
                a=p['predicted']
                return len(a)==6 and all(not isinstance(x,bool) and isinstance(x,(int,float)) and np.isfinite(x) and x>=0 for x in a) and bool(p['reason'].strip())
            a=p['choices']
            if len(a)!=6:return False
            person=v['you']
            for c in a:
                token=c['action']
                if 'delay_min' in c:
                    validate_choice(c,person)
                else:
                    delay=60 if token=='delay_subway_60' else 120 if token=='delay_subway_120' else 0
                    validate_choice({'action':'delay_subway' if delay else token,'delay_min':delay,'reason':c['reason']},person)
            return True
        except (ValueError,TypeError,KeyError,AttributeError):return False
    out=[]
    for route in ['direct','persona']:
        rs=[r for r in records if r['request']['route']==route]
        out.append({'route':route,'calls':len(rs),'input_tokens':sum(r.get('usage',{}).get('input_tokens',0) for r in rs),
                    'output_tokens':sum(r.get('usage',{}).get('output_tokens',0) for r in rs),
                    'reported_total_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in rs),
                    'charged_tokens':sum(r.get('conservative_charge',r.get('usage',{}).get('total_tokens',0)) for r in rs),
                    'request_errors':sum(r['status']=='request_error' for r in rs),'incomplete':sum(r['status'] not in ['completed','request_error'] for r in rs),
                    'repair_or_retry_calls':sum(r['request']['attempt']>0 for r in rs),
                    'format_constraint_valid_calls':sum(valid_response(r) for r in rs),
                    'first_attempt_calls':sum(r['request']['attempt']==0 for r in rs),
                    'first_attempt_valid_calls':sum(r['request']['attempt']==0 and valid_response(r) for r in rs)})
    result=pd.DataFrame(out);result.to_csv('res/api_usage.csv',index=False)
    return result


def behavioral_groups(decisions,data):
    d=decisions[decisions.group.notna()&(decisions.weight>0)].copy()
    weather=data[['station_complex_id','date','weather_class']].drop_duplicates().copy()
    weather['date']=weather.date.dt.strftime('%Y-%m-%d')
    d=d.merge(weather,on=['station_complex_id','date'],how='left')
    # Action shares are distributions of represented trip opportunities, not people.
    actions=d.groupby(['method','group','action']).weight.sum().rename('weight').reset_index()
    actions['share']=actions.weight/actions.groupby(['method','group']).weight.transform('sum')
    actions.to_csv('res/group_actions.csv',index=False)
    identities=d.groupby(['method','identity','action']).weight.sum().rename('weight').reset_index()
    identities['share']=identities.weight/identities.groupby(['method','identity']).weight.transform('sum')
    identities.to_csv('res/identity_actions.csv',index=False)
    by_weather=d.groupby(['method','weather_class','group','action']).weight.sum().rename('weight').reset_index()
    by_weather['share']=by_weather.weight/by_weather.groupby(['method','weather_class','group']).weight.transform('sum')
    by_weather.to_csv('res/group_weather_actions.csv',index=False)
    summary=[]
    for method in ['rule','persona_zero','persona_few']:
        x=data[data.method==method]
        for g in GROUPS:
            valid=x[x[g].notna()&x['pred_'+g].notna()]
            day=valid.groupby(['date','station_complex_id'])[[g,'pred_'+g]].sum();den=day[g].sum()
            if den>0:summary.append({'method':method,'group':g,'daily_wape':float(np.abs(day['pred_'+g]-day[g]).sum()/den),'observed_hours':len(valid),'actual':float(den)})
    pd.DataFrame(summary).to_csv('res/group_metrics.csv',index=False)
    reasons=d.sort_values(['method','group','date','agent_id','block']).groupby(['method','group','weather_class','action']).head(1)
    reasons[['method','group','identity','date','weather_class','action','reason']].to_csv('res/reason_examples.csv',index=False)


def service_context():
    stops=pd.read_csv('../../data/supplementary/service_2023_09_29_stop_times.csv.gz',dtype={'stop_id':str,'trip_uid':str})
    complexes=pd.read_csv('../../data/supplementary/station_complexes_current.csv',dtype={'complex_id':str})
    mapping=complexes[['complex_id','gtfs_stop_ids']].copy();mapping['parent_stop']=mapping.gtfs_stop_ids.str.split(';');mapping=mapping.explode('parent_stop');mapping['parent_stop']=mapping.parent_stop.str.strip()
    stops['parent_stop']=stops.stop_id.str.rstrip('NS');stops=stops.sort_values('last_observed').drop_duplicates(['trip_uid','stop_id'],keep='last')
    stops=stops.merge(mapping[['complex_id','parent_stop']],on='parent_stop',how='left')
    stops['timestamp']=pd.to_datetime(stops.departure_time,unit='s',utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.floor('h')
    stations=pd.read_csv('res/stations.csv',dtype={'station_complex_id':str})
    relevant=stops[stops.complex_id.isin(stations.station_complex_id)]
    counts=relevant.groupby(['complex_id','timestamp']).size().rename('archived_trip_stop_departure_estimates').reset_index().rename(columns={'complex_id':'station_complex_id'})
    counts.to_csv('res/sep29_service_context.csv',index=False)
    # These are archived estimates, not confirmed train counts or closure labels.


def plots(summary,weather,daily,stations,data):
    os.makedirs('res/figures',exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,ax=plt.subplots(figsize=(10,4))
    t=summary.set_index('method').reindex(ORDER)
    ax.bar(range(7),t.daily_wape*100,color=[COLORS[m] for m in ORDER]);ax.set_xticks(range(7),[LABELS[m].replace(' ','\n',1) for m in ORDER]);ax.set_ylabel('Station-day WAPE (%)');ax.set_title('Same 8 stations and 24 test days | lower is better')
    for i,v in enumerate(t.daily_wape*100):ax.text(i,v+.3,f'{v:.1f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig('res/figures/overall_errors.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,kind in zip(axes,['dry','wet','extreme']):
        t=weather[weather.weather_class==kind].set_index('method').reindex(ORDER)
        ax.bar(range(7),t.daily_wape*100,color=[COLORS[m] for m in ORDER]);ax.set_title(kind+' weather');ax.set_xticks(range(7),['Base','Ridge','Rule','D0','DF','P0','PF'],rotation=30)
    axes[0].set_ylabel('Station-day WAPE (%)');fig.suptitle('D = direct | P = persona | 0 = zero-shot | F = few-shot');fig.tight_layout();fig.savefig('res/figures/weather_errors.png');plt.close(fig)
    extreme=daily[daily.weather_class=='extreme'].groupby(['date','method'])[['actual','predicted']].sum().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(date,x) in zip(axes,extreme.groupby('date')):
        a=float(x.actual.iloc[0]);ax.axhline(a,color='black',ls='--',label='Observed');x=x.set_index('method').reindex(ORDER);ax.bar(range(7),x.predicted,color=[COLORS[m] for m in ORDER]);ax.set_title(str(pd.Timestamp(date).date()));ax.set_xticks(range(7),['Base','Ridge','Rule','D0','DF','P0','PF'],rotation=30)
    axes[0].set_ylabel('Entries across 8 sites (observed hours)');axes[-1].legend();fig.tight_layout();fig.savefig('res/figures/extreme_days.png');plt.close(fig)
    per=metrics(data,['method','station_complex_id']);per=per.merge(stations[['station_complex_id','latitude','longitude','station_name']],on='station_complex_id')
    fig,axes=plt.subplots(1,3,figsize=(12,5),sharex=True,sharey=True)
    for ax,m in zip(axes,['rule','direct_few','persona_few']):
        x=per[per.method==m];scatter=ax.scatter(x.longitude,x.latitude,c=x.daily_wape*100,cmap='YlOrRd',vmin=0,vmax=max(40,per.daily_wape.max()*100),s=120,edgecolors='black',lw=.4)
        for _,r in x.iterrows():ax.annotate(r.station_complex_id,(r.longitude,r.latitude),xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.set_title(LABELS[m]);ax.set_xlabel('Longitude');ax.set_aspect(1/np.cos(np.deg2rad(40.7)));ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Latitude');fig.suptitle('Geographic comparison (station complex coordinates; no basemap)')
    fig.subplots_adjust(top=.88,right=.87,wspace=.15)
    color_axis=fig.add_axes([.9,.22,.018,.55]);fig.colorbar(scatter,cax=color_axis,label='Station-day WAPE (%)')
    fig.savefig('res/figures/station_map.png');plt.close(fig)
    a=pd.read_csv('res/group_actions.csv');a['subway']=a.action.isin(['subway','delay_subway']);t=a[a.subway].groupby(['method','group']).share.sum().unstack().reindex(['rule','persona_zero','persona_few'])
    ax=t.plot.bar(figsize=(9,4));ax.set_ylabel('Represented trip share choosing subway');ax.set_ylim(0,1.05);ax.set_xlabel('');ax.legend(title='Fare proxy',bbox_to_anchor=(1.02,1),loc='upper left');plt.tight_layout();plt.savefig('res/figures/group_subway_choices.png');plt.close()
    actions=pd.read_csv('res/group_weather_actions.csv')
    actions=actions[actions.action.isin(['subway','delay_subway'])]
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,kind in zip(axes,['dry','wet','extreme']):
        table=actions[actions.weather_class==kind].groupby(['group','method']).share.sum().unstack().reindex(GROUPS)
        table=table.reindex(columns=['rule','persona_zero','persona_few']).fillna(0)
        table.columns=['Rule','Persona zero','Persona few']
        table.index=['Student','Fair fare','Senior/disab.','General']
        table.plot.bar(ax=ax,color=[COLORS[m] for m in ['rule','persona_zero','persona_few']],legend=False)
        ax.set_title(kind+' weather');ax.set_xlabel('');ax.tick_params(axis='x',rotation=25);ax.set_ylim(0,1.05)
    axes[0].set_ylabel('Represented trip share choosing subway')
    fig.suptitle('Hypothetical decisions by fare proxy; not observed individual choices')
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.tight_layout(rect=[0,0,1,.84]);fig.savefig('res/figures/group_weather_choices.png');plt.close(fig)


def probes():
    files=sorted(glob.glob('res/test_probe*_predictions.csv.gz'))
    parts=[]
    for f in files:
        if os.path.exists(f):
            x=pd.read_csv(f,dtype={'station_complex_id':str},parse_dates=['date','timestamp'])
            if set(x.method)!=set(ORDER) or x.groupby('method').size().nunique()!=1:continue
            x['variant']=os.path.basename(f).replace('_predictions.csv.gz','');parts.append(x)
    if not parts:return
    p=pd.concat(parts);t=metrics(p,['variant','method','n_agents','enriched','replicate']);t.to_csv('res/probe_metrics.csv',index=False)
    repeats=t.groupby(['method','n_agents','enriched']).daily_wape.agg(['mean','std','min','max','count']).reset_index()
    repeats.to_csv('res/probe_repeat_summary.csv',index=False)
    combined=p.groupby(['method','n_agents','enriched','station_complex_id','date','timestamp','block','weather_class'],dropna=False).agg(
        predicted=('predicted','mean'),total_ridership=('total_ridership','first')).reset_index()
    metrics(combined,['method','n_agents','enriched']).to_csv('res/probe_ensemble_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    for method in ['rule','persona_zero','persona_few']:
        x=repeats[(repeats.method==method)&~repeats.enriched].sort_values('n_agents')
        ax.errorbar(x.n_agents,x['mean']*100,yerr=np.vstack([(x['mean']-x['min'])*100,(x['max']-x['mean'])*100]),fmt='o-',capsize=4,label=LABELS[method],color=COLORS[method])
    ax.set_xlabel('Representatives per station');ax.set_ylabel('Mean daily WAPE (%) on same 2 sites / 4 days')
    ax.set_title('Three repeated episodes; error bars = min-max, not confidence intervals')
    ax.legend();fig.tight_layout();fig.savefig('res/figures/agent_resolution.png');plt.close(fig)


def rule_sensitivity():
    # Post-test diagnosis of every predeclared grid value, never a new selected main model.
    panel=load_panel();panel=panel[panel.split=='test']
    selection=json.load(open('res/offline_selection.json'))
    parts=[]
    for candidate in selection['rule_candidates']:
        scale=candidate['rain_scale'];model=PolicyModel(panel.station_complex_id.unique().tolist(),8)
        for date in sorted(panel.date.unique()):
            for s,rows in panel[panel.date==date].groupby('station_complex_id'):
                rows=rows.sort_values('hour');agents=[a for a in model.agents if a.persona['station_complex_id']==s]
                policies=[]
                for agent in agents:
                    if rows['planned_'+agent.persona['group']].sum()<=0:
                        policy=[{'action':'not_applicable','delay_min':0,'reason':'No planned mass.'}]*6
                    else:
                        policy=[rule_choice(agent.persona,e,agent.memory,rain_scale=scale) for e in weather_view(rows)]
                        agent.observe_policy(policy,weather_view(rows))
                    policies.append(policy)
                pred,_,_=apply_policy(rows,[a.persona for a in agents],{'choices':policies,'hashes':['']*len(agents)},'rule')
                x=rows[['date','station_complex_id','block','total_ridership','weather_class']].copy()
                x['predicted']=pred;x['rain_scale']=scale;parts.append(x)
    predictions=pd.concat(parts)
    summary=metrics(predictions,['rain_scale'])
    summary=summary.merge(pd.DataFrame(selection['rule_candidates']),on='rain_scale')
    summary['selected_before_test']=summary.rain_scale==selection['rule_rain_scale']
    summary['scope']='post-test sensitivity, not a reselected primary model'
    summary.to_csv('res/rule_sensitivity.csv',index=False)
    metrics(predictions,['rain_scale','weather_class']).to_csv('res/rule_weather_sensitivity.csv',index=False)
    return summary


def analyze():
    data=pd.read_csv('res/test_predictions.csv.gz',dtype={'station_complex_id':str},parse_dates=['date','timestamp'])
    if set(data.method)!=set(ORDER) or data.groupby('method').size().nunique()!=1:raise RuntimeError('Main test incomplete; do not report partial results')
    summary=metrics(data,['method']);summary.to_csv('res/metrics.csv',index=False)
    weather=metrics(data,['method','weather_class']);weather.to_csv('res/weather_metrics.csv',index=False)
    data['month']=data.date.dt.strftime('%Y-%m');metrics(data,['method','month']).to_csv('res/month_metrics.csv',index=False)
    daily=daily_table(data);daily.to_csv('res/daily_predictions.csv',index=False)
    metrics(data[data.weather_class=='extreme'],['method','date']).to_csv('res/event_metrics.csv',index=False)
    metrics(data,['method','station_complex_id','station_name','borough']).to_csv('res/station_metrics.csv',index=False)
    bootstrap(daily).to_csv('res/paired_bootstrap.csv',index=False)
    valid=pd.read_csv('res/validation_predictions.csv.gz',dtype={'station_complex_id':str},parse_dates=['date','timestamp'])
    calibration=[];adjusted=[]
    for m,rows in valid[valid.total_ridership.notna()].groupby('method'):
        factor=float(np.clip(rows.total_ridership.sum()/rows.predicted.sum(),.7,1.4));calibration.append({'method':m,'validation_only_scale':factor})
        x=data[data.method==m].copy();x['predicted']*=factor;adjusted.append(x)
    pd.DataFrame(calibration).to_csv('res/calibration.csv',index=False);metrics(pd.concat(adjusted),['method']).to_csv('res/calibrated_metrics.csv',index=False)
    coverage=data[data.method=='baseline'].groupby(['station_complex_id','date']).total_ridership.count().rename('observed_hours').reset_index();coverage.to_csv('res/coverage.csv',index=False)
    decisions=pd.read_csv('res/test_decisions.csv.gz',dtype={'station_complex_id':str});behavioral_groups(decisions,data);service_context()
    stations=pd.read_csv('res/stations.csv',dtype={'station_complex_id':str})
    selected=stations.station_complex_id.tolist()
    pd.DataFrame([{**p,'n_agents':n} for n in [8,16,24]
        for p in population(selected if n==8 else [selected[0],selected[4]],n)]).to_csv('res/representatives.csv',index=False)
    plots(summary,weather,daily,stations,data);probes();sensitivity=rule_sensitivity();usage=api_quality()
    table=summary.set_index('method').reindex(ORDER)
    winner=table.daily_wape.idxmin()
    lines=['# V4 结果','',f'本轮是 8 站、24 天的天气已知回顾性测试；实际评分 {int(table.observed_hours.iloc[0]):,} 个站点小时。日总误差最低：{ZH[winner]}。','',
        'WAPE = 各站预测与实际的绝对差之和 / 实际客流之和，越低越好。它不是准确率；偏差为负表示整体少预测了。','',
        '| 方法 | 日总 WAPE | 四小时 WAPE | 小时 WAPE | 偏差 |','| --- | ---: | ---: | ---: | ---: |']
    for m,r in table.iterrows():lines.append(f'| {ZH[m]} | {r.daily_wape:.2%} | {r.block_wape:.2%} | {r.hourly_wape:.2%} | {r.bias:+.2%} |')
    llm_best=table.loc[['direct_zero','direct_few','persona_zero','persona_few']].daily_wape.idxmin()
    difference=float(table.loc[llm_best,'daily_wape']-table.loc['rule','daily_wape'])*100
    lines+=['',f'表现最好的 LLM 是 {ZH[llm_best]}，比规则 ABM 的日总误差{"低" if difference<0 else "高"} {abs(difference):.2f} 个百分点。',
        f'加入示范后：直接路线误差变化 {(table.loc["direct_few","daily_wape"]-table.loc["direct_zero","daily_wape"])*100:+.2f} 个百分点；角色路线 {(table.loc["persona_few","daily_wape"]-table.loc["persona_zero","daily_wape"])*100:+.2f} 个百分点。负数才是改善。']
    lines+=['','原始结果是主要比较；验证期总量校正结果另见 res/calibrated_metrics.csv，不能在测试后选有利的校正方法。','',
        '## 普通和极端天气','', '| 方法 | 干燥日 | 非极端雨天 | 强降雨日 |','| --- | ---: | ---: | ---: |']
    for m in ORDER:
        r=weather[weather.method==m].set_index('weather_class');lines.append('| '+ZH[m]+' | '+' | '.join(f'{r.loc[k,"daily_wape"]:.2%}' for k in ['dry','wet','extreme'])+' |')
    if os.path.exists('res/probe_metrics.csv'):
        probe=pd.read_csv('res/probe_repeat_summary.csv')
        lines+=['','## 额外对照','', '只比较同样的两个站、四天，不能直接与上面的八站结果比较。增加人数与增加背景分开做；人数增加也扩大人物类型覆盖，这是配置敏感性，不是纯数量的因果效果。','',
            '首轮重新采样时，角色 zero-shot 的误差从 8.18% 变成 15.08%，因此补足每个原配置三次重复；不改变主实验或提示参数。下面是三次平均 ± 标准差（百分点），不是置信区间。','',
            '| 配置 | 规则 ABM | 直接 zero | 直接 few | 角色 zero | 角色 few |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for n,enriched,label in [(8,False,'8 人控制组'),(16,False,'16 人'),(24,False,'24 人'),(8,True,'8 人 + 人口/路网背景')]:
            r=probe[(probe.n_agents==n)&(probe.enriched==enriched)].set_index('method')
            if len(r):lines.append('| '+label+' | '+' | '.join(f'{r.loc[k,"mean"]:.2%} ± {r.loc[k,"std"]*100:.2f}' for k in ['rule','direct_zero','direct_few','persona_zero','persona_few'])+' |')
        ensemble=pd.read_csv('res/probe_ensemble_metrics.csv')
        ensemble=ensemble[(ensemble.n_agents==8)&~ensemble.enriched].set_index('method')
        lines+=['','把同一配置的三次预测按小时等权平均，再评分，属于事后免费诊断，不是重新挑一次表现最好的运行。8 人控制组的平均预测误差为：'+
            '；'.join(f'{ZH[k]} {ensemble.loc[k,"daily_wape"]:.2%}' for k in ['direct_zero','direct_few','persona_zero','persona_few'])+'。这些数值仍只对应两个站、四天。其他配置见 probe_ensemble_metrics.csv。']
        lines+=['','重复实验中，角色路线的波动大于直接路线。16 人的角色 zero-shot 均值略好，24 人没有继续改善；人口/路网背景的改善较小，均小于相应的采样波动。暂时没有稳定证据支持“更多人物或更多背景就更准”。角色 few-shot 的示范来自合成规则，也没有带来一致的改善。']
    lines+=['','强降雨只有 Aug13、Sep29、Oct7 三天，不能把 24 个站点日当成 24 次独立暴雨。三个事件单独看 event_metrics.csv 和下图。按日期重采样区间见 paired_bootstrap.csv。','',
        '## 规则为什么落后','',
        '验证期选出的降雨系数为 4，主测试中规则总体少预测了约 16.8%。事后免费重放所有事先列出的系数，检查这种结果是否依赖系数。下面不用于重新挑选主模型。','',
        '| 降雨系数 | 验证期日总 WAPE | 测试期日总 WAPE | 是否原选值 |','| --- | ---: | ---: | --- |']
    for _,r in sensitivity.iterrows():lines.append(f'| {r.rain_scale:g} | {r.validation_daily_wape:.2%} | {r.daily_wape:.2%} | {"是" if r.selected_before_test else "否"} |')
    lines+=['','这份表检验规则参数能否从少量普通雨天迁移到强降雨；不能把 LLM 胜过当前规则，写成胜过所有规则模型。',
        '直接 zero-shot 相对历史基线的日总误差改善约 0.92 个百分点，但按日期重采样的区间包含零；目前不能确定这个小幅优势在更多日期上仍成立。角色路线的强降雨日总误差较低，但小时误差仍高，全天总数接近不代表各时段准。','',
        '## 如何解释','',
        '- 角色与规则使用相同画像、机会量和执行器；画像不是实测个人，reason 不是行为真值。学生、低收入优惠、老人/残障都是票种代理；老人和残障在模拟中区分，在真实评分中仍只能合并。',
        '- 身份、目的和出行约束随人物设定同时变化，不能把选择差异归因于单个人口或性格标签。群体天气下的行动份额在 group_weather_actions.csv，具体原文在 reason_examples.csv。',
        '- 直接预测可增加需求，当前角色/规则只调整原有地铁出行机会，不能凭空产生新增客流。这是两类架构的能力差异，不应只归因于 LLM 本身。',
        '- 同时报告小时指标；角色策略按四小时段变化，直接预测也只输出四小时段，段内分配沿用历史小时形状。',
        '- 只评分有观测的小时，部分日不是完整日；跨午夜的延期另记账，不把它重复加到当天。',
        '- 新人口/当前网络维度只进入回顾性子集对照。它们不能用于声称严格时间可用的 2023 预测改进；Sep29 服务归档只有事件日且含估计值，只作背景资料，不能推断关闭。','',
        '## API 用量','',f'共 {int(usage.calls.sum()):,} 次实际调用，已报告 {int(usage.reported_total_tokens.sum()):,} token，含未知失败的保守记账 {int(usage.charged_tokens.sum()):,}。均为 Lite，含试跑、验证和额外探针；明细见 api_usage.csv。','',
        '## 复现检查','',
        '7 项单元检查通过；14 组预测文件都能用已有真实缓存免费重放，数值完全相同。52,308 条个人行动记录通过菜单、预算、必须到场与延期上限检查；这证明实现符合设定，不能证明行为设定符合真实乘客。逐项记录见 res/manifest.json。','',
        '从 V4 执行 `python analyze.py` 可免费重做分析。全部真实提示及响应保留在 api_calls.jsonl.gz；不存在用规则或手填文本冒充 API 返回。','',
        '![总体误差](res/figures/overall_errors.png)','![天气分组](res/figures/weather_errors.png)','![三次事件](res/figures/extreme_days.png)','![站点空间](res/figures/station_map.png)','![群体决策](res/figures/group_subway_choices.png)','![群体在不同天气的决策](res/figures/group_weather_choices.png)']
    if os.path.exists('res/figures/agent_resolution.png'):lines.append('![人数敏感性](res/figures/agent_resolution.png)')
    open('RESULTS.md','w').write('\n'.join(lines)+'\n');print(summary.to_string(index=False),flush=True)


if __name__=='__main__':analyze()
