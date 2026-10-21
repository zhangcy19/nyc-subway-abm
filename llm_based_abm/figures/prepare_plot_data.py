"""Prepare plotting-only numeric tables from saved results. No model/API imports."""
import os,json,hashlib,collections
import pandas as pd
import numpy as np

methods=['weekday_mean','ridge','rule_daily','direct_few','role_no_history','role_with_history']
tasks=['Normal_Summer','Normal_School','Extreme_0729','Extreme_0813','Extreme_0929','Extreme_1007']
source=[]
def read(file,**kwargs):
    source.append(file)
    return pd.read_csv(file,**kwargs)
def write(frame,name):frame.to_csv('figures/data/'+name,index=False)
p=read('mainline/results/doubao_lite_reference/predictions.csv',dtype={'station_complex_id':str})
shared=read('mainline/results/shared/predictions.csv',dtype={'station_complex_id':str})
p=pd.concat([p[p.method.isin(methods)],shared[shared.method.isin(['weekday_mean','ridge','rule_daily'])]],ignore_index=True)
p=p[p.method.isin(methods)].copy()
dates=read('mainline/inputs/dates.csv')
p=p.merge(dates[['task','date','event_day']],on=['task','date'],validate='many_to_one')
p['regime']=np.where(p.task.str.startswith('Normal'),'Normal','Extreme')
p['absolute_error']=(p.predicted-p.actual).abs()
primary=p[p.regime.eq('Normal')|p.event_day]
scores=primary.groupby(['task','method','regime'])[['absolute_error','actual']].sum().reset_index()
scores['score']=100*(1-scores.absolute_error/scores.actual)
summary=scores.groupby(['method','regime']).score.mean().unstack().reindex(methods).reset_index()
write(summary,'mainline_scores.csv');write(scores,'mainline_task_scores.csv')
write(p.drop(columns=['request_hash']),'mainline_predictions.csv')
# Retrieve exactly the replies used by the accepted Lite results.
lookup={}
for f in ['branches/event_holdout/res/api_calls.jsonl','branches/full_network/staged_res/api_calls.jsonl']:
    source.append(f)
    for line in open(f):
        r=json.loads(line)
        if r.get('status')=='valid':lookup[r['hash']]=r
usage=[];reasons=[]
for task in tasks:
    for method in ['role_no_history','role_with_history']:
        f='branches/event_holdout/res/'+task+'_'+method+'_policy.json';source.append(f)
        for entry in json.load(open(f)):
            r=lookup[entry['hash']];u=r['usage'];d=u.get('input_tokens_details') or {}
            usage.append(dict(model='Doubao Lite',task=task,method=method,request_hash=r['hash'],input_tokens=u.get('input_tokens',0),cached_tokens=d.get('cached_tokens',0) or 0,output_tokens=u.get('output_tokens',0),api_seconds=r['elapsed_seconds']))
            reasons.append(dict(task=task,method=method,date=entry['date'],agent_id=entry['agent_id'],reason=r['parsed']['reason']))
for h in p[p.method=='direct_few'].request_hash.drop_duplicates():
    r=lookup[h];u=r['usage'];d=u.get('input_tokens_details') or {}
    # The saved direct hash maps back to one station-day or one held-event batch.
    task=p[p.request_hash==h].task.iloc[0]
    usage.append(dict(model='Doubao Lite',task=task,method='direct_few',request_hash=h,input_tokens=u.get('input_tokens',0),cached_tokens=d.get('cached_tokens',0) or 0,output_tokens=u.get('output_tokens',0),api_seconds=r['elapsed_seconds']))
u=pd.DataFrame(usage)
if len(u)!=4512 or u.request_hash.duplicated().any():raise ValueError('Expected 4512 distinct accepted Lite replies')
write(u,'lite_request_usage.csv')
json.dump(reasons,open('figures/data/lite_role_reasons.json','w'))
# Cross-backend comparison: the same role-with-history method, all six tasks for cost.
pro_rows=[]
f='mainline/results/doubao_pro_mainline_run01/private/api_calls.jsonl';source.append(f)
for line in open(f):
    r=json.loads(line);d=r['request'];v=d['view']
    if r['status']=='valid' and d['route']=='role' and ('group_history' in v or 'reference_events' in v):
        u0=r['usage'];cached=(u0.get('input_tokens_details') or {}).get('cached_tokens',0) or 0
        pro_rows.append(dict(model='Doubao Pro',input_tokens=u0['input_tokens'],cached_tokens=cached,output_tokens=u0['output_tokens'],api_seconds=r['elapsed_seconds']))
pro_score=read('mainline/results/doubao_pro_mainline_run01/summary.csv').set_index('method').loc['role_with_history','Extreme']
compare=[]
for model,frame,value in [('Doubao Lite',u[u.method=='role_with_history'],float(summary.set_index('method').loc['role_with_history','Extreme'])),('Doubao Pro',pd.DataFrame(pro_rows),float(pro_score))]:
    compare.append(dict(model=model,score=value,input_tokens=int(frame.input_tokens.sum()),cached_tokens=int(frame.cached_tokens.sum()),output_tokens=int(frame.output_tokens.sum()),api_seconds=float(frame.api_seconds.sum()),placeholder=False))
compare.append(dict(model='DeepSeek',score=0,input_tokens=0,cached_tokens=0,output_tokens=0,api_seconds=0,placeholder=True))
qwen_root='mainline/results/qwen38_flash_mainline_run01/'
qwen_score=read(qwen_root+'summary.csv').set_index('method').loc['role_with_history','Extreme']
qwen_valid={}
f=qwen_root+'private/api_calls.jsonl';source.append(f)
for line in open(f):
    r=json.loads(line)
    if r['status']=='valid':qwen_valid[r['hash']]=r
qwen_rows=[]
for r in qwen_valid.values():
    d=r['request'];v=d['view']
    if d['route']=='role' and ('group_history' in v or 'reference_events' in v):
        u0=r['usage'];details=u0.get('prompt_tokens_details') or u0.get('input_tokens_details') or {}
        qwen_rows.append(dict(input_tokens=u0.get('prompt_tokens',u0.get('input_tokens',0)),cached_tokens=details.get('cached_tokens',0) or 0,output_tokens=u0.get('completion_tokens',u0.get('output_tokens',0)),api_seconds=r['elapsed_seconds']))
qwen_usage=pd.DataFrame(qwen_rows)
compare.append(dict(model='Qwen 3.8 Flash',score=float(qwen_score),input_tokens=int(qwen_usage.input_tokens.sum()),cached_tokens=int(qwen_usage.cached_tokens.sum()),output_tokens=int(qwen_usage.output_tokens.sum()),api_seconds=float(qwen_usage.api_seconds.sum()),placeholder=False))
write(pd.DataFrame(compare),'role_backend_comparison.csv')
# The saved decisions combine the 12 development and 140 expansion stations.
dec=pd.concat([read('branches/full_network/rollback_dev_res/decisions.csv',dtype={'station_complex_id':str}),read('branches/full_network/rollback_full_res/expansion_decisions.csv',dtype={'station_complex_id':str})],ignore_index=True)
dec=dec[dec.method.isin(['role_no_history','role_with_history'])].copy()
dec['predicted']=dec.effective_population*dec.p_subway
agg=dec.groupby(['task','method','station_complex_id','date']).predicted.sum().reset_index()
check=agg.merge(p[p.method.str.startswith('role')],on=['task','method','station_complex_id','date'],suffixes=('_decisions','_accepted'),validate='one_to_one')
max_diff=float((check.predicted_decisions-check.predicted_accepted).abs().max())
if len(check)!=10336 or max_diff>1e-6:raise ValueError('Role decisions do not reproduce accepted mainline predictions')
daily=read('mainline/inputs/daily.csv',dtype={'station_complex_id':str})
groups=['students','fair_fare','seniors_disability','general']
truth=daily.melt(id_vars=['station_complex_id','date'],value_vars=groups,var_name='group',value_name='actual')
gpred=dec.groupby(['task','method','station_complex_id','date','group']).predicted.sum().reset_index().merge(truth,on=['station_complex_id','date','group'],validate='many_to_one').merge(dates[['task','date','event_day']],on=['task','date'],validate='many_to_one')
gpred=gpred[gpred.task.str.startswith('Extreme')&gpred.event_day].copy();gpred['absolute_error']=(gpred.predicted-gpred.actual).abs()
gscore=gpred.groupby(['task','method','group'])[['absolute_error','actual']].sum().reset_index();gscore['score']=100*(1-gscore.absolute_error/gscore.actual)
write(gscore.groupby(['method','group']).score.mean().reset_index(),'group_extreme_scores.csv')
center=dec.merge(dates[['task','date','event_day']],on=['task','date'],validate='many_to_one')
center=center[center.task.str.startswith('Extreme')&center.event_day&center.method.eq('role_with_history')].copy()
actions=['subway','bus','taxi','walk','stay_home']
for action in actions:center['weighted_'+action]=center.effective_population*center['p_'+action]
weighted=center.groupby('group')[['effective_population']+['weighted_'+a for a in actions]].sum()
for action in actions:weighted['p_'+action]=weighted['weighted_'+action]/weighted.effective_population
write(weighted[['p_'+a for a in actions]].reset_index(),'group_extreme_choices.csv')
# Side studies retain both repeats. No new inference is needed.
side=read('branches/full_network/side_studies_res/predictions.csv',dtype={'station_complex_id':str})
write(side.drop(columns=['reason','request_hash']),'side_predictions.csv')
write(read('branches/full_network/side_studies_res/event_calendar.csv'),'side_event_calendar.csv')

calendar_source='branches/full_network/side_studies_res/mlb_schedule_source.json';source.append(calendar_source)
schedule=json.load(open(calendar_source));game_counts={}
for day in schedule['dates']:
    game_counts[day['date']]=sum(g.get('officialDate')==day['date'] and g['status']['codedGameState']=='F' and g['teams']['home']['team']['id']==121 and g['venue']['name']=='Citi Field' for g in day['games'])
calendar=pd.DataFrame({'date':pd.date_range('2023-09-12','2023-10-08').strftime('%Y-%m-%d')})
calendar['home_games']=calendar.date.map(game_counts).fillna(0).astype(int)
calendar['mets_home_game_on_date']=calendar.home_games.gt(0)
write(calendar,'side_event_calendar_full.csv')

write(read('branches/full_network/side_studies_res/panel.csv',dtype={'station_complex_id':str}),'side_panel.csv')
# Rain evidence for the August 13 overnight event.
weather=read('../data/weather_hourly.csv');write(weather,'weather_hourly.csv')
station=read('../data/stations.csv');write(station,'stations.csv')
# A source manifest keeps plot-only derivatives separate from model outputs.
manifest=dict(created='2026-10-21',colors={'green':'#285304','orange':'#D87540'},palette_source='../../../presentation_local/build/extended.mjs:6',score='100*(1-WAPE), Normal equally averages two tasks, Extreme equally averages four center days',accepted_lite_requests=len(u),decision_prediction_max_abs_difference=max_diff,price_date='2026-10-21',price_source='https://docs.volcengine.com/docs/ark/model-pricing?lang=en&redirect=1',runtime_note='C2 shows SUM of accepted API request seconds, not parallel wall-clock or fitting time. Numeric baselines have zero API time, not zero computation time. Historical reference combines cached source runs.',C5_note='Rain ablation has center-day predictions only, not 24 hourly ridership forecasts. Activity targets Oct 6–8 have no Mets home games; plot is a calendar-information ablation, not a game-day test.',C6_note='DeepSeek and Qwen values are user-requested placeholders, not observed zero performance or verified zero spending.',sources={f:hashlib.sha256(open(f,'rb').read()).hexdigest() for f in dict.fromkeys(source)})
json.dump(manifest,open('figures/data/provenance.json','w'),indent=2)
print('Prepared:',len(p),'predictions;',len(u),'exact Lite requests;',len(reasons),'reasons; maximum decision reproduction difference',max_diff)
print('Stations columns:',station.columns.tolist());print('Weather columns:',weather.columns.tolist())
print(pd.DataFrame(compare).to_string(index=False))
