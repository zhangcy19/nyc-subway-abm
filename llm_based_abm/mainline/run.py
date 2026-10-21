"""Run from mainline/: preflight, shared baselines, or one isolated LLM experiment."""
import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import api
import role_history
from design import inputs, normal_history, event_data, batch_view, direct_view_for, EVENTS, SEASON
from experiment import PEOPLE, weekday_mean, feature
from model import fit_population, predict
from timing import timed_rule_policy

PHASES=['Normal_Summer','Normal_School']+list(EVENTS)
LLM_METHODS=['role_no_history','role_with_history','direct_few']
BASELINES=['weekday_mean','ridge','rule_daily','rule_timing']
api.BATCH_SYSTEM += '\nReference events may be later than the targets; the whole target event is withheld. This is retrospective event generalization.'
# Direct prompts in the accepted run used the full-network protocol, not role history.
api.DIRECT_SYSTEM=api.DIRECT_SYSTEM.split('\nEvent references may occur later')[0]+'\nReference events may be later than the target; the whole target event is withheld. This is retrospective event generalization.'


def fingerprint(config):
    names=['run.py','api.py','design.py','role_history.py','model.py','experiment.py','timing.py',
           'inputs/daily.csv','inputs/dates.csv','inputs/station_ids.json','inputs/role_donors.csv',
           'inputs/frozen_parameters.csv','../../data/weather_hourly.csv']
    files={f:hashlib.sha256(open(f,'rb').read()).hexdigest() for f in names}
    science={k:v for k,v in config.items() if k not in ['workers','token_cap','run_id']}
    signature=hashlib.sha256(json.dumps(dict(config=science,files=files),sort_keys=True).encode()).hexdigest()
    return dict(signature=signature,config=config,files=files)


def configure(filename):
    c=json.load(open(filename))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',c['run_id']):raise ValueError('run_id must be a simple folder name')
    if c['run_id'] in ['shared','doubao_lite_reference']:raise ValueError('Reserved result folder')
    if c['api_mode'] not in ['chat','responses']:raise ValueError('api_mode must be chat or responses')
    if c.get('methods') and not set(c['methods']).issubset(LLM_METHODS):raise ValueError('Unknown LLM method')
    if c.get('walking_policy','strict') not in ['strict','zero_walk_renormalize']:raise ValueError('Unknown walking policy')
    if c.get('taxi_policy','strict') not in ['strict','zero_taxi_renormalize']:raise ValueError('Unknown taxi policy')
    out='results/'+c['run_id'];os.makedirs(out+'/private',exist_ok=True)
    manifest=fingerprint(c);path=out+'/manifest.json'
    if os.path.exists(path) and json.load(open(path))['signature']!=manifest['signature']:
        raise ValueError('Model, protocol, data or station scope changed: use a new run_id')
    json.dump(manifest,open(path,'w'),indent=2)
    api.MODEL=c['model'];api.BASE_URL=c['base_url'];api.API_MODE=c['api_mode']
    api.API_KEY=os.environ.get(c['key_env'],'local' if c['base_url'].startswith(('http://localhost:','http://127.0.0.1:')) else '')
    api.EXTRA_BODY=c.get('extra_body',{});api.CAP=c['token_cap'];api.LOGDIR=out+'/private'
    api.REQUEST_WINDOW=c.get('request_window')
    api.WALKING_POLICY=c.get('walking_policy','strict')
    api.TAXI_POLICY=c.get('taxi_policy','strict')
    api.TEMPERATURE=c['temperature'];api.MAX_OUTPUT=c['max_output_tokens'];api.BATCH_OUTPUT=c['batch_max_output_tokens']
    return c,out


def progress_map(call,jobs,workers,label):
    # Keep result order while advancing whenever any request finishes.
    pool=ThreadPoolExecutor(max_workers=workers)
    futures={pool.submit(call,job):i for i,job in enumerate(jobs)}
    results=[None]*len(jobs)
    try:
        with tqdm(total=len(jobs),desc=label,unit='request',dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                results[futures[future]]=future.result()
                bar.update(1)
    finally:
        # Ctrl+C cancels queued work and lets only in-flight replies save.
        pool.shutdown(wait=True,cancel_futures=True)
    return results


def get_policy(donors,w,phase,method,workers,offline=False):
    part=donors[donors.task==phase].drop_duplicates('date')
    def call(job):
        row,person=job
        view=role_history.role_view_for(donors,w,row,person,phase,method=='role_with_history')
        return (row.date,person['agent_id']),api.request('role',view,person,offline=offline)
    jobs=[(r,person) for _,r in part.iterrows() for person in PEOPLE]
    return dict(progress_map(call,jobs,workers,phase+' '+method))


def direct(p,w,phase,workers,offline=False):
    target=p[(p.task==phase)&p.split.eq('test')]
    jobs=[g for _,g in target.groupby('station_complex_id')] if phase in EVENTS else [pd.DataFrame([r]) for _,r in target.iterrows()]
    def call(g):
        if phase in EVENTS:
            answer=api.request('batch',batch_view(p,w,g,phase),offline=offline)
            values={v['date']:v['entries'] for v in answer['predictions']}
        else:
            r=g.iloc[0];answer=api.request('direct',direct_view_for(p,w,r,phase),offline=offline)
            values={r.date:answer['entries']}
        return [dict(station_complex_id=r.station_complex_id,date=r.date,actual=r.total_ridership,predicted=values[r.date]) for _,r in g.iterrows()]
    return [r for block in progress_map(call,jobs,workers,phase+' direct_few') for r in block]


def summarize(out):
    x=pd.read_csv(out+'/predictions.csv',dtype={'station_complex_id':str})
    keys=['task','method','station_complex_id','date']
    if x.duplicated(keys).any() or not np.isfinite(x.predicted).all() or x.predicted.lt(0).any():raise ValueError('Invalid prediction file')
    dates=pd.read_csv('inputs/dates.csv');x=x.merge(dates,on=['task','date'],validate='many_to_one')
    x['regime']=np.where(x.task.str.startswith('Normal'),'Normal','Extreme')
    x['absolute_error']=abs(x.predicted-x.actual)
    x.to_csv(out+'/prediction_details.csv',index=False)
    def scores(data,group):
        g=data.groupby(group)[['absolute_error','actual']].sum()
        g['score']=100*(1-g.absolute_error/g.actual)
        return g.reset_index()
    primary=x[(x.regime=='Normal')|x.event_day]
    t=scores(primary,['regime','task','method']);t.to_csv(out+'/task_scores.csv',index=False)
    t.groupby(['method','regime']).score.mean().unstack().to_csv(out+'/summary.csv')
    scores(x,['regime','task','method']).groupby(['method','regime']).score.mean().unstack().to_csv(out+'/window_summary.csv')
    scores(primary,['station_complex_id','regime','task','method']).to_csv(out+'/station_scores.csv',index=False)


def execute(c,out,baseline=False,offline=False):
    if not baseline:
        provenance=json.load(open('results/shared/provenance.json'))
        for name,digest in provenance['files'].items():
            if hashlib.sha256(open(name,'rb').read()).hexdigest()!=digest:raise ValueError('Shared baseline inputs or implementation changed: '+name)
    ids=c.get('station_ids')
    p,w=inputs(ids);donors=pd.read_csv('inputs/role_donors.csv',dtype={'station_complex_id':str})
    params=pd.read_csv('inputs/frozen_parameters.csv').set_index(['task','method']).to_dict('index')
    methods=BASELINES if baseline else c.get('methods',LLM_METHODS)
    names=['predictions','populations','runtime','decisions']
    records={n:pd.read_csv(out+'/'+n+'.csv',dtype={'station_complex_id':str}).to_dict('records') if os.path.exists(out+'/'+n+'.csv') else [] for n in names}
    for phase in PHASES:
        season=phase if phase.startswith('Normal') else SEASON[phase]
        train=normal_history(p,season);test=p[(p.task==phase)&p.split.eq('test')]
        for method in methods:
            present=pd.DataFrame([r for r in records['predictions'] if r['task']==phase and r['method']==method])
            expected=set(zip(test.station_complex_id,test.date))
            if len(present):
                if set(zip(present.station_complex_id,present.date))!=expected:raise ValueError('Incomplete result block: new run_id required')
                continue
            start=time.perf_counter();before=sum(api.charge(r) for r in api.ledger()) if not baseline else 0
            stage=PHASES.index(phase)*len(methods)+methods.index(method)+1
            print('Starting',str(stage)+'/'+str(len(PHASES)*len(methods)),phase,method,flush=True)
            if method=='weekday_mean':rows=weekday_mean(train,test)
            elif method=='ridge':
                learn=train if phase.startswith('Normal') else pd.concat([train]+[event_data(p,t) for t in EVENTS if t!=phase]).drop_duplicates(['station_complex_id','date'])
                rows=[]
                for sid,g in learn.groupby('station_complex_id'):
                    model=make_pipeline(StandardScaler(),Ridge(alpha=params[(season,method)]['alpha']))
                    model.fit(np.array([feature(r) for _,r in g.iterrows()]),np.log1p(g.total_ridership))
                    for _,r in test[test.station_complex_id==sid].iterrows():
                        rows.append(dict(station_complex_id=sid,date=r.date,actual=r.total_ridership,predicted=max(0,float(np.expm1(model.predict([feature(r)])[0])))))
            elif method=='direct_few':rows=direct(p,w,phase,c['workers'],offline)
            else:
                if method.startswith('role'):
                    policy=get_policy(donors,w,phase,method,c['workers'],offline)
                else:policy=timed_rule_policy(p[p.task==phase].drop_duplicates('date'),w,params[(phase,method)]['rain_scale'],method=='rule_timing')
                if phase.startswith('Normal'):
                    cfg=params[(phase,method)]
                    fitted=fit_population(train,policy,PEOPLE,cfg['shrink'],cfg['calendar_prior'])
                    records['populations'].extend(dict(task=phase,method=method,**r) for r in fitted)
                else:fitted=[r for r in records['populations'] if r['task']==season and r['method']==method]
                rows,decisions=predict(test,fitted,policy,PEOPLE)
                records['decisions'].extend(dict(task=phase,method=method,population_source=season,**{k:v for k,v in r.items() if k not in ['reason','hash']}) for r in decisions)
            records['predictions'].extend(dict(task=phase,method=method,**r) for r in rows)
            records['runtime'].append(dict(task=phase,method=method,seconds=time.perf_counter()-start,new_tokens=0 if baseline else sum(api.charge(r) for r in api.ledger())-before))
            for n in names:
                if records[n]:pd.DataFrame(records[n]).to_csv(out+'/'+n+'.csv',index=False)
            summarize(out)
    if not baseline:
        shared=pd.read_csv('results/shared/predictions.csv',dtype={'station_complex_id':str})
        shared=shared[shared.station_complex_id.isin(p.station_complex_id.unique())]
        compare=out+'/comparison';os.makedirs(compare,exist_ok=True)
        pd.concat([shared,pd.DataFrame(records['predictions'])],ignore_index=True).to_csv(compare+'/predictions.csv',index=False)
        summarize(compare)


def export_usage(out):
    rows=api.ledger()
    result=dict(calls=len(rows),valid_calls=sum(r.get('status')=='valid' for r in rows),
                error_calls=sum(r.get('status')!='valid' for r in rows),
                charged_tokens=sum(api.charge(r) for r in rows),
                input_tokens=sum(r.get('usage',{}).get('input_tokens',r.get('usage',{}).get('prompt_tokens',0)) for r in rows),
                output_tokens=sum(r.get('usage',{}).get('output_tokens',r.get('usage',{}).get('completion_tokens',0)) for r in rows),
                api_seconds=sum(r.get('elapsed_seconds',0) for r in rows),
                estimated_charge_calls=sum('charged_tokens' in r for r in rows),
                reused_calls=sum(bool(r.get('reused_from_run_id')) for r in rows),
                walking_adjusted_calls=sum(r.get('status')=='valid' and r.get('parsed',{}).get('probability_adjustment',{}).get('removed_walk_probability',0)>0 for r in rows),
                taxi_adjusted_calls=sum(r.get('status')=='valid' and r.get('parsed',{}).get('probability_adjustment',{}).get('removed_taxi_probability',0)>0 for r in rows),
                removed_taxi_probability_sum=sum(r.get('parsed',{}).get('probability_adjustment',{}).get('removed_taxi_probability',0) for r in rows if r.get('status')=='valid'),
                removed_walk_probability_sum=sum(r.get('parsed',{}).get('probability_adjustment',{}).get('removed_walk_probability',0) for r in rows if r.get('status')=='valid'))
    json.dump(result,open(out+'/usage.json','w'),indent=2)


def preflight(c,out):
    p,w=inputs(c.get('station_ids'));donors=pd.read_csv('inputs/role_donors.csv',dtype={'station_complex_id':str})
    counts=[];unique=set();size=0
    for phase in PHASES:
        for method in c.get('methods',LLM_METHODS):
            if method.startswith('role'):
                requests=[('role',role_history.role_view_for(donors,w,r,person,phase,method=='role_with_history')) for _,r in donors[donors.task==phase].drop_duplicates('date').iterrows() for person in PEOPLE]
            elif phase in EVENTS:requests=[('batch',batch_view(p,w,g,phase)) for _,g in event_data(p,phase).groupby('station_complex_id')]
            else:requests=[('direct',direct_view_for(p,w,r,phase)) for _,r in p[(p.task==phase)&p.split.eq('test')].iterrows()]
            counts.append(dict(task=phase,method=method,requests=len(requests)))
            for route,view in requests:
                text={'role':api.ROLE_SYSTEM,'direct':api.DIRECT_SYSTEM,'batch':api.BATCH_SYSTEM}[route]+json.dumps(view,sort_keys=True)
                if text not in unique:unique.add(text);size+=len(text.encode())
    report=dict(stations=p.station_complex_id.nunique(),unique_requests=len(unique),estimated_tokens_low=int(size*.35+len(unique)*60),estimated_tokens_high=int(size*.6+len(unique)*200),token_cap=c['token_cap'],scope=counts)
    json.dump(report,open(out+'/preflight.json','w'),indent=2);print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','llm','baselines','report'])
    parser.add_argument('--config',default='configs/qwen_local.json');parser.add_argument('--offline',action='store_true')
    parser.add_argument('--output',default='results/shared')
    args=parser.parse_args()
    if args.command=='report':summarize(args.output)
    elif args.command=='baselines':
        os.makedirs(args.output,exist_ok=True)
        execute({},args.output,baseline=True)
    else:
        config,output=configure(args.config)
        if args.command=='preflight':preflight(config,output)
        else:
            if not api.API_KEY:raise ValueError('Set '+config['key_env']+' in the environment')
            try:execute(config,output,offline=args.offline)
            except KeyboardInterrupt:
                print('\nStopped. Saved replies will be reused when you run the same command again.',flush=True)
                raise SystemExit(130)
            finally:export_usage(output)
