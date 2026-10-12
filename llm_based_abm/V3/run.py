"""Run from V3. Each HTTP request contains just one synthetic traveler."""
import argparse
import copy
import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from model import GROUPS, population, TravelerModel, make_plan, rule_choice, validate_choice

MODELS={'pro':'doubao-seed-2-1-pro-260915','lite':'doubao-seed-2-1-lite-260915'}
CAPS={'pro':530000,'lite':460000}
WINDOWS={'feedback':pd.date_range('2023-09-08','2023-09-10'),
    'test_ordinary':pd.date_range('2023-09-22','2023-09-24'),
    'test_storm':pd.date_range('2023-09-28','2023-09-30')}
SYSTEM='''You ARE the single hypothetical traveler described below, not a traffic forecaster.
Choose ONE action for YOUR one planned trip, using your needs, available options and own experience.
Do not estimate crowd counts, demographic shares, activity multipliers or probabilities.
No verified station closure or future weather forecast is supplied; do not invent either.
Profiles, transport costs and memories are synthetic assumptions, not real passenger observations.
Respect available_actions, attendance, budget and mobility constraints. Delay must fit max_delay_min.
An older adult can travel independently; a discount fare does not imply unemployment.
Return JSON with action, delay_min, reason. Use a short first-person reason of at most20 words.
All non-delay actions use delay_min0; delay_subway uses60 or120 within the stated limit.'''
EXAMPLES=[{'person':'Must attend; no remote; delay limit0; taxi unavailable. Light rain.',
    'decision':{'action':'subway','delay_min':0,'reason':'I must arrive on time and subway is affordable.'}},
    {'person':'Optional errand; cancellation allowed; high rain concern; no need to attend today. Heavy rain.',
    'decision':{'action':'cancel','delay_min':0,'reason':'I can do the errand another day without taking the rain risk.'}}]
LOCK=threading.Lock();RATE_LOCK=threading.Lock();LAST_START=0.
RESERVED=dict.fromkeys(MODELS,0);CACHE=None


def ledger():
    records=[]
    for name in ['res/api_calls.jsonl.gz','res/api_calls.jsonl']:
        if os.path.exists(name):
            opener=gzip.open if name.endswith('.gz') else open
            with opener(name,'rt') as f:records.extend(json.loads(x) for x in f)
    return records


def api_choice(view, person, provider, offline=False, sample=0, few=False):
    global CACHE,LAST_START
    if CACHE is None:
        with LOCK:
            if CACHE is None:CACHE=ledger()
    content={'model':MODELS[provider],'system':SYSTEM,'view':view,'temperature':.7,
        'max_output_tokens':320,'sample':sample,'few_examples':EXAMPLES if few else []}
    prompt=json.dumps({'you':view,**({'examples_of_other_hypothetical_people':EXAMPLES} if few else {})},separators=(',',':'),allow_nan=False)
    for attempt in range(3):
        request={**content,'prompt':prompt,'repair_attempt':attempt}
        digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        with LOCK:
            cached=next((r for r in CACHE if r['hash']==digest and r['status'] in ['completed','incomplete']),None)
            failed=any(r['hash']==digest and r['status']=='request_error' for r in CACHE)
        if cached:
            payload=cached['parsed']
        else:
            if offline:
                if failed:continue
                raise RuntimeError('Missing actual cached individual decision '+digest)
            reserve=len(SYSTEM+prompt)//3+320+300
            with LOCK:
                used=sum(r.get('budget_charge_tokens',r.get('usage',{}).get('total_tokens',0)) for r in CACHE if r['provider']==provider)
                if used+RESERVED[provider]+reserve>CAPS[provider]:raise RuntimeError(provider+' budget guard at '+str(used))
                RESERVED[provider]+=reserve
            with open('../key.txt') as f:key=f.read().strip().strip('\"\'')
            client=OpenAI(base_url='https://ark.cn-beijing.volces.com/api/v3',api_key=key,timeout=120,max_retries=0)
            with RATE_LOCK:
                time.sleep(max(0,2-(time.monotonic()-LAST_START)))
                LAST_START=time.monotonic()
            started=time.time()
            try:
                response=client.responses.create(model=MODELS[provider],input=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
                    temperature=.7,max_output_tokens=320,extra_body={'thinking':{'type':'disabled'}})
                raw=response.output_text.strip()
                if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
                try:payload=json.loads(raw)
                except json.JSONDecodeError:payload=None
                if response.status!='completed':payload=None
                record={'hash':digest,'provider':provider,'request':request,'status':response.status,
                    'parsed':payload,'raw':response.output_text,'usage':response.usage.model_dump(),
                    'elapsed_seconds':time.time()-started,'response_id':response.id}
            except Exception as error:
                record={'hash':digest,'provider':provider,'request':request,'status':'request_error','parsed':None,
                    'usage':{},'budget_charge_tokens':reserve,'error_type':type(error).__name__}
                with LOCK:
                    CACHE.append(record)
                    with open('res/api_calls.jsonl','a') as f:f.write(json.dumps(record)+'\n')
                    RESERVED[provider]-=reserve
                if isinstance(error,(RateLimitError,APIConnectionError,APITimeoutError)) and attempt<2:
                    time.sleep(10*(attempt+1));continue
                raise
            with LOCK:
                CACHE.append(record)
                with open('res/api_calls.jsonl','a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                RESERVED[provider]-=reserve
                total=sum(r.get('usage',{}).get('total_tokens',0) for r in CACHE if r['provider']==provider)
                if len(CACHE)%24==0:print('Actual API calls',len(CACHE),provider,'reported tokens',total,flush=True)
        try:
            if payload is None:raise ValueError('Response must be complete JSON')
            return validate_choice(payload,person),digest
        except (ValueError,AttributeError,TypeError) as error:
            if attempt==2:raise
            prompt=json.dumps({'you':view,**({'examples_of_other_hypothetical_people':EXAMPLES} if few else {}),
                'format_repair':str(error),'previous_real_response':payload,
                'instruction':'Regenerate your own choice respecting the supplied options and delay limits.'},separators=(',',':'))
    raise RuntimeError('No completed individual decision')


def load_panel():
    return pd.read_csv('res/panel.csv.gz',dtype={'station_complex_id':str},parse_dates=['timestamp','date'])


def simulate(panel, stations, method, dates, block, n=24, offline=False, rain_scale=1.):
    personas=population(stations,n)
    model=TravelerModel(personas)
    records=[]
    for date in dates:
        jobs=[];decisions={};hashes={}
        for agent in model.agents:
            plan=make_plan(panel,agent.persona,date)
            if plan is None:continue
            if method=='rule':
                decisions[agent.persona['agent_id']]=rule_choice(agent.persona,plan['weather'],agent.memory,rain_scale)
            elif method=='no_change':
                decisions[agent.persona['agent_id']]={'action':'subway','delay_min':0,'reason':'All planned trips enter by subway.'}
            else:
                jobs.append((agent.persona['agent_id'],agent.perceive(plan),agent.persona))
        if jobs:
            errors=[]
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending={pool.submit(api_choice,v,p,method,offline):agent_id for agent_id,v,p in jobs}
                for future in as_completed(pending):
                    try:choice,digest=future.result()
                    except Exception as error:errors.append(str(error));continue
                    decisions[pending[future]]=choice;hashes[pending[future]]=digest
            if errors:raise RuntimeError(str(len(errors))+' individual decisions need resume: '+errors[0])
        daily=model.advance(panel,date,decisions,method,block,n,0)
        for r in daily:r['request_hash']=hashes.get(r['agent_id'],'')
        records.extend(daily)
    return pd.DataFrame(records)


def aggregate(panel, decisions):
    output=[]
    for (method,block,n,sample),rows in decisions.groupby(['method','block','n_agents','sample']):
        dates=pd.to_datetime(rows.date.unique())
        actual=panel[panel.date.isin(dates)&panel.station_complex_id.isin(rows.station_complex_id.unique())].copy()
        entered=rows[rows.action.isin(['subway','delay_subway'])].copy()
        entered['timestamp']=pd.to_datetime(entered.actual_entry_timestamp)
        totals=entered.groupby(['station_complex_id','timestamp']).weight.sum().rename('predicted')
        actual=actual.merge(totals,on=['station_complex_id','timestamp'],how='left')
        actual['predicted']=actual.predicted.fillna(0)
        for g in GROUPS:
            group=entered[entered.group==g].groupby(['station_complex_id','timestamp']).weight.sum().rename('pred_'+g)
            actual=actual.merge(group,on=['station_complex_id','timestamp'],how='left')
            actual['pred_'+g]=actual['pred_'+g].fillna(0)
        actual['method'],actual['block'],actual['n_agents'],actual['sample']=method,block,n,sample
        output.append(actual)
    return pd.concat(output,ignore_index=True)


def fit_rule(panel,stations):
    scores=[]
    for scale in [.5,1.,1.5,2.]:
        choices=simulate(panel,stations,'rule',WINDOWS['feedback'],'feedback',rain_scale=scale)
        pred=aggregate(panel,choices)
        # Daily totals: hourly error is affected by finite representative population timing.
        day=pred.groupby(['station_complex_id','date'])[['predicted','total_ridership']].sum()
        wape=float(np.abs(day.predicted-day.total_ridership).sum()/day.total_ridership.sum())
        scores.append({'rain_scale':scale,'feedback_daily_wape':wape})
    best=min(scores,key=lambda x:x['feedback_daily_wape'])
    with open('res/rule_selection.json','w') as f:json.dump({'candidates':scores,'selected':best,
        'selected_on':'Sep8-10 daily totals only; test never used'},f,indent=2)
    return best['rain_scale']


def matched_checks(panel,main,offline=False):
    station=str(main.station_complex_id.iloc[0])
    people={p['agent_id']:p for p in population([station],24)}
    work=[]
    # Each probe uses a copy of the real main agent's PRE-choice state.
    # Counterfactual probes do not feed outcomes into subsequent main history.
    for method in MODELS:
        source=main[(main.method==method)&(main.station_complex_id==station)]
        for block,rows in source.groupby('block'):
            model=TravelerModel(list(people.values()))
            for date,day in rows.sort_values(['date','agent_id']).groupby('date'):
                date=pd.Timestamp(date)
                for agent in model.agents:
                    original=day[day.agent_id==agent.persona['agent_id']]
                    if original.empty:continue
                    r=original.iloc[0];plan=make_plan(panel,agent.persona,date)
                    if date.strftime('%Y-%m-%d') in ['2023-09-22','2023-09-29']:
                        for variant in ['no_memory','few_shot','repeat']:
                            probe=copy.copy(agent);probe.memory=copy.deepcopy(agent.memory)
                            work.append((probe,plan,method,variant,block,r.action,int(r.delay_min)))
                    agent.execute(plan,{'action':r.action,'delay_min':int(r.delay_min),'reason':r.reason},method,block,24,0)
    def one(item):
        agent,plan,method,variant,block,reference_action,reference_delay=item
        view=agent.perceive(plan,memory_on=variant!='no_memory')
        choice,digest=api_choice(view,agent.persona,method,offline,sample=1 if variant=='repeat' else 0,few=variant=='few_shot')
        record=agent.execute(plan,choice,method+'_'+variant,block,24,1 if variant=='repeat' else 0,memory_on=variant!='no_memory')
        record['request_hash']=digest;record['reference_action']=reference_action;record['reference_delay_min']=reference_delay
        return record
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,item) for item in work]
        for future in as_completed(futures):
            try:results.append(future.result())
            except Exception as error:errors.append(str(error));continue
            pd.DataFrame(results).to_csv('res/matched_checks.csv.gz',index=False)
    if errors:raise RuntimeError(str(len(errors))+' matched choices need resume: '+errors[0])
    return pd.DataFrame(results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','feedback','test','checks','evaluate']);parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    os.makedirs('res',exist_ok=True)
    panel=load_panel();stations=pd.read_csv('res/stations.csv',dtype={'station_complex_id':str}).station_complex_id.tolist()
    if args.stage=='pilot':
        people=population(stations,24)
        model=TravelerModel(people)
        records=[]
        for provider in MODELS:
            for group in GROUPS:
                for kind,date in [('fixed_attendance','2023-09-08'),('optional_visit','2023-09-29')]:
                    agent=next(a for a in model.agents if a.persona['group']==group and a.persona['type']==kind)
                    plan=make_plan(panel,agent.persona,pd.Timestamp(date))
                    choice,digest=api_choice(agent.perceive(plan),agent.persona,provider,args.offline)
                    records.append({'provider':provider,'group':group,'type':kind,**choice,'request_hash':digest})
        pd.DataFrame(records).to_csv('res/pilot.csv',index=False);print(pd.DataFrame(records)[['provider','group','action','reason']].to_string(index=False));return
    if args.stage=='feedback':
        scale=fit_rule(panel,stations)
        output=[simulate(panel,stations,method,WINDOWS['feedback'],'feedback',offline=args.offline,rain_scale=scale) for method in ['no_change','rule','pro','lite']]
        pd.concat(output,ignore_index=True).to_csv('res/feedback_decisions.csv.gz',index=False)
        with open('res/manifest.json','w') as f:json.dump({'main_n_agents_per_station':24,'groups':GROUPS,'train_end':'2023-09-07',
            'windows':{k:[str(d.date()) for d in v] for k,v in WINDOWS.items()},'model_ids':MODELS,'caps':CAPS,
            'rule_scale':scale,'test_used_for_selection':False,'decision':'single personal action; never aggregate demand multiplier',
            'profile_weights':'synthetic profiles, observed training fare shares, same pre-trip planned mass across methods',
            'memory':'last 3 synthetic personal outcomes; common training seed, reset per independent episode',
            'code_sha256':hashlib.sha256(open('run.py','rb').read()+open('model.py','rb').read()+open('prepare.py','rb').read()).hexdigest()},f,indent=2)
        return
    with open('res/rule_selection.json') as f:scale=json.load(f)['selected']['rain_scale']
    if args.stage=='test':
        output=[]
        for block in ['test_ordinary','test_storm']:
            for method in ['no_change','rule','pro','lite']:
                frame=simulate(panel,stations,method,WINDOWS[block],block,offline=args.offline,rain_scale=scale)
                output.append(frame)
                pd.concat(output,ignore_index=True).to_csv('res/decisions.csv.gz',index=False)
                print('Completed episode',block,method,len(frame),'personal decisions',flush=True)
        return
    decisions=pd.read_csv('res/decisions.csv.gz',dtype={'station_complex_id':str})
    if args.stage=='checks':
        matched_checks(panel,decisions,args.offline).to_csv('res/matched_checks.csv.gz',index=False)
        output=[]
        for n in [24,48,96]:
            for method in ['no_change','rule','pro']:
                output.append(simulate(panel,[stations[0]],method,pd.DatetimeIndex(['2023-09-29']),'resolution_probe',n,args.offline,scale))
                pd.concat(output,ignore_index=True).to_csv('res/resolution_decisions.csv.gz',index=False)
        return
    aggregate(panel,decisions).to_csv('res/predictions.csv.gz',index=False)
    print('Saved main station-hour predictions; each action came from one traveler',flush=True)


if __name__=='__main__':main()
