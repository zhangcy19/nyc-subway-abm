"""Fresh Lite experiment; no previous replies, coefficients or predictions are loaded."""
import argparse,json,time,hashlib,os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from api import request,MODEL,ROLE_SYSTEM,DIRECT_SYSTEM,ledger,charge
from experiment import PEOPLE,RAIN,choose_population,weekday_mean,numerical,examples,save_policy,wape
from model import predict
from timing import hourly_weather,person_view,weather_view,timed_rule_policy

METHODS=['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few']

def role_jobs(dates,train,weather,few):
    return [(r.date,p,person_view(p,r,train,weather,few,True)) for _,r in dates.iterrows() for p in PEOPLE]

def direct_view(row,train,weather):
    past=train[(train.station_complex_id==row.station_complex_id)&train.total_ridership.notna()]
    selected=examples(past,row,8)
    return {'target':weather_view(row,weather,True),'observed_training_examples':[{'input':weather_view(e,weather,True),'entries':round(float(e.total_ridership),1)} for _,e in selected.iterrows()]}

def preflight():
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});dates=pd.read_csv('res/dates.csv');weather=hourly_weather();unique={};counts=[]
    for task,part in panel.groupby('task'):
        train=part[part.split=='train'];test=part[part.split=='test'];td=dates[dates.task==task]
        for few in [False,True]:
            for date,person,view in role_jobs(td,train,weather,few):
                text=ROLE_SYSTEM+json.dumps(view,separators=(',',':'));unique[hashlib.sha256(text.encode()).hexdigest()]=len(text.encode())
        for _,r in test.iterrows():
            text=DIRECT_SYSTEM+json.dumps(direct_view(r,train,weather),separators=(',',':'));unique[hashlib.sha256(text.encode()).hexdigest()]=len(text.encode())
        counts.append(dict(task=task,days=len(td),train_days=len(td[td.split=='train']),validation_days=len(td[td.split=='validation']),test_days=len(td[td.split=='test']),observed_test_station_days=int(test.total_ridership.notna().sum())))
    estimate=sum(v/3+200 for v in unique.values())
    report=dict(unique_planned_requests=len(unique),estimated_tokens=round(estimate),estimate_definition='UTF-8 input bytes / 3 + 200 output tokens per unique request; estimate only, actual usage has a hard cap',token_cap=5000000,model=MODEL)
    json.dump(report,open('res/preflight.json','w'),indent=2);pd.DataFrame(counts).to_csv('res/dataset_summary.csv',index=False);print(report);print(pd.DataFrame(counts).to_string(index=False))

def run(offline=False):
    protocol=json.load(open('res/protocol.json'))
    for f,h in protocol['files'].items():
        if hashlib.sha256(open(f,'rb').read()).hexdigest()!=h:raise ValueError('Frozen input changed: '+f)
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});dates=pd.read_csv('res/dates.csv');weather=hourly_weather()
    predictions=[];decisions=[];populations=[];choices=[];runtime=[];completed=set()
    if not offline and os.path.exists('res/predictions.csv'):
        predictions=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str}).to_dict('records')
        decisions=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str}).to_dict('records')
        populations=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str}).to_dict('records')
        choices=pd.read_csv('res/validation.csv').to_dict('records')
        runtime=pd.read_csv('res/runtime.csv').to_dict('records')
        completed={(r['task'],r['method']) for r in runtime}
    for task,part in panel.groupby('task'):
        train=part[part.split=='train'];valid=part[part.split=='validation'];test=part[part.split=='test'];td=dates[dates.task==task]
        for method in METHODS:
            if (task,method) in completed:continue
            start=time.perf_counter();tokens_before=sum(charge(r) for r in ledger());print(task,method,flush=True)
            if method=='weekday_mean':rows=weekday_mean(train,test)
            elif method=='ridge':
                rows,candidates,_=numerical(train,valid,test);choices.extend(dict(task=task,**r) for r in candidates)
            elif method=='direct_few':
                def call(item):
                    _,r=item;answer=request('direct',direct_view(r,train,weather),offline=offline)
                    return dict(station_complex_id=r.station_complex_id,date=r.date,actual=r.total_ridership,predicted=answer['entries'],request_hash=answer['hash'],reason=answer['reason'])
                with ThreadPoolExecutor(max_workers=int(os.environ.get("ABM_WORKERS","6"))) as pool:rows=list(pool.map(call,test.iterrows()))
            else:
                if method.startswith('rule_'):policies={scale:timed_rule_policy(td,weather,scale,method=='rule_timing') for scale in RAIN}
                else:
                    jobs=role_jobs(td,train,weather,method=='role_with_history')
                    def call(job):
                        date,person,view=job
                        return (date,person['agent_id']),request('role',view,person,offline=offline)
                    with ThreadPoolExecutor(max_workers=int(os.environ.get("ABM_WORKERS","6"))) as pool:policies={0.:dict(pool.map(call,jobs))}
                weights,policy,candidates,best=choose_population(train,valid,policies,method)
                save_policy(task,method,policy);rows,dec=predict(test,weights,policy,PEOPLE)
                decisions.extend(dict(task=task,method=method,**r) for r in dec);populations.extend(dict(task=task,method=method,**r) for r in weights);choices.extend(dict(task=task,**r) for r in candidates)
            predictions.extend(dict(task=task,method=method,**r) for r in rows)
            runtime.append(dict(task=task,method=method,seconds=time.perf_counter()-start,new_tokens=sum(charge(r) for r in ledger())-tokens_before))
            for filename,data in [('predictions',predictions),('decisions',decisions),('populations',populations),('validation',choices),('runtime',runtime)]:
                pd.DataFrame(data).to_csv('res/'+('replay_runtime' if offline and filename=='runtime' else filename)+'.csv',index=False)
            print('Completed',task,method,'WAPE',round(wape(rows),4),'tokens',sum(charge(r) for r in ledger()),flush=True)
    pd.DataFrame(PEOPLE).to_csv('res/personas.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--offline',action='store_true');args=p.parse_args()
    if args.preflight:preflight()
    else:run(args.offline)
