"""Daily prediction with hourly rainfall exposure; run from current/."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from model import profiles,rule_prob,predict
from experiment import environment,examples,group_history,role_view,choose_population,RAIN,wape

TASKS=['Extreme_0813','Extreme_0929']
STATIONS=['360','37']
PEOPLE=profiles()

def hourly_weather():
    w=pd.read_csv('../../../data/weather_hourly.csv',parse_dates=['timestamp'])
    if w.timestamp.duplicated().any() or w.rain_mm.isna().any() or (w.rain_mm<0).any():raise ValueError('Invalid hourly weather')
    w['date']=w.timestamp.dt.strftime('%Y-%m-%d');w['hour']=w.timestamp.dt.hour
    p=w.pivot(index='date',columns='hour',values='rain_mm').reindex(columns=range(24))
    if p.isna().any().any():raise ValueError('Need 24 weather hours per day')
    return {d:r.to_numpy(float) for d,r in p.iterrows()}

def schedule(person,weekday):
    k=person['kind'];weekend=weekday>=5
    hours=([9,10,11] if weekend else [7,8,9]) if k==0 else list(range(10,17)) if k==1 else list(range(12,21)) if weekend else list(range(17,22))
    q=np.zeros(24);q[hours]=1/len(hours)
    return q

def exposure(person,row,rain):
    q=schedule(person,int(row['weekday']))
    return min(3.,float(q@rain/rain.mean())) if rain.mean()>0 else 0.

def timed_rule_policy(dates,weather,scale,timed):
    result={}
    for _,r in dates.iterrows():
        for person in PEOPLE:
            factor=exposure(person,r,weather[r.date]) if timed else 1.
            result[(r.date,person['agent_id'])]=dict(probabilities=rule_prob(person,r,scale*factor).tolist(),reason='Daily utility with schedule-weighted rainfall exposure.' if timed else 'Daily utility without rainfall timing.',hash='')
    return result

def weather_view(row,weather,timed):
    value=environment(row)
    if timed:value['hourly_rain_mm_00_to_23']=np.round(weather[row['date']],3).tolist()
    return value

def history_view(train,group,row,weather,timed):
    past=train[(train.observed_hours==24)&(train.date<row['date'])].copy()
    means=past.groupby('station_complex_id')[group].transform('mean')
    past['entry_index']=past[group]/means.replace(0,np.nan)
    daily=past.groupby('date').agg(**{k:(k,'first') for k in ['weekday','school_term','holiday','rain_mm','peak_mm','six_mm','temp_c','wind_ms']},entry_index=('entry_index','mean')).reset_index().dropna()
    return [{'calendar_weather':weather_view(r,weather,timed),'observed_group_entry_index':round(float(r.entry_index),3)} for _,r in examples(daily,row,4).iterrows()]

def person_view(person,row,train,weather,few,timed):
    v=role_view(person,row,train,False);v['target']=weather_view(row,weather,timed)
    q=schedule(person,int(row['weekday']))
    v['person']['hypothetical_departure_hours_local']=np.flatnonzero(q).tolist()
    v['schedule_definition']='Conditional on making ONE potential journey, departure is equally likely in the listed hours. This schedule is an assumption, not observed behavior. No return trip or known closures are modeled. All hours are New York local time.'
    if few:
        v['group_history']=history_view(train,person['group'],row,weather,timed)
        v['history_definition']='Earlier training fare-group entry indices, station-normalized; not personal choices or action probabilities.'
    return v

def offline_rules():
    weather=hourly_weather();panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});dates=pd.read_csv('res/dates.csv')
    predictions=[];decisions=[];weights=[];choices=[];runtime=[]
    for task,part in panel.groupby('task'):
        train=part[part.split=='train'];valid=part[part.split=='validation'];test=part[part.split=='test']
        for timed in [False,True]:
            start=time.perf_counter();method='rule_timing' if timed else 'rule_daily'
            policies={s:timed_rule_policy(dates[dates.task==task],weather,s,timed) for s in RAIN}
            fit,policy,candidates,best=choose_population(train,valid,policies,method)
            p,d=predict(test,fit,policy,PEOPLE)
            predictions.extend(dict(task=task,method=method,**r) for r in p);decisions.extend(dict(task=task,method=method,**r) for r in d)
            weights.extend(dict(task=task,method=method,**r) for r in fit);choices.extend(dict(task=task,**r) for r in candidates)
            runtime.append(dict(task=task,method=method,seconds=time.perf_counter()-start))
            print(task,method,'WAPE',round(wape(p),4),'chosen',best,flush=True)
    for name,rows in [('rule_predictions',predictions),('rule_decisions',decisions),('rule_populations',weights),('validation',choices),('runtime',runtime)]:pd.DataFrame(rows).to_csv('res/timing/'+name+'.csv',index=False)
    pd.DataFrame([dict(date=d,**{f'rain_{h:02}':v for h,v in enumerate(r)}) for d,r in weather.items()]).to_csv('res/timing/hourly_weather.csv',index=False)

def pro(offline=False):
    from timing_api import request
    weather=hourly_weather();panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str})
    jobs=[]
    for task in TASKS:
        part=panel[panel.task==task];train=part[part.split=='train'];targets=part[(part.split=='test')&part.event_day&part.station_complex_id.isin(STATIONS)]
        row=targets.iloc[0]
        for timed in [False,True]:
            for few in [False,True]:
                for person in PEOPLE:jobs.append(('role',task,timed,few,person,row,train,targets))
            for _,target in targets.iterrows():jobs.append(('direct',task,timed,True,None,target,train,targets))
    def call(job):
        route,task,timed,few,person,row,train,targets=job
        if route=='role':view=person_view(person,row,train,weather,few,timed)
        else:
            selected=examples(train[(train.station_complex_id==row.station_complex_id)&train.total_ridership.notna()],row,8)
            view={'target':weather_view(row,weather,timed),'observed_training_examples':[{'input':weather_view(e,weather,timed),'entries':round(float(e.total_ridership),1)} for _,e in selected.iterrows()]}
        answer=request(route,view,person,offline=offline)
        return job,answer
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(call,jobs))
    predictions=[];decisions=[];responses=[]
    for job,answer in results:
        route,task,timed,few,person,row,train,targets=job
        arm='timing' if timed else 'daily';method='direct_few' if route=='direct' else 'role_with_history' if few else 'role_no_history'
        responses.append(dict(task=task,method=method,arm=arm,date=row.date,agent_id=person['agent_id'] if person else '',station_complex_id='' if person else row.station_complex_id,**answer))
        if route=='direct':predictions.append(dict(task=task,method=method,arm=arm,date=row.date,station_complex_id=row.station_complex_id,actual=row.total_ridership,predicted=answer['entries'],request_hash=answer['hash']))
    for task in TASKS:
        targets=panel[(panel.task==task)&(panel.split=='test')&panel.event_day&panel.station_complex_id.isin(STATIONS)]
        for method in ['role_no_history','role_with_history']:
            fitted=weights[(weights.task==task)&(weights.method==method)&weights.station_complex_id.isin(STATIONS)].to_dict('records')
            for arm in ['daily','timing']:
                policy={(r['date'],r['agent_id']):r for r in responses if r['task']==task and r['method']==method and r['arm']==arm}
                p,d=predict(targets,fitted,policy,PEOPLE)
                predictions.extend(dict(task=task,method=method,arm=arm,**r) for r in p);decisions.extend(dict(task=task,method=method,arm=arm,**r) for r in d)
    pd.DataFrame(predictions).to_csv('res/timing/pro_predictions.csv',index=False);pd.DataFrame(decisions).to_csv('res/timing/pro_decisions.csv',index=False)
    json.dump(responses,open('res/timing/responses.json','w'),ensure_ascii=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['rules','pro']);parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    if args.phase=='rules':offline_rules()
    else:pro(args.offline)
