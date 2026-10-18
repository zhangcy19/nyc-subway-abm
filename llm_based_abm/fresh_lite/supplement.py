"""Fixed-weight rainfall-information ablation and a future synthetic scenario.
Run from fresh_lite/. Uses a separate, capped 1M-token supplemental ledger.
"""
import argparse,json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import supplement_api as api
from experiment import PEOPLE,examples
from timing import hourly_weather,person_view,weather_view
from model import predict


def jobs():
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});w=hourly_weather();out=[]
    for task,part in panel.groupby('task'):
        train=part[part.split=='train'];test=part[(part.split=='test')&part.total_ridership.notna()]
        for _,r in test.drop_duplicates('date').iterrows():
            for p in PEOPLE:out.append(dict(task=task,route='role',date=r.date,person=p,view=person_view(p,r,train,w,True,False)))
        for _,r in test.iterrows():
            past=train[(train.station_complex_id==r.station_complex_id)&train.total_ridership.notna()]
            v={'target':weather_view(r,w,False),'observed_training_examples':[{'input':weather_view(e,w,False),'entries':round(float(e.total_ridership),1)} for _,e in examples(past,r,8).iterrows()]}
            out.append(dict(task=task,route='direct',date=r.date,station_complex_id=r.station_complex_id,view=v))
    return panel,out


def ablation(offline=False):
    panel,work=jobs();start=time.perf_counter()
    def call(j):return j,api.request(j['route'],j['view'],j.get('person'),offline=offline)
    with ThreadPoolExecutor(max_workers=6) as pool:result=list(pool.map(call,work))
    weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});pred=[];dec=[]
    for task,part in panel.groupby('task'):
        test=part[(part.split=='test')&part.total_ridership.notna()]
        policy={(j['date'],j['person']['agent_id']):a for j,a in result if j['task']==task and j['route']=='role'}
        fitted=weights[(weights.task==task)&(weights.method=='role_with_history')].to_dict('records')
        p,d=predict(test,fitted,policy,PEOPLE)
        pred.extend(dict(task=task,method='role_few_daily_input',**r) for r in p);dec.extend(dict(task=task,method='role_few_daily_input',**r) for r in d)
    for j,a in result:
        if j['route']=='direct':
            r=panel[(panel.task==j['task'])&(panel.date==j['date'])&(panel.station_complex_id==j['station_complex_id'])].iloc[0]
            pred.append(dict(task=j['task'],method='direct_few_daily_input',date=j['date'],station_complex_id=j['station_complex_id'],actual=r.total_ridership,predicted=a['entries'],reason=a['reason'],request_hash=a['hash']))
    pd.DataFrame(pred).to_csv('res/supplement/ablation_predictions.csv',index=False);pd.DataFrame(dec).to_csv('res/supplement/ablation_decisions.csv',index=False)
    json.dump(dict(seconds=time.perf_counter()-start,scope='TEST input ablation; original timed TRAIN weights frozen; not a retrained daily-only pipeline',offline=offline),open('res/supplement/ablation_runtime.json','w'),indent=2)


def scenario(offline=False):
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});train=panel[(panel.task=='Extreme_0929')&(panel.split=='train')];weather=hourly_weather();future='2026-10-23';result=[];pred=[];dec=[]
    # Hypothetical Friday, matched dry control; no claim this is an actual weather forecast.
    rain=np.array([0,0,0,0,0,1,4,10,20,18,12,8,4,2,1,0,0,0,0,0,0,0,0,0.],float)
    weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});weights=weights[(weights.task=='Extreme_0929')&(weights.method=='role_with_history')].to_dict('records')
    for name,hours in [('dry',np.zeros(24)),('storm',rain)]:
        r=dict(date=future,weekday=4,school_term=1,holiday=0,rain_mm=float(hours.sum()),peak_mm=float(hours.max()),six_mm=float(pd.Series(hours).rolling(6,min_periods=1).sum().max()),temp_c=18.,wind_ms=10.,total_ridership=np.nan)
        weather[future]=hours
        policy={}
        for p in PEOPLE:
            v=person_view(p,r,train,weather,True,True);v['scenario_context']='Hypothetical 2026-10-23 weather, NOT an official forecast. Hold 2023 traveler composition and demand scale fixed; no closures assumed.'
            a=api.request('role',v,p,offline=offline);policy[(future,p['agent_id'])]=a;result.append(dict(scenario=name,date=future,agent_id=p['agent_id'],**a))
        target=pd.DataFrame([dict(station_complex_id=s,**r) for s in sorted(train.station_complex_id.unique())]);rows,ds=predict(target,weights,policy,PEOPLE)
        pred.extend(dict(scenario=name,**x) for x in rows);dec.extend(dict(scenario=name,**x) for x in ds)
        pd.DataFrame([dict(scenario=name,hour=h,rain_mm=float(v)) for h,v in enumerate(hours)]).to_csv('res/supplement/weather_'+name+'.csv',index=False)
    json.dump(result,open('res/supplement/scenario_policy_private.json','w'),indent=2)
    public=[{**r,'reason':'Original decision summary retained locally.'} for r in result]
    json.dump(public,open('res/supplement/scenario_policy.json','w'),indent=2)
    pd.DataFrame(pred).to_csv('res/supplement/scenario_predictions.csv',index=False);pd.DataFrame(dec).to_csv('res/supplement/scenario_decisions.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','ablation','scenario']);p.add_argument('--offline',action='store_true');a=p.parse_args()
    if a.phase=='preflight':
        _,work=jobs();est=sum(len(json.dumps(j['view']).encode())/3+250 for j in work)
        print('Calls',len(work),'estimated tokens',round(est),'plus 24 scenario calls and one planning call; hard cap 1M')
    else:globals()[a.phase](a.offline)
    print('Supplement charged',sum(api.charge(r) for r in api.ledger()))
