"""Read-only numerical replay of saved primary decisions and provenance checks."""
import hashlib
import glob
import json
import numpy as np
import pandas as pd
from api import ledger,charge,validate
from experiment import load_policy,choose_population,rule_policy,RAIN,PEOPLE,numerical,weekday_mean
from model import predict

def audit():
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});saved=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str})
    weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});dates=pd.read_csv('res/dates.csv');calls=ledger();known={r['hash'] for r in calls if r['status']=='valid'}
    replay=[]
    for task,part in panel.groupby('task'):
        train=part[part.split=='train'];valid=part[part.split=='validation'];test=part[part.split=='test']
        for method,pred in [('weekday_mean',weekday_mean(train,test)),('ridge',numerical(train,valid,test)[0])]:
            r=pd.DataFrame(pred).merge(saved[(saved.task==task)&(saved.method==method)],on=['station_complex_id','date'],suffixes=('_replay','_saved'))
            delta=float(abs(r.predicted_replay-r.predicted_saved).max())
            if delta>1e-7:raise ValueError('Numerical model replay mismatch')
            replay.append(dict(task=task,method=method,max_absolute_difference=delta))
        for method in ['rule_abm','role_no_history','role_with_history']:
            policy=load_policy(task,method);test=part[part.split=='test'];w=weights[(weights.task==task)&(weights.method==method)].to_dict('records')
            policies={scale:rule_policy(dates[dates.task==task],scale) for scale in RAIN} if method=='rule_abm' else {0.:policy}
            fitted,_,_,_=choose_population(train,valid,policies,method)
            fitted=pd.DataFrame(fitted).merge(pd.DataFrame(w),on=['station_complex_id','agent_id'],suffixes=('_replay','_saved'))
            if abs(fitted.effective_population_replay-fitted.effective_population_saved).max()>1e-6:raise ValueError('Training/validation population replay mismatch')
            pred,_=predict(test,w,policy,PEOPLE);r=pd.DataFrame(pred).merge(saved[(saved.task==task)&(saved.method==method)],on=['station_complex_id','date'],suffixes=('_replay','_saved'))
            max_error=float(abs(r.predicted_replay-r.predicted_saved).max())
            replay.append(dict(task=task,method=method,max_absolute_difference=max_error))
            if max_error>1e-7:raise ValueError('Replay mismatch')
            if method!='rule_abm' and any(v['hash'] not in known for v in policy.values()):raise ValueError('Unbacked policy')
    for r in calls:
        computed=hashlib.sha256(json.dumps(r['request'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if computed!=r['hash']:raise ValueError('API request hash mismatch')
        if r['status']=='valid' and r['request']['route']=='role':
            validate(r['parsed'],'role',r['request']['view']['person'])
    if sum(charge(r) for r in calls)>4000000:raise ValueError('Budget exceeded')
    byhash={r['hash']:r for r in calls if r['status']=='valid'}
    for f in ['res/predictions.csv']+glob.glob('res/*_predictions.csv'):
        p=pd.read_csv(f)
        for _,r in p[p.method=='direct_few'].iterrows():
            if r.request_hash not in byhash or abs(byhash[r.request_hash]['parsed']['entries']-r.predicted)>1e-7:raise ValueError('Direct prediction not backed by its actual API result')
    errors=[]
    for task,rows in saved.groupby('task'):
        sets=[set(zip(p.date[p.actual.notna()],p.station_complex_id[p.actual.notna()])) for _,p in rows.groupby('method')]
        if any(s!=sets[0] for s in sets):raise ValueError('Unequal scoring sample')
    result=dict(primary_replay=replay,training_and_validation_refit_matches=True,all_direct_outputs_match_real_API=True,api_hashes_valid=True,valid_role_outputs_pass_constraints=True,
        equal_observed_samples_across_methods=True,total_calls=len(calls),charged_tokens=sum(charge(r) for r in calls),
        missing_station_days_excluded=int(panel.total_ridership.isna().sum()))
    json.dump(result,open('res/audit.json','w'),indent=2);print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
