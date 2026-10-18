"""Re-fit and replay published numerical results; no raw API log or key required."""
import pandas as pd
import numpy as np
from experiment import load_policy,choose_population,numerical,weekday_mean,PEOPLE,RAIN
from timing import timed_rule_policy,hourly_weather
from model import predict

panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str})
dates=pd.read_csv('res/dates.csv');saved=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str});weather=hourly_weather();checks=[]
for task,part in panel.groupby('task'):
    train=part[part.split=='train'];valid=part[part.split=='validation'];test=part[part.split=='test']
    for method in ['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few']:
        reference=saved[(saved.task==task)&(saved.method==method)]
        if method=='weekday_mean':rows=weekday_mean(train,test)
        elif method=='ridge':rows=numerical(train,valid,test)[0]
        elif method=='direct_few':
            checks.append(dict(task=task,method=method,max_error=0.,scope='Saved numeric LLM forecasts; no new inference or raw-response verification'))
            continue
        else:
            policies={s:timed_rule_policy(dates[dates.task==task],weather,s,method=='rule_timing') for s in RAIN} if method.startswith('rule_') else {0.:load_policy(task,method)}
            weights,policy,_,_=choose_population(train,valid,policies,method)
            rows,_=predict(test,weights,policy,PEOPLE)
        result=pd.DataFrame(rows).merge(reference,on=['date','station_complex_id'],suffixes=('_replay','_saved'))
        error=float(abs(result.predicted_replay-result.predicted_saved).max())
        np.testing.assert_allclose(result.predicted_replay,result.predicted_saved,atol=1e-6,rtol=1e-8)
        checks.append(dict(task=task,method=method,max_error=error,scope='Refitted from published numeric policies and training/validation data'))
    print('Verified',task,flush=True)
pd.DataFrame(checks).to_csv('res/numeric_replay.csv',index=False)
print('Numerical replay passed. Direct LLM forecasts are reused; raw API provenance is retained locally, not verified by this public replay.')
