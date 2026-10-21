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
    w=pd.read_csv('../../data/weather_hourly.csv',parse_dates=['timestamp'])
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
