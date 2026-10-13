"""Run from V3. Public totals define planned opportunities, not personal histories."""
import os
import json
import numpy as np
import pandas as pd
from model import GROUPS, population


def prepare():
    os.makedirs('res',exist_ok=True)
    x=pd.read_csv('../../data/station_hourly.csv.gz',dtype={'station_complex_id':str},parse_dates=['timestamp'])
    # Keep the published V3 experiment on its original August–September window.
    x=x[(x.timestamp>='2023-08-01')&(x.timestamp<'2023-10-01')]
    old=pd.concat([pd.read_csv('../V1/experiment_data/stations.csv',dtype={'station_complex_id':str}),
        pd.read_csv('../V2/results/stations.csv',dtype={'station_complex_id':str})]).station_complex_id
    train=x[x.timestamp<'2023-09-08']
    summary=train.groupby(['borough','station_complex_id']).agg(n=('timestamp','size'),total=('total_ridership','sum')).reset_index()
    chosen=[]
    for borough in ['Manhattan','Brooklyn']:
        candidates=summary[(summary.borough==borough)&(summary.n>=.98*38*24)&~summary.station_complex_id.isin(old)].sort_values(['total','station_complex_id'])
        chosen.append(candidates.iloc[len(candidates)//2].station_complex_id)
    x=x[x.station_complex_id.isin(chosen)].copy()
    stations=x[['station_complex_id','station_name','borough','latitude','longitude']].drop_duplicates('station_complex_id')
    stations.to_csv('res/stations.csv',index=False)
    weather=pd.read_csv('../../data/weather_hourly.csv',parse_dates=['timestamp']).sort_values('timestamp')
    weather=weather[(weather.timestamp>='2023-08-01')&(weather.timestamp<'2023-10-01')].copy()
    weather['rain_6h_mm']=weather.rain_mm.rolling(6,min_periods=1).sum()
    weather['weather_context']=np.select([weather.rain_6h_mm>=30,(weather.rain_mm>=5)|(weather.rain_6h_mm>=10),
        (weather.rain_mm>.1)|(weather.rain_6h_mm>.1)],['extreme','moderate','light'],default='dry')
    # Plan for all hours without knowing which future observations will be missing.
    grid=pd.MultiIndex.from_product([chosen,weather.timestamp],names=['station_complex_id','timestamp']).to_frame(index=False)
    x=grid.merge(x.drop(columns=['station_name','borough','latitude','longitude']),on=['station_complex_id','timestamp'],how='left').merge(stations,on='station_complex_id')
    x=x.drop(columns=['rain_mm','temp_c','wind_ms','weather_available','weather_station']).merge(weather,on='timestamp')
    channels={'students':['metrocard_students'],'fair_fare':['metrocard_fair_fare'],
        'seniors_disability':['metrocard_seniors_disability','omny_seniors_disability'],
        'general':['metrocard_full_fare','metrocard_unlimited_30day','metrocard_unlimited_7day','metrocard_other','omny_full_fare','omny_other']}
    for group,cols in channels.items():x[group]=x[cols].sum(axis=1,min_count=1)
    x['date']=x.timestamp.dt.normalize();x['hour']=x.timestamp.dt.hour;x['weekday']=x.timestamp.dt.dayofweek
    train=x[x.timestamp<'2023-09-08']
    profiles=train.groupby(['station_complex_id','weekday','hour'])[['total_ridership']+GROUPS].mean()
    index=pd.MultiIndex.from_frame(x[['station_complex_id','weekday','hour']])
    prior=profiles.reindex(index).reset_index(drop=True)
    indexed=x.set_index(['station_complex_id','timestamp'])
    previous=pd.MultiIndex.from_arrays([x.station_complex_id,x.timestamp-pd.Timedelta(days=7)])
    # A completed week's same hour is known before the trip. No current-day truth enters planning.
    x['planned_total']=indexed.total_ridership.reindex(previous).to_numpy()
    x['planned_total']=x.planned_total.fillna(pd.Series(prior.total_ridership.to_numpy(),index=x.index))
    group_profile=indexed[GROUPS].reindex(previous).reset_index(drop=True)
    # Preserve known recent school-term fare patterns; missing fare observations are unknown.
    # Allocation uses past observed-hour means where that week's fare cell was missing.
    group_profile=group_profile.fillna(prior[GROUPS]).fillna(0).to_numpy()
    denominator=group_profile.sum(axis=1)
    for i,g in enumerate(GROUPS):x['planned_'+g]=x.planned_total*np.divide(group_profile[:,i],denominator,out=np.zeros(len(x)),where=denominator>0)
    x.loc[denominator==0,'planned_general']=x.loc[denominator==0,'planned_total']
    x.to_csv('res/panel.csv.gz',index=False)
    for n in [24,48,96]:pd.DataFrame(population(chosen,n)).to_csv('res/personas_'+str(n)+'.csv',index=False)
    with open('res/source_summary.json','w') as f:json.dump({'train_end':'2023-09-07','stations':chosen,
        'group_profile':'previous-week observed fares, falling back to training observed-hour means; original group NaNs retained for evaluation',
        'planned_total':'previous week same hour, falling back to training weekday-hour profile',
        'weight':'planned daily group trip opportunities / representatives in that group',
        'not_observed':['individual profiles','personality','OD','personal history','mode alternatives','experience delays']},f,indent=2)
    print('Prepared two new stations',stations[['station_complex_id','station_name']].to_dict('records'),flush=True)


if __name__=='__main__':prepare()
