"""Run from current: daily observations and weather-only chronological splits."""
import glob
import json
import os
import numpy as np
import pandas as pd

GROUPS = ['students', 'fair_fare', 'seniors_disability', 'general']
CHANNELS = {'students':['metrocard_students'], 'fair_fare':['metrocard_fair_fare'],
    'seniors_disability':['metrocard_seniors_disability','omny_seniors_disability'],
    'general':['metrocard_full_fare','metrocard_unlimited_30day','metrocard_unlimited_7day','metrocard_other','omny_full_fare','omny_other']}
STATIONS = ['360','37','228','269']

def prepare():
    os.makedirs('res/figures', exist_ok=True)
    w = pd.read_csv('../../../data/weather_hourly.csv', parse_dates=['timestamp']).sort_values('timestamp')
    w['six'] = w.rain_mm.rolling(6, min_periods=1).sum()
    weather = w.groupby(w.timestamp.dt.strftime('%Y-%m-%d')).agg(rain_mm=('rain_mm','sum'),
        peak_mm=('rain_mm','max'), six_mm=('six','max'), temp_c=('temp_c','mean'), wind_ms=('wind_ms','max')).rename_axis('date').reset_index()
    weather['weekday'] = pd.to_datetime(weather.date).dt.dayofweek
    weather['weekend'] = (weather.weekday >= 5).astype(int)
    # Calendar signals, not inferred from ridership. School calendar is deliberately coarse.
    weather['school_term'] = ((weather.date >= '2023-09-07') & (weather.weekday < 5)).astype(int)
    weather['holiday'] = weather.date.isin(['2023-07-04','2023-09-04','2023-10-09']).astype(int)
    weather['day_index'] = (pd.to_datetime(weather.date)-pd.Timestamp('2023-07-01')).dt.days
    peaks = weather[(weather.rain_mm >= 25) | (weather.peak_mm >= 10)].date.tolist()
    rows=[]
    windows={
        'Normal_Summer':[('train','2023-07-10','2023-07-26'),('validation','2023-08-01','2023-08-07'),('test','2023-08-17','2023-08-23')],
        'Normal_School':[('train','2023-09-12','2023-10-04'),('validation','2023-10-10','2023-10-16'),('test','2023-10-23','2023-10-29')],
        'Extreme_0729':[('train','2023-07-01','2023-07-19'),('validation','2023-07-20','2023-07-26'),('test','2023-07-27','2023-07-31')],
        'Extreme_0813':[('train','2023-07-18','2023-08-03'),('validation','2023-08-04','2023-08-10'),('test','2023-08-11','2023-08-15')],
        'Extreme_0929':[('train','2023-09-07','2023-09-19'),('validation','2023-09-20','2023-09-26'),('test','2023-09-27','2023-10-01')],
        'Extreme_1007':[('train','2023-09-07','2023-09-29'),('validation','2023-09-30','2023-10-04'),('test','2023-10-05','2023-10-09')],
    }
    event_dates={'Extreme_0729':'2023-07-29','Extreme_0813':'2023-08-13','Extreme_0929':'2023-09-29','Extreme_1007':'2023-10-07'}
    excluded={str(d.date()) for peak in peaks for d in pd.date_range(pd.Timestamp(peak)-pd.Timedelta(days=2),pd.Timestamp(peak)+pd.Timedelta(days=2))}
    for task,parts in windows.items():
        for split,start,end in parts:
            eligible=weather[(weather.date>=start)&(weather.date<=end)]
            if task.startswith('Normal'):
                eligible=eligible[(eligible.rain_mm<10)&(eligible.peak_mm<10)&~eligible.date.isin(excluded)]
            for d in eligible.date:
                event=event_dates.get(task,'')
                rows.append(dict(task=task,date=d,split=split,event=event,event_day=d==event))
    dates=pd.DataFrame(rows).merge(weather,on='date').sort_values(['task','date'])
    dates.to_csv('res/dates.csv',index=False);weather.to_csv('res/weather_daily.csv',index=False)
    parts=[]
    for f in ['../../../data/station_hourly.csv.gz']+sorted(glob.glob('../../../data/extensions/*.csv.gz')):
        for x in pd.read_csv(f,dtype={'station_complex_id':str},parse_dates=['timestamp'],chunksize=100000):
            x=x[x.station_complex_id.isin(STATIONS)].copy();x['date']=x.timestamp.dt.strftime('%Y-%m-%d')
            # Sparse fare-category pivots: recorded categories sum EXACTLY to the hourly total.
            # Missing category cells therefore contribute zero; missing entire hours remain missing.
            for g,cols in CHANNELS.items():x[g]=x[cols].sum(axis=1).where(x.total_ridership.notna())
            parts.append(x[['station_complex_id','date','timestamp','total_ridership']+GROUPS])
    x=pd.concat(parts,ignore_index=True)
    daily=x.groupby(['station_complex_id','date'])[['total_ridership']+GROUPS].sum(min_count=24).reset_index()
    coverage=x.groupby(['station_complex_id','date']).total_ridership.count().rename('observed_hours').reset_index()
    daily=daily.merge(coverage).merge(weather,on='date');daily.to_csv('res/daily.csv',index=False)
    stations=pd.read_csv('../../../data/stations.csv',dtype={'station_complex_id':str})
    stations[stations.station_complex_id.isin(STATIONS)].to_csv('res/stations.csv',index=False)
    panel=dates[['task','date','split','event','event_day']].merge(daily,on='date')
    panel.to_csv('res/panel.csv',index=False)
    print(dates.groupby(['task','split']).agg(days=('date','size'),rain_min=('rain_mm','min'),rain_max=('rain_mm','max')).to_string())
    print('Incomplete station-days:',panel[panel.observed_hours!=24][['task','date','station_complex_id','observed_hours']].to_dict('records'))
    json.dump({'events':peaks,'stations':STATIONS,'budget_tokens':2000000,'normal_rain_daily_lt':10,
        'event_trigger':'daily >=25 mm OR peak hour >=10 mm; NOT official extreme classification',
        'event_split':'four chronological event folds; same summer/school regime; +/-2-day test windows',
        'missing':'exclude incomplete 24-hour labels; never fill missing ridership with zero',
        'forecast':'daily entries, realized daily weather assumed known; no target ridership input',
        'reuse':'retrospective redesign on previously inspected 2023 data, not a pristine prospective holdout'},open('res/design.json','w'),indent=2)

if __name__=='__main__':prepare()
