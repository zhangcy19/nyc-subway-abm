"""Keep original task windows and labels; replace only weather with local estimates."""
import json
import numpy as np
import pandas as pd

ids=json.load(open('res/station_ids.json'))
w=pd.read_csv('res/spatial_hourly.csv',dtype={'station_complex_id':str},parse_dates=['timestamp']).sort_values(['station_complex_id','timestamp'])
w['date']=w.timestamp.dt.strftime('%Y-%m-%d');w['hour']=w.timestamp.dt.hour
# Preserve the original daily aggregation definitions. Incomplete rain days are omitted.
w['six']=w.groupby('station_complex_id').rain_mm.transform(lambda s:s.rolling(6,min_periods=1).sum())
d=w.groupby(['station_complex_id','date']).agg(rain_mm=('rain_mm',lambda a:a.sum(min_count=24)),peak_mm=('rain_mm','max'),six_mm=('six','max'),temp_c=('temp_c','mean'),wind_ms=('wind_ms','max'),weather_hours=('rain_mm','count')).reset_index()
d['weather_complete']=d.weather_hours.eq(24)
old=pd.read_csv('../fresh_lite/res/panel.csv',dtype={'station_complex_id':str})
cols=['rain_mm','peak_mm','six_mm','temp_c','wind_ms']
p=old[old.station_complex_id.isin(ids)].drop(columns=cols).merge(d,on=['station_complex_id','date'],validate='many_to_one')
p[~p.weather_complete].to_csv('res/excluded_weather_days.csv',index=False)
p=p[p.weather_complete].copy();p.to_csv('res/panel.csv',index=False);d.to_csv('res/weather_daily.csv',index=False)
old[old.station_complex_id.isin(ids)].merge(p[['task','station_complex_id','date']],on=['task','station_complex_id','date'],validate='one_to_one').to_csv('res/jfk_panel_matched.csv',index=False)
# Preserve the original pooled fare-group history convention: change weather source, not the demographic scope.
valid_dates=set(d[d.weather_complete].date);old[old.date.isin(valid_dates)].to_csv('res/history_pool.csv',index=False)
pd.read_csv('../../../data/stations.csv',dtype={'station_complex_id':str}).query('station_complex_id in @ids').to_csv('res/stations.csv',index=False)
oldw=pd.read_csv('../../../data/weather_hourly.csv',parse_dates=['timestamp']);oldw['date']=oldw.timestamp.dt.strftime('%Y-%m-%d')
j=oldw.groupby('date').rain_mm.sum().rename('jfk_rain_mm').reset_index();q=d.merge(j,on='date');q=q[q.date.isin(['2023-07-29','2023-08-13','2023-09-29','2023-10-07'])];q.to_csv('res/event_weather_comparison.csv',index=False)
print(q[['station_complex_id','date','rain_mm','peak_mm','jfk_rain_mm']].to_string(index=False));print(p.groupby(['station_complex_id','task','split']).size().to_string())
