"""Read all source rows once. Keep absent hours missing; select a complete evaluation cohort."""
import glob,json,os
import numpy as np
import pandas as pd
GROUPS=['students','fair_fare','seniors_disability','general']
CHANNELS={'students':['metrocard_students'],'fair_fare':['metrocard_fair_fare'],
 'seniors_disability':['metrocard_seniors_disability','omny_seniors_disability'],
 'general':['metrocard_full_fare','metrocard_unlimited_30day','metrocard_unlimited_7day','metrocard_other','omny_full_fare','omny_other']}
EVENTS={'Extreme_0729':'2023-07-29','Extreme_0813':'2023-08-13','Extreme_0929':'2023-09-29','Extreme_1007':'2023-10-07'}

def prepare():
 os.makedirs('prepared',exist_ok=True)
 w=pd.read_csv('../../../data/weather_hourly.csv',parse_dates=['timestamp']).sort_values('timestamp')
 w['six']=w.rain_mm.rolling(6,min_periods=1).sum()
 weather=w.groupby(w.timestamp.dt.strftime('%Y-%m-%d')).agg(rain_mm=('rain_mm','sum'),peak_mm=('rain_mm','max'),six_mm=('six','max'),temp_c=('temp_c','mean'),wind_ms=('wind_ms','max')).rename_axis('date').reset_index()
 weather['weekday']=pd.to_datetime(weather.date).dt.dayofweek
 weather['school_term']=((weather.date>='2023-09-07')&(weather.weekday<5)).astype(int)
 weather['holiday']=weather.date.isin(['2023-07-04','2023-09-04','2023-10-09']).astype(int)
 peaks=weather[(weather.rain_mm>=25)|(weather.peak_mm>=10)].date
 excluded={str(d.date()) for peak in peaks for d in pd.date_range(pd.Timestamp(peak)-pd.Timedelta(days=2),pd.Timestamp(peak)+pd.Timedelta(days=2))}
 windows={'Normal_Summer':[('train','2023-07-10','2023-07-26'),('validation','2023-08-01','2023-08-07'),('test','2023-08-17','2023-08-23')],
 'Normal_School':[('train','2023-09-12','2023-10-04'),('validation','2023-10-10','2023-10-16'),('test','2023-10-23','2023-10-29')]}
 dates=[]
 for task,parts in windows.items():
  for split,a,b in parts:
   ds=weather[(weather.date>=a)&(weather.date<=b)&(weather.rain_mm<10)&(weather.peak_mm<10)&~weather.date.isin(excluded)]
   dates.extend(dict(task=task,date=d,split=split,event_day=False) for d in ds.date)
 for task,date in EVENTS.items():
  dates.extend(dict(task=task,date=str(d.date()),split='test',event_day=str(d.date())==date) for d in pd.date_range(pd.Timestamp(date)-pd.Timedelta(days=2),pd.Timestamp(date)+pd.Timedelta(days=2)))
 dates=pd.DataFrame(dates);dates.to_csv('prepared/dates.csv',index=False)
 cols=['station_complex_id','timestamp','total_ridership']+sum(CHANNELS.values(),[]);parts=[]
 for f in ['../../../data/station_hourly.csv.gz']+sorted(glob.glob('../../../data/extensions/*.csv.gz')):
  for x in pd.read_csv(f,usecols=cols,dtype={'station_complex_id':str},chunksize=100000):
   for g,cs in CHANNELS.items():x[g]=x[cs].sum(axis=1).where(x.total_ridership.notna())
   parts.append(x[['station_complex_id','timestamp','total_ridership']+GROUPS])
 x=pd.concat(parts,ignore_index=True)
 if x.duplicated(['station_complex_id','timestamp']).any():raise ValueError('Duplicate station-hour keys')
 if not np.allclose(x[GROUPS].sum(axis=1),x.total_ridership,equal_nan=True):raise ValueError('Fare groups do not sum to total')
 x['date']=x.timestamp.str[:10]
 daily=x.groupby(['station_complex_id','date'])[['total_ridership']+GROUPS].sum(min_count=24).reset_index()
 coverage=x.groupby(['station_complex_id','date']).total_ridership.count().rename('observed_hours').reset_index()
 stations=pd.read_csv('../../../data/stations.csv',dtype={'station_complex_id':str})
 grid=pd.MultiIndex.from_product([stations.station_complex_id.unique(),sorted(dates.date.unique())],names=['station_complex_id','date']).to_frame(index=False)
 coverage=grid.merge(coverage,how='left');coverage['observed_hours']=coverage.observed_hours.fillna(0).astype(int);coverage['missing_hours']=24-coverage.observed_hours
 daily=coverage.merge(daily,how='left').merge(weather,on='date')
 daily.loc[daily.observed_hours!=24,['total_ridership']+GROUPS]=np.nan
 daily.to_csv('prepared/daily.csv',index=False)
 audit=coverage.groupby('station_complex_id').agg(incomplete_days=('missing_hours',lambda s:int((s>0).sum())),missing_hours=('missing_hours','sum'),fully_absent_days=('observed_hours',lambda s:int((s==0).sum())),max_missing_hours_in_day=('missing_hours','max')).reset_index().merge(stations[['station_complex_id','station_name']],on='station_complex_id')
 audit.to_csv('prepared/missing_by_station.csv',index=False);coverage[coverage.missing_hours>0].to_csv('prepared/missing_by_day.csv',index=False)
 complete=audit[audit.missing_hours==0].station_complex_id.tolist()
 # Keep the old two pilot stations outside the fresh full cohort.
 selected=[s for s in complete if s not in ['256','385']]
 json.dump(selected,open('prepared/station_ids.json','w'),indent=2)
 for col,bins in [('incomplete_days',[-1,0,2,7,14,30,74]),('missing_hours',[-1,0,2,10,24,72,168,1776])]:
  print(col,audit.groupby(pd.cut(audit[col],bins),observed=False).size().to_dict())
 print('Complete stations',len(complete),'full cohort excluding pilot stations',len(selected),'absent entire station-days',int((coverage.observed_hours==0).sum()))

if __name__=='__main__':prepare()
