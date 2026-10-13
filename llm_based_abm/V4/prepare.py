"""Run from V4. All historical demand features precede their target day."""
import os
import json
import numpy as np
import pandas as pd

GROUPS = ['students', 'fair_fare', 'seniors_disability', 'general']
CHANNELS = {'students':['metrocard_students'], 'fair_fare':['metrocard_fair_fare'],
    'seniors_disability':['metrocard_seniors_disability','omny_seniors_disability'],
    'general':['metrocard_full_fare','metrocard_unlimited_30day','metrocard_unlimited_7day','metrocard_other','omny_full_fare','omny_other']}
VALIDATION = ['2023-08-04','2023-08-05','2023-08-06','2023-08-07']
TEST = ['2023-08-'+str(i).zfill(2) for i in [12,13,14,18,19,20,25,26]] + ['2023-09-'+str(i).zfill(2) for i in [22,23,24,25,27,28,29,30]] + ['2023-10-'+str(i).zfill(2) for i in [6,7,8,9,13,14,20,21]]


def supplementary(stations):
    county={'005':'Bronx','047':'Brooklyn','061':'Manhattan','081':'Queens','085':'Staten Island'}
    features = stations[['station_complex_id','borough','latitude','longitude']].copy()
    for table,columns,label in [
        ('age_B01001',[f'B01001_E{i:03}' for i in list(range(20,26))+list(range(44,50))],'resident_age65_share'),
        ('school_enrollment_B14001',['B14001_E002'],'resident_enrolled_share'),
        ('disability_B18101',[f'B18101_E{i:03}' for i in [4,7,10,13,16,19,23,26,29,32,35,38]],'resident_disability_share')]:
        x=pd.read_csv('../../data/supplementary/acs_2023_nyc_tract_'+table+'.csv',dtype={'GEO_ID':str})
        x['borough']=x.GEO_ID.str[11:14].map(county)
        prefix=columns[0].split('_')[0];x['numerator']=x[columns].sum(axis=1);x['denominator']=x[prefix+'_E001']
        totals=x.groupby('borough')[['numerator','denominator']].sum()
        features=features.merge((totals.numerator/totals.denominator).rename(label),on='borough',how='left')
    current=pd.read_csv('../../data/supplementary/station_complexes_current.csv',dtype={'complex_id':str,'gtfs_stop_ids':str})
    edges=pd.read_csv('../../data/supplementary/network_edges_current.csv',dtype={'from_stop_id':str,'to_stop_id':str,'route_id':str})
    current['current_route_count']=current.daytime_routes.str.split().str.len()
    current['current_platform_count']=current.gtfs_stop_ids.str.split(';').str.len()
    current['current_network_degree']=current.gtfs_stop_ids.map(lambda ids: int(edges[edges.from_stop_id.str.rstrip('NS').isin([v.strip() for v in ids.split(';')])].to_stop_id.nunique()))
    features=features.merge(current[['complex_id','current_route_count','current_platform_count','current_network_degree','ada']].rename(columns={'complex_id':'station_complex_id','ada':'current_ada_code'}),on='station_complex_id',how='left')
    features.to_csv('res/supplementary_context.csv',index=False)


def prepare():
    os.makedirs('res',exist_ok=True)
    files=['../../data/station_hourly.csv.gz']
    import glob
    files+=sorted(glob.glob('../../data/extensions/*.csv.gz'))
    # Only July is needed to select stations; do not select using later test outcomes.
    july=pd.concat([pd.read_csv(f,dtype={'station_complex_id':str},parse_dates=['timestamp']) for f in files if '/extensions/' in f and '_07_' in f],ignore_index=True)
    summary=july.groupby(['borough','station_complex_id']).agg(n=('timestamp','size'),volume=('total_ridership','sum')).reset_index()
    selections=[]
    for borough in ['Bronx','Brooklyn','Manhattan','Queens']:
        candidates=summary[(summary.borough==borough)&(summary.n>=.98*744)].sort_values(['volume','station_complex_id'])
        selections.append(candidates.iloc[[len(candidates)//3,2*len(candidates)//3]])
    selected=pd.concat(selections).station_complex_id.tolist()
    stations=pd.read_csv('../../data/stations.csv',dtype={'station_complex_id':str})
    stations=stations[stations.station_complex_id.isin(selected)].copy().sort_values(['borough','station_complex_id'])
    stations['alias']=['site_'+str(i) for i in range(len(stations))];stations.to_csv('res/stations.csv',index=False)
    parts=[]
    for f in files:
        for chunk in pd.read_csv(f,dtype={'station_complex_id':str},parse_dates=['timestamp'],chunksize=100000):
            parts.append(chunk[chunk.station_complex_id.isin(selected)])
    x=pd.concat(parts,ignore_index=True).sort_values(['station_complex_id','timestamp'])
    weather=pd.read_csv('../../data/weather_hourly.csv',parse_dates=['timestamp']).sort_values('timestamp')
    weather['rain_6h_mm']=weather.rain_mm.rolling(6,min_periods=1).sum()
    grid=pd.MultiIndex.from_product([stations.station_complex_id,weather.timestamp],names=['station_complex_id','timestamp']).to_frame(index=False)
    x=grid.merge(x.drop(columns=['station_name','borough','latitude','longitude','rain_mm','temp_c','wind_ms','weather_available','weather_station']),on=['station_complex_id','timestamp'],how='left').merge(weather,on='timestamp').merge(stations,on='station_complex_id')
    for g,cols in CHANNELS.items():x[g]=x[cols].sum(axis=1,min_count=1)
    x['date']=x.timestamp.dt.normalize();x['weekday']=x.timestamp.dt.dayofweek;x['hour']=x.timestamp.dt.hour;x['block']=x.hour//4
    x['split']=np.select([x.date<'2023-08-01',x.date.isin(pd.to_datetime(VALIDATION)),x.date.isin(pd.to_datetime(TEST))],['train','validation','test'],default='history')
    daily_weather=weather.groupby(weather.timestamp.dt.normalize()).agg(rain_day_mm=('rain_mm','sum'),rain_max_mm=('rain_mm','max'),rain_6h_max_mm=('rain_6h_mm','max'))
    daily_weather['weather_class']=np.select([(daily_weather.rain_max_mm>=10)|(daily_weather.rain_6h_max_mm>=30),daily_weather.rain_day_mm>=1],['extreme','wet'],default='dry')
    x=x.merge(daily_weather,left_on='date',right_index=True)
    # Past same weekdays only: no current-day ridership enters any planned mass.
    for s,rows in x.groupby('station_complex_id',sort=False):
        values=rows.set_index('timestamp')[['total_ridership']+GROUPS]
        train=rows[rows.split=='train'].groupby(['weekday','hour'])[['total_ridership']+GROUPS].mean()
        prior=train.reindex(pd.MultiIndex.from_frame(rows[['weekday','hour']])).to_numpy()
        lags=np.stack([values.reindex(rows.timestamp-pd.Timedelta(days=d)).to_numpy() for d in [7,14,21,28]])
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);base=np.nanmedian(lags,axis=0)
        base=np.where(np.isnan(base),prior,base)
        # During early-July training, fallback profiles are fitted within training only.
        # Evaluation dates have at least four historical weeks.
        base=np.nan_to_num(base,nan=0.)
        group_mass=base[:,1:];denominator=group_mass.sum(axis=1)
        fractions=np.divide(group_mass,denominator[:,None],out=np.zeros_like(group_mass),where=denominator[:,None]>0)
        fractions[denominator==0,3]=1
        x.loc[rows.index,'planned_total']=base[:,0]
        for i,g in enumerate(GROUPS):x.loc[rows.index,'planned_'+g]=base[:,0]*fractions[:,i]
    x.to_csv('res/panel.csv.gz',index=False)
    daily=x.groupby(['date','weather_class','split']).agg(station_hours=('timestamp','size'),observed_hours=('total_ridership','count'),rain_mm=('rain_day_mm','first'),max_hour_rain_mm=('rain_max_mm','first'),max_6h_rain_mm=('rain_6h_max_mm','first')).reset_index()
    daily[daily.split.isin(['validation','test'])].to_csv('res/test_dates.csv',index=False)
    supplementary(stations)
    json.dump({'train':'2023-07-01 to 2023-07-31','validation':VALIDATION,'test':TEST,'selection':'July coverage >=98%, two volume tercile stations per borough','main_n_agents':8,'lite_model':'doubao-seed-2-1-lite-260915','token_cap':5000000,'max_authorized_cap':10000000,'weather_definition':'research threshold: hourly rain >=10mm or trailing six-hour rain >=30mm','forecast_task':'conditional on realized target-day weather; not an operational ex-ante forecast','supplementary':'2023 ACS released after 2023; 2026 current network/ADA only retrospective contextual probes, excluded from primary forecasts'},open('res/design.json','w'),indent=2)
    print('Selected',stations[['station_complex_id','station_name','borough']].to_dict('records'),flush=True)
    print('Evaluation dates',daily[daily.split.isin(['validation','test'])][['date','weather_class']].to_dict('records'),flush=True)


if __name__=='__main__':prepare()
