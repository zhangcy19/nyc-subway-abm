"""Select four new stations using only the earliest event fold's training dates."""
import glob,json
import pandas as pd
parts=[]
for f in sorted(glob.glob('../../../data/extensions/station_hourly_2023_07_*.csv.gz')):
    if not any(f.endswith('2023_07_'+d+'.csv.gz') for d in ['01','08','15']):continue
    for x in pd.read_csv(f,usecols=['timestamp','station_complex_id','total_ridership'],dtype={'station_complex_id':str},chunksize=100000):
        x=x[(x.timestamp>='2023-07-01')&(x.timestamp<'2023-07-20')];parts.append(x)
x=pd.concat(parts);summary=x.groupby('station_complex_id').total_ridership.agg(['count','mean']).reset_index()
stations=pd.read_csv('../../../data/stations.csv',dtype={'station_complex_id':str});s=stations.merge(summary);s=s[(s['count']>=19*24*.98)&~s.station_complex_id.isin(['360','37','228','269'])]
chosen=[]
for borough in ['Bronx','Brooklyn','Manhattan','Queens']:
    p=s[s.borough==borough].sort_values(['mean','station_complex_id']);chosen.append(p.iloc[(len(p)-1)//2])
chosen=pd.DataFrame(chosen);chosen.to_csv('res/selected_stations.csv',index=False)
json.dump(chosen.station_complex_id.tolist(),open('res/station_ids.json','w'))
print(chosen.to_string(index=False))
