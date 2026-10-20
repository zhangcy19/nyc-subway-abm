"""Read the supplied ZIP and independently check all station-hour interpolations."""
import json,zipfile,hashlib
import numpy as np
import pandas as pd

STATIONS=['256','385']
FIELDS=['rain_mm','temp_c','wind_ms']
archive='../../../data/weather/subway_weather_hourly.zip'
z=zipfile.ZipFile(archive)
meta=json.load(open('../../../data/weather/station_inventory.json'))
gauges=[s for s in meta['stations'] if s.get('exported')]
codes=[s['station'] for s in gauges]
times=pd.date_range('2023-07-01','2023-11-01',freq='h',inclusive='left').strftime('%Y-%m-%d %H:%M:%S').tolist()
frames=[pd.read_csv(z.open(s['filename'])) for s in gauges]
for g in frames:
 if g.timestamp.tolist()!=times:raise ValueError('Gauge clock mismatch')
values=np.stack([g[FIELDS].to_numpy(float) for g in frames],axis=1)
stations=pd.read_csv('../../../data/stations.csv',dtype={'station_complex_id':str}).set_index('station_complex_id')
lat=np.radians([s['latitude'] for s in gauges]);lon=np.radians([s['longitude'] for s in gauges])
count=0;seen=set();errors=[];missing=[];selected=[];catalog=[]
for d in pd.read_csv(z.open('subway_weather_hourly.csv'),dtype={'station_complex_id':str},chunksize=2952):
 sid=d.station_complex_id.iloc[0]
 if sid in seen or d.station_complex_id.nunique()!=1 or d.timestamp.tolist()!=times:raise ValueError('Duplicate key or incomplete clock')
 seen.add(sid);s=stations.loc[sid]
 for c in ['station_name','borough','latitude','longitude']:
  if not d[c].eq(s[c]).all():raise ValueError('Station metadata mismatch')
 a=np.sin((lat-np.deg2rad(s.latitude))/2)**2+np.cos(lat)*np.cos(np.deg2rad(s.latitude))*np.sin((lon-np.deg2rad(s.longitude))/2)**2
 distances=2*6371.0088*np.arctan2(np.sqrt(a),np.sqrt(1-a));order=np.argsort(distances);rank=order[:3]
 if sid in STATIONS:catalog.append(dict(station_complex_id=sid,nearest=[dict(code=codes[i],km=float(distances[i])) for i in rank]))
 for vi,c in enumerate(FIELDS):
  source=values[:,:,vi];valid=np.isfinite(source);use=np.zeros_like(valid);use[:,rank]=valid[:,rank]
  fallback=(~use.any(axis=1))&valid.any(axis=1);best=order[np.argmax(valid[:,order],axis=1)];use[np.flatnonzero(fallback),best[fallback]]=True
  weight=use/distances**2;den=weight.sum(axis=1);weight=np.divide(weight,den[:,None],out=np.zeros_like(weight),where=den[:,None]>0)
  expected=(np.where(use,source,0)*weight).sum(axis=1);expected[den==0]=np.nan
  actual=d[c].to_numpy(float);np.testing.assert_allclose(actual,expected,atol=5.0001e-7,rtol=0,equal_nan=True)
  errors.append(float(np.nanmax(abs(actual-expected))))
 if not np.array_equal(d.weather_available.to_numpy(),d[FIELDS].notna().all(axis=1).to_numpy()):raise ValueError('Availability mismatch')
 if (d.rain_mm.dropna()<0).any() or (d.wind_ms.dropna()<0).any():raise ValueError('Negative rain/wind')
 missing.extend(d.loc[d.rain_mm.isna(),['station_complex_id','timestamp']].to_dict('records'))
 if sid in STATIONS:selected.append(d)
 count+=len(d)
 if len(seen)%100==0:print('Checked',len(seen),'stations',flush=True)
if seen!=set(stations.index):raise ValueError('Station coverage mismatch')
pd.concat(selected).to_csv('res/spatial_hourly.csv',index=False)
report=dict(status='passed_numerical_reconciliation',source_sha256=hashlib.sha256(open(archive,'rb').read()).hexdigest(),rows=count,stations=len(seen),rain_missing_rows=len(missing),missing_timestamps=sorted(set(x['timestamp'] for x in missing)),max_error=max(errors),selected_stations=catalog,limitations=['Spatial estimates from six gauges, not subway-site observations','Unflagged JRB rain remains potentially underestimated','METAR accumulation windows do not exactly coincide with clock hours','Original research splits retained; Normal/Extreme names refer to original JFK-based task selection'])
json.dump(report,open('res/weather_audit.json','w'),indent=2)
json.dump(STATIONS,open('res/station_ids.json','w'))
print(json.dumps(report,indent=2))
