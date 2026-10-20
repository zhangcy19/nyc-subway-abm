"""One shared-weather protocol; weighted group histories; no held-event labels."""
import json
import numpy as np
import pandas as pd
import api
from prepare import EVENTS,GROUPS
from experiment import PEOPLE,examples
from timing import person_view,weather_view,hourly_weather
METHODS=['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few']
SEASON={t:'Normal_Summer' if d<'2023-09-01' else 'Normal_School' for t,d in EVENTS.items()}
HISTORY_DEFINITION='Index = sum of observed entries in this fare group / sum of each contributing station\'s Normal-training same-weekday mean. If that weekday has no training data, use the station\'s available Normal-training mean. Index 1 means usual same-weekday entries. These are aggregate observations, not personal choice probabilities. Normal and event records use this SAME definition.'

def inputs(ids=None):
 if ids is None:ids=json.load(open('inputs/station_ids.json'))
 daily=pd.read_csv('inputs/daily.csv',dtype={'station_complex_id':str});dates=pd.read_csv('inputs/dates.csv')
 p=dates.merge(daily[daily.station_complex_id.isin(ids)],on='date')
 if p.total_ridership.isna().any() or p.observed_hours.ne(24).any():raise ValueError('Selected cohort contains incomplete days')
 if set(p.station_complex_id)!=set(ids):raise ValueError('Missing selected station')
 return p,hourly_weather()

def normal_history(p,season):return p[(p.task==season)&(p.split=='train')].copy()
def event_data(p,task):return p[(p.task==task)&(p.split=='test')].copy()

def compact_weather(row,w):
 v=weather_view(row,w,False);rain=w[row.date]
 v['rain_mm_by_6h_00_06_12_18']=[round(float(rain[i:i+6].sum()),3) for i in range(0,24,6)]
 v['peak_hour_local']=int(np.argmax(rain)) if rain.max()>0 else None
 v['wet_hours_local']=np.flatnonzero(rain>0).tolist()
 return v

def group_index(block,base,group):
 obs=0.;expected=0.;n=0
 for _,r in block.iterrows():
  b=base[base.station_complex_id==r.station_complex_id];same=b[b.weekday==r.weekday]
  mean=(same if len(same) else b)[group].mean()
  if pd.notna(r[group]) and pd.notna(mean) and mean>0:
   obs+=float(r[group]);expected+=float(mean);n+=1
 return (round(obs/expected,3) if expected>0 else None),n

REF_CACHE={}
NORMAL_CACHE={}
def clear_cache():REF_CACHE.clear();NORMAL_CACHE.clear()

def normal_records(p,w,row,season,group):
 key=(season,row.date,group)
 if key in NORMAL_CACHE:return NORMAL_CACHE[key]
 base=normal_history(p,season)
 # No future Normal observations for a Normal target; retrospective seasonal
 # calibration is allowed for an event target, including the July holdout.
 if str(row.task).startswith('Normal'):base=base[base.date<row.date]
 ds=base.drop_duplicates('date').copy()
 if ds.empty:return []
 ds['distance']=abs(np.log1p(ds.rain_mm)-np.log1p(row.rain_mm))+(ds.weekday!=row.weekday)
 selected=ds.sort_values(['distance','date'],ascending=[True,False]).head(4).sort_values('date')
 result=[]
 for _,r in selected.iterrows():
  ratio,n=group_index(base[base.date==r.date],base,group)
  result.append(dict(date=r.date,weather=compact_weather(r,w),group_entry_index=ratio,contributing_stations=n))
 NORMAL_CACHE[key]=result
 return result

def references(p,w,held,group=None,sid=None,compact=True):
 key=(held,group,sid,compact)
 if key in REF_CACHE:return REF_CACHE[key]
 refs=[]
 for task,event in EVENTS.items():
  if task==held:continue
  block=event_data(p,task);base=normal_history(p,SEASON[task])
  if sid is not None:block=block[block.station_complex_id==sid]
  records=[]
  for date,g in block.groupby('date'):
   r=g.iloc[0];rec=dict(date=date,relative_day=int((pd.Timestamp(date)-pd.Timestamp(event)).days),weather=compact_weather(r,w) if compact else weather_view(r,w,True))
   if sid is None:rec['group_entry_index'],rec['contributing_stations']=group_index(g,base,group)
   else:rec['observed_entries']=round(float(r.total_ridership),1)
   records.append(rec)
  refs.append(dict(event_date=event,season=SEASON[task],days=records))
 REF_CACHE[key]=refs
 return refs

def role_view_for(p,w,row,person,phase,few):
 season=phase if phase.startswith('Normal') else SEASON[phase]
 v=person_view(person,row,normal_history(p,season),w,False,True)
 if few:
  v['normal_reference']=normal_records(p,w,row,season,person['group'])
  v['history_definition']=HISTORY_DEFINITION
  if phase in EVENTS:v['reference_events']=references(p,w,phase,group=person['group'])
 return v

def direct_view_for(p,w,row,phase,compact=True):
 season=phase if phase.startswith('Normal') else SEASON[phase]
 train=normal_history(p,season);train=train[train.station_complex_id==row.station_complex_id]
 if phase.startswith('Normal'):selected=examples(train,row,8)
 else:
  train=train.copy();train['distance']=abs(np.log1p(train.rain_mm)-np.log1p(row.rain_mm))+(train.weekday!=row.weekday)
  selected=train.sort_values(['distance','date'],ascending=[True,False]).head(4).sort_values('date')
 v=dict(target=weather_view(row,w,True),observed_training_examples=[dict(input=compact_weather(r,w) if compact else weather_view(r,w,True),entries=round(float(r.total_ridership),1)) for _,r in selected.iterrows()])
 if phase in EVENTS:v['reference_events']=references(p,w,phase,sid=row.station_complex_id,compact=compact)
 return v

def batch_view(p,w,targets,phase,compact=True):
 center=targets[targets.event_day].iloc[0]
 v=direct_view_for(p,w,center,phase,compact);v.pop('target')
 v['targets']=[dict(date=r.date,**weather_view(r,w,True)) for _,r in targets.sort_values('date').iterrows()]
 return v
