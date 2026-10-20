"""Shared-weather, frozen-Normal population, leave-one-event-out design."""
import json
import pandas as pd
import numpy as np
import api
from experiment import PEOPLE,examples
from timing import person_view,weather_view,hourly_weather

METHODS=['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few']
EVENTS={'Extreme_0729':'2023-07-29','Extreme_0813':'2023-08-13','Extreme_0929':'2023-09-29','Extreme_1007':'2023-10-07'}
SEASON={t:'Normal_Summer' if d<'2023-09-01' else 'Normal_School' for t,d in EVENTS.items()}
api.ROLE_SYSTEM=api.ROLE_SYSTEM.replace('Historical group entry indices, when supplied, are aggregate observations, NOT your past choices or action probabilities.','Historical group entry indices, when supplied, are aggregate observations, NOT your past choices or action probabilities. Event references may occur later than the target: this is retrospective leave-one-event-out generalization. The entire target event is withheld. Do not assume a reference-event demand change was caused only by rain.')
api.DIRECT_SYSTEM += '\nEvent references may occur later than the target: this is retrospective leave-one-event-out generalization. The entire target event is withheld. Do not assume reference-event changes are caused only by rain.'

def normal_history(p,season):return p[(p.task==season)&(p.split=='train')].copy()

def event_data(p,task):return p[(p.task==task)&(p.split=='test')].copy()

REFERENCE_CACHE={}
def references(p,weather,held,group=None,sid=None):
 key=(held,group,sid)
 if key in REFERENCE_CACHE:return REFERENCE_CACHE[key]
 refs=[]
 for task,event in EVENTS.items():
  if task==held:continue
  block=event_data(p,task);base=normal_history(p,SEASON[task]);records=[]
  if sid is not None:block=block[block.station_complex_id==sid];base=base[base.station_complex_id==sid]
  for date,g in block.groupby('date'):
   row=g.iloc[0];rec=dict(date=date,relative_day=int((pd.Timestamp(date)-pd.Timestamp(event)).days),weather=weather_view(row,weather,True))
   if sid is not None:
    rec['observed_entries']=round(float(row.total_ridership),1) if pd.notna(row.total_ridership) else None
   else:
    ratios=[]
    for _,r in g.iterrows():
     b=base[(base.station_complex_id==r.station_complex_id)&(base.weekday==r.weekday)][group].dropna()
     if b.empty:b=base[base.station_complex_id==r.station_complex_id][group].dropna()
     mean=b.mean()
     if pd.notna(r[group]) and pd.notna(mean) and mean>0:ratios.append(float(r[group])/mean)
    rec['observed_group_entries_relative_to_normal_weekday']=round(float(np.mean(ratios)),3) if ratios else None
    rec['contributing_stations']=len(ratios)
   records.append(rec)
  refs.append(dict(event_date=event,season=SEASON[task],days=records))
 REFERENCE_CACHE[key]=refs
 return refs

def role_view_for(p,weather,row,person,phase,few,held=None):
 season=phase if phase.startswith('Normal') else SEASON[phase]
 train=normal_history(p,season)
 if phase.startswith('Normal'):return person_view(person,row,train,weather,few,True)
 view=person_view(person,row,train,weather,False,True)
 if few:
  view['reference_events']=references(p,weather,held or phase,group=person['group'])
  view['history_definition']='Three OTHER complete events, days -2 to +2. Group entry ratios use each station and season Normal-training same-weekday mean, then average across stations. Ratio 1 means usual entries; null means a positive normal reference is unavailable. These are group observations, not personal action probabilities. Later reference events are allowed; the target event is excluded.'
 return view
