"""Shared-weather, frozen-Normal population, leave-one-event-out design."""
import sys,json
import pandas as pd
import numpy as np
sys.path.insert(0,'../fresh_lite')
import api
from experiment import PEOPLE,examples
from timing import person_view,weather_view,hourly_weather

METHODS=['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few']
EVENTS={'Extreme_0729':'2023-07-29','Extreme_0813':'2023-08-13','Extreme_0929':'2023-09-29','Extreme_1007':'2023-10-07'}
SEASON={t:'Normal_Summer' if d<'2023-09-01' else 'Normal_School' for t,d in EVENTS.items()}
api.ROLE_SYSTEM=api.ROLE_SYSTEM.replace('Historical group entry indices, when supplied, are aggregate observations, NOT your past choices or action probabilities.','Historical group entry indices, when supplied, are aggregate observations, NOT your past choices or action probabilities. Event references may occur later than the target: this is retrospective leave-one-event-out generalization. The entire target event is withheld. Do not assume a reference-event demand change was caused only by rain.')
api.DIRECT_SYSTEM += '\nEvent references may occur later than the target: this is retrospective leave-one-event-out generalization. The entire target event is withheld. Do not assume reference-event changes are caused only by rain.'

def inputs():
 ids=json.load(open('res/station_ids.json'))
 p=pd.read_csv('../fresh_lite/res/panel.csv',dtype={'station_complex_id':str});p=p[p.station_complex_id.isin(ids)&(p.task.str.startswith('Normal')|p.split.eq('test'))].copy()
 return p,hourly_weather()

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

def direct_view_for(p,weather,row,phase):
 season=phase if phase.startswith('Normal') else SEASON[phase];train=normal_history(p,season);train=train[(train.station_complex_id==row.station_complex_id)&train.total_ridership.notna()]
 if phase.startswith('Normal'):selected=examples(train,row,8)
 else:
  train=train.copy();train['distance']=abs(np.log1p(train.rain_mm)-np.log1p(row.rain_mm))+(train.weekday!=row.weekday)
  selected=train.sort_values(['distance','date'],ascending=[True,False]).head(4).sort_values('date')
 view=dict(target=weather_view(row,weather,True),observed_training_examples=[dict(input=weather_view(e,weather,True),entries=round(float(e.total_ridership),1)) for _,e in selected.iterrows()])
 if phase in EVENTS:view['reference_events']=references(p,weather,phase,sid=row.station_complex_id)
 return view

def jobs(p,weather,phase,method):
 part=p[p.task==phase];dates=part.drop_duplicates('date') if phase.startswith('Normal') else event_data(p,phase).drop_duplicates('date')
 if method=='direct_few':return [(r,None,direct_view_for(p,weather,r,phase)) for _,r in part[(part.split=='test')&part.total_ridership.notna()].iterrows()]
 return [(r,person,role_view_for(p,weather,r,person,phase,method=='role_with_history')) for _,r in dates.iterrows() for person in PEOPLE]

def prepare():
 p,w=inputs();p.to_csv('res/panel.csv',index=False)
 blocks=set(p[(p.task.isin(EVENTS))&(p.split=='test')].date)
 normals=p[p.task.str.startswith('Normal')]
 if blocks & set(normals.date):raise ValueError('Normal data overlaps event holdouts')
 rows=[]
 for held in EVENTS:
  for ref in references(p,w,held,group='general'):
   if ref['event_date']==EVENTS[held]:raise ValueError('Held event in references')
   rows.extend(dict(held_event=EVENTS[held],reference_event=ref['event_date'],reference_date=r['date']) for r in ref['days'])
 pd.DataFrame(rows).to_csv('res/fold_reference_dates.csv',index=False)
 p.groupby(['task','split']).agg(days=('date','nunique'),station_days=('date','size')).reset_index().to_csv('res/dataset_summary.csv',index=False)
 print('Prepared two stations; Normal and all four full event windows disjoint.')

if __name__=='__main__':prepare()
