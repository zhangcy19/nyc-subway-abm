"""Independent input checks; no API calls."""
import json
import pandas as pd
from side_studies import build,EVENTS,P
jobs=build();held_errors=[];bad_lengths=[];event_errors=[]
for j in jobs:
 v=j['view']
 if j['study']=='rain':
  held={str(d.date()) for d in pd.date_range(pd.Timestamp(EVENTS[j['task']])-pd.Timedelta(days=2),pd.Timestamp(EVENTS[j['task']])+pd.Timedelta(days=2))}
  historic=[r['date'] for e in v.get('reference_events',[]) for r in e['days']]+[r['input']['date'] for r in v.get('observed_training_examples',[])]
  if held & set(historic):held_errors.append(j['key'])
 else:
  historic=[r['input']['date'] for r in v.get('observed_training_examples',[])]
  if set(historic)&{'2023-10-05','2023-10-06','2023-10-07','2023-10-08','2023-10-09'}:held_errors.append(j['key'])
  if j['condition']=='with_event' and v['target']['mets_calendar']['mets_home_game_on_date']:event_errors.append(j['key'])
 def walk(a):
  if isinstance(a,dict):
   for k,b in a.items():
    if k=='hourly_rain_mm_00_to_23' and len(b)!=24:bad_lengths.append(j['key'])
    walk(b)
  elif isinstance(a,list):
   for b in a:walk(b)
 walk(v)
report=dict(station_agent_request_rows=len(jobs),held_event_history_violations=len(held_errors),invalid_hourly_array_lengths=len(bad_lengths),incorrect_target_game_flags=len(event_errors),status='passed' if not(held_errors or bad_lengths or event_errors) else 'failed')
json.dump(report,open('side_studies_res/input_audit.json','w'),indent=2);print(report)
