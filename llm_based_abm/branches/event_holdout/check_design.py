"""Input leakage checks; no API calls."""
import json
import pandas as pd
import design as d
p,w=d.inputs();blocked=set(p[p.task.isin(d.EVENTS)].date)
if blocked & set(p[p.task.str.startswith('Normal')].date):raise ValueError('Normal overlap')
for phase in d.EVENTS:
 target=d.event_data(p,phase);held=set(target.date)
 for person in d.PEOPLE:
  v=d.role_view_for(p,w,target.iloc[0],person,phase,True)
  seen={r['date'] for event in v['reference_events'] for r in event['days']}
  if seen & held or len(v['reference_events'])!=3:raise ValueError('Invalid event references')
  json.dumps(v,allow_nan=False)
 # Replacing ALL held-event outcome labels must not affect any prompt.
 copy=p.copy();mask=copy.task.eq(phase)
 copy.loc[mask,['total_ridership','students','fair_fare','seniors_disability','general']]=99999999
 for person in [d.PEOPLE[0],d.PEOPLE[-1]]:
  d.REFERENCE_CACHE.clear();a=d.role_view_for(p,w,target.iloc[0],person,phase,True)
  d.REFERENCE_CACHE.clear();b=d.role_view_for(copy,w,target.iloc[0],person,phase,True)
  if a!=b:raise ValueError('Target labels leaked into role input')
 d.REFERENCE_CACHE.clear();a=d.direct_view_for(p,w,target.iloc[0],phase)
 d.REFERENCE_CACHE.clear();b=d.direct_view_for(copy,w,target.iloc[0],phase)
 if a!=b:raise ValueError('Target labels leaked into direct input')
print('PASS: all Normal dates disjoint from event windows; three reference events per fold; held-event label perturbation leaves inputs unchanged.')
