"""Offline design and API-schema checks. Never calls the network."""
import json
import numpy as np
import pandas as pd
import api
import design as d
from timing import timed_rule_policy,revised_rule_policy

def verify():
 p,w=d.inputs();ids=sorted(p.station_complex_id.unique())
 if len(ids)!=len(json.load(open('prepared/station_ids.json'))):raise ValueError('Wrong cohort')
 normal=p[p.task.str.startswith('Normal')];event=p[p.task.isin(d.EVENTS)]
 if set(normal.date)&set(event.date):raise ValueError('Normal/event date overlap')
 if p.duplicated(['task','station_complex_id','date']).any():raise ValueError('Duplicate panel keys')
 if not np.allclose(p[d.GROUPS].sum(axis=1),p.total_ridership):raise ValueError('Group sum mismatch')
 # Label perturbation invariance on all four held events.
 for phase in d.EVENTS:
  target=d.event_data(p,phase);held=set(target.date);copy=p.copy();copy.loc[copy.task==phase,['total_ridership']+d.GROUPS]=99999999
  for person in d.PEOPLE[::3]:
   row=target.iloc[0];d.clear_cache();a=d.role_view_for(p,w,row,person,phase,True)
   d.clear_cache();b=d.role_view_for(copy,w,row,person,phase,True)
   if a!=b:raise ValueError('Held outcome leaks into role input')
   if any(r['date'] in held for e in a['reference_events'] for r in e['days']):raise ValueError('Held date in references')
  g=target[target.station_complex_id==ids[0]];d.clear_cache();a=d.batch_view(p,w,g,phase);d.clear_cache();b=d.batch_view(copy,w,g,phase)
  if a!=b:raise ValueError('Held outcome leaks into direct batch')
 # Normal test labels must not affect normal histories or event histories.
 changed=p.copy();changed.loc[changed.task.str.startswith('Normal')&changed.split.eq('test'),d.GROUPS+['total_ridership']]=88888888
 row=normal[normal.split=='test'].iloc[0]
 d.clear_cache();a=d.role_view_for(p,w,row,d.PEOPLE[0],row.task,True)
 d.clear_cache();b=d.role_view_for(changed,w,row,d.PEOPLE[0],row.task,True)
 if a!=b:raise ValueError('Normal test labels leak')
 # Hand-check the pooling definition: (10+100)/(5+100), not (2+1)/2.
 base=pd.DataFrame([dict(station_complex_id='a',weekday=0,students=5),dict(station_complex_id='b',weekday=0,students=100)])
 block=base.copy();block.students=[10,100]
 if d.group_index(block,base,'students')[0]!=round(110/105,3):raise ValueError('Pooling definition')
 dry=normal[normal.rain_mm==0].drop_duplicates('date').head(1)
 old=timed_rule_policy(dry,w,1,True);new=revised_rule_policy(dry,w,1,True)
 if any(not np.allclose(old[k]['probabilities'],new[k]['probabilities']) for k in old):raise ValueError('Dry utility changed')
 valid=api.validate({'predictions':[{'date':'2023-10-07','entries':123}],'reason':'test'},'batch')
 try:api.validate({'predictions':[{'date':'2023-10-07','entries':-1}],'reason':'test'},'batch')
 except ValueError:pass
 else:raise ValueError('Negative batch prediction accepted')
 d.clear_cache();print('PASS:',len(ids),'stations,',len(p),'complete station-days; group sums, event/Normal label isolation, weighted indices, dry-utility identity, batch schema.')

if __name__=='__main__':verify()
