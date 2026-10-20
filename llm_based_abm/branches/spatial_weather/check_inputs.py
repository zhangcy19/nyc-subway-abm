"""Offline checks of local weather joins and historical information boundaries."""
import json
import numpy as np
import run_spatial as r
p,h,w=r.inputs();count=0
for (sid,task),part in p.groupby(['station_complex_id','task']):
 train=part[part.split=='train']
 if train.date.max()>=part[part.split=='validation'].date.min():raise ValueError('Invalid split chronology')
 if part[part.split=='validation'].date.max()>=part[part.split=='test'].date.min():raise ValueError('Invalid test chronology')
 for _,row in part.iterrows():
  if not np.isclose(row.rain_mm,sum(w[sid][row.date])):raise ValueError('Weather mismatch')
 for method in ['role_no_history','role_with_history','direct_few']:
  for row,person,v in r.views(part,h,w[sid],method):
   json.dumps(v,allow_nan=False)
   if not np.allclose(v['target']['hourly_rain_mm_00_to_23'],np.round(w[sid][row.date],3)):raise ValueError('Target weather mismatch')
   if method=='role_no_history' and 'group_history' in v:raise ValueError('Zero-shot history')
   if method=='direct_few' and len(v['observed_training_examples'])>8:raise ValueError('Too many examples')
   count+=1
print('PASS:',count,'views; chronological splits, local weather joins, finite payloads, zero-shot separation')
