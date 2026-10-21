"""Exploratory prompt-only test. No target ridership is sent to the API."""
import json, os
import pandas as pd
import side_studies as s
import api

s.OUT='role_prompt_diagnosis_res'
os.makedirs(s.OUT,exist_ok=True)
api.LOGDIR=s.OUT
api.CAP=1300000
api.LEDGER=None
api.ROLE_SYSTEM += '''
Additional decision guidance:
Evaluate whether the supplied local event information changes YOUR need for a journey, separately from mode preference. A home game is neither a universal trip requirement nor a reason for every traveler to stay home when absent. Apply event relevance to the supplied persona's stated purpose; do not invent a fan identity, ticket, work shift, destination, or attendance rate. Preserve unrelated regular trips.
Compare supplied historical examples with and without games when available, while considering weekday and weather differences. A single observed Sunday cannot identify a usual non-game Sunday. An aggregate index of 1 is a within-weekday reference, not evidence of equal absolute demand across weekdays or of personal travel certainty.
Keep the original five outcomes and their meanings. Do not reinterpret subway as probability of visiting one station, or stay_home as using another station. Where event context is supplied, briefly state whether it affects your stated purpose in the final decision summary. Do not force an increase or decrease in probabilities.'''
jobs=[j for j in s.build() if j['study']=='activity' and j['method']=='role_with_history' and j['condition']=='with_event']
for j in jobs:j['condition']='event_reason_prompt'
json.dump(dict(kind='Exploratory known-case diagnostic, not independent evaluation',stations=s.IDS,dates=['2023-10-06','2023-10-07','2023-10-08'],replicates=s.REPS,requests=len(jobs),change='Only system prompt; inputs, populations, action meanings unchanged',baseline='side_studies_res, role_with_history, with_event; earlier two independent draws',system=api.ROLE_SYSTEM),open(s.OUT+'/protocol.json','w'),indent=2)
s.execute(jobs,{j['key']:j for j in jobs})
usage=json.load(open(s.OUT+'/usage.json'))
usage.pop('cap_adjustment',None)
usage['token_cap']=api.CAP
usage['note']='Prompt-only diagnostic; 1.3M token cap within the existing monetary budget, no population refitting.'
json.dump(usage,open(s.OUT+'/usage.json','w'),indent=2)
old=pd.read_csv('side_studies_res/predictions.csv',dtype={'station_complex_id':str})
old=old[(old.study=='activity')&(old.method=='role_with_history')&(old.condition=='with_event')].copy()
new=pd.read_csv(s.OUT+'/predictions.csv',dtype={'station_complex_id':str})
all_rows=pd.concat([old,new],ignore_index=True)
summary=[]
for (station,condition,rep),g in all_rows.groupby(['station_complex_id','condition','replicate']):
 summary.append(dict(station=station,condition=condition,replicate=rep,score=100*(1-abs(g.predicted-g.actual).sum()/g.actual.sum())))
pd.DataFrame(summary).to_csv(s.OUT+'/comparison.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
print(all_rows.groupby(['station_complex_id','date','condition'])[['actual','predicted']].mean().to_string())
