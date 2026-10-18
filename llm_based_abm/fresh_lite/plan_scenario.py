"""A separate Lite planning conversation grounded in numerical scenario outputs."""
import json,time,hashlib,os
import pandas as pd
from openai import OpenAI
import supplement_api as api
outfile='../../../presentation_local/build/evidence/planning_private.json'
os.makedirs('../../../presentation_local/build/evidence',exist_ok=True)
if os.path.exists(outfile):
 print('Planning response already saved; no new call');raise SystemExit
pred=pd.read_csv('res/supplement/scenario_predictions.csv').drop(columns='actual').to_dict('records');groups=pd.read_csv('res/supplement/scenario_decisions.csv');summary=[]
for (scenario,group),p in groups.groupby(['scenario','group']):
 summary.append(dict(scenario=scenario,group=group,subway_entries=float((p.effective_population*p.p_subway).sum()),stay_home=float((p.effective_population*p.p_stay_home).sum())))
system='You are a transport planning analyst. Return JSON only: {"headline":string,"findings":[three brief strings],"recommendations":[{"action":string,"evidence":string,"condition":string}],"limits":[strings]}. Give 3 practical conditional recommendations, not chain-of-thought. Use only supplied evidence. Never assert actual station closures, flooding, capacity shortfalls, observed OD or verified forecasts. Do not convert simulated journey opportunities into unique people. No operational deployment is authorized.'
view=dict(scenario_date='2026-10-23',assumptions='HYPOTHETICAL 80 mm rain, peak 20 mm/h, concentrated 06:00–14:00; dry matched control. 2023 fitted demand scale carried to 2026, no growth adjustment. Shared role probabilities; no closures or route disruptions modeled.',station_predictions=pred,group_outcomes=summary,station_names=pd.read_csv('res/stations.csv').to_dict('records'))
prompt=json.dumps(view);reserve=len((system+prompt).encode())+1600
if sum(api.charge(r) for r in api.ledger())+reserve>api.CAP:raise RuntimeError('Supplement cap')
key=open('../key.txt').read().strip().strip('\"\'');start=time.perf_counter();row=dict(hash=hashlib.sha256(prompt.encode()).hexdigest(),request=dict(route='planner',model=api.MODEL,system=system,view=view))
try:
 with OpenAI(api_key=key,base_url='https://ark.cn-beijing.volces.com/api/v3',timeout=90,max_retries=0) as c:
  r=c.responses.create(model=api.MODEL,input=[dict(role='system',content=system),dict(role='user',content=prompt)],temperature=.3,max_output_tokens=1200,extra_body={'thinking':{'type':'disabled'}})
 row.update(raw=r.output_text,usage=r.usage.model_dump(),response_id=r.id,status='valid');parsed=json.loads(r.output_text.strip().removeprefix('```json').removesuffix('```').strip());row['parsed']=parsed
 json.dump(dict(input=view,output=parsed),open(outfile,'w'),indent=2)
except Exception as e:
 row.update(status='error',error=type(e).__name__)
 if 'usage' not in row:row['charged_tokens']=reserve
 raise
finally:
 row['elapsed_seconds']=time.perf_counter()-start
 with open('res/supplement/api_calls.jsonl','a') as f:f.write(json.dumps(row)+'\n')
print('Planning completed; all supplemental charges',sum(api.charge(r) for r in api.ledger())+api.charge(row))
