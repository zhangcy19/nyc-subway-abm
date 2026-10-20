"""Summarize the known stadium-case diagnostic without altering main results."""
import json
import pandas as pd

folder='role_prompt_diagnosis_res'
old=pd.read_csv('side_studies_res/predictions.csv',dtype={'station_complex_id':str})
old=old[(old.study=='activity')&(old.method=='role_with_history')&(old.condition=='with_event')]
new=pd.read_csv(folder+'/predictions.csv',dtype={'station_complex_id':str})
p=pd.concat([old,new],ignore_index=True)
summary=[]
for (sid,condition,rep),g in p.groupby(['station_complex_id','condition','replicate']):
 summary.append(dict(station=sid,condition=condition,replicate=rep,score=100*(1-abs(g.predicted-g.actual).sum()/g.actual.sum())))
s=pd.DataFrame(summary)
s.to_csv(folder+'/comparison.csv',index=False)
daily=p.groupby(['station_complex_id','date','condition'])[['actual','predicted']].mean().reset_index()
daily.to_csv(folder+'/daily_comparison.csv',index=False)
rows=[json.loads(line) for line in open(folder+'/api_calls.jsonl')]
u=json.load(open(folder+'/usage.json'))
u.update(reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in rows),input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in rows),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in rows),valid_calls=sum(r['status']=='valid' for r in rows),token_cap=1300000)
u['estimated_cny']=u['input_tokens']*.8/1e6+u['output_tokens']*2.7/1e6
u['note']='Prompt-only diagnostic; 1.3M token cap within the existing monetary budget, no population refitting. Cost uses previous rate estimates, not an account receipt.'
json.dump(u,open(folder+'/usage.json','w'),indent=2)
# Verify new requests contain precisely the prior input views, with no target truth added.
prior=[json.loads(line) for line in open('side_studies_res/api_calls.jsonl')]
old_views={json.dumps(r['request']['view'],sort_keys=True) for r in prior if r.get('status')=='valid' and r['request']['route']=='role'}
audit=dict(expected_rows=18,actual_rows=len(new),duplicates=int(new.duplicated(['station_complex_id','date','replicate']).sum()),all_inputs_previously_used=all(json.dumps(r['request']['view'],sort_keys=True) in old_views for r in rows if r.get('status')=='valid'),new_valid_calls=u['valid_calls'])
json.dump(audit,open(folder+'/audit.json','w'),indent=2)
print(s.groupby(['station','condition']).score.agg(['mean','min','max']).to_string())
print(daily.to_string(index=False))
print(u)
print(audit)
