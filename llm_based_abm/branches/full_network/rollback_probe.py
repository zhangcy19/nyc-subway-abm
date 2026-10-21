"""Development-only rollback replay. No API calls; preserves every original result.
Old role decisions are shared strategies built from the original two donor stations.
Direct LLM comparator retains the existing full-network predictions, not an old rerun.
"""
import os,json,hashlib
import pandas as pd
import run as runner
import api
from design import inputs,METHODS
from staged import metrics

order=pd.read_csv('staged_res/batches.csv',dtype={'station_complex_id':str}).sort_values('position')
ids=order.station_complex_id.head(12).tolist()
runner.OUT='rollback_dev_res';os.makedirs(runner.OUT,exist_ok=True)
runner.CONFIG=dict(station_ids=ids,methods=METHODS,revised_rule=False)
api.LEDGER=[]
api.request=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('This diagnostic forbids API calls'))

def old_policy(p,w,phase,method,offline):
 records=json.load(open(f'../event_holdout/res/{phase}_{method}_policy.json'))
 return {(r['date'],r['agent_id']):dict(probabilities=r['probabilities'],hash=r['hash'],reason='Original event_holdout cached response') for r in records}
existing=pd.read_csv('staged_res/predictions.csv',dtype={'station_complex_id':str})
def direct(p,w,phase,offline):
 return existing[(existing.task==phase)&existing.method.eq('direct_few')&existing.station_complex_id.isin(ids)].drop(columns=['task','method']).to_dict('records')
runner.policy_calls=old_policy;runner.direct_calls=direct
json.dump(dict(station_ids=ids,stage='development',api_calls=0,role_policy_source='../event_holdout/res',direct_source='staged_res/predictions.csv',extreme_population='seasonal Normal only',rule_formula='original',note='Previously evaluated dates and stations: development replay, not independent new validation',script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest()),open(runner.OUT+'/protocol.json','w'),indent=2)
runner.run(offline=True)
p,w=inputs(ids);out=pd.read_csv(runner.OUT+'/predictions.csv',dtype={'station_complex_id':str});out=out.merge(p[['task','station_complex_id','date','event_day']],on=['task','station_complex_id','date'],validate='many_to_one');out['regime']=out.task.str.startswith('Normal').map({True:'Normal',False:'Extreme'})
t=metrics(out[(out.regime=='Normal')|out.event_day]);t.to_csv(runner.OUT+'/task_scores.csv',index=False)
s=t.groupby(['regime','method']).score.mean().unstack('regime');s.to_csv(runner.OUT+'/summary.csv')
gates={r:dict(few_beats_zero=bool(s.loc['role_with_history',r]>s.loc['role_no_history',r]),llm_is_best=bool(s.loc[['direct_few','role_no_history','role_with_history'],r].max()>=s[r].max()-1e-9)) for r in ['Normal','Extreme']}
json.dump(gates,open(runner.OUT+'/gate.json','w'),indent=2)
print(s.round(3).to_string());print(gates)
