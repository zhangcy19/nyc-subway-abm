"""Summarize pilot evidence without making API calls."""
import json
import numpy as np
import pandas as pd
from design import EVENTS

def score(g):return 100*(1-abs(g.predicted-g.actual).sum()/g.actual.sum())
def main():
 old=pd.read_csv('../event_holdout/res/predictions.csv',dtype={'station_complex_id':str})
 dates=pd.read_csv('prepared/dates.csv');rows=[]
 rules=pd.read_csv('pilot_res/rule_predictions.csv')
 for (version,task),g in rules.groupby(['version','task']):
  center=g if task.startswith('Normal') else g[g.date==EVENTS[task]]
  rows.append(dict(method='rule_timing',arm=version,task=task,center_score=score(center),window_score=score(g)))
 new=pd.read_csv('pilot_res/role_predictions.csv',dtype={'station_complex_id':str})
 for arm,g in [('old',old[(old.task=='Extreme_1007')&(old.method=='role_with_history')]),('new',new)]:
  rows.append(dict(method='role_with_history',arm=arm,task='Extreme_1007',center_score=score(g[g.date=='2023-10-07']),window_score=score(g)))
 direct=pd.read_csv('pilot_res/direct_predictions.csv',dtype={'station_complex_id':str})
 baseline=old[(old.task=='Extreme_1007')&(old.method=='direct_few')].copy();baseline['arm']='old_single_full'
 direct=pd.concat([direct,baseline],ignore_index=True)
 ledger=[json.loads(s) for s in open('pilot_res/api_calls.jsonl')];old_ledger=[json.loads(s) for s in open('../event_holdout/res/api_calls.jsonl')]
 lookup={r['hash']:r for r in old_ledger+ledger}
 for arm,g in direct.groupby('arm'):
  hashes=set(g.request_hash);tokens=sum(lookup[h]['usage']['total_tokens'] for h in hashes)
  rows.append(dict(method='direct_few',arm=arm,task='Extreme_1007',center_score=score(g[g.date=='2023-10-07']),window_score=score(g),valid_reply_tokens=tokens,valid_calls=len(hashes)))
 summary=pd.DataFrame(rows);summary.to_csv('pilot_res/comparison.csv',index=False)
 probe=pd.read_csv('pilot_res/context_probe.csv').pivot(index=['station_complex_id','date'],columns='context',values='predicted').reset_index();probe['change_pct']=100*(probe.event/probe.normal-1);probe.to_csv('pilot_res/context_probe_comparison.csv',index=False)
 print(summary.round(3).to_string(index=False));print(probe.to_string(index=False))
 print('API',json.load(open('pilot_res/usage.json')))

if __name__=='__main__':main()
