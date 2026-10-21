"""Measure numerical replay locally; all network request methods are blocked."""
import os,sys,json,time,platform
from unittest.mock import patch
import pandas as pd
import numpy as np
os.chdir('mainline');sys.path.insert(0,'.')
import run
p=pd.read_csv('results/doubao_lite_reference/predictions.csv',dtype={'station_complex_id':str})
pol=pd.read_csv('results/doubao_lite_reference/policies.csv')
def policy(d,w,phase,method,*args):
 g=pol[(pol.task==phase)&(pol.method==method)]
 return {(r.date,r.agent_id):dict(probabilities=[r[k] for k in ['p_subway','p_bus','p_taxi','p_walk','p_stay_home']],hash='',reason='Saved numeric policy') for _,r in g.iterrows()}
def direct(frame,w,phase,*args):
 return p[(p.task==phase)&(p.method=='direct_few')].drop(columns=['task','method']).to_dict('records')
start=time.perf_counter()
with patch.object(run.api,'request',side_effect=RuntimeError('Network disabled for plotting benchmark')),patch.object(run.api,'ledger',return_value=[]):
 out='../figures/private/offline_benchmark_baselines';os.makedirs(out,exist_ok=True);run.execute({},out,baseline=True)
 out2='../figures/private/offline_benchmark_llm';os.makedirs(out2,exist_ok=True)
 with patch.object(run,'get_policy',side_effect=policy),patch.object(run,'direct',side_effect=direct):
  run.execute(dict(workers=1),out2,offline=True)
 got=pd.read_csv(out2+'/predictions.csv',dtype={'station_complex_id':str})
 joined=got.merge(p,on=['task','method','station_complex_id','date'],suffixes=('_new','_accepted'),validate='one_to_one')
 difference=float((joined.predicted_new-joined.predicted_accepted).abs().max())
 if len(joined)!=15504 or difference>1e-6:raise ValueError('Offline benchmark does not reproduce published predictions')
 runtime=pd.concat([pd.read_csv(out+'/runtime.csv'),pd.read_csv(out2+'/runtime.csv')],ignore_index=True)
 runtime.to_csv('../figures/data/offline_compute_runtime.csv',index=False)
 json.dump(dict(machine=platform.platform(),processor=platform.machine(),cpu_count=os.cpu_count(),elapsed_seconds=time.perf_counter()-start,api_calls=0,mainline_prediction_max_abs_difference=difference,note='One local numerical replay. Final chart adds saved API service seconds, a serial-equivalent work estimate rather than original parallel wall-clock time.'),open('../figures/data/runtime_provenance.json','w'),indent=2)
print('Offline benchmark finished; prediction max difference',difference,flush=True)
