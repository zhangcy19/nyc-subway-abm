"""Offline regression: reference replay on three stations; no model requests."""
import json
import numpy as np
import pandas as pd
import run
from model import fit_population,predict
from timing import timed_rule_policy
from experiment import PEOPLE
from design import inputs,normal_history,SEASON

ids=['103','119','120']
ids=json.load(open('inputs/station_ids.json'))[:3]
p,w=inputs(ids)
reference=pd.concat([pd.read_csv('results/'+folder+'/predictions.csv',dtype={'station_complex_id':str}) for folder in ['shared','doubao_lite_reference']])
policies=pd.read_csv('results/doubao_lite_reference/policies.csv')
cfg=pd.read_csv('inputs/frozen_parameters.csv').set_index(['task','method']).to_dict('index')
weights={};diffs=[]
for phase in run.PHASES:
    season=phase if phase.startswith('Normal') else SEASON[phase]
    part=p[p.task==phase];train=normal_history(p,season);test=part[part.split=='test']
    for method in ['role_no_history','role_with_history','rule_daily','rule_timing']:
        if method.startswith('role'):
            z=policies[(policies.task==phase)&(policies.method==method)]
            policy={(r.date,r.agent_id):dict(probabilities=[getattr(r,k) for k in ['p_subway','p_bus','p_taxi','p_walk','p_stay_home']],hash='',reason='offline replay') for _,r in z.iterrows()}
        else:policy=timed_rule_policy(part.drop_duplicates('date'),w,cfg[(phase,method)]['rain_scale'],method=='rule_timing')
        if phase.startswith('Normal'):
            c=cfg[(phase,method)];weights[(phase,method)]=fit_population(train,policy,PEOPLE,c['shrink'],c['calendar_prior'])
        rows,_=predict(test,weights[(season,method)],policy,PEOPLE)
        got=pd.DataFrame(rows).merge(reference[(reference.task==phase)&(reference.method==method)],on=['station_complex_id','date'],suffixes=('_new','_old'),validate='one_to_one')
        delta=abs(got.predicted_new-got.predicted_old).max();diffs.append(float(delta))
        if len(got)!=len(test) or not np.allclose(got.predicted_new,got.predicted_old,rtol=1e-6,atol=.01):raise ValueError((phase,method,delta))
# Held event never enters role references; ordinary histories precede each target.
donors=pd.read_csv('inputs/role_donors.csv',dtype={'station_complex_id':str})
for phase in run.PHASES:
    for _,r in donors[donors.task==phase].drop_duplicates('date').iterrows():
        view=run.role_history.role_view_for(donors,w,r,PEOPLE[0],phase,True)
        if phase in run.EVENTS and any(x['event_date']==run.EVENTS[phase] for x in view['reference_events']):raise ValueError('Held-event leak')
for folder in ['shared','doubao_lite_reference']:run.summarize('results/'+folder)
s=pd.concat([pd.read_csv('results/'+f+'/summary.csv') for f in ['shared','doubao_lite_reference']]).set_index('method')
expected={'weekday_mean':(91.71,79.20),'ridge':(91.15,85.79),'rule_daily':(92.07,84.41),'rule_timing':(91.69,58.08),'direct_few':(92.38,84.34),'role_no_history':(87.71,78.86),'role_with_history':(91.74,86.29)}
for m,(n,e) in expected.items():
    if abs(s.loc[m,'Normal']-n)>.011 or abs(s.loc[m,'Extreme']-e)>.011:raise ValueError('Reference score drift: '+m)
print('PASS: 3 stations, 6 tasks, 4 ABM variants; maximum prediction difference',max(diffs))
print('PASS: all 152-station headline scores reproduced; held-event isolation checked.')
