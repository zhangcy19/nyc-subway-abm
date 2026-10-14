"""Offline checks of weekly structure and persona concentration."""
import json
import numpy as np
import pandas as pd
from model import predict, profiles

def metrics(p):
    p=p.dropna(subset=['actual','predicted']);e=p.predicted-p.actual
    return dict(n=len(p),actual=p.actual.sum(),predicted=p.predicted.sum(),WAPE_pct=100*abs(e).sum()/p.actual.sum(),bias_pct=100*e.sum()/p.actual.sum())

def run():
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str})
    weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str})
    rows=[]
    for (task,method),w in weights.groupby(['task','method']):
        from experiment import load_policy
        policy=load_policy(task,method)
        part=panel[panel.task==task]
        pred,_=predict(part,w.to_dict('records'),policy,profiles())
        p=pd.DataFrame(pred).merge(part[['date','station_complex_id','split','weekday']],on=['date','station_complex_id'])
        for (split,day),r in p.groupby(['split','weekday']):
            rows.append(dict(task=task,method=method,split=split,weekday=day,**metrics(r)))
    pd.DataFrame(rows).to_csv('res/repair_weekdays_after.csv',index=False)
    weights['share']=weights.effective_population/weights.groupby(['task','method','station_complex_id','group']).effective_population.transform('sum')
    shares=weights.groupby(['task','method','station_complex_id','group']).share.agg(['min','max']).reset_index()
    shares.to_csv('res/repair_shares_after.csv',index=False)
    if weights.share.min()<.099999 or weights.share.max()>.600001:raise ValueError('Concentration constraint failed')
    primary=pd.read_csv('res/predictions.csv')
    repeats=pd.concat([primary]+[pd.read_csv(f'res/repeat{i}_predictions.csv') for i in [1,2]],ignore_index=True)
    dates=pd.read_csv('res/dates.csv');events=dates[dates.event_day][['task','date']]
    samples={'window':primary,'event':repeats.merge(events,on=['task','date']),'dry_sunday':primary[(primary.task=='Extreme_1007')&(primary.date=='2023-10-08')]}
    rows=[]
    for scope,p in samples.items():
        for (task,method,replicate),r in p.groupby(['task','method','replicate']):rows.append(dict(scope=scope,task=task,method=method,replicate=replicate,**metrics(r)))
    after=pd.DataFrame(rows)
    before=pd.read_csv('res/repair_reference.csv')
    compare=before.merge(after,on=['scope','task','method','replicate'],suffixes=('_before','_after'))
    compare.to_csv('res/repair_comparison.csv',index=False)
    print(compare[(compare.task=='Extreme_1007')&(compare.method.isin(['rule_abm','role_no_history','role_with_history']))][['scope','method','replicate','WAPE_pct_before','WAPE_pct_after','predicted_after']].to_string(index=False))
    print('Persona share range:',weights.share.min(),weights.share.max())

if __name__=='__main__':run()
