"""Test-only prompt-context ablation. Frozen fitted weights; not a replacement main model."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from api import request
from experiment import PEOPLE,WEATHER,role_view,environment
from model import predict

def run():
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});dates=pd.read_csv('res/dates.csv')
    part=panel[panel.task=='Extreme_1007'];train=part[part.split=='train'];test=part[(part.split=='test')&(part.date=='2023-10-07')]
    target=dates[(dates.task=='Extreme_1007')&(dates.date=='2023-10-07')].iloc[0]
    weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str});weights=weights[(weights.task=='Extreme_1007')&(weights.method=='role_with_history')].to_dict('records')
    predictions=[];decisions=[];start=time.perf_counter()
    for replicate in range(3):
        def call(person):
            view=role_view(person,target,train,True)
            past=train[(train.observed_hours==24)&(train.date<target.date)].copy();group=person['group']
            past['entry_index']=past[group]/past.groupby('station_complex_id')[group].transform('mean').replace(0,float('nan'))
            daily=past.groupby('date').agg(**{k:(k,'first') for k in WEATHER},entry_index=('entry_index','mean')).reset_index().dropna()
            storm=daily.sort_values(['rain_mm','date']).iloc[-1]
            example={'calendar_weather':environment(storm),'observed_group_entry_index':round(float(storm.entry_index),3)}
            view['group_history']=view['group_history'][:3]+[example]
            return (target.date,person['agent_id']),request('role',view,person,replicate)
        with ThreadPoolExecutor(max_workers=10) as pool:policy=dict(pool.map(call,PEOPLE))
        pred,dec=predict(test,weights,policy,PEOPLE)
        predictions.extend(dict(task='Extreme_1007',method='role_storm_context',replicate=replicate,**r) for r in pred)
        decisions.extend(dict(task='Extreme_1007',method='role_storm_context',replicate=replicate,**r) for r in dec)
    pd.DataFrame(predictions).to_csv('res/context_probe_predictions.csv',index=False)
    pd.DataFrame(decisions).to_csv('res/context_probe_decisions.csv',index=False)
    json.dump({'seconds':time.perf_counter()-start,'event':'2023-10-07','forced_example':'2023-09-29','test_only':True},open('res/context_probe.json','w'),indent=2)

if __name__=='__main__':run()
