"""Post-hoc weather-only response probe. Never used to select or replace forecasts."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from api import request
from experiment import PEOPLE,role_view
from model import rule_prob

def probe():
    panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str})
    train=panel[(panel.task=='Extreme')&(panel.split=='train')]
    base=dict(date='2023-09-29',weekday=4,school_term=1,holiday=0,rain_mm=200,peak_mm=25,six_mm=120,temp_c=15,wind_ms=5)
    jobs=[(p,rain,method) for p in PEOPLE for rain in [0,10,50,200] for method in ['role_no_history','role_with_history']]
    def run(job):
        p,rain,method=job
        # History retrieval is held fixed too: only the supplied target rainfall changes.
        view=role_view(p,base,train,method=='role_with_history')
        view['target'].update(rain_mm=rain,peak_mm=rain/8,six_mm=rain*.6)
        answer=request('role',view,p)
        return dict(method=method,agent_id=p['agent_id'],group=p['group'],rain_mm=rain,p_subway=answer['probabilities'][0],reason=answer['reason'],request_hash=answer['hash'])
    with ThreadPoolExecutor(max_workers=12) as pool:rows=list(pool.map(run,jobs))
    val=pd.read_csv('res/validation.csv');scale=val[(val.task=='Extreme')&(val.method=='rule_abm')].sort_values('validation_wape',kind='stable').iloc[0].rain_scale
    for p in PEOPLE:
        for rain in [0,10,50,200]:
            r={**base,'rain_mm':rain,'peak_mm':rain/8,'six_mm':rain*.6}
            rows.append(dict(method='rule_abm',agent_id=p['agent_id'],group=p['group'],rain_mm=rain,p_subway=float(rule_prob(p,r,scale)[0]),reason='Utility rule',request_hash=''))
    r=pd.DataFrame(rows);r.to_csv('res/weather_probe_decisions.csv',index=False)
    weights=pd.read_csv('res/populations.csv');weights=weights[weights.task=='Extreme'].groupby(['method','agent_id']).effective_population.sum().reset_index()
    r=r.merge(weights,on=['method','agent_id'])
    summary=r.groupby(['method','group','rain_mm']).apply(lambda x:pd.Series({'p_subway':np.average(x.p_subway,weights=x.effective_population)}),include_groups=False).reset_index()
    summary.to_csv('res/weather_probe_summary.csv',index=False)
    fig,axes=plt.subplots(1,4,figsize=(14,4),sharey=True)
    for ax,(group,q) in zip(axes,summary.groupby('group')):
        for method,x in q.groupby('method'):ax.plot(x.rain_mm,x.p_subway,'o-',label=method)
        ax.set(title=group,xlabel='Synthetic daily rain (mm)',ylabel='Weighted P(subway)',ylim=(0,1))
    axes[-1].legend(fontsize=7);fig.suptitle('Weather-only sensitivity: fixed Friday, term, profiles, history examples and fitted populations')
    fig.tight_layout();fig.savefig('res/figures/12_weather_sensitivity.png',dpi=150);plt.close(fig)
    json.dump({'purpose':'post-hoc diagnostic, not accuracy evaluation or tuning','scenarios_mm':[0,10,50,200],
        'fixed':'Friday, school term, 15 C, wind 5 m/s, each profile, same historical examples, Extreme fitted populations',
        'not_claimed':'real individual preferences, causal human response, or observed choice probabilities'},open('res/weather_probe_design.json','w'),indent=2)

if __name__=='__main__':probe()
