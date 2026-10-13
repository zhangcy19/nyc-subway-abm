"""Run from V4. Only Lite is called; every real response is cached and budgeted."""
import argparse
import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from model import GROUPS, PolicyModel, allowed_actions, rule_choice, validate_choice

MODEL='doubao-seed-2-1-lite-260915'
CAP=5000000
LOCK=threading.Lock();RATE=threading.Lock();RESERVED=0;LAST_START=0.;CACHE=None
DIRECT_SYSTEM='''Return JSON only. Forecast six four-hour subway entry totals for one anonymous site.
This is a retrospective weather-conditioned test: the supplied six-period target weather is assumed known.
All demand history precedes the target day. No target-day entries or confirmed closures are supplied.
Do not invent station shutdowns, named events or a weather forecast. Site identity and date are hidden.
Use the numerical baseline, previous same-weekday counts, recent history, weather and calendar.
Return {"predicted":[six finite nonnegative entry counts],"reason":"at most 20 words"}.
Counts are trips, not unique people. You may increase or decrease demand relative to baseline.'''
ROLE_SYSTEM='''You ARE the ONE hypothetical traveler described, not a crowd forecaster.
Choose your mode for six mutually exclusive possible departure periods of ONE planned trip.
These are alternative scenarios, NOT six trips. Decide independently for each period, respecting your
budget, attendance, mobility needs, available_actions and delay limit. The common prior assumes subway.
Weather for each departure period is given as a retrospective condition; no verified closures are supplied.
Do not invent closures, future weather or other people's behavior. Traits and memory are synthetic.
Return {"choices":[six objects in period order]}.
Each object: {"action":"one available action token","reason":"first person, at most 8 words"}.
subway means depart now; delay_subway_60 and delay_subway_120 are distinct delayed options.
Never output a separate delay_min field; the chosen token specifies the delay. Do not output counts, demographic shares or crowd probabilities.'''
METHODS=['baseline','ridge','rule','direct_zero','direct_few','persona_zero','persona_few']


def read_ledger():
    rows=[]
    for f in ['res/api_calls.jsonl.gz','res/api_calls.jsonl']:
        if os.path.exists(f):
            with (gzip.open(f,'rt') if f.endswith('.gz') else open(f)) as source:
                rows.extend(json.loads(line) for line in source)
    return rows


def charge(row):
    return row.get('conservative_charge',row.get('usage',{}).get('total_tokens',0))


def request(system, view, route, persona=None, replicate=0, offline=False):
    global CACHE,RESERVED,LAST_START
    with LOCK:
        if CACHE is None:CACHE=read_ledger()
    prompt=json.dumps(view,separators=(',',':'),allow_nan=False)
    for attempt in range(3):
        descriptor={'model':MODEL,'system':system,'prompt':prompt,'temperature':.3,'max_output_tokens':650,
                    'replicate':replicate,'attempt':attempt,'route':route}
        digest=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
        with LOCK:
            cached=next((x for x in CACHE if x['hash']==digest and x['status']=='completed'),None)
            failed_before=any(x['hash']==digest and x['status']=='request_error' for x in CACHE)
        if cached:
            payload=cached['parsed']
        else:
            # Replay the recorded error -> retry path rather than stopping at attempt zero.
            if failed_before and attempt<2:continue
            if offline:raise RuntimeError('Missing actual API cache '+digest)
            reserve=len((system+prompt).encode())//2+950
            with LOCK:
                used=sum(charge(x) for x in CACHE)
                if used+RESERVED+reserve>CAP:raise RuntimeError('Token budget guard; actual/conservative charged '+str(used))
                RESERVED+=reserve
            with RATE:
                time.sleep(max(0,.18-(time.monotonic()-LAST_START)));LAST_START=time.monotonic()
            with open('../key.txt') as f:key=f.read().strip().strip('\"\'')
            client=OpenAI(api_key=key,base_url='https://ark.cn-beijing.volces.com/api/v3',timeout=90,max_retries=0)
            started=time.time()
            try:
                response=client.responses.create(model=MODEL,input=[{'role':'system','content':system},{'role':'user','content':prompt}],
                    temperature=.3,max_output_tokens=650,extra_body={'thinking':{'type':'disabled'}})
                text=response.output_text.strip()
                if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
                try:payload=json.loads(text)
                except json.JSONDecodeError:payload=None
                if response.status!='completed':payload=None
                row={'hash':digest,'request':descriptor,'status':response.status,'parsed':payload,
                     'raw':response.output_text,'usage':response.usage.model_dump(),'response_id':response.id,
                     'elapsed_seconds':round(time.time()-started,3)}
            except Exception as error:
                row={'hash':digest,'request':descriptor,'status':'request_error','parsed':None,'usage':{},
                     'conservative_charge':reserve,'error_type':type(error).__name__}
                payload=None
            finally:client.close()
            with LOCK:
                CACHE.append(row)
                with open('res/api_calls.jsonl','a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                RESERVED-=reserve
                if len(CACHE)%32==0:print('API calls',len(CACHE),'charged tokens',sum(charge(x) for x in CACHE),flush=True)
            if row['status']=='request_error':
                if attempt<2:time.sleep(3*(attempt+1));continue
                raise RuntimeError('API request failed: '+row['error_type'])
        try:
            if route=='direct':
                values=payload['predicted']
                if len(values)!=6 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in values):raise ValueError('Need six finite nonnegative counts')
                if not isinstance(payload.get('reason'),str) or not payload['reason'].strip():raise ValueError('Need reason')
                return {'predicted':[float(v) for v in values],'reason':payload['reason']},digest
            choices=payload['choices']
            if len(choices)!=6:raise ValueError('Need six choices in period order')
            normalized=[]
            for c in choices:
                token=c['action']
                delay=60 if token=='delay_subway_60' else 120 if token=='delay_subway_120' else 0
                normalized.append(validate_choice({'action':'delay_subway' if delay else token,'delay_min':delay,'reason':c['reason']},persona))
            return {'choices':normalized},digest
        except (TypeError,ValueError,KeyError,AttributeError) as error:
            if attempt==2:raise RuntimeError('Invalid API output: '+str(error))
            prompt=json.dumps({'input':view,'format_repair':str(error),'previous_response':payload},separators=(',',':'),allow_nan=False)
    raise RuntimeError('No valid actual response')


def load_panel():
    return pd.read_csv('res/panel.csv.gz',dtype={'station_complex_id':str},parse_dates=['timestamp','date'])


def weather_view(rows):
    return [dict(block=int(b),hours=f'{int(b)*4:02}-{int(b)*4+3:02}',
        rain_mm=round(float(x.rain_mm.max()),2),rain_6h_mm=round(float(x.rain_6h_mm.max()),2),
        rain_period_mm=round(float(x.rain_mm.sum()),2),temp_c=round(float(x.temp_c.mean()),1),wind_ms=round(float(x.wind_ms.max()),1))
        for b,x in rows.groupby('block',sort=True)]


def direct_view(panel,rows,few=False,enriched=False):
    s=rows.station_complex_id.iloc[0];date=rows.date.iloc[0]
    history=panel[(panel.station_complex_id==s)&(panel.date<date)]
    same=[]
    for lag in [7,14,21,28]:
        past=history[history.date==date-pd.Timedelta(days=lag)]
        same.append({'lag_days':lag,'entries':past.groupby('block').total_ridership.sum(min_count=1).fillna(0).round(1).tolist(),
                     'observed_hours':int(past.total_ridership.notna().sum())})
    recent=history[history.date>=date-pd.Timedelta(days=7)].groupby('date').agg(entries=('total_ridership','sum'),observed_hours=('total_ridership','count'))
    view={'weekday':int(date.dayofweek),'baseline':rows.groupby('block').planned_total.sum().round(1).tolist(),
          'weather':weather_view(rows),'previous_same_weekdays':same,'previous_seven_days':recent.round(1).to_dict('records')}
    if few:
        pool=history[history.date<'2023-08-01'].groupby('date').agg(rain=('rain_day_mm','first'),weekday=('weekday','first'),n=('total_ridership','count'))
        pool=pool[pool.n==24].copy()
        pool['distance']=np.abs(np.log1p(pool.rain)-np.log1p(rows.rain_day_mm.iloc[0]))+.3*(pool.weekday!=date.dayofweek)
        dates=pool.sort_values(['distance']).head(3).index
        view['demonstrations']=[]
        for d in dates:
            example=panel[(panel.station_complex_id==s)&(panel.date==d)]
            view['demonstrations'].append({'input':{'weekday':int(d.dayofweek),'baseline':example.groupby('block').planned_total.sum().round(1).tolist(),'weather':weather_view(example)},
                'output':{'predicted':example.groupby('block').total_ridership.sum(min_count=1).round(1).tolist(), 'reason':'Observed aggregate training-day entry counts; not individual behavior.'}})
    if enriched:view['retrospective_context']=context(s)
    return view


def context(station):
    x=pd.read_csv('res/supplementary_context.csv',dtype={'station_complex_id':str})
    r=x[x.station_complex_id==station].iloc[0]
    return {**{k:round(float(r[k]),3) for k in r.index if k not in ['station_complex_id','borough'] and pd.notna(r[k])},
        'definitions':{'resident_shares':'fractions 0-1 of borough residents; not rider shares',
            'current_ada_code':'0 not accessible; 1 fully accessible; 2 partially accessible',
            'current_network_degree':'distinct adjacent GTFS directional stops; not inter-station transfer options'},
        'scope':'resident borough aggregates and current network/ADA; retrospective only, not observed rider traits or historical closures'}


def persona_view(agent,rows,few=False,enriched=False):
    p=agent.persona
    view={'you':{k:p[k] for k in ['identity','type','purpose','attendance_required','max_delay_min','cancel_allowed','remote_allowed','mobility_support','risk_aversion','subway_habit','travel_budget','access_walk_min']},
        'weekday':int(rows.weekday.iloc[0]),'usual_mode':'subway','available_actions':[a for a in allowed_actions(p) if a!='delay_subway'] + (['delay_subway_60'] if p['max_delay_min']>=60 else []) + (['delay_subway_120'] if p['max_delay_min']>=120 else []),
        'scenario_costs':{'subway':3,'bus':3,'taxi':20,'walk':0},
        'scenario_minutes':{'subway':25+p['access_walk_min'],'bus':45,'taxi':25,'walk':65},
        'past_own_simulated_experience':agent.memory,'alternative_departure_weather':weather_view(rows)}
    if few:
        demo=[]
        for rain,six in [(0,0),(12,45)]:
            environment={'rain_mm':rain,'rain_6h_mm':six}
            c=rule_choice(p,environment,[])
            token='delay_subway_'+str(c['delay_min']) if c['action']=='delay_subway' else c['action']
            demo.append({'weather':environment,'choice':{'action':token,'reason':'Illustrative rule-generated synthetic choice.'}})
        view['synthetic_demonstrations']=demo
        view['demonstration_scope']='Format/action examples generated by the transparent rule, not observed human labels; you may disagree.'
    if enriched:view['retrospective_context']=context(p['station_complex_id'])
    return view


def apply_policy(rows,persons,decisions,method,n=8,phase='test'):
    predicted=np.zeros(24);group_predictions={g:np.zeros(24) for g in GROUPS};records=[];planned=0.;removed=0.;spill=0.
    for person,choices,digest in zip(persons,decisions['choices'],decisions['hashes']):
        for _,r in rows.sort_values('hour').iterrows():
            weight=float(r['planned_'+person['group']])*person['within_group_weight'];choice=choices[int(r.block)]
            planned+=weight
            if choice['action'] in ['subway','delay_subway']:
                destination=int(r.hour)+choice['delay_min']//60
                if destination<24:
                    predicted[destination]+=weight
                    group_predictions[person['group']][destination]+=weight
                else:spill+=weight
            else:removed+=weight
        for b,c in enumerate(choices):
            weight=float(rows.loc[rows.block==b,'planned_'+person['group']].sum())*person['within_group_weight']
            records.append({**{k:person[k] for k in ['agent_id','group','identity','type','risk_aversion','mobility_support','station_complex_id']},
                'date':str(rows.date.iloc[0].date()),'block':b,'method':method,'n_agents':n,'phase':phase,
                'weight':weight,**c,'request_hash':digest})
    audit={'date':str(rows.date.iloc[0].date()),'station_complex_id':str(rows.station_complex_id.iloc[0]),'method':method,'phase':phase,'n_agents':n,
           'planned':planned,'entered_same_day':float(predicted.sum()),'non_subway':removed,'spillover_next_day':spill,'mass_error':float(predicted.sum()+removed+spill-planned),'group_predictions':{g:v.tolist() for g,v in group_predictions.items()}}
    return predicted,records,audit


def spread_direct(rows,blocks):
    predicted=np.zeros(24)
    for b,value in enumerate(blocks):
        part=rows[rows.block==b];mass=part.planned_total.to_numpy();den=float(mass.sum())
        predicted[part.hour.to_numpy()]=value*(mass/den if den>0 else np.ones(4)/4)
    return predicted


def features(x):
    return np.column_stack([x.planned_total,x.rain_mm,x.rain_6h_mm,x.temp_c,x.wind_ms,x.weekday,
        np.sin(x.hour*2*np.pi/24),np.cos(x.hour*2*np.pi/24),x.planned_total*x.rain_mm])


def fit_offline(panel):
    train=panel[(panel.split=='train')&(panel.date>='2023-07-15')&panel.total_ridership.notna()&(panel.planned_total>0)]
    validation=panel[(panel.split=='validation')&panel.total_ridership.notna()&(panel.planned_total>0)]
    candidates=[];fits={}
    for alpha in [1,10,100,1000]:
        reg=make_pipeline(StandardScaler(),Ridge(alpha=alpha));reg.fit(features(train),train.total_ridership/train.planned_total)
        pred=np.clip(reg.predict(features(validation)),.1,3)*validation.planned_total
        day=pd.DataFrame({'date':validation.date,'station':validation.station_complex_id,'actual':validation.total_ridership,'pred':pred}).groupby(['date','station'])[['actual','pred']].sum()
        error=float(np.abs(day.pred-day.actual).sum()/day.actual.sum());candidates.append({'alpha':alpha,'validation_daily_wape':error});fits[alpha]=reg
    selected=min(candidates,key=lambda r:r['validation_daily_wape'])['alpha']
    people=PolicyModel(panel.station_complex_id.unique().tolist(),8);rules=[]
    for scale in [0,.5,1,2,4]:
        predictions=[]
        for (s,date),rows in panel[panel.split=='validation'].groupby(['station_complex_id','date']):
            persons=[a.persona for a in people.agents if a.persona['station_complex_id']==s]
            choices=[[rule_choice(p,e,[],rain_scale=scale) for e in weather_view(rows)] for p in persons]
            pred,_,_=apply_policy(rows,persons,{'choices':choices,'hashes':['']*len(persons)},'rule')
            observed=rows.total_ridership.notna().to_numpy();predictions.append((np.sum(pred[observed]),rows.total_ridership.sum()))
        a=np.array(predictions);rules.append({'rain_scale':scale,'validation_daily_wape':float(np.abs(a[:,0]-a[:,1]).sum()/a[:,1].sum())})
    scale=min(rules,key=lambda r:r['validation_daily_wape'])['rain_scale']
    json.dump({'ridge_candidates':candidates,'ridge_alpha':selected,'rule_candidates':rules,'rule_rain_scale':scale,'fit':'July15-31; selections only Aug4-7 validation'},open('res/offline_selection.json','w'),indent=2)
    return fits[selected],scale


def run(phase,offline=False,n=8,subset=False,enriched=False,replicate=0):
    panel=load_panel();reg,scale=fit_offline(panel)
    selected=pd.read_csv('res/stations.csv',dtype={'station_complex_id':str}).station_complex_id.tolist()
    dates=sorted(panel[panel.split==('validation' if phase=='pilot' else phase)].date.unique())
    if phase=='pilot':selected=selected[:1];dates=dates[:1]
    if subset:
        selected=[selected[0],selected[4]]
        dates=pd.to_datetime(['2023-08-13','2023-09-29','2023-10-07','2023-10-13']).tolist()
    methods=METHODS if not subset else ['baseline','ridge','rule','direct_zero','direct_few','persona_zero','persona_few']
    forecasts=[];decisions=[];audits=[]
    for method in methods:
        model=PolicyModel(selected,n)
        for date in dates:
            date=pd.Timestamp(date);rows_by={s:panel[(panel.station_complex_id==s)&(panel.date==date)].sort_values('hour') for s in selected}
            agents_by={s:[a for a in model.agents if a.persona['station_complex_id']==s] for s in selected}
            results={};jobs={}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for s,rows in rows_by.items():
                    if method.startswith('direct'):
                        view=direct_view(panel,rows,method.endswith('few'),enriched)
                        future=pool.submit(request,DIRECT_SYSTEM,view,'direct',None,replicate,offline);jobs[future]=(s,None)
                    elif method.startswith('persona'):
                        for agent in agents_by[s]:
                            if rows['planned_'+agent.persona['group']].sum()<=0:
                                results[(s,agent.persona['agent_id'])]=({'choices':[{'action':'not_applicable','delay_min':0,'reason':'No planned mass; no API call.'}]*6},'not_applicable_zero_mass')
                                continue
                            view=persona_view(agent,rows,method.endswith('few'),enriched)
                            future=pool.submit(request,ROLE_SYSTEM,view,'persona',agent.persona,replicate,offline);jobs[future]=(s,agent)
                for future in as_completed(jobs):
                    s,agent=jobs[future];payload,digest=future.result();results[(s,agent.persona['agent_id'] if agent else '')]=(payload,digest)
            for s,rows in rows_by.items():
                persons=[a.persona for a in agents_by[s]]
                group_forecasts=None
                if method=='baseline':pred=rows.planned_total.to_numpy()
                elif method=='ridge':pred=np.clip(reg.predict(features(rows)),.1,3)*rows.planned_total.to_numpy()
                elif method.startswith('direct'):
                    payload,digest=results[(s,'')];pred=spread_direct(rows,payload['predicted'])
                    decisions.append({'method':method,'station_complex_id':s,'date':str(date.date()),'reason':payload['reason'],'request_hash':digest,'phase':phase,'n_agents':n})
                else:
                    choices=[];hashes=[]
                    for agent in agents_by[s]:
                        if rows['planned_'+agent.persona['group']].sum()<=0:
                            policy=[{'action':'not_applicable','delay_min':0,'reason':'No planned mass; no API call.'}]*6;digest='not_applicable_zero_mass'
                        elif method=='rule':policy=[rule_choice(agent.persona,e,agent.memory,rain_scale=scale) for e in weather_view(rows)];digest=''
                        else:payload,digest=results[(s,agent.persona['agent_id'])];policy=payload['choices']
                        choices.append(policy);hashes.append(digest)
                        if digest!='not_applicable_zero_mass':agent.observe_policy(policy,weather_view(rows))
                    pred,records,audit=apply_policy(rows,persons,{'choices':choices,'hashes':hashes},method,n,phase)
                    group_forecasts=audit.pop('group_predictions')
                    decisions.extend(records);audits.append(audit)
                out=rows[['timestamp','date','station_complex_id','station_name','borough','hour','block','weather_class','total_ridership','planned_total']+GROUPS].copy()
                out['predicted']=pred
                for g in GROUPS:out['pred_'+g]=group_forecasts[g] if group_forecasts else np.nan
                out['method']=method;out['phase']=phase;out['n_agents']=n;out['enriched']=enriched;out['replicate']=replicate
                forecasts.append(out)
            print(phase,method,str(date.date()),'stations',len(selected),flush=True)
        # Save only complete method/day outputs; resume always replays the real cache.
        label=phase+('_probe'+str(n) if subset else '')+('_enriched' if enriched else '')+('_repeat'+(str(replicate) if replicate>1 else '') if replicate else '')
        pd.concat(forecasts,ignore_index=True).to_csv('res/'+label+'_predictions.csv.gz',index=False)
        pd.DataFrame(decisions).to_csv('res/'+label+'_decisions.csv.gz',index=False)
        pd.DataFrame(audits).to_csv('res/'+label+'_mass.csv',index=False)
    json.dump({'reported_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in (CACHE or read_ledger())),
               'charged_tokens':sum(charge(r) for r in (CACHE or read_ledger())),'cap':CAP,'calls':len(CACHE or read_ledger()),'model':MODEL},open('res/budget.json','w'),indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pilot','validation','test']);parser.add_argument('--offline',action='store_true');parser.add_argument('--subset',action='store_true');parser.add_argument('--enriched',action='store_true');parser.add_argument('--replicate',type=int,default=0);parser.add_argument('--agents',type=int,default=8);parser.add_argument('--budget',type=int,default=5000000)
    args=parser.parse_args()
    if args.budget>10000000:parser.error('User-authorized maximum is 10,000,000 tokens')
    CAP=args.budget
    run(args.phase,args.offline,args.agents,args.subset,args.enriched,args.replicate)
