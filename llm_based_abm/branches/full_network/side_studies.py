"""Paired information ablations; real Lite calls, fixed populations, no test tuning.
Run --prepare first, then --execute. Shared strategies are deduplicated explicitly.
"""
import argparse,copy,hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import api
from design import inputs,EVENTS,SEASON,normal_history
from experiment import PEOPLE
from timing import person_view,weather_view
from model import predict

OUT='side_studies_res';IDS=['1','152','448'];REPS=[0,1]
api.LOGDIR=OUT;api.CAP=3200000;api.MODEL='doubao-seed-2-1-lite-260915'
# Use explicit common instructions; only experimental fields vary between arms.
api.ROLE_SYSTEM += '\nEvents may be later than the target in this retrospective experiment. All target-window observed entries are withheld. Use only supplied information; do not assume demand changes are caused only by weather.'
api.DIRECT_SYSTEM += '\nEvents may be later than the target in this retrospective experiment. All target-window observed entries are withheld. Do not use outside knowledge of actual target ridership.'
P,W=inputs(IDS)
DONOR=pd.read_csv('../event_holdout/res/panel.csv',dtype={'station_complex_id':str})
POPS=pd.concat([pd.read_csv('rollback_dev_res/populations.csv',dtype={'station_complex_id':str}),pd.read_csv('rollback_full_res/expansion_populations.csv',dtype={'station_complex_id':str})],ignore_index=True)
SCHEDULE=json.load(open(OUT+'/mlb_schedule_source.json'))
GAMES={d['date']:[g for g in d['games'] if g['teams']['home']['team']['id']==121 and g['venue']['name']=='Citi Field'] for d in SCHEDULE['dates']}

def weather(row,hourly):
 v=weather_view(row,W,hourly);v.pop('peak_mm',None);v.pop('six_mm',None);v['date']=row.date
 return v

def event_fact(date):
 games=[g for g in GAMES.get(date,[]) if g.get('officialDate')==date and g['status']['codedGameState']=='F']
 return dict(mets_home_game_on_date=bool(games),home_games=len(games),start_local=[str(pd.Timestamp(g['gameDate']).tz_convert('America/New_York')) for g in games],scope='Historical completed Mets home games on their official date; excludes postponed fixtures and unplayed resumptions. Target dates have no Mets fixture. No attendance or game outcome provided; other event types not covered.')

def refs(held,group,hourly,sid=None):
 data=DONOR if sid is None else P[P.station_complex_id==sid]
 out=[]
 for task,center in EVENTS.items():
  if task==held:continue
  block=data[(data.task==task)&data.split.eq('test')];base=normal_history(data,SEASON[task]);days=[]
  for date,g in block.groupby('date'):
   r=g.iloc[0];v=dict(date=date,relative_day=int((pd.Timestamp(date)-pd.Timestamp(center)).days),weather=weather(r,hourly))
   if sid is not None:v['observed_entries']=float(r.total_ridership)
   else:
    ratios=[]
    for _,a in g.iterrows():
     b=base[base.station_complex_id==a.station_complex_id];same=b[b.weekday==a.weekday];mean=(same if len(same) else b)[group].mean()
     if pd.notna(mean) and mean>0:ratios.append(float(a[group])/mean)
    v['observed_group_entries_relative_to_normal_weekday']=round(float(np.mean(ratios)),3) if ratios else None
    v['contributing_stations']=len(ratios)
   days.append(v)
  out.append(dict(event_date=center,season=SEASON[task],days=days))
 return out

def base_role(row,person):
 v=person_view(person,row,normal_history(P,SEASON[row.task]),W,False,True)
 return v

def rain_role(row,person,few,hourly):
 v=base_role(row,person);v['target']=weather(row,hourly)
 if few:
  v['reference_events']=refs(row.task,person['group'],hourly)
  v['history_definition']='Three OTHER complete events, days -2 to +2. Group entry ratios use each donor station and season Normal-training same-weekday mean, then average across two donor stations. Ratio 1 means usual entries. Aggregate observations, not personal action probabilities. Entire target event excluded.'
 return v

def rain_direct(row,hourly):
 base=normal_history(P,SEASON[row.task]);base=base[base.station_complex_id==row.station_complex_id].copy()
 base['distance']=abs(np.log1p(base.rain_mm)-np.log1p(row.rain_mm))+(base.weekday!=row.weekday)
 selected=base.sort_values(['distance','date'],ascending=[True,False]).head(4).sort_values('date')
 return dict(target=weather(row,hourly),observed_training_examples=[dict(input=weather(r,hourly),entries=float(r.total_ridership)) for _,r in selected.iterrows()],reference_events=refs(row.task,None,hourly,row.station_complex_id))

def activity_view(row,person,few,information):
 # Identical local history in both arms. Event annotations are the only difference.
 base=normal_history(P,'Normal_School');base=base[base.station_complex_id==row.station_complex_id]
 v=base_role(row,person) if person else {}
 v['target']=weather(row,True)
 if few:
  records=[]
  for _,r in base.sort_values('date').iterrows():
   rec=dict(input=weather(r,True))
   if person:
    same=base[base.weekday==r.weekday];denom=same[person['group']].mean();rec['observed_group_entry_index']=round(float(r[person['group']]/denom),3) if denom>0 else None
   else:rec['entries']=float(r.total_ridership)
   if information:rec['mets_calendar']=event_fact(r.date)
   records.append(rec)
  v['observed_training_examples']=records
  if person:v['history_definition']='Same-station fare-group observed entries divided by its Normal-training same-weekday mean. Index 1 means usual same-weekday entries; not an individual probability. These 15 training days are identical in both arms.'
 if information:
  v['target']['mets_calendar']=event_fact(row.date)
  v['local_event_context']=dict(venue='Citi Field',station_relation='station directly serves the ballpark' if row.station_complex_id=='448' else 'comparison station not directly serving the ballpark; no measured event effect supplied',scope='Known baseball calendar and geographic relationship only. Do not assume no games means zero regular demand. No attendance, target ridership, or expected demand multiplier supplied.')
 return v

def build():
 jobs=[]
 for study in ['rain','activity']:
  target=P[P.task.isin(EVENTS)&P.event_day] if study=='rain' else P[(P.task=='Extreme_1007')&P.date.isin(['2023-10-06','2023-10-07','2023-10-08'])]
  for _,row in target.iterrows():
   for method in ['role_no_history','role_with_history','direct_few']:
    for condition in (['daily','hourly'] if study=='rain' else ['without_event','with_event']):
     for rep in REPS:
      for person in ([None] if method=='direct_few' else PEOPLE):
       few=method!='role_no_history'
       view=(rain_role(row,person,few,condition=='hourly') if person else rain_direct(row,condition=='hourly')) if study=='rain' else activity_view(row,person,few,condition=='with_event')
       route='role' if person else 'direct';signature=json.dumps(dict(route=route,view=view,rep=rep),sort_keys=True)
       jobs.append(dict(study=study,task=row.task,date=row.date,station_complex_id=row.station_complex_id,method=method,condition=condition,replicate=rep,agent_id=person['agent_id'] if person else None,person=person,route=route,view=view,key=hashlib.sha256(signature.encode()).hexdigest()))
 return jobs

def prepare(jobs):
 unique={j['key']:j for j in jobs};estimated=sum((len(json.dumps(j['view']))/3.2+450+80) for j in unique.values())
 # Paired payloads must differ only by rainfall fields or known event metadata.
 def strip(value):
  if isinstance(value,dict):return {k:strip(v) for k,v in value.items() if k not in ['hourly_rain_mm_00_to_23','mets_calendar','local_event_context']}
  if isinstance(value,list):return [strip(v) for v in value]
  return value
 pairs={}
 for j in jobs:
  key=(j['study'],j['task'],j['date'],j['station_complex_id'],j['method'],j['replicate'],j['agent_id'])
  clean=json.dumps(strip(j['view']),sort_keys=True)
  if key in pairs and pairs[key]!=clean:raise ValueError('Uncontrolled paired prompt difference')
  pairs[key]=clean
 protocol=dict(stations=IDS,rain_dates=list(EVENTS.values()),activity_dates=['2023-10-06','2023-10-07','2023-10-08'],replicates=REPS,unique_requests=len(unique),estimated_tokens=round(estimated),token_cap=api.CAP,estimated_cny_at_previous_rates=round(estimated*.95/1e6,2),budget_cny=10,model=api.MODEL,rain='Only full 24-hour rainfall array added; daily total/temp/wind/calendar and observed history unchanged. Derived peak/six-hour fields absent in both arms. Target and all historical weather use the assigned representation.',activity='Known MLB home-game flags/times and venue relationship added to target and same 15 training days; no attendance, results or target entries. Focus 448, controls 1 and 152. Historical schedule retrospective; no claim of as-issued forecast.',population='Frozen per-station Normal weights from rollback; no refitting in either arm. Role decisions real fresh API calls; no old responses reused.',role_rain_history='Original two donor stations event_holdout indices, same in both arms',scoring='Score each event, average four event scores for rain. Activity score three dates pooled; station448 separately and controls separately. Both repeats shown, do not select best.',paired_payload_check='passed',source='https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=121&startDate=2023-07-01&endDate=2023-10-31')
 json.dump(protocol,open(OUT+'/protocol.json','w'),indent=2)
 P.to_csv(OUT+'/panel.csv',index=False)
 facts=[dict(date=d,**event_fact(d)) for d in sorted(set(normal_history(P,'Normal_School').date)|{'2023-10-06','2023-10-07','2023-10-08'})];pd.DataFrame(facts).to_csv(OUT+'/event_calendar.csv',index=False)
 print(protocol,flush=True)
 return unique

def execute(jobs,unique):
 api.NETWORK_FAILED=False
 answers={};ordered=list(unique.items());rng=np.random.default_rng(5011);rng.shuffle(ordered)
 def call(item):
  key,j=item
  return key,api.request(j['route'],j['view'],j['person'],replicate=j['replicate'])
 start=time.perf_counter()
 with ThreadPoolExecutor(max_workers=3) as pool:
  for n,(key,answer) in enumerate(pool.map(call,ordered),1):
   answers[key]=answer
   if n%40==0:print('Completed',n,'/',len(unique),'requests',flush=True)
 records=[];decisions=[];mapping=[]
 for j in jobs:
  a=answers[j['key']];meta={k:j[k] for k in ['study','task','date','station_complex_id','method','condition','replicate','agent_id']};mapping.append(dict(**meta,request_hash=a['hash']))
  if j['route']=='direct':
   actual=float(P[(P.task==j['task'])&(P.date==j['date'])&(P.station_complex_id==j['station_complex_id'])].iloc[0].total_ridership)
   records.append(dict(**{k:v for k,v in meta.items() if k!='agent_id'},actual=actual,predicted=a['entries'],request_hash=a['hash'],reason=a['reason']))
  else:
   season=SEASON[j['task']];weights=POPS[(POPS.task==season)&(POPS.method==j['method'])&(POPS.station_complex_id==j['station_complex_id'])&(POPS.agent_id==j['agent_id'])].iloc[0]
   weekday=pd.Timestamp(j['date']).weekday();effective=float(weights.effective_population*weights[f'weekday_{weekday}'])
   decisions.append(dict(**meta,effective_population=effective,p_subway=a['probabilities'][0],predicted=effective*a['probabilities'][0],request_hash=a['hash'],reason=a['reason']))
 dec=pd.DataFrame(decisions);keys=['study','task','date','station_complex_id','method','condition','replicate']
 agg=dec.groupby(keys).predicted.sum().reset_index();truth=P[['task','date','station_complex_id','total_ridership']].rename(columns={'total_ridership':'actual'});agg=agg.merge(truth,on=['task','date','station_complex_id'],validate='many_to_one')
 pred=pd.concat([pd.DataFrame(records),agg],ignore_index=True);pred.to_csv(OUT+'/predictions.csv',index=False);dec.to_csv(OUT+'/decisions_private.csv',index=False);pd.DataFrame(mapping).to_csv(OUT+'/request_map.csv',index=False)
 rows=api.ledger();usage=dict(reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in rows),input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in rows),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in rows),budget_tokens=sum(api.charge(r) for r in rows),seconds=time.perf_counter()-start)
 usage['elapsed_wall_seconds']=time.time()-os.stat(OUT+'/api_calls.jsonl').st_birthtime;usage['last_resume_seconds']=usage['seconds'];usage['seconds']=usage['elapsed_wall_seconds'];usage['cap_adjustment']='Initial 3M conservative cap reached near completion; increased to 3.2M within authorized 10 CNY budget to finish missing requests.';usage['estimated_cny']=usage['input_tokens']*.8/1e6+usage['output_tokens']*2.7/1e6;json.dump(usage,open(OUT+'/usage.json','w'),indent=2);print(usage,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');args=a.parse_args();jobs=build();unique=prepare(jobs)
 if args.execute:execute(jobs,unique)
