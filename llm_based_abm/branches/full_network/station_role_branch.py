"""Single-station structural diagnostic; run here with --execute or --report."""
import argparse, copy, json, os, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import mesa
import side_studies as s
import api

FOLDER='station_role_branch_res'
os.makedirs(FOLDER,exist_ok=True)
api.LOGDIR=FOLDER;api.CAP=3000000;api.LEDGER=None
api.ROLE_SYSTEM='''You simulate the supplied hypothetical traveler, not a demand forecaster. Return TWO probabilities and a brief decision summary. Follow the task's precise probability definitions. Each probability must lie between 0 and 1. Their product is the unconditional outcome probability; they do NOT sum to one. Consider calendar, weather, purpose and supplied events. No trip, ticket, work shift or closure is guaranteed. Historical fare-group counts describe aggregates, NOT personal choices or population size. Do not derive a personal probability by dividing historical counts by an invented population. No target observed demand is supplied. Do not use remembered actual target demand. Event absence does not eliminate unrelated trips. The baseball calendar covers only Mets games; other activities may exist. Return JSON {"p_opportunity":number,"p_conditional":number,"reason":"brief first-person decision summary, at most 25 words"}.'''

def validate(payload,route,person=None):
 if not isinstance(payload.get('reason'),str) or not payload['reason'].strip():raise ValueError('Need brief reason')
 for k in ['p_opportunity','p_conditional']:
  x=payload[k]
  if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<=x<=1:raise ValueError('Invalid probability')
 return payload
api.validate=validate
# Reuse one connection pool per worker; the common API helper otherwise closes
# a newly created client after every short call. This changes transport only.
Client=api.OpenAI
local=threading.local()
clients=[]
def worker_client(**kwargs):
 if not hasattr(local,'client'):
  local.client=Client(**kwargs)
  local.client.close=lambda:None
  clients.append(local.client)
 return local.client
api.OpenAI=worker_client
train=s.normal_history(s.P,'Normal_School').query("station_complex_id=='448'").sort_values('date')
test=s.P.query("station_complex_id=='448' and task=='Extreme_1007' and date in ['2023-10-06','2023-10-07','2023-10-08']").sort_values('date')
people=copy.deepcopy(s.PEOPLE)
relations=[
 'Nearby resident: this is your usual local subway access station. Regular work, education or other obligations elsewhere may generate an outbound entry here. Weekend obligations are not guaranteed.',
 'Local-area errands or appointment traveler: you may visit or leave this neighborhood for ordinary errands, appointments or social shopping. No specific booking or daily visit is guaranteed.',
 'Nonresident potential Citi Field event visitor: a ballpark visit is your principal reason to use this area. No ticket or daily attendance is guaranteed. Consider a supplied event and your constraints. Without a relevant event, do not invent attendance; unrelated visits remain possible, not certain.'
]

def calendar(date):
 fact=s.event_fact(date)
 fact['scope']='Completed Mets home games on this date only; excludes postponed fixtures and unplayed resumptions. Other event types are not covered. No attendance or game outcome supplied.'
 return fact

def view(row,person,variant):
 target=s.weather(row,True)
 target['mets_calendar']=calendar(row.date)
 history=[]
 for _,h in train.iterrows():
  if h.date==row.date:continue # training-day target label also excluded from its request
  history.append(dict(date=h.date,weekday=pd.Timestamp(h.date).day_name(),rain_mm=float(h.rain_mm),observed_fare_group_entries=float(h[person['group']]),mets_calendar=calendar(h.date)))
 persona={k:person[k] for k in ['identity','routine','flexibility','weather_caution','budget_usd','mobility_limited','usual_mode','access_walk_minutes']}
 if variant=='station':persona['relationship_to_station']=relations[person['kind']]
 task=('p_opportunity = probability of making any journey today. p_conditional = probability of choosing subway somewhere in NYC conditional on that journey. This control predicts general subway use, not a specific station.' if variant=='generic' else 'p_opportunity = probability of being in the Mets-Willets Point station area today with a potential departure journey, whether as resident or visitor. p_conditional = probability of entering THIS station to board the subway conditional on that opportunity. Their product predicts at least one entry HERE for this representative opportunity, not subway use anywhere. An arrival exit is NOT an entry: an event visitor may enter here for their return departure. Other stations, other modes or no departure are non-entry outcomes. Do not assume every visitor uses subway.')
 return dict(task=task,person=persona,station=dict(name='Mets-Willets Point',borough='Queens',line='7',nearby_venue='Citi Field',relationship='station serves the ballpark and surrounding area; not all users are spectators'),target=target,observed_training_examples=history,history_definition='Same 15 Normal-training dates, absolute observed fare-group entries. For training-day requests that same date is excluded. No test-window observations. Historical examples are contextual aggregate evidence, not individual behavior labels. Known case exploratory experiment.',assumptions='Personas and station relationships are hypotheses, not observed individual records. One entry opportunity per representative; fitted counts are effective coefficients, not resident population.')

def jobs():
 return [dict(variant=v,replicate=rep,date=r.date,agent_id=p['agent_id'],person=p,view=view(r,p,v)) for v in ['generic','station'] for rep in [0,1] for _,r in pd.concat([train,test]).iterrows() for p in people]

class Representative(mesa.Agent):
 def __init__(self,model,weight,prob):super().__init__(model);self.weight=weight;self.prob=prob
 def step(self):self.model.entries+=self.weight*self.prob
class StationDay(mesa.Model):
 def __init__(self,weights,probs):
  super().__init__(seed=5011);self.entries=0.
  for w,p in zip(weights,probs):Representative(self,w,p)
 def step(self):self.agents.do('step')

def report():
 decisions=pd.read_csv(FOLDER+'/decisions_private.csv')
 weights=[];pred=[];fit=[]
 for (variant,rep),block in decisions.groupby(['variant','replicate']):
  coefficients={}
  for group in s.POPS.group.unique():
   ids=[p['agent_id'] for p in people if p['group']==group]
   x=block.pivot(index='date',columns='agent_id',values='p_entry').reindex(index=train.date,columns=ids).to_numpy()
   y=train[group].to_numpy(float);scale=max(float(y.mean()),1.)
   # Pre-fixed small ridge penalty. No weekday residual factors and no forced visitor shares.
   c=lsq_linear(np.vstack([x,np.sqrt(.01)*np.eye(3)]),np.r_[y/scale,np.zeros(3)],bounds=(0,np.inf),tol=1e-10).x*scale
   coefficients.update(zip(ids,c))
   weights.extend(dict(variant=variant,replicate=rep,group=group,agent_id=a,effective_count=float(w)) for a,w in zip(ids,c))
  for _,r in pd.concat([train,test]).iterrows():
   b=block[block.date==r.date].set_index('agent_id').reindex([p['agent_id'] for p in people]);w=[coefficients[a] for a in b.index]
   m=StationDay(w,b.p_entry.to_numpy());m.step()
   rec=dict(variant=variant,replicate=rep,date=r.date,actual=r.total_ridership,predicted=m.entries)
   (pred if r.date in set(test.date) else fit).append(rec)
 p=pd.DataFrame(pred);p.to_csv(FOLDER+'/predictions.csv',index=False);pd.DataFrame(fit).to_csv(FOLDER+'/training_fit.csv',index=False);pd.DataFrame(weights).to_csv(FOLDER+'/weights.csv',index=False)
 summary=[]
 for (variant,rep),b in p.groupby(['variant','replicate']):summary.append(dict(variant=variant,replicate=rep,wape_pct=100*abs(b.predicted-b.actual).sum()/b.actual.sum(),score=100*(1-abs(b.predicted-b.actual).sum()/b.actual.sum())))
 pd.DataFrame(summary).to_csv(FOLDER+'/scores.csv',index=False)
 logs=api.ledger();usage=dict(tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in logs),input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in logs),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in logs),valid_calls=sum(r['status']=='valid' for r in logs))
 usage['estimated_cny']=usage['input_tokens']*.8/1e6+usage['output_tokens']*2.7/1e6
 json.dump(usage,open(FOLDER+'/usage.json','w'),indent=2)
 print(p.groupby(['variant','date'])[['actual','predicted']].mean().to_string());print(pd.DataFrame(summary).to_string(index=False));print(usage)

def run():
 work=jobs();json.dump(dict(station='448',training_dates=train.date.tolist(),test_dates=test.date.tolist(),replicates=2,unique_requests=len(work),token_cap=api.CAP,calibration='Per fare group nonnegative least squares, normalized target, fixed ridge .01, no weekday factors or profile-share constraints. Identical in both arms.',meaning='Generic trip x subway mode vs local opportunity x THIS-station entry. Station arm additionally specifies resident, local errands, potential event visitor relationships.',limitations='Known-case diagnostic; changed personas and output semantics together. No causal attribution to a single change. No added activity data. Previous full-network predictions unchanged.',system=api.ROLE_SYSTEM),open(FOLDER+'/protocol.json','w'),indent=2)
 print('Requests',len(work),'estimated prompt tokens',sum(len(json.dumps(j['view']))/3+400 for j in work),flush=True)
 if not args.execute:return
 records=[]
 def call(j):return j,api.request('station_role',j['view'],j['person'],replicate=j['replicate'])
 with ThreadPoolExecutor(max_workers=3) as pool:
  for n,(j,a) in enumerate(pool.map(call,work),1):
   records.append({**{k:j[k] for k in ['variant','replicate','date','agent_id']},**a,'p_entry':a['p_opportunity']*a['p_conditional']})
   if n%60==0:print('Completed',n,'/',len(work),flush=True)
 for client in clients:Client.close(client)
 pd.DataFrame(records).to_csv(FOLDER+'/decisions_private.csv',index=False);report()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--report',action='store_true');args=parser.parse_args()
 if args.report:report()
 else:run()
