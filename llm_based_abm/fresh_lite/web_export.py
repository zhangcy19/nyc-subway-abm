"""All-network synthetic animation data. No claim to observed OD or validated network forecasts."""
import os,json,glob,heapq,collections
import numpy as np
import pandas as pd
from prepare import CHANNELS,GROUPS
from experiment import load_policy
from model import fit_population,predict,profiles
os.makedirs('web_data',exist_ok=True)
os.makedirs('../../../presentation_local/build/evidence',exist_ok=True)
stations=pd.read_csv('../../data/stations.csv',dtype={'station_complex_id':str});stations.to_csv('web_data/stations.csv',index=False)
dates=pd.read_csv('res/dates.csv');days=dates[(dates.task=='Extreme_0929')&(dates.split=='train')]
chunks=[]
for f in ['../../data/station_hourly.csv.gz']+glob.glob('../../data/extensions/*.csv.gz'):
 for x in pd.read_csv(f,dtype={'station_complex_id':str},chunksize=100000):
  x['date']=x.timestamp.str[:10];x=x[x.date.isin(days.date)].copy()
  if x.empty:continue
  for g,cols in CHANNELS.items():x[g]=x[cols].sum(axis=1)
  chunks.append(x[['station_complex_id','date','total_ridership']+GROUPS])
x=pd.concat(chunks);train=x.groupby(['station_complex_id','date'])[['total_ridership']+GROUPS].sum(min_count=24).reset_index().dropna().merge(days,on='date')
val=pd.read_csv('res/validation.csv');best=val[(val.task=='Extreme_0929')&(val.method=='role_with_history')].sort_values('validation_wape').iloc[0]
weights=fit_population(train,load_policy('Extreme_0929','role_with_history'),profiles(),best.shrink,best.calendar_prior)
pd.DataFrame(weights).to_csv('web_data/network_population_weights.csv',index=False)
scenario=json.load(open('res/supplement/scenario_policy.json'));counts=[];dec=[]
for name in ['dry','storm']:
 policy={(r['date'],r['agent_id']):r for r in scenario if r['scenario']==name}
 frame=stations[['station_complex_id']].copy();frame['date']='2026-10-23';frame['weekday']=4;frame['total_ridership']=np.nan
 rows,d=predict(frame,weights,policy,profiles());counts.extend(dict(scenario=name,provenance='synthetic scenario; 2023 demand scale; transferred role probabilities',**r) for r in rows);dec.extend(dict(scenario=name,**r) for r in d)
flow=pd.DataFrame(counts).drop(columns='actual');fitted_ids=set(r['station_complex_id'] for r in weights);flow['fit_available']=flow.station_complex_id.isin(fitted_ids);flow.loc[~flow.fit_available,'predicted']=np.nan;flow.to_csv('web_data/station_flows.csv',index=False)
# Current GTFS complex graph. Complex aggregation assumes transfers; no accessibility claim.
meta=pd.read_csv('../../data/supplementary/station_complexes_current.csv',dtype=str).fillna('');mapping={}
for _,r in meta.iterrows():
 for stop in r.gtfs_stop_ids.split(';'):mapping[stop]=r.complex_id
edges=pd.read_csv('../../data/supplementary/network_edges_current.csv',dtype={'from_stop_id':str,'to_stop_id':str,'route_id':str});graph=collections.defaultdict(dict)
for _,r in edges.iterrows():
 a=mapping.get(r.from_stop_id.rstrip('NS'));b=mapping.get(r.to_stop_id.rstrip('NS'))
 if a and b and a!=b:
  value=(float(r.scheduled_travel_time_s),r.route_id)
  if b not in graph[a] or value[0]<graph[a][b][0]:graph[a][b]=value
xy={r.station_complex_id:[float(r.longitude),float(r.latitude)] for _,r in meta.rename(columns={'complex_id':'station_complex_id'}).iterrows() if r.longitude and r.latitude}
features=[]
for a,nexts in graph.items():
 for b,(t,route) in nexts.items():
  if a in xy and b in xy:features.append(dict(type='Feature',properties=dict(from_station=a,to_station=b,route_id=route,seconds=t,geometry_type='straight station-to-station schematic',accessible_path='unknown'),geometry=dict(type='LineString',coordinates=[xy[a],xy[b]])))
json.dump(dict(type='FeatureCollection',features=features),open('web_data/network.geojson','w'))

def paths(origin):
 dist={origin:0.};prev={};q=[(0.,origin)]
 while q:
  t,a=heapq.heappop(q)
  if t!=dist[a]:continue
  for b,(cost,line) in graph[a].items():
   new=t+cost
   if new<dist.get(b,float('inf')):dist[b]=new;prev[b]=a;heapq.heappush(q,(new,b))
 return dist,prev
attraction=train.groupby('station_complex_id').total_ridership.mean().to_dict();out=[];routes={};coverage=[]
for origin in stations.station_complex_id:
 dist,prev=paths(origin);choices=[d for d in stations.station_complex_id if d!=origin and d in dist and dist[d]>0]
 candidates=sorted(choices,key=lambda d:attraction.get(d,0)/(dist[d]/60+5)**1.4,reverse=True)[:12]
 score=np.array([attraction.get(d,0)/(dist[d]/60+5)**1.4 for d in candidates]);score=score/score.sum() if score.sum() else score
 coverage.append(dict(station_complex_id=origin,reachable_destinations=len(choices),exported_destinations=len(candidates)))
 if origin not in fitted_ids:continue
 for dest,share in zip(candidates,score):
  path=[dest]
  while path[-1]!=origin:path.append(prev[path[-1]])
  path=path[::-1];rid=origin+'_'+dest;elapsed=[0.]
  for a,b in zip(path[:-1],path[1:]):elapsed.append(elapsed[-1]+graph[a][b][0])
  routes[rid]=dict(route_id=rid,stations=path,coordinates=[xy[s] for s in path],elapsed_seconds=elapsed,geometry_type='schematic GTFS connections, not track geometry')
  for scenario_name in ['dry','storm']:
   volume=float(flow[(flow.station_complex_id==origin)&(flow.scenario==scenario_name)].predicted.iloc[0])
   out.append(dict(scenario=scenario_name,origin=origin,destination=dest,route_id=rid,expected_trips=volume*float(share),destination_probability=float(share),travel_seconds=dist[dest],od_source='synthetic gravity allocation; not observed OD'))
pd.DataFrame(out).to_csv('web_data/od_flows.csv',index=False);pd.DataFrame(coverage).to_csv('web_data/coverage.csv',index=False)
json.dump(list(routes.values()),open('web_data/routes.json','w'))
# Individuals are illustrative members, not estimated people. One fixed quantile per person.
D=pd.DataFrame(dec);individuals=[]
for who,aid,origin,age,u in [('adult','general_0','311',35,.50),('student','students_0','385',14,.65),('older_adult','seniors_disability_2','256',68,.60)]:
 dest=next((r['destination'] for r in out if r['origin']==origin),None)
 for name in ['dry','storm']:
  r=D[(D.station_complex_id==origin)&(D.agent_id==aid)&(D.scenario==name)].iloc[0]
  modes=['subway','bus','taxi','walk','stay_home'];ps=np.array([r['p_'+m] for m in modes]);mode=modes[min(np.searchsorted(ps.cumsum(),u),4)]
  individuals.append(dict(individual_id=who,synthetic_age=age,agent_id=aid,scenario=name,origin=origin,destination=dest,decision=mode,choice_quantile=u,probabilities=dict(zip(modes,ps.tolist())),route_id=origin+'_'+dest if mode=='subway' and dest else None,reason=r.reason,provenance='Illustrative synthetic individual; age, destination and sampled action are not observed',departure_hour=8 if aid.endswith('_0') else 18))
# Text summaries are private/local under the user's publishing preference.
json.dump(individuals,open('../../../presentation_local/build/evidence/individuals_private.json','w'),indent=2)
public=[{k:v for k,v in r.items() if k!='reason'} for r in individuals];json.dump(public,open('web_data/individuals.json','w'),indent=2)
# Check conservation and route continuity.
od=pd.DataFrame(out);check=od.groupby(['scenario','origin']).expected_trips.sum().reset_index().merge(flow,left_on=['scenario','origin'],right_on=['scenario','station_complex_id']);err=float(abs(check.expected_trips-check.predicted).max())
if err>1e-6:raise ValueError('OD does not conserve origin demand')
json.dump(dict(stations=len(stations),fitted_stations=len(set(r['station_complex_id'] for r in weights)),od_origins=od.origin.nunique(),unroutable=[r['station_complex_id'] for r in coverage if not r['exported_destinations']],max_conservation_error=err,unfitted_stations=sorted(set(stations.station_complex_id)-fitted_ids),forecast_evaluated_stations=4,scenario_date='2026-10-23',warning='Full-network output is a synthetic transfer demonstration, not a validated all-station evaluation'),open('web_data/validation.json','w'),indent=2)
print('Network export complete',len(out),'OD rows',len(routes),'routes; conservation error',err)
