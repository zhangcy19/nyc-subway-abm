"""Export saved Doubao Lite results. Run from mainline/; no API calls."""
import collections, hashlib, heapq, json, os
import numpy as np
import pandas as pd
from model import profiles, ACTIONS
from design import EVENTS
os.makedirs('../web_data',exist_ok=True)
def js(name,data):
    with open('../web_data/'+name,'w') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def csv(name,frame):frame.to_csv('../web_data/'+name,index=False)
def quantile(s):return int(hashlib.sha256(s.encode()).hexdigest()[:12],16)/16**12

# Use accepted Lite output, including the matching saved reasons.
d=pd.concat([pd.read_csv('../branches/full_network/'+f,dtype={'station_complex_id':str}) for f in ['rollback_dev_res/decisions.csv','rollback_full_res/expansion_decisions.csv']])
d=d[d.task.str.startswith('Extreme') & d.method.eq('role_with_history')].copy()
p=pd.read_csv('results/doubao_lite_reference/policies.csv')
p=p[p.task.str.startswith('Extreme') & p.method.eq('role_with_history')].copy()
p=p.merge(pd.DataFrame(json.load(open('../figures/data/lite_role_reasons.json'))),on=['task','method','date','agent_id'],validate='one_to_one')
p['policy_id']=p.task+'_'+p.date+'_'+p.agent_id
keys=['task','date','agent_id'];d=d.merge(p[keys+['policy_id']],on=keys,validate='many_to_one').sort_values(['task','date','station_complex_id','agent_id'])
if d.duplicated(['task','date','station_complex_id','agent_id']).any():raise ValueError('Duplicate decisions')
for a in ACTIONS:
    check=d.merge(p[keys+['p_'+a]],on=keys,suffixes=('_saved','_policy'))
    if not np.allclose(check['p_'+a+'_saved'],check['p_'+a+'_policy']):raise ValueError('Policy mismatch')
ids=set(d.station_complex_id)
stations=pd.read_csv('../figures/data/stations.csv',dtype={'station_complex_id':str})
csv('stations.csv',stations[stations.station_complex_id.isin(ids)])
js('profiles.json',profiles())
js('policies.json',[dict(policy_id=r.policy_id,task=r.task,date=r.date,agent_id=r.agent_id,probabilities={a:float(getattr(r,'p_'+a)) for a in ACTIONS},reason=r.reason) for r in p.itertuples()])
csv('station_decisions.csv',d[['task','date','station_complex_id','agent_id','policy_id','effective_population']])
f=pd.read_csv('results/doubao_lite_reference/predictions.csv',dtype={'station_complex_id':str})
f=f[f.task.str.startswith('Extreme') & f.method.eq('role_with_history')][['task','date','station_complex_id','actual','predicted']]
replay=d.assign(entries=d.effective_population*d.p_subway).groupby(['task','date','station_complex_id']).entries.sum().reset_index()
check=f.merge(replay,on=['task','date','station_complex_id'],validate='one_to_one');error=float(abs(check.predicted-check.entries).max())
if len(check)!=3040 or not np.allclose(check.predicted,check.entries,atol=.01):raise ValueError('Flow mismatch or incomplete coverage')
csv('station_flows.csv',f)
js('events.json',[dict(task=t,event_date=date,dates=sorted(f[f.task==t].date.unique())) for t,date in EVENTS.items()])
w=pd.read_csv('../../data/weather_hourly.csv');csv('weather_hourly.csv',w[w.timestamp.str[:10].isin(f.date.unique())])

# Schematic paths through the current GTFS graph, separate from LLM decisions.
m=pd.read_csv('../../data/supplementary/station_complexes_current.csv',dtype=str).fillna('')
mapping={stop.strip():r.complex_id for r in m.itertuples() for stop in r.gtfs_stop_ids.split(';')}
xy={r.complex_id:[float(r.longitude),float(r.latitude)] for r in m.itertuples() if r.longitude and r.latitude}
graph=collections.defaultdict(dict)
edges=pd.read_csv('../../data/supplementary/network_edges_current.csv',dtype={'from_stop_id':str,'to_stop_id':str,'route_id':str})
for r in edges.itertuples():
    a=mapping.get(r.from_stop_id.rstrip('NS'));b=mapping.get(r.to_stop_id.rstrip('NS'))
    if a and b and a!=b and a in xy and b in xy:
        value=(float(r.scheduled_travel_time_s),r.route_id)
        if b not in graph[a] or value[0]<graph[a][b][0]:graph[a][b]=value
js('network.geojson',dict(type='FeatureCollection',features=[dict(type='Feature',properties=dict(from_station=a,to_station=b,line=line,seconds=seconds),geometry=dict(type='LineString',coordinates=[xy[a],xy[b]])) for a,adj in graph.items() for b,(seconds,line) in adj.items()]))
csv('network_stations.csv',m[['complex_id','display_name','latitude','longitude']].rename(columns={'complex_id':'station_complex_id','display_name':'station_name'}))
panel=pd.read_csv('inputs/daily.csv',dtype={'station_complex_id':str});dates=pd.read_csv('inputs/dates.csv')
train_dates=dates[dates.task.str.startswith('Normal') & dates.split.eq('train')].date.unique()
attraction=panel[panel.date.isin(train_dates)].groupby('station_complex_id').total_ridership.mean().to_dict()
routes=[];candidates={};unroutable=[]
for origin in sorted(ids):
    dist={origin:0.};prev={};queue=[(0.,origin)]
    while queue:
        t,a=heapq.heappop(queue)
        if t!=dist[a]:continue
        for b,(seconds,line) in graph[a].items():
            new=t+seconds
            if new<dist.get(b,float('inf')):dist[b]=new;prev[b]=a;heapq.heappush(queue,(new,b))
    available=[dest for dest in sorted(ids) if dest!=origin and dest in dist and dist[dest]>0]
    selected=sorted(available,key=lambda dest:attraction.get(dest,0)/(dist[dest]/60+5)**1.4,reverse=True)[:12]
    if not selected:unroutable.append(origin);continue
    weights=np.array([attraction.get(dest,0)/(dist[dest]/60+5)**1.4 for dest in selected]);weights/=weights.sum()
    candidates[origin]=(selected,weights.cumsum())
    for dest in selected:
        path=[dest]
        while path[-1]!=origin:path.append(prev[path[-1]])
        path.reverse();elapsed=[0.];lines=[]
        for a,b in zip(path[:-1],path[1:]):
            seconds,line=graph[a][b];elapsed.append(elapsed[-1]+seconds);lines.append(line)
        routes.append(dict(route_id=origin+'_'+dest,stations=path,coordinates=[xy[s] for s in path],elapsed_seconds=elapsed,lines=lines))
js('routes.json',routes)
people={p['agent_id']:p for p in profiles()};individuals=[]
for r in d.itertuples():
    if r.effective_population<=0:continue
    who=r.station_complex_id+'_'+r.agent_id;u=quantile(who+'_mode')
    probabilities=np.array([getattr(r,'p_'+a) for a in ACTIONS]);mode=ACTIONS[min(int(np.searchsorted(probabilities.cumsum(),u)),4)]
    dest=None;route=None
    if mode=='subway' and r.station_complex_id in candidates:
        destinations,cumulative=candidates[r.station_complex_id]
        dest=destinations[min(int(np.searchsorted(cumulative,quantile(who+'_destination'))),len(destinations)-1)];route=r.station_complex_id+'_'+dest
    hours=[[7,8,9],[10,11,12,13,14,15,16],[17,18,19,20]][people[r.agent_id]['kind']]
    hour=hours[int(quantile(who+'_hour')*len(hours))] if mode!='stay_home' else None
    individuals.append(dict(individual_id=who,task=r.task,date=r.date,origin=r.station_complex_id,agent_id=r.agent_id,policy_id=r.policy_id,decision=mode,destination=dest,route_id=route,departure_hour=hour,display_weight=r.effective_population))
js('individuals.json',individuals)
route_ids={r['route_id'] for r in routes}
if any(i['route_id'] and i['route_id'] not in route_ids for i in individuals):raise ValueError('Missing route')
js('manifest.json',dict(model='doubao-seed-2-1-lite-260915',method='role_with_history',stations=len(ids),events=4,dates=20,profiles=12,policies=len(p),station_decisions=len(d),individuals=len(individuals),routes=len(routes),unroutable_origins=unroutable,max_flow_replay_error=error,source='mainline/results/doubao_lite_reference',routes_source='current GTFS graph; synthetic gravity destinations; straight schematic segments',individual_source='fixed quantile sampling from saved probabilities; synthetic representatives, not observed travelers',reason_source='saved short decision summaries, not chain-of-thought',timezone='America/New_York'))
print('Exported',len(d),'decisions,',len(individuals),'individual-day records,',len(routes),'routes; flow error',error)
