"""Small synthetic animation feasibility sample, NOT inferred observed OD or routes."""
import json
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ACTIONS

def make_demo():
    os.makedirs('res/animation_demo',exist_ok=True)
    stations=pd.read_csv('../../data/supplementary/station_complexes_current.csv',dtype={'complex_id':str,'gtfs_stop_ids':str})
    edges=pd.read_csv('../../data/supplementary/network_edges_current.csv',dtype=str)
    nodes=stations[['complex_id','display_name','latitude','longitude']].rename(columns={'complex_id':'station_id','display_name':'name'})
    nodes.to_csv('res/animation_demo/nodes.csv',index=False)
    mapping={stop.strip():row.complex_id for _,row in stations.iterrows() for stop in row.gtfs_stop_ids.split(';')}
    graph=nx.DiGraph();graph.add_nodes_from(nodes.station_id)
    for _,r in edges.iterrows():
        a=mapping.get(r.from_stop_id.rstrip('NS'));b=mapping.get(r.to_stop_id.rstrip('NS'))
        if a is None or b is None or a==b:continue
        seconds=float(r.scheduled_travel_time_s)
        if not graph.has_edge(a,b) or seconds<graph[a][b]['seconds']:graph.add_edge(a,b,seconds=seconds,route=r.route_id)
    pd.DataFrame([dict(from_station=a,to_station=b,seconds=v['seconds'],route=v['route']) for a,b,v in graph.edges(data=True)]).to_csv('res/animation_demo/edges.csv',index=False)
    dec=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str})
    dec=dec[(dec.task=='Extreme')&(dec.date=='2023-09-29')&(dec.method=='role_no_history')]
    rng=np.random.default_rng(5011);agents=[];waypoints=[];paths=[];unroutable=0
    coords=nodes.set_index('station_id');people=pd.read_csv('res/personas.csv').set_index('agent_id')
    for origin,part in dec.groupby('station_complex_id'):
        distance,route=nx.single_source_dijkstra(graph,origin,weight='seconds')
        destinations=[s for s,t in distance.items() if 300<=t<=3600]
        for i in range(20):
            chosen=part.iloc[rng.choice(len(part),p=part.effective_population/part.effective_population.sum())]
            action=rng.choice(ACTIONS,p=chosen[['p_'+a for a in ACTIONS]].to_numpy(float))
            aid=f'demo_{origin}_{i:02d}';profile=people.loc[chosen.agent_id]
            person=dict(id=aid,origin_station=origin,profile_id=chosen.agent_id,group=chosen.group,
                identity=profile.identity,action=action,represented_opportunities=part.effective_population.sum()/20,
                reason=chosen.reason,synthetic=True)
            agents.append(person)
            if action!='subway':continue
            if not destinations:unroutable+=1;continue
            destination=str(rng.choice(destinations));path=route[destination]
            departure=int(rng.integers(7*3600,20*3600));elapsed=0
            paths.append(dict(agent_id=aid,origin_station=origin,destination_station=destination,departure_seconds=departure,arrival_seconds=departure+distance[destination],route_station_ids=';'.join(path),destination_source='seeded synthetic reachable destination'))
            for order,station in enumerate(path):
                if order:elapsed+=graph[path[order-1]][station]['seconds']
                waypoints.append(dict(agent_id=aid,sequence=order,station_id=station,seconds=departure+elapsed,
                    longitude=float(coords.loc[station,'longitude']),latitude=float(coords.loc[station,'latitude'])))
    pd.DataFrame(agents).to_csv('res/animation_demo/agents.csv',index=False)
    pd.DataFrame(paths).to_csv('res/animation_demo/trips.csv',index=False)
    pd.DataFrame(waypoints).to_csv('res/animation_demo/waypoints.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,8))
    for a,b in graph.edges:
        ax.plot([coords.loc[a,'longitude'],coords.loc[b,'longitude']],[coords.loc[a,'latitude'],coords.loc[b,'latitude']],color='#d5dbe0',linewidth=.5,zorder=1)
    ax.scatter(nodes.longitude,nodes.latitude,s=4,color='#9fa9af',zorder=2)
    for _,r in pd.DataFrame(waypoints).groupby('agent_id'):
        ax.plot(r.longitude,r.latitude,alpha=.55,linewidth=1.2,color='#527fb3',zorder=3)
    for station in dec.station_complex_id.unique():
        x,y=coords.loc[station,['longitude','latitude']];ax.scatter([x],[y],marker='*',s=100,color='#cb6a53',zorder=4);ax.annotate(station,(x,y),xytext=(5,5),textcoords='offset points')
    ax.set(title=f'Animation feasibility: {len(agents)} synthetic people, {len(paths)} sampled subway trips\nCurrent network; synthetic destinations; schematic paths',xlabel='Longitude',ylabel='Latitude')
    ax.set_aspect(1/np.cos(np.deg2rad(40.7)));fig.tight_layout();fig.savefig('res/figures/11_animation_feasibility.png',dpi=160);plt.close(fig)
    checks=dict(nodes=len(nodes),directed_edges=graph.number_of_edges(),weak_components=nx.number_weakly_connected_components(graph),
        synthetic_agents=len(agents),subway_trips=len(paths),unroutable_subway_choices=unroutable,
        all_coordinates_finite=bool(np.isfinite(nodes[['latitude','longitude']].to_numpy()).all()),
        paths_follow_edges=all(graph.has_edge(a,b) for p in paths for a,b in zip(p['route_station_ids'].split(';')[:-1],p['route_station_ids'].split(';')[1:])),
        scope='Current network; synthetic destinations/departure times; schematic station-to-station interpolation; no verified 2023 closures, OD, platform transfer feasibility, or true paths.')
    json.dump(checks,open('res/animation_demo/feasibility.json','w'),indent=2);print(checks)

if __name__=='__main__':make_demo()
