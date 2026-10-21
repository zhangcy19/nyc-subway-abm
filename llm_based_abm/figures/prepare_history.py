import pandas as pd,json,glob,hashlib
p=pd.read_csv('figures/data/mainline_predictions.csv',dtype={'station_complex_id':str})
scores=pd.read_csv('figures/data/mainline_scores.csv').set_index('method')
selected=[]
for regime in ['Normal','Extreme']:
 method=scores.loc[['direct_few','role_no_history','role_with_history'],regime].idxmax()
 x=p[(p.regime==regime)&(p.method==method)].copy();x['absolute_error']=(x.predicted-x.actual).abs()
 g=x.groupby(['task','station_complex_id'])[['absolute_error','actual']].sum().reset_index();g['window_score']=100*(1-g.absolute_error/g.actual)
 best=g.sort_values(['window_score','task','station_complex_id'],ascending=[False,True,True]).iloc[0]
 selected.append(dict(regime=regime,method=method,task=best.task,station_complex_id=best.station_complex_id,window_score=best.window_score,selection='Best full-test-window station case for the LLM method with best regime headline score; illustrative, selected after evaluation.'))
pd.DataFrame(selected).to_csv('figures/data/selected_cases.csv',index=False)
ids={str(r['station_complex_id']) for r in selected}|{'448'}
files=['../data/station_hourly.csv.gz']+sorted(glob.glob('../data/extensions/station_hourly_2023_*.csv.gz'))
parts=[]
for f in files:
 for x in pd.read_csv(f,usecols=['timestamp','station_complex_id','total_ridership'],dtype={'station_complex_id':str},chunksize=250000):
  parts.append(x[x.station_complex_id.isin(ids)])
h=pd.concat(parts,ignore_index=True).drop_duplicates(['timestamp','station_complex_id'],keep='last');h['date']=h.timestamp.str[:10]
d=h.groupby(['station_complex_id','date']).agg(total_ridership=('total_ridership','sum'),observed_hours=('timestamp','nunique')).reset_index()
d=d[d.observed_hours==24];d.to_csv('figures/data/history_daily.csv',index=False)
manifest=json.load(open('figures/data/provenance.json'))
manifest['history_selection']=selected
manifest['history_sources']={f:hashlib.sha256(open(f,'rb').read()).hexdigest() for f in files}
manifest['runtime_note']='C2 adds one measured local numerical replay to SUM of accepted API service seconds, a serial-equivalent estimate, not original parallel wall-clock.'
json.dump(manifest,open('figures/data/provenance.json','w'),indent=2)
print(pd.DataFrame(selected).to_string(index=False));print('Historical daily rows:',len(d),'selected IDs:',ids)
