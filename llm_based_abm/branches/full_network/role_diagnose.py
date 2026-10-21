import pandas as pd,json
D=pd.read_csv('prepared/dates.csv');pop=pd.read_csv('staged_res/populations.csv');dec=pd.read_csv('staged_res/decisions.csv');pred=pd.read_csv('staged_res/prediction_details.csv')
for season in ['Normal_Summer','Normal_School']:
 print('\n',season)
 for method in ['role_no_history','role_with_history']:
  pol=pd.DataFrame(json.load(open(f'staged_res/{season}_{method}_policy.json')));pol['p']=pol.probabilities.str[0];pol=pol.merge(D[D.task==season][['date','split']],on='date');pol['dow']=pd.to_datetime(pol.date).dt.weekday
  print(method,'probabilities',pol.groupby('split').p.mean().round(3).to_dict())
  train=pol[pol.split=='train'];means=train.groupby(['agent_id','dow']).p.mean()
  q=dec[(dec.task==season)&(dec.method==method)].copy();q['dow']=pd.to_datetime(q.date).dt.weekday
  q['ref_p']=[means.get((a,d),train[train.agent_id==a].p.mean()) for a,d in zip(q.agent_id,q.dow)]
  q['cf']=q.effective_population*q.ref_p
  out=q.groupby(['station_complex_id','date']).cf.sum().reset_index().merge(pred[(pred.task==season)&(pred.method==method)][['station_complex_id','date','actual']],on=['station_complex_id','date'])
  print('fixed weekday train-prob replay score',100*(1-abs(out.cf-out.actual).sum()/out.actual.sum()))
  print(pol.groupby(['agent_id','split']).p.mean().unstack().round(3).to_string())
old=pd.read_csv('../event_holdout/res/task_metrics.csv');print(old[(old.scope=='primary') & old.task.str.startswith('Normal') & old.method.str.startswith('role')].to_string(index=False))
