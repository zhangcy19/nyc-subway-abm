"""One frozen full-cohort experiment, evaluated/released in twenty-station batches.

Shared LLM policies and Normal calibration are computed once. Only station-specific
paid calls expand batch by batch. No model/prompt changes based on batch test scores.
"""
import argparse,json,os,time
import numpy as np
import pandas as pd
import api
import run as runner
from design import inputs,METHODS,EVENTS

PHASES=['Normal_Summer','Normal_School']+list(EVENTS)

def batches(p,out):
 path=out+'/batches.csv'
 if os.path.exists(path):return pd.read_csv(path,dtype={'station_complex_id':str})
 stations=pd.read_csv('../../../data/stations.csv',dtype={'station_complex_id':str})
 train=p[p.task.str.startswith('Normal')&p.split.eq('train')]
 meta=train.groupby('station_complex_id').total_ridership.mean().rename('training_mean').reset_index().merge(stations,on='station_complex_id')
 # Geography/volume stratification uses training features only, never test errors.
 meta['volume_band']=pd.qcut(meta.training_mean,4,labels=False,duplicates='drop')
 rng=np.random.default_rng(5011);groups=[]
 for _,g in meta.groupby(['borough','volume_band']):groups.append(list(rng.permutation(g.station_complex_id.to_numpy())))
 sequence=[]
 while any(groups):
  order=rng.permutation(len(groups))
  for i in order:
   if groups[i]:sequence.append(groups[i].pop())
 order=pd.DataFrame({'station_complex_id':sequence,'position':range(len(sequence))});order['batch']=order.position//20+1
 order.merge(meta,on='station_complex_id').to_csv(path,index=False)
 return order.merge(meta,on='station_complex_id')

def metrics(frame):
 rows=[]
 for (regime,task,method),g in frame.groupby(['regime','task','method']):
  error=abs(g.predicted-g.actual);rows.append(dict(regime=regime,task=task,method=method,score=100*(1-error.sum()/g.actual.sum())))
 return pd.DataFrame(rows)

def evaluate(out,p,ids,batch):
 y=pd.DataFrame(out['predictions']);y=y[y.station_complex_id.isin(ids)]
 y=y.merge(p[['task','station_complex_id','date','event_day']],on=['task','station_complex_id','date'],validate='many_to_one')
 expected=len(ids)*34*7
 if len(y)!=expected or y.duplicated(['task','station_complex_id','date','method']).any():raise ValueError('Batch prediction coverage mismatch')
 if not np.isfinite(y.predicted).all() or (y.predicted<0).any():raise ValueError('Invalid prediction')
 y['regime']=np.where(y.task.str.startswith('Normal'),'Normal','Extreme')
 primary=y[(y.regime=='Normal')|y.event_day]
 detail=metrics(primary);summary=detail.groupby(['regime','method']).score.mean().reset_index();summary['rank']=summary.groupby('regime').score.rank(ascending=False,method='min')
 station=pd.concat([metrics(g).assign(station_complex_id=s) for s,g in primary.groupby('station_complex_id')]).groupby(['station_complex_id','regime','method']).score.mean().reset_index()
 folder=runner.OUT+'/batch_'+str(batch).zfill(2);os.makedirs(folder,exist_ok=True)
 summary.to_csv(folder+'/summary.csv',index=False);detail.to_csv(folder+'/events.csv',index=False);station.to_csv(folder+'/stations.csv',index=False)
 y.to_csv(folder+'/predictions.csv',index=False)
 # Predetermined broad-failure gate; never exclude outliers from headline metrics.
 issues=[];diagnostics=[]
 numerical=['weekday_mean','ridge','rule_daily','rule_timing']
 for regime,g in summary.groupby('regime'):
  scores=g.set_index('method').score
  baseline=scores[numerical].idxmax();pivot=station[station.regime==regime].pivot(index='station_complex_id',columns='method',values='score')
  direct_gap=float(scores[baseline]-scores['direct_few']);fraction=float(((pivot[baseline]-pivot.direct_few)>10).mean())
  best_llm=max(float(scores[m]) for m in ['direct_few','role_with_history','role_no_history'])
  bad_fraction=float((pivot[['direct_few','role_with_history','role_no_history']].max(axis=1)<50).mean())
  diagnostics.append(dict(regime=regime,best_numerical=baseline,direct_score_gap=direct_gap,fraction_stations_direct_gap_gt_10=fraction,best_llm_score=best_llm,fraction_stations_all_llm_below_50=bad_fraction))
  if direct_gap>10 and fraction>=.6:issues.append(regime+': direct LLM >10 points behind best numerical and >60% stations similarly behind')
  if best_llm<50 and bad_fraction>=.6:issues.append(regime+': all LLMs broadly below 50')
  # Removing two worst stations is diagnostic only; primary outputs retain them.
  worst=pivot.direct_few.nsmallest(min(2,len(pivot))).index.tolist();diagnostics[-1]['two_worst_direct_stations']=worst
 rows=api.ledger();usage=dict(reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in rows),conservative_tokens=sum(api.charge(r) for r in rows),input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in rows),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in rows))
 usage['estimated_cny_uncached']=usage['input_tokens']/1e6*.8+usage['output_tokens']/1e6*2.7
 report=dict(batch=batch,stations=len(ids),status='stop_for_review' if issues else 'pass',issues=issues,diagnostics=diagnostics,usage=usage)
 json.dump(report,open(folder+'/gate.json','w'),indent=2)
 print('BATCH',batch,report,flush=True);print(summary.pivot(index='method',columns='regime',values='score').round(2).to_string(),flush=True)
 return report

def stage(config,limit):
 runner.configure(config);p,w=inputs();order=batches(p,runner.OUT)
 # Freeze all shared history and calibration on the full eligible cohort, once.
 runner.CONFIG['methods']=METHODS[:-1]
 runner.run()
 runner.CONFIG.pop('methods')
 out=json.load(open(runner.OUT+'/checkpoint.json'));out.setdefault('batch_progress',[]);out.setdefault('batch_gates',[])
 completed={(r['batch'],r['task']) for r in out['batch_progress']}
 gates={r['batch']:r for r in out['batch_gates']}
 for batch,g in order.groupby('batch',sort=True):
  batch=int(batch)
  if limit is not None and batch>limit:break
  if batch in gates:
   if gates[batch]['status']!='pass':raise RuntimeError('Existing failed gate requires review; no further API calls')
   continue
  ids=g.station_complex_id.tolist()
  for phase in PHASES:
   if (batch,phase) in completed:continue
   start=time.perf_counter();before=sum(api.charge(r) for r in api.ledger())
   rows=runner.direct_calls(p,w,phase,False,ids)
   out['predictions'].extend(dict(task=phase,method='direct_few',**r) for r in rows)
   out['batch_progress'].append(dict(batch=batch,task=phase,seconds=time.perf_counter()-start,new_tokens=sum(api.charge(r) for r in api.ledger())-before))
   runner.save(out);print('Direct batch',batch,phase,'complete',flush=True)
  report=evaluate(out,p,ids,batch);out['batch_gates'].append(report);runner.save(out)
  if report['status']!='pass':print('STOP: batch review required; models unchanged.',flush=True);return
 if len(out['batch_gates'])==order.batch.nunique() and all(g['status']=='pass' for g in out['batch_gates']):
  for phase in PHASES:
   if any(r['task']==phase and r['method']=='direct_few' for r in out['runtime']):continue
   records=[r for r in out['batch_progress'] if r['task']==phase]
   out['runtime'].append(dict(task=phase,method='direct_few',seconds=sum(r['seconds'] for r in records),new_tokens=sum(r['new_tokens'] for r in records)))
  runner.save(out);print('ALL BATCHES COMPLETE',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--config',default='staged_config.json');a.add_argument('--through-batch',type=int,default=1);args=a.parse_args();stage(args.config,args.through_batch)
