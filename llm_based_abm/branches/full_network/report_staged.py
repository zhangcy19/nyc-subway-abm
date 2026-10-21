"""Comparable results only: include stations whose complete seven-method batch finished."""
import json,os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from design import inputs,METHODS
from staged import metrics

c=json.load(open('staged_config.json'));folder=c['output'];out=json.load(open(folder+'/checkpoint.json'))
gates=out.get('batch_gates',[])
if not gates:raise RuntimeError('No complete comparison batch yet')
order=pd.read_csv(folder+'/batches.csv',dtype={'station_complex_id':str});done={g['batch'] for g in gates};ids=order[order.batch.isin(done)].station_complex_id.tolist()
p,_=inputs();y=pd.DataFrame(out['predictions']);y=y[y.station_complex_id.isin(ids)].merge(p[['task','station_complex_id','date','event_day']],on=['task','station_complex_id','date'],validate='many_to_one')
y['regime']=np.where(y.task.str.startswith('Normal'),'Normal','Extreme');primary=y[(y.regime=='Normal')|y.event_day]
tasks=metrics(primary);summary=tasks.groupby(['regime','method']).score.mean().unstack('regime').reindex(METHODS)
y.to_csv(folder+'/completed_station_predictions.csv',index=False);tasks.to_csv(folder+'/completed_task_metrics.csv',index=False);summary.to_csv(folder+'/completed_summary.csv')
windows=metrics(y).groupby(['regime','method']).score.mean().unstack('regime').reindex(METHODS);windows.to_csv(folder+'/completed_window_summary.csv')
progress=[]
for last in sorted(done):
 included=order[order.batch.le(last)].station_complex_id
 z=primary[primary.station_complex_id.isin(included)]
 m=metrics(z).groupby(['regime','method']).score.mean().reset_index()
 m['through_batch']=last;m['stations']=len(included);progress.append(m)
pd.concat(progress,ignore_index=True).to_csv(folder+'/cumulative_metrics.csv',index=False)
station_scores=pd.concat([metrics(g).assign(station_complex_id=sid) for sid,g in primary.groupby('station_complex_id')]).groupby(['station_complex_id','regime','method']).score.mean().reset_index()
station_scores.to_csv(folder+'/station_primary_scores.csv',index=False)
station_scores.groupby(['regime','method']).score.mean().reset_index().to_csv(folder+'/equal_station_summary.csv',index=False)
ledger=[json.loads(s) for s in open(folder+'/api_calls.jsonl')];usage=dict(input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in ledger),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in ledger),total_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in ledger),budget_tokens=sum(r.get('charged_tokens',r.get('usage',{}).get('total_tokens',0)) for r in ledger))
usage['estimated_cny_uncached']=usage['input_tokens']*.8/1e6+usage['output_tokens']*2.7/1e6
usage['end_to_end_minutes']=(os.stat(folder+'/checkpoint.json').st_mtime-os.stat(folder+'/manifest.json').st_birthtime)/60
json.dump(usage,open(folder+'/cost_summary.json','w'),indent=2)
labels=['Weekday mean','Ridge','Rule daily','Rule timing','Role zero-shot','Role few-shot','Direct few-shot'];colors=['#adb5bd','#929da8','#d6b58d','#b99671','#93c5cb','#25828d','#db8060']
fig,axes=plt.subplots(1,2,figsize=(12,5.5));plt.rcParams.update({'font.size':10})
for ax,regime in zip(axes,['Normal','Extreme']):
 bars=ax.bar(range(7),summary[regime],color=colors)
 ax.set_xticks(range(7),labels,rotation=35,ha='right');ax.set_title(regime);ax.set_ylabel('100 × (1 − WAPE)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.1f}',ha='center',fontsize=9)
 ax.set_ylim(min(0,float(summary[regime].min())-8),105)
fig.suptitle(f'{len(ids)} completed stations · fixed event-holdout protocol');fig.tight_layout();fig.savefig(folder+'/completed_performance.png',dpi=180);plt.close(fig)
text=['# 分批实验结果','',f'完成完整方法比较的站点：{len(ids)} / 152。共享角色计算已完成不代表直接预测全部完成；本报告只比较同一组完整站点。','', '| 方法 | Normal | Extreme中心日 | Extreme五天 |','|---|---:|---:|---:|']
for method,label in zip(METHODS,labels):text.append(f'| {label} | {summary.loc[method,"Normal"]:.2f} | {summary.loc[method,"Extreme"]:.2f} | {windows.loc[method,"Extreme"]:.2f} |')
text+=['','分数为100×(1−WAPE)，不是分类正确率；Normal两任务、Extreme四事件等权。没有排除异常站点。','', '## 批次检查','']
for gate in gates:text.append(f'- 第{gate["batch"]}批：{gate["status"]}；'+('; '.join(gate['issues']) or '未触发预设大范围失败条件。'))
text+=['',f'API报告token：{usage["total_tokens"]:,}；保守预算账本：{usage["budget_tokens"]:,} / 12,000,000。按输入0.8元/百万、输出2.7元/百万估算未优惠成本：{usage["estimated_cny_uncached"]:.2f}元。不是账户实际扣款。','', '角色策略和全站Normal校准一次完成，逐站直接预测分批推进。参数、提示词及共享参考不随批次结果修改。若提前停止，本结果仅代表已完成样本，不能外推成152站完整结论。']
open(folder+'/RESULTS.md','w').write('\n'.join(text)+'\n');print(summary.round(2).to_string());print(usage)
