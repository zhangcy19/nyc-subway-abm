"""Audit and report the frozen rollback expansion without making API requests."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from design import inputs,METHODS
from staged import metrics
folder='rollback_full_res'
p=pd.read_csv(folder+'/predictions.csv',dtype={'station_complex_id':str})
ids=p.station_complex_id.unique().tolist();source,w=inputs(ids)
x=p.merge(source[['task','station_complex_id','date','event_day']],on=['task','station_complex_id','date'],validate='many_to_one');x['regime']=np.where(x.task.str.startswith('Normal'),'Normal','Extreme');x['abs_error']=abs(x.predicted-x.actual)
x.to_csv(folder+'/prediction_details.csv',index=False)
g=metrics(x[(x.regime=='Normal')|x.event_day]);s=g.groupby(['regime','method']).score.mean().unstack('regime')
window=metrics(x).groupby(['regime','method']).score.mean().unstack('regime')
window.to_csv(folder+'/window_summary.csv')
gates=json.load(open(folder+'/gates.json'));passed=all(r['batch_pass'] and r['cumulative_expansion_pass'] for r in gates)
orig=pd.read_csv('staged_res/predictions.csv',dtype={'station_complex_id':str});keys=['task','method','station_complex_id','date'];o=orig[orig.station_complex_id.isin(ids)]
a=p.merge(o[keys+['actual']],on=keys,suffixes=('','_original'),validate='one_to_one')
prior=pd.read_csv('rollback_half_res/predictions.csv',dtype={'station_complex_id':str});c=p.merge(prior[keys+['predicted']],on=keys,suffixes=('','_prior'),validate='one_to_one')
pop=pd.read_csv(folder+'/expansion_populations.csv');dec=pd.read_csv(folder+'/expansion_decisions.csv')
probs=['p_subway','p_bus','p_taxi','p_walk','p_stay_home']
audit=dict(stations=len(ids),prediction_rows=len(p),expected_rows=len(ids)*34*7,no_duplicate_keys=not p.duplicated(keys).any(),actuals_unchanged=bool((a.actual==a.actual_original).all()),valid_predictions=bool(np.isfinite(p.predicted).all() and p.predicted.ge(0).all()),previous_32_station_max_prediction_difference=float(abs(c.predicted-c.predicted_prior).max()),only_normal_population_fits=bool(pop.task.str.startswith('Normal').all()),decision_probabilities_sum_to_one=bool(np.allclose(dec[probs].sum(axis=1),1)),gates_passed=passed,completed_full_cohort=len(ids)==152,new_api_calls=0,new_tokens=0,estimated_new_api_cny=0)
json.dump(audit,open(folder+'/audit.json','w'),indent=2)
labels={'weekday_mean':'Weekday mean','ridge':'Ridge','rule_daily':'Rule daily (original)','rule_timing':'Rule timing (original)','role_no_history':'Role zero-shot (original policy)','role_with_history':'Role few-shot (original policy)','direct_few':'Direct few-shot (existing)'}
lines=['# 回退方案全量分批结果','',f'完成{len(ids)}站，含12个开发站和{len(ids)-12}个扩展站。新增API调用0，token 0，费用0元。'+('全部阶段通过放宽后的门槛。' if passed else '触发门槛，已停止进一步扩量。'),'', '## 方法与口径','', '恢复event_holdout旧版角色真实缓存回复，在各站Normal训练数据上重新拟合人口。超参数固定为12个开发站验证集选出的值，Extreme沿用对应季节Normal人口。旧角色参考来自原两个站点，概率全站共享；直接LLM保留现有full_network逐站预测；规则恢复旧公式。这是混合配置的缓存策略迁移实验，不是重新调用LLM的稳定性复验。','', '152站和这些日期此前已评价过，不是新的独立盲测。各站不因成绩被删除；评分为100×(1−WAPE)，Normal两段、Extreme四中心日各自等权；每任务内按逐站绝对误差汇总。','', '## 全部已完成站点','', '| 方法 | Normal | Extreme中心日 | Extreme五日窗口 |','|---|---:|---:|---:|']
for m in METHODS:lines.append(f'| {labels[m]} | {s.loc[m,"Normal"]:.2f} | {s.loc[m,"Extreme"]:.2f} | {window.loc[m,"Extreme"]:.2f} |')
lines += ['', '## 扩展站单独统计（不含开发12站）','', '| 方法 | Normal | Extreme |','|---|---:|---:|']
e=pd.read_csv(folder+'/expansion_summary.csv').set_index('method')
for m in METHODS:lines.append(f'| {labels[m]} | {e.loc[m,"Normal"]:.2f} | {e.loc[m,"Extreme"]:.2f} |')
lines += ['', '## 分阶段门槛','', '每批及累计扩展分别要求：角色few-shot在Normal、Extreme都高于角色zero-shot；两类任务中至少一类最佳LLM第一，另一类至少第二。排名按所有7个配置计算。用户在已完成32站之后放宽了门槛；本次没有按后续结果继续改变门槛。','']
for b in gates:lines.append(f'- 扩展批次{b["batch"]}：{len(b["stations"])}站；本批通过={b["batch_pass"]}，累计扩展通过={b["cumulative_expansion_pass"]}。')
lines += ['', '复现：在full_network目录用dev环境运行rollback_full.py，再运行report_rollback_full.py；二者均不请求API。旧结果保留，未同步GitHub。','', '数值文件：all_tasks.csv分六任务；prediction_details.csv逐站日；expansion_populations.csv与expansion_decisions.csv为新增站点权重及决策，开发站对应文件在rollback_dev_res。']
open(folder+'/RESULTS.md','w').write('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(13,5.8));colors=['#adb5bd','#687b90','#c3a787','#a57c4d','#74b7ba','#147d86','#d96848']
for ax,r in zip(axes,['Normal','Extreme']):
 vals=s.loc[METHODS,r];bars=ax.barh(range(7),vals,color=colors);ax.set_yticks(range(7),[labels[m] for m in METHODS]);ax.invert_yaxis();ax.set_xlim(0,104);ax.set_title(r);ax.set_xlabel('100 × (1 − WAPE); higher is better');ax.spines[['top','right']].set_visible(False)
 for b,v in zip(bars,vals):ax.text(v+.5,b.get_y()+b.get_height()/2,f'{v:.2f}',va='center',fontsize=9,fontweight='bold' if v==max(vals) else 'normal')
fig.suptitle(f'{len(ids)} stations · frozen original role policies · no station exclusion');fig.tight_layout();fig.savefig(folder+'/performance.png',dpi=170);plt.close(fig)
print(s.round(3).to_string());print(audit)
