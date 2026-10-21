"""Report all paired ablations and both replicates, including negative findings."""
import json,os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F='side_studies_res'
p=pd.read_csv(F+'/predictions.csv',dtype={'station_complex_id':str});p['ae']=abs(p.predicted-p.actual);p['error']=p.predicted-p.actual
rows=[]
for (study,method,condition,rep),g in p.groupby(['study','method','condition','replicate']):
 cohorts={'all_3':g}
 for sid,z in g.groupby('station_complex_id'):cohorts[sid]=z
 if study=='activity':cohorts['controls']=g[g.station_complex_id!='448']
 for cohort,z in cohorts.items():
  blocks=z.groupby('task') if study=='rain' else [('three_days',z)]
  for block,a in blocks:
   rows.append(dict(study=study,cohort=cohort,method=method,condition=condition,replicate=rep,block=block,score=100*(1-a.ae.sum()/a.actual.sum()),bias=100*a.error.sum()/a.actual.sum(),n=len(a)))
t=pd.DataFrame(rows);t.to_csv(F+'/scores_by_event_replicate.csv',index=False)
r=t.groupby(['study','cohort','method','condition','replicate'])[['score','bias']].mean().reset_index();r.to_csv(F+'/scores_by_replicate.csv',index=False)
s=r.groupby(['study','cohort','method','condition']).agg(score=('score','mean'),score_min=('score','min'),score_max=('score','max'),bias=('bias','mean')).reset_index();s.to_csv(F+'/summary.csv',index=False)
# Check expected unique station-day predictions and numeric invariants.
keys=['study','task','date','station_complex_id','method','condition','replicate']
mp=pd.read_csv(F+'/request_map.csv',dtype={'station_complex_id':str});logs=[json.loads(l) for l in open(F+'/api_calls.jsonl')];byhash={a['hash']:a for a in logs if a['status']=='valid'}
checks=dict(rows=len(p),expected_rows=252,no_duplicate_prediction_keys=not p.duplicated(keys).any(),all_finite_nonnegative=bool(np.isfinite(p.predicted).all() and p.predicted.ge(0).all()),all_requests_valid=bool(set(mp.request_hash)<=set(byhash)),all_three_stations=sorted(p.station_complex_id.unique().tolist()),two_replicates=sorted(p.replicate.unique().tolist()),paired_prompt_check=json.load(open(F+'/protocol.json'))['paired_payload_check'])
json.dump(checks,open(F+'/audit.json','w'),indent=2)
methods=['direct_few','role_no_history','role_with_history'];labels=['Direct few-shot','Role zero-shot','Role few-shot'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12.5,5.3))
for ax,study,cohort,conds,title in [(axes[0],'rain','all_3',['daily','hourly'],'Rainfall information | 3 stations, 4 events'),(axes[1],'activity','448',['without_event','with_event'],'Baseball calendar | Mets-Willets Point, Oct 6–8')]:
 z=s[(s.study==study)&(s.cohort==cohort)]
 for j,cond in enumerate(conds):
  a=z[z.condition==cond].set_index('method').loc[methods];pos=np.arange(3)+(j-.5)*.34
  bars=ax.bar(pos,a.score,width=.32,color=['#9aabb9','#16858b'][j],label=cond.replace('_',' '));ax.errorbar(pos,a.score,yerr=np.vstack([a.score-a.score_min,a.score_max-a.score]),fmt='none',ecolor='#283747',capsize=4,lw=1)
  for bar,v,lo,hi in zip(bars,a.score,a.score_min,a.score_max):ax.annotate(f'{v:.1f}',(bar.get_x()+bar.get_width()/2,hi if v>=0 else lo),xytext=(0,5 if v>=0 else -13),textcoords='offset points',ha='center',fontsize=10)
 ax.axhline(0,color='#9ca3af',lw=.8);ax.set_xticks(range(3),labels,rotation=12);ax.set_title(title,fontsize=11);ax.set_ylabel('100 × (1 − WAPE)');ax.legend(frameon=False);ax.margins(y=.22)
fig.suptitle('Fresh Lite calls · fixed population weights · mean of two runs (whiskers: run range)',fontsize=11);fig.tight_layout();fig.savefig(F+'/comparison.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True);z=p[(p.study=='activity')&(p.station_complex_id=='448')]
for ax,m,label in zip(axes,methods,labels):
 a=z[z.method==m];dates=sorted(a.date.unique());actual=a.drop_duplicates('date').set_index('date').loc[dates,'actual'];ax.plot(dates,actual,color='#192f44',marker='o',lw=2.5,label='Observed')
 for cond,c in [('without_event','#9aabb9'),('with_event','#16858b')]:
  v=a[a.condition==cond].groupby('date').predicted.agg(['mean','min','max']).loc[dates];ax.plot(dates,v['mean'],marker='o',color=c,label=cond.replace('_',' '));ax.fill_between(dates,v['min'],v['max'],color=c,alpha=.18)
 ax.set_xticks(dates,[d[5:] for d in dates]);ax.set_title(label);ax.set_ylabel('Daily entries');ax.grid(alpha=.15)
handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False);fig.suptitle('Mets-Willets Point | Oct 6 dry · Oct 7 rainy · Oct 8 dry | no Mets home fixtures');fig.tight_layout(rect=[0,.1,1,.93]);fig.savefig(F+'/activity_predictions.png',dpi=180);plt.close(fig)
# Save verbatim representative requests and brief decisions, no credentials.
examples=[]
for study,date,sid,method in [('rain','2023-09-29','448','role_with_history'),('activity','2023-10-07','448','role_with_history'),('activity','2023-10-07','448','direct_few')]:
 subset=mp[(mp.study==study)&(mp.date==date)&(mp.station_complex_id==sid)&(mp.method==method)&(mp.replicate==0)]
 if method.startswith('role'):subset=subset[subset.agent_id=='general_1']
 for _,a in subset.iterrows():
  b=byhash[a.request_hash];examples+=['## '+study+' '+method+' '+a.condition,'','```text',b['request']['system'],'```','','```json',json.dumps(b['request']['view'],indent=2),'```','','Actual response:','```json',b['raw'],'```','']
open(F+'/prompt_examples_private.md','w').write('\n'.join(examples))
# Precise repeat-level gains, not best-of-two selection.
gains=[]
for study,conds in [('rain',['daily','hourly']),('activity',['without_event','with_event'])]:
 a=r[r.study==study].pivot(index=['cohort','method','replicate'],columns='condition',values='score');a['gain']=a[conds[1]]-a[conds[0]];a=a.reset_index();a['study']=study;gains.append(a)
pd.concat(gains).to_csv(F+'/paired_gains.csv',index=False)
usage=json.load(open(F+'/usage.json'))
lines=['# 三站小支线：雨量时段与球赛信息','', '两类实验均为本轮真实Doubao Lite调用。两个重复全部保留，主表报告重复得分均值，不挑最好一次。角色人口和星期系数固定；只替换信息，不重新校准。','', '站点：Astoria-Ditmars Blvd（1）、135 St（152）、Mets-Willets Point（448）。活动支线以448为主，另外两站为地理对照；不是活动因果效应估计。','', '## 降雨信息','', '四个Extreme中心日：7/29、8/13、9/29、10/7。两组均给日雨量、温度、风速、日历；小时组额外提供目标与所有历史日完整24小时雨量。两组均不另外给小时峰值/六小时峰值，因此信息差只在完整小时序列。角色few-shot保留旧两站的群体事件参考，直接few-shot给本站历史。','', '| 方法 | 日汇总 | 完整小时 | 差值 |','|---|---:|---:|---:|']
for m,label in zip(methods,labels):
 a=s[(s.study=='rain')&(s.cohort=='all_3')&(s.method==m)].set_index('condition');lines.append(f'| {label} | {a.loc["daily","score"]:.2f} | {a.loc["hourly","score"]:.2f} | {a.loc["hourly","score"]-a.loc["daily","score"]:+.2f} |')
lines+=['','## 活动信息','', '预测10/6–10/8：一个雨日及前后两个干日。使用相同15个非暑期Normal训练日；有信息组仅增加Mets主场球赛日历、时间和本站与球场关系，没有提供入场人数、目标客流或人工客流修正系数。历史延期赛和未实际续赛记录排除。只覆盖棒球，不代表其他活动不存在。','', '| Mets-Willets Point | 无活动信息 | 有活动信息 | 差值 |','|---|---:|---:|---:|']
for m,label in zip(methods,labels):
 a=s[(s.study=='activity')&(s.cohort=='448')&(s.method==m)].set_index('condition');lines.append(f'| {label} | {a.loc["without_event","score"]:.2f} | {a.loc["with_event","score"]:.2f} | {a.loc["with_event","score"]-a.loc["without_event","score"]:+.2f} |')
lines+=['', '## 解释边界','', '得分为100×(1−WAPE)，可为负数。活动信息同时标注历史与目标，这是信息包消融，不能区分单独哪条标注的贡献。固定画像没有球迷角色，固定人口/星期系数可能已吸收历史活动量；若角色方案仍失败，不能靠修改测试人口掩盖。两次重复不足以证明显著性，且这些站点/日期此前已分析过，属于探索性小实验。','', '不加入“LLM必须赢”的停止条件，不按结果选择示例、重复或日期。不能据此说规则模型无法接入活动特征；本次没有测试有活动特征的规则模型。','', f'实际报告tokens：{usage["reported_tokens"]:,}；按此前输入0.8/输出2.7元每百万估算{usage["estimated_cny"]:.2f}元（非账户账单）；保守计数{usage["budget_tokens"]:,}。失败和已被数据修正替代的早期请求也计入用量。','', '来源：[MLB官方赛程接口](https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=121&startDate=2023-07-01&endDate=2023-10-31)。原始源文件和清洗后的event_calendar.csv保留。','', '文件：summary.csv、scores_by_event_replicate.csv、paired_gains.csv、predictions.csv；图comparison.png、activity_predictions.png。原始请求与理由仅本地，未上传GitHub。']
lines += ['', '## 可用于展示的结论', '', '1. 小时雨量没有稳定的整体优势：三个方法的两次配对得分变化均一正一负。8/13三种方法的平均得分均改善；9/29直接预测改善而角色方案退步。不要概括为小时信息必然更好。', '', '2. 球赛信息对直接预测的改善集中在10/8晴天：真实1933，无信息两次均预测8144；有信息两次预测2100和3000，均值2550。无信息模型明确引用唯一训练周日9/17的8144；有信息模型明确区分无球赛周日。', '', '3. 原来最关心的10/7暴雨日没有被直接预测修好：真实1323，无信息平均3560，有信息平均3985，反而略差。不能把三天整体改善写成暴雨日成功。角色few-shot这一天从4735降至3958，仍严重高估。', '', '4. 角色方案的三天总体改善只有约3分，两次重复方向相反，且仍大幅高估。固定画像没有专门球迷群体，人口/星期系数保持不变；这是值得后续验证的结构限制，尚未单独证明因果。', '', '完整逐日表见activity_daily_means.csv；所有重复变化见paired_gains.csv。']
activity_mean=p[p.study=='activity'].groupby(['station_complex_id','date','method','condition']).agg(actual=('actual','first'),predicted_mean=('predicted','mean'),predicted_min=('predicted','min'),predicted_max=('predicted','max')).reset_index();activity_mean.to_csv(F+'/activity_daily_means.csv',index=False)
open(F+'/RESULTS.md','w').write('\n'.join(lines)+'\n')
print(s[(s.cohort=='all_3')|(s.cohort=='448')].round(3).to_string(index=False));print(checks);print(usage)
