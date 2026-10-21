"""Audit and plot the isolated station-role case study; no additional API calls."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import station_role_branch as b

folder=b.FOLDER
p=pd.read_csv(folder+'/predictions.csv');d=pd.read_csv(folder+'/decisions_private.csv');w=pd.read_csv(folder+'/weights.csv')
logs={r['hash']:r for r in b.api.ledger() if r['status']=='valid'}
used=[logs[h] for h in d['hash']]
violations=[]
for r in used:
 v=r['request']['view'];date=v['target']['date']
 if any(h['date']==date or h['date'] in set(b.test.date) for h in v['observed_training_examples']):violations.append(r['hash'])
 if 'Target dates have no Mets fixture' in json.dumps(v):violations.append(r['hash'])
audit=dict(expected_decisions=864,actual_decisions=len(d),unique_decisions=len(d.drop_duplicates(['variant','replicate','date','agent_id'])),expected_test_predictions=12,actual_test_predictions=len(p),history_leak_or_calendar_text_violations=len(violations),all_finite=bool(np.isfinite(p.predicted).all()),nonnegative_coefficients=bool((w.effective_count>=0).all()),zero_test_dates_in_fit=not bool(set(b.train.date)&set(b.test.date)))
json.dump(audit,open(folder+'/audit.json','w'),indent=2)
x=d.merge(w,on=['variant','replicate','agent_id']);x['contribution']=x.p_entry*x.effective_count
x['relationship']=np.where(x.variant.eq('station'),x.agent_id.str[-1].map({'0':'Resident / regular','1':'Local errands','2':'Potential event visitor'}),x.agent_id.str[-1].map({'0':'Regular attendance','1':'Flexible errands','2':'Discretionary social / shopping'}))
x.to_csv(folder+'/contributions_private.csv',index=False)
x.groupby(['variant','date','relationship'])[['p_opportunity','p_conditional','p_entry','contribution']].mean().to_csv(folder+'/relationship_probabilities.csv')
contrib=x.groupby(['variant','replicate','date','relationship']).contribution.sum().reset_index();contrib.to_csv(folder+'/relationship_contributions.csv',index=False)
old=pd.read_csv('side_studies_res/predictions.csv',dtype={'station_complex_id':str});old=old.query("study=='activity' and station_complex_id=='448' and condition=='with_event' and method=='role_with_history'")
fig,ax=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.25,1]})
dates=sorted(p.date.unique());xx=np.arange(3)
ax[0].plot(xx,p.groupby('date').actual.first().reindex(dates),'-o',color='#1c2639',lw=2.5,label='Observed')
ax[0].plot(xx,old.groupby('date').predicted.mean().reindex(dates),'--',color='#9c9c9c',label='Previous role model')
for variant,color,label in [('generic','#547cad','Generic role, refitted'),('station','#e28042','Station-specific role')]:
 g=p[p.variant==variant].groupby('date').predicted.agg(['mean','min','max']).reindex(dates)
 ax[0].plot(xx,g['mean'],'-o',color=color,label=label);ax[0].fill_between(xx,g['min'],g['max'],color=color,alpha=.12)
ax[0].set(xticks=xx,xticklabels=['Oct 6','Oct 7\nRain','Oct 8'],ylabel='Daily entries',title='Mets-Willets Point: known-case diagnostic');ax[0].legend(fontsize=8,frameon=False)
c=contrib[(contrib.variant=='station')&contrib.date.isin(dates)].groupby(['date','relationship']).contribution.mean().unstack().reindex(dates)
bottom=np.zeros(3)
for label,color in [('Resident / regular','#547cad'),('Local errands','#60a99c'),('Potential event visitor','#e28042')]:
 ax[1].bar(xx,c[label],bottom=bottom,color=color,label=label,width=.6);bottom+=c[label].to_numpy()
ax[1].set(xticks=xx,xticklabels=['Oct 6','Oct 7','Oct 8'],ylabel='Predicted entries',title='Station-role contributions');ax[1].legend(fontsize=8,frameon=False)
ax[0].set_ylim(bottom=0);ax[1].set_ylim(0,float(bottom.max())*1.32)
for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
fig.text(.02,.01,'Two Lite runs; fixed training-only calibration. Persona relationships are assumed, not observed demographic labels.',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(folder+'/comparison.png',dpi=180);plt.close(fig)
print(audit)
print(p.groupby(['variant','date'])[['actual','predicted']].mean().to_string())
print(c.to_string())
scores=pd.read_csv(folder+'/scores.csv')
means=p.groupby(['variant','date']).predicted.mean().unstack(0)
oldmeans=old.groupby('date').predicted.mean()
text='''# Mets-Willets Point 站点级角色支线

本实验只研究站点448、2023年10月6–8日，属于已经看到过误差的案例诊断，不是独立泛化检验。未改变主实验或同步GitHub。

## 结果

所有人数是两次独立Lite调用及各自训练拟合后的均值。

|日期|真实客流|上一轮角色模型（有赛程）|匹配对照：一般地铁选择|本站进站选择|
|---|---:|---:|---:|---:|
'''
for date in dates:
 text+=f"|{date}|{p[p.date==date].actual.iloc[0]:,.0f}|{oldmeans[date]:,.0f}|{means.loc[date,'generic']:,.0f}|{means.loc[date,'station']:,.0f}|\n"
text+='\n三天整体得分为100×(1−WAPE)，每次重复先算分再平均。负值表示绝对误差大于真实客流总量。\n\n|方案|重复1|重复2|均值|\n|---|---:|---:|---:|\n'
for variant,label in [('generic','一般地铁选择，重新拟合'),('station','本站进站选择，重新拟合')]:
 a=scores[scores.variant==variant].sort_values('replicate').score
 text+=f'|{label}|{a.iloc[0]:.2f}|{a.iloc[1]:.2f}|{a.mean():.2f}|\n'
text+='''

本站方案相对匹配对照的三天WAPE由144.30%降至51.77%，相对减少64.1%；两次重复都改善，但绝对误差仍大。10/7仍预测2,584人，真实1,323人，不能据此声称暴雨预测已经解决。

一个明确的角色机制例子：普通成人潜在球场访客，9/17比赛日的本站进站概率两次为24.48%、24.50%；10/8无比赛时两次均为4.80%。回复明确指出“非居民、没有比赛、缺少访问这里的理由”。这是不同日期的说明性对比，不是保持天气和历史完全不变的独立赛程干预。

剩余问题也有具体线索：10/7三类角色的预测贡献约为居民557、普通办事775、潜在活动访客1,251。没有Mets比赛时，访客仍保留少量到访概率，乘较大的拟合代表系数后占近一半预测。这提示活动访客的剩余机会概率与有效系数还需更多非活动周末检验；没有真实个人或访客标签，不能把该项直接解释为实际球迷人数。
'''
text+='''
## 具体改了什么

两组使用相同15个Normal训练日、群体绝对客流、历史及目标日Mets赛程、目标日完整24小时雨量。历史雨量只保留日总量以减少重复token；两个组完全一致。原模型的历史是同星期指数，新两组都使用绝对群体客流，因此匹配对照用于区分“站点角色定义”与共同的输入、拟合变化。

- 一般地铁对照：P(今天出行) × P(出行时选择地铁)，仍是原12个通用画像。
- 站点方案：P(今天在本站周边产生离开该区域的出行机会) × P(有该机会时从本站进站)。
- 站点方案每个票价群体有3类假设关系：附近居民、普通办事者、潜在球场活动访客，共12个代表。原有预算、行动限制等属性保留。
- 不把进站和出站混淆：游客乘地铁到达球场时是出站，离开球场时才可能在本站进站。
- 没有手工指定“没比赛概率为0”，没有指定访客占比，没有把真实测试客流或期待概率传给LLM。

## 人数如何校准

两组分别在相同训练集拟合非负代表系数；每个票价群体拟合3个系数。目标先按训练均值缩放，固定岭惩罚0.01，不根据测试结果调参。取消原来的星期残差系数与10%–60%的画像占比限制，允许不存在的画像权重接近0。星期信息由LLM读取。

改变概率含义后重新拟合系数是必要的。两组执行同一种拟合，所以新方案相对“匹配对照”的差异比相对旧模型更有解释力。但站点方案同时改变出行机会定义和画像与车站的关系，不能将变化归因于某一句prompt。

所有训练日也先生成LLM概率，随后拟合人数。为防止同日客流直接进入自己的概率，训练日请求会删除该日历史记录；测试日只使用完整15个训练日。Mesa为每个代表创建agent，汇总代表系数×两步概率的乘积。

## 限制与使用方式

- 这是看到问题后针对性设计的单站支线；不能与原盲测分数混在一起，也不能据此证明所有站点都有效。
- 没有新增普通周末，原训练集周末样本不足仍存在；需要更多有活动/无活动周末验证。
- 人数系数是有效代表量，不是实际居民数、球迷数或人口占比。
- 活动表只覆盖Mets棒球赛。9/20等日期可能还有其他活动，本轮保持输入范围一致，没有偷偷补数据。
- 两次重复用于观察生成波动，不能作为充分的统计显著性检验。
- 不根据本次结果继续挑选惩罚、概率或人物比例；保留全部结果。

## 文件

- `../station_role_branch.py`：画像、prompt、真实API调用、训练期校准、Mesa汇总。
- `protocol.json`：固定方案、训练与测试日期、system prompt。
- `predictions.csv`、`scores.csv`：每次重复的结果。
- `weights.csv`、`training_fit.csv`：训练所得系数与拟合结果。
- `relationship_contributions.csv`：三类站点关系的预测贡献。
- `comparison.png`：预测曲线与站点角色贡献。
- `audit.json`：历史目标日期排除、结果覆盖和系数检查。
- `api_calls.jsonl`、`decisions_private.csv`、`contributions_private.csv`：本地原始记录，不上传。

复现：在本目录的父目录 full_network 运行 `python station_role_branch.py --execute`，相同请求使用缓存；再运行 `python report_station_role_branch.py`。使用dev环境。运行其他日期或改变方案需重新设置并明确预算。
'''
usage=json.load(open(folder+'/usage.json'))
text+=f"\n本支线账本累计报告 {usage['tokens']:,} token，估算 {usage['estimated_cny']:.2f} 元，包含早期被修正说明的调用及有返回用量的失败调用。价格沿用此前估值，并非账户账单。额外无返回的中断费用无法由API用量核实。\n"
open(folder+'/RESULTS.md','w').write(text)
