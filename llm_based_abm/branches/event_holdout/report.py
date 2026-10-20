"""Human-readable results and cost summary from completed local artifacts."""
import json,hashlib
import pandas as pd
s=pd.read_csv('res/summary.csv');m=pd.read_csv('res/task_metrics.csv');a=json.load(open('res/result_audit.json'));r=pd.read_csv('res/runtime.csv');y=pd.read_csv('res/prediction_details.csv')
order=['weekday_mean','ridge','rule_daily','rule_timing','role_no_history','role_with_history','direct_few'];names=['星期均值','Ridge','规则 ABM：日汇总','规则 ABM：加入时段','角色 LLM：zero-shot','角色 LLM：few-shot','直接 LLM：few-shot'];tab=s[s.scope=='primary'].pivot(index='method',columns='regime',values='accuracy_percent')
t='''# 固定 Normal 人口、事件留一：两站实验结果

**本次没有在任何 Extreme 任务重新拟合人口。角色 few-shot 在四个事件中心日的平均得分最高，但事件差异明显，不能据此说所有暴雨都预测得好。**

站点：Parsons Blvd（256）与176 St（385）。统一使用原JFK天气，模型为Doubao Lite。Normal人口按暑期/非暑期各校准一次；四个事件的角色few-shot和直接few-shot均获得另外三个事件的完整五日参考。允许参考事件晚于测试事件；这是回顾性未见事件泛化，不是严格按时间向前预报。

## 平均结果

得分为100×(1−WAPE)，越高越好，不是分类正确率。Normal先在两个任务内汇总两站，再对任务等权平均。Extreme先在每个中心日汇总两站，再对四个事件等权平均。共28个Normal测试站点日、8个Extreme中心站点日。

| 方法 | Normal | Extreme中心日 |
|---|---:|---:|
'''
for method,label in zip(order,names):
 vals=[f'**{tab.loc[method,c]:.2f}**' if tab.loc[method,c]==tab[c].max() else f'{tab.loc[method,c]:.2f}' for c in ['Normal','Extreme']];t+='| '+label+' | '+' | '.join(vals)+' |\n'
t+='''
![平均表现](res/performance.png)

角色few-shot比角色zero-shot的Extreme得分提高14.54个百分点；比Ridge高2.15个百分点。这里两者的Normal校准概率和人口参数分别拟合，故14.54个百分点是完整方案的差异，并不能全部归因于加入事件示例这一单项变化。直接few-shot的Extreme表现与星期均值接近。

## 四个事件分别看

| 方法 | 7/29 | 8/13 | 9/29 | 10/7 |
|---|---:|---:|---:|---:|
'''
e=m[(m.scope=='primary')&(m.regime=='Extreme')].pivot(index='method',columns='task',values='accuracy_percent');tasks=['Extreme_0729','Extreme_0813','Extreme_0929','Extreme_1007']
for method,label in zip(order,names):
 vals=[f'**{e.loc[method,c]:.2f}**' if e.loc[method,c]==e[c].max() else f'{e.loc[method,c]:.2f}' for c in tasks];t+='| '+label+' | '+' | '.join(vals)+' |\n'
t+='''
![事件客流](res/events.png)

- 9/29角色few-shot达到94.19%，但8/13只有43.85%。后者主要是高估客流，并不是一律过度取消出行。
- 7/29规则日汇总与星期均值表现较好；10/7角色few-shot较好。不存在一个方法每个事件都赢。
- 时段规则在10/7仅19.61%。该折基于其余三个事件选择了降雨系数2.0；未根据10/7成绩改档位。小样本选参、假设出发时段与固定人口之间的配合可能不稳定，不能把“加入时段”视为必然改进。

## 前后五天整体表现

如果把事件前后四天也一起计入，各方法会得到更高得分。这不应替代中心日的主结果：

| 方法 | 五日窗口得分 |
|---|---:|
'''
w=s[(s.scope=='all_test_days')&(s.regime=='Extreme')].set_index('method')
for method,label in zip(order,names):t+=f'| {label} | {w.loc[method,"accuracy_percent"]:.2f} |\n'
t+='''
五日窗口里星期均值最好（86.79%）；角色few-shot为82.18%。说明角色few-shot的平均优势集中在事件当天，并没有扩展到整个前后窗口。

## 固定人口是否与新的历史上下文兼容

额外选择每季一个无雨工作日、一个无雨周末日，保留相同天气、画像、人口，只把Normal历史换成各折另外三个事件的历史。此诊断不调整任何主结果。

| 留出事件的历史配置 | 干燥探针预测变化均值 | 最小至最大 |
|---|---:|---:|
'''
q=pd.read_csv('res/context_probe_predictions.csv');probe=q.pivot(index=['held_event','station_complex_id','date'],columns='context',values='predicted').reset_index();probe['change_pct']=100*(probe.other_events_history/probe.normal_history-1);probe.to_csv('res/context_probe_comparison.csv',index=False)
for held,g in probe.groupby('held_event'):t+=f'| {held} | {g.change_pct.mean():+.2f}% | {g.change_pct.min():+.2f}% 至 {g.change_pct.max():+.2f}% |\n'
t+='''
总体范围约−2%至+14%。因此即使人口参数完全固定，换历史上下文仍可能改变普通天气下的决策水平。这里只各调用一次，差异包含LLM采样噪声，尚不能精确分离上下文效应。不能声称固定人口已经消除了所有迁移问题。

## 具体如何保证没有重拟合或泄漏

- 人口参数文件只包含两个Normal任务，共192条画像参数；所有Extreme决策逐条核对，代表人数和星期系数与对应Normal记录一致。
- Normal所有日期与四个五日事件窗口互不重叠。目标事件的两站五天客流一并排除，不能从其他站“看到”同一事件。
- 输入检查将留出事件客流全部替换成异常值，角色和直接预测的提示输入不变。
- 角色zero-shot不提供历史。角色few-shot提供另外三个事件的人群相对客流，不是个人决策标签。
- 规则模型只在另外三个事件中心日上选择降雨响应系数，人口不变。Ridge固定Normal选出的正则化参数，用Normal训练集及另外三个事件的五日数据拟合回归系数；Ridge没有人口参数。
- 七种方法的评价日期完全一致，概率、人口份额10%–60%约束、Mesa加权求和、预算均通过检查。

## 成本与文件

'''
main=int(r.new_tokens.sum());t+=f'主实验消耗 **{main:,} token**；普通天气上下文诊断额外 **{a["charged_tokens"]-main:,} token**；合计 **{a["charged_tokens"]:,} token**，低于500万额度。没有使用reset，没有提交或推送GitHub。\n\n'
t+='运行说明见 [README](README.md)。数值结果在 `res/summary.csv`、`res/task_metrics.csv`、`res/prediction_details.csv`；群体概率和参数在 `res/decisions.csv`、`res/populations.csv`；完整性检查见 `res/result_audit.json`。原始API日志保留本地并已忽略。\n\n两站、四个已研究过的事件仍是小样本；本轮分数不是未触碰过的数据集上的最终泛化证明。统一JFK天气也不是各站真实局地天气。\n'
open('RESULTS.md','w').write(t)
ledger=[json.loads(line) for line in open('res/api_calls.jsonl')];cost=[]
for x in ledger:
 view=x['request']['view'];route=x['request']['route'];kind='event_context' if 'reference_events' in view else 'normal_history' if ('group_history' in view or route=='direct') else 'no_history'
 cost.append(dict(route=route,context=kind,attempts=1,charged_tokens=x.get('charged_tokens',x.get('usage',{}).get('total_tokens',0)),summed_request_seconds=x['elapsed_seconds']))
pd.DataFrame(cost).groupby(['route','context']).sum().reset_index().to_csv('res/api_cost.csv',index=False)
protocol=json.load(open('res/protocol.json'))
for file,h in protocol['files'].items():
 if hashlib.sha256(open(file,'rb').read()).hexdigest()!=h:raise ValueError('Frozen input changed '+file)
print('Report written; protocol hashes unchanged; cost',a['charged_tokens'])
