"""Offline post-hoc sensitivity; run from full_network. Original results are unchanged."""
import os
import pandas as pd
import numpy as np

os.makedirs('staged_res/role_outliers', exist_ok=True)
p = pd.read_csv('staged_res/prediction_details.csv', dtype={'station_complex_id': str})
names = pd.read_csv('../../../data/stations.csv', dtype={'station_complex_id': str})
roles = ['role_no_history', 'role_with_history']
p = p[p.method.isin(roles + ['weekday_mean'])].copy()
p['ae'] = abs(p.predicted - p.actual)
p['signed_error'] = p.predicted - p.actual
primary = p[(p.regime == 'Normal') | p.event_day.astype(str).str.lower().eq('true')].copy()
t = primary.groupby(['regime', 'task', 'station_complex_id', 'method']).agg(actual=('actual','sum'), predicted=('predicted','sum'), ae=('ae','sum'), signed_error=('signed_error','sum')).reset_index()
t['wape'] = 100*t.ae/t.actual
t['bias'] = 100*t.signed_error/t.actual
t['score'] = 100-t.wape
t.merge(names,on='station_complex_id').to_csv('staged_res/role_outliers/station_task_errors.csv', index=False)
s = t.groupby(['regime','station_complex_id','method'])[['wape','bias','score']].mean().reset_index()
s = s.merge(names,on='station_complex_id')
s.to_csv('staged_res/role_outliers/station_errors.csv',index=False)
results=[];counts=[];rankings=[];contributions=[]
for regime in ['Normal','Extreme']:
    z = primary[primary.regime.eq(regime)]
    w = s[s.regime.eq(regime)].pivot(index='station_complex_id',columns='method',values='wape')
    w['worst_role_wape'] = w[roles].max(axis=1)
    w['few_minus_zero_wape'] = w.role_with_history-w.role_no_history
    w = w.sort_values('worst_role_wape',ascending=False)
    rank = w.reset_index().merge(names,on='station_complex_id')
    rank['regime']=regime
    rankings.append(rank)
    counts.append(dict(regime=regime,stations=len(w),few_worse=int((w.few_minus_zero_wape>0).sum()),median_few_minus_zero=float(w.few_minus_zero_wape.median()),zero_over_50=int((w.role_no_history>50).sum()),few_over_50=int((w.role_with_history>50).sum())))
    scenarios = {'all':[], 'remove_448':['448']}
    for k in [1,3,5]:scenarios[f'remove_worst_{k}']=list(w.index[:k])
    scenarios['remove_either_role_wape_over_50']=list(w.index[w.worst_role_wape>50])
    for scenario,removed in scenarios.items():
        for scope,src in [('primary',z),('all_test_days',p[p.regime.eq(regime)])]:
            kept=src[~src.station_complex_id.isin(removed)]
            q=kept.groupby(['task','method'])[['ae','actual']].sum()
            q['score']=100*(1-q.ae/q.actual)
            for method,score in q.groupby('method').score.mean().items():
                results.append(dict(regime=regime,scope=scope,scenario=scenario,removed_count=len(removed),retained_count=kept.station_complex_id.nunique(),removed_ids=';'.join(removed),method=method,score=score))
    # Additive contributions to the official few-shot minus zero-shot WAPE gap.
    u=t[t.regime.eq(regime)&t.method.isin(roles)].copy()
    u['denominator']=u.groupby(['task','method']).actual.transform('sum')
    u['contribution']=100*u.ae/u.denominator/u.task.nunique()
    c=u.groupby(['station_complex_id','method']).contribution.sum().unstack()
    c['few_minus_zero_contribution']=c.role_with_history-c.role_no_history
    c=c.sort_values('few_minus_zero_contribution',ascending=False).reset_index().merge(names,on='station_complex_id')
    c['regime']=regime;contributions.append(c)
result=pd.DataFrame(results)
result.to_csv('staged_res/role_outliers/sensitivity.csv',index=False)
pd.DataFrame(counts).to_csv('staged_res/role_outliers/paired_summary.csv',index=False)
pd.concat(rankings).to_csv('staged_res/role_outliers/ranked_stations.csv',index=False)
pd.concat(contributions).to_csv('staged_res/role_outliers/error_contributions.csv',index=False)
for regime in ['Normal','Extreme']:
    print('\n',regime)
    print(result[(result.regime==regime)&(result.scope=='primary')].pivot(index=['scenario','removed_count','retained_count'],columns='method',values='score').round(3).to_string())
    print(pd.concat(rankings).query('regime == @regime').head(10)[['station_complex_id','station_name',*roles,'few_minus_zero_wape']].round(2).to_string(index=False))
    print('Largest contributors to few-shot deterioration:')
    print(pd.concat(contributions).query('regime == @regime').head(5)[['station_complex_id','station_name','few_minus_zero_contribution']].round(3).to_string(index=False))
print(pd.DataFrame(counts).to_string(index=False))
# A compact handoff report; all exclusions are post-hoc, not a revised test set.
def markdown(df):
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(rows)
report=['# 角色 LLM：异常站点敏感性分析','',
'原始152站结果不变。本分析不调用API，不重新训练，也不修改预测。按每站两个角色方法中较大的WAPE排序，Normal和Extreme分别剔除最差1、3、5站；同一行所有方法使用相同保留站点。站点排名按任务等权，汇总得分仍先在每个任务内用客流加权WAPE，再对任务等权平均。Normal为两个任务，Extreme为四个事件中心日。',
'', '得分 = 100 × (1 − WAPE)，不是分类准确率。极端误差可能超过100%，对应负分。剔除根据测试误差选择，只能说明异常站影响，不能作为新的正式模型成绩。','']
for regime in ['Normal','Extreme']:
    report+=['## '+regime,'',markdown(result[(result.regime==regime)&(result.scope=='primary')].pivot(index=['scenario','removed_count'],columns='method',values='score').reset_index()),'', '误差最高的10站（以下数值是WAPE%，越低越好）：','',markdown(pd.concat(rankings).query('regime == @regime').head(10)[['station_complex_id','station_name',*roles]]),'']
report+=['## few-shot退步是否集中在个别站点','',markdown(pd.DataFrame(counts)),'']
a=primary.groupby(['task','method'])[['ae','actual','signed_error']].sum()
a['score']=100*(1-a.ae/a.actual)
a['bias_pct']=100*a.signed_error/a.actual
a=a.reset_index();a=a[a.method.isin(roles)]
a.to_csv('staged_res/role_outliers/task_scores_and_bias.csv',index=False)
report += [markdown(a[['task','method','score','bias_pct']]),'',
'Normal的主要问题集中在暑期：few-shot得分84.52，zero-shot为91.97；非暑期few-shot反而更好（92.65 vs 91.92）。暑期few-shot总体高估12.17%，zero-shot高估4.96%。因此不能笼统说历史信息总是有害，也不能把当前退步归因于一两个异常站。',
'', 'Mets-Willets Point（448）确实异常：10月7日实际1323，zero-shot预测8152，few-shot预测6201。但它在7月29日和8月13日的few-shot误差分别只有1.20%和0.05%。不能把整个站点所有日期都当作无效数据。当前仅证实预测异常，尚未证明原始数据错误。',
'', '文件：station_errors.csv为全站分方法分天气汇总；station_task_errors.csv为各时段/事件明细；ranked_stations.csv为排序；sensitivity.csv含每种剔除方案、剔除站ID和恢复期结果；error_contributions.csv分解各站对few-shot退步的贡献。']
open('staged_res/role_outliers/README.md','w').write('\n'.join(report)+'\n')
