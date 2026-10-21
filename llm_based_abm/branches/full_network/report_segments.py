"""Report frozen full-network results by season and individual held-out event."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('staged_res/segments',exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
methods=['weekday_mean','ridge','rule_daily','rule_timing','direct_few','role_no_history','role_with_history']
labels=['Weekday mean','Ridge','Rule: daily','Rule: timing','LLM direct: few-shot','LLM role: zero-shot','LLM role: few-shot']
colors=['#9ca3af','#687b90','#c3a787','#a57c4d','#d96848','#74b7ba','#147d86']
norm=['Normal_Summer','Normal_School'];ext=['Extreme_0729','Extreme_0813','Extreme_0929','Extreme_1007']
titles={'Normal_Summer':'Summer | Aug 17–23','Normal_School':'School term | Oct 23–29','Extreme_0729':'Jul 29 | 16.26 mm/day','Extreme_0813':'Aug 13 | 16.00 mm/day','Extreme_0929':'Sep 29 | 207.01 mm/day','Extreme_1007':'Oct 7 | 47.75 mm/day'}
s=pd.read_csv('staged_res/completed_task_metrics.csv')
p=pd.read_csv('staged_res/prediction_details.csv')
primary=p[(p.regime=='Normal')|p.event_day].copy()
g=primary.groupby(['task','method'])[['actual','predicted']].sum();g['bias_pct']=100*(g.predicted/g.actual-1)
s=s.merge(g.reset_index(),on=['task','method']);s.to_csv('staged_res/segments/scores_and_bias.csv',index=False)
for tasks,filename in [(norm,'normal_scores'),(ext,'extreme_scores')]:
 fig,axes=plt.subplots(len(tasks)//2,2,figsize=(13,4.4*(len(tasks)//2)),squeeze=False)
 for ax,task in zip(axes.flat,tasks):
  scores=s[s.task==task].set_index('method').loc[methods,'score']
  bars=ax.barh(np.arange(7),scores,color=colors,height=.67)
  ax.set_yticks(np.arange(7),labels);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_title(titles[task],loc='left',fontweight='bold',pad=12)
  ax.set_xlabel('Score = 100 × (1 − WAPE); higher is better');ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
  for bar,value in zip(bars,scores):ax.text(value+.7,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontweight='bold' if value==scores.max() else 'normal')
 fig.suptitle('152 stations · all sites retained · '+('7 test days per season' if tasks==norm else 'each event center day scored separately'),fontsize=13)
 fig.tight_layout(rect=[0,0,1,.95]);fig.savefig('staged_res/segments/'+filename+'.png',dpi=180);plt.close(fig)
for tasks,filename in [(norm,'normal_daily_flows'),(ext,'extreme_daily_flows')]:
 fig,axes=plt.subplots(len(tasks)//2,2,figsize=(13,4.2*(len(tasks)//2)),squeeze=False)
 for ax,task in zip(axes.flat,tasks):
  z=p[p.task==task];a=z.drop_duplicates(['station_complex_id','date']).groupby('date').actual.sum();dates=list(a.index);x=np.arange(len(dates))
  ax.plot(x,a.values/1e6,color='#172c40',lw=3,label='Observed',marker='o',markersize=4,zorder=10)
  for method,label,color in zip(methods,labels,colors):
   v=z[z.method==method].groupby('date').predicted.sum().reindex(dates)
   ax.plot(x,v.values/1e6,color=color,lw=1.6,label=label,ls='--' if method in ['rule_timing','role_no_history'] else '-')
  if task in ext:ax.axvspan(1.7,2.3,color='#f7d5ca',alpha=.55,zorder=0)
  ax.set_xticks(x,[d[5:] for d in dates]);ax.set_title(titles[task],loc='left',fontweight='bold');ax.set_ylabel('Entries across 152 stations (million)');ax.grid(alpha=.13)
 handles,labs=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=4,frameon=False)
 fig.suptitle('Daily total flows · curves may hide offsetting station errors; scores use station-level absolute errors',fontsize=11)
 fig.tight_layout(rect=[0,.13 if tasks==norm else .08,1,.94]);fig.savefig('staged_res/segments/'+filename+'.png',dpi=180);plt.close(fig)
wide=s.pivot(index='method',columns='task',values='score').reindex(methods)[norm+ext]
lines=['# 分时段结果','', '152站全部保留，不重新训练、不调用API。得分为100×(1−WAPE)，不是分类准确率；Normal按每段7个测试日计算，Extreme按中心日计算。曲线包含Extreme前后恢复日，但中心日得分不混入恢复日。','', '| 方法 | 暑期 | 非暑期 | 7/29 | 8/13 | 9/29 | 10/7 |','|---|---:|---:|---:|---:|---:|---:|']
for method,label in zip(methods,labels):lines.append('| '+label+' | '+' | '.join(('**'+f'{v:.2f}'+'**') if v==wide[col].max() else f'{v:.2f}' for col,v in wide.loc[method].items())+' |')
lines += ['', '规则时段相对日汇总变化：暑期−0.55，非暑期−0.59，7/29 +2.14，8/13 +0.12，9/29 −3.44，10/7 −23.54。10/7时段规则总体低估35.12%，日汇总高估6.01%。', '', '时段规则使用假设的画像出发时间及25%全天/75%出发时段暴露组合；人数和系数分别校准，因此这里是完整方案比较，不是只增加天气字段的严格消融。10/7时段方案在其他事件上选中的降雨系数为1，日汇总为0.5；两种公式系数的含义不同，不能简单解释为惩罚翻倍。', '', 'Normal暑期角色few-shot高估12.17%，非暑期低估3.23%；9/29直接LLM总体偏差−2.22%，但站点绝对误差仍19.21%，总量接近不代表各站准确。', '', '图：normal_scores.png、extreme_scores.png为各任务得分；normal_daily_flows.png、extreme_daily_flows.png为逐日总量。数值见scores_and_bias.csv。']
open('staged_res/segments/README.md','w').write('\n'.join(lines)+'\n')
print(wide.round(2).to_string())
