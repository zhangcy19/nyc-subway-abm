"""Offline audit, summary tables and figures for this independent experiment."""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from api import ledger,charge,validate
from experiment import load_policy,PEOPLE,choose_population,RAIN
from model import predict,ACTIONS
from timing import timed_rule_policy,hourly_weather

NAMES={'weekday_mean':'星期均值','ridge':'Ridge','rule_daily':'规则ABM（日汇总）','rule_timing':'规则ABM（时段）','role_no_history':'Lite角色zero-shot（时段）','role_with_history':'Lite角色few-shot（时段）','direct_few':'Lite直接few-shot（时段）'}
ENGLISH={'weekday_mean':'Weekday mean','ridge':'Ridge','rule_daily':'Rule daily','rule_timing':'Rule timing','role_no_history':'Role zero-shot','role_with_history':'Role few-shot','direct_few':'Direct few-shot'}
ORDER=list(NAMES)

def metrics(p):
    p=p.dropna(subset=['actual','predicted']);e=p.predicted-p.actual;den=p.actual.sum()
    return dict(n=len(p),actual=den,predicted=p.predicted.sum(),WAPE_pct=100*abs(e).sum()/den if den>0 else np.nan,MAE=abs(e).mean(),RMSE=np.sqrt((e**2).mean()),bias_pct=100*e.sum()/den if den>0 else np.nan)

def table(p,keys):
    return pd.DataFrame([{**dict(zip(keys,k if isinstance(k,tuple) else [k])),**metrics(r)} for k,r in p.groupby(keys)])

def main():
    protocol=json.load(open('res/protocol.json'))
    for f,h in protocol['files'].items():
        if hashlib.sha256(open(f,'rb').read()).hexdigest()!=h:raise ValueError('Frozen file changed: '+f)
    p=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str});panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});dates=pd.read_csv('res/dates.csv')
    dec=pd.read_csv('res/decisions.csv',dtype={'station_complex_id':str});weights=pd.read_csv('res/populations.csv',dtype={'station_complex_id':str})
    if set(p.method)!=set(ORDER) or p.groupby('task').method.nunique().min()!=7 or p.task.nunique()!=6:raise ValueError('Incomplete experiment')
    calls=ledger();cost=sum(charge(r) for r in calls);byhash={r['hash']:r for r in calls if r['status']=='valid'}
    if cost>5000000:raise ValueError('Over budget')
    def logical(r):
        d=r['request'].copy();d.pop('attempt');return json.dumps(d,sort_keys=True)
    first_valid={}
    for r in calls:
        if r['status']=='valid':first_valid.setdefault(logical(r),r)
    if len(first_valid)!=json.load(open('res/preflight.json'))['unique_planned_requests']:raise ValueError('Planned request coverage incomplete')
    used=set(p.loc[p.method=='direct_few','request_hash']).union(dec.loc[dec.method.str.startswith('role_'),'request_hash'])
    for digest in used:
        if first_valid[logical(byhash[digest])]['hash']!=digest:raise ValueError('Later valid duplicate used instead of earliest response')
    for r in calls:
        if hashlib.sha256(json.dumps(r['request'],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=r['hash']:raise ValueError('Request hash mismatch')
        if r['status']=='valid':validate(r['parsed'],r['request']['route'],r['request']['view'].get('person'))
    for _,r in p[p.method=='direct_few'].iterrows():
        if r.predicted!=byhash[r.request_hash]['parsed']['entries']:raise ValueError('Direct output mismatch')
    for _,r in dec[dec.method.str.startswith('role_')].iterrows():np.testing.assert_allclose([r['p_'+a] for a in ACTIONS],byhash[r.request_hash]['parsed']['probabilities'],atol=1e-12)
    for task,part in p.groupby('task'):
        expected=panel[(panel.task==task)&(panel.split=='test')&panel.total_ridership.notna()]
        keys=set(zip(expected.station_complex_id,expected.date))
        for method,r in part.groupby('method'):
            r=r.dropna(subset=['actual'])
            if r.duplicated(['station_complex_id','date']).any() or set(zip(r.station_complex_id,r.date))!=keys:raise ValueError('Unequal scoring samples')
    replay=[];weather=hourly_weather()
    for (task,method),w in weights.groupby(['task','method']):
        policy=load_policy(task,method)
        if method.startswith('role_'):
            for result in policy.values():
                np.testing.assert_allclose(result['probabilities'],byhash[result['hash']]['parsed']['probabilities'],atol=1e-12)
                if first_valid[logical(byhash[result['hash']])]['hash']!=result['hash']:raise ValueError('Later duplicate in fitted policy')
        part=panel[panel.task==task];train=part[part.split=='train'];valid=part[part.split=='validation'];test=part[part.split=='test']
        policies={scale:timed_rule_policy(dates[dates.task==task],weather,scale,method=='rule_timing') for scale in RAIN} if method.startswith('rule_') else {0.:policy}
        refit,_,_,_=choose_population(train,valid,policies,method)
        refit=pd.DataFrame(refit).merge(w,on=['station_complex_id','agent_id'],suffixes=('_refit','_saved'))
        for column in ['effective_population']+[f'weekday_{i}' for i in range(7)]:np.testing.assert_allclose(refit[column+'_refit'],refit[column+'_saved'],atol=1e-6,rtol=1e-8)
        rows,_=predict(test,w.to_dict('records'),policy,PEOPLE)
        check=pd.DataFrame(rows).merge(p[(p.task==task)&(p.method==method)],on=['date','station_complex_id'],suffixes=('_replay','_saved'));error=float(abs(check.predicted_replay-check.predicted_saved).max());replay.append(error)
        if error>1e-7:raise ValueError('Replay mismatch')
    shares=weights.effective_population/weights.groupby(['task','method','station_complex_id','group']).effective_population.transform('sum')
    if shares.min()<.099999 or shares.max()>.600001:raise ValueError('Persona bounds failed')
    p=p.merge(dates[['task','date','event_day']],on=['task','date']);whole=table(p,['task','method']);whole.to_csv('res/metrics.csv',index=False)
    events=table(p[p.event_day],['task','method']);events.to_csv('res/event_metrics.csv',index=False)
    table(p,['task','method','station_complex_id']).to_csv('res/station_metrics.csv',index=False)
    observed=events[events.method=='weekday_mean'][['task','actual','predicted']].copy()
    observed['actual_vs_training_weekday_pct']=100*(observed.actual/observed.predicted-1)
    observed.to_csv('res/observed_weather_change.csv',index=False)
    summary=pd.concat([whole[whole.task.str.startswith('Normal')].groupby('method').WAPE_pct.mean().rename('Normal'),events.groupby('method').WAPE_pct.mean().rename('Extreme'),events[events.task!='Extreme_0929'].groupby('method').WAPE_pct.mean().rename('Extreme_without_0929')],axis=1).reindex(ORDER)
    summary.to_csv('res/summary.csv')
    dec['predicted']=dec.effective_population*dec.p_subway
    g=dec.groupby(['task','method','station_complex_id','date','group']).predicted.sum().reset_index()
    actual=panel[panel.split=='test'].melt(id_vars=['task','station_complex_id','date'],value_vars=['students','fair_fare','seniors_disability','general'],var_name='group',value_name='actual')
    table(g.merge(actual,on=['task','station_complex_id','date','group']),['task','method','group']).to_csv('res/group_metrics.csv',index=False)
    runtime=pd.read_csv('res/runtime.csv');runtime_scope='含新API调用的已完成阶段' if runtime.new_tokens.sum()>0 else '离线缓存回放阶段';budget=dict(model='doubao-seed-2-1-lite-260915',requests=len(calls),valid_responses=len(byhash),unique_valid_requests=len(first_valid),unused_later_valid_responses=len(byhash)-len(first_valid),failed_requests=sum(r['status']!='valid' for r in calls),charged_tokens=cost,cap=5000000,previous_experiment_replies_loaded=0,runtime_scope=runtime_scope,completed_stage_seconds=runtime.seconds.sum(),api_elapsed_seconds_sum=sum(r['elapsed_seconds'] for r in calls),reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in calls),conservative_unreported_charge=sum(r.get('charged_tokens',0) for r in calls))
    json.dump(budget,open('res/budget.json','w'),indent=2);json.dump(dict(earliest_valid_response_used=True,unique_valid_requests=len(first_valid),unused_later_valid_responses=len(byhash)-len(first_valid),training_validation_refit_matches=True,protocol_hashes_match=True,api_hashes_match=True,all_outputs_backed_by_this_runs_API=True,equal_scoring_samples=True,max_replay_error=max(replay),persona_share_min=float(shares.min()),persona_share_max=float(shares.max())),open('res/audit.json','w'),indent=2)
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,col in zip(axes,summary.columns):
        ax.barh([ENGLISH[m] for m in ORDER],summary[col],color=['#89939e','#bc8d42','#cb6a53','#c99586','#378d78','#855da8','#527fb3']);ax.invert_yaxis();ax.set(title=col,ylabel='',xlabel='Mean task/event WAPE % (lower is better)')
        for i,v in enumerate(summary[col]):ax.text(v+.3,i,f'{v:.1f}',va='center',fontsize=8)
        ax.set_xlim(0,summary[col].max()*1.18)
    fig.tight_layout();fig.savefig('res/figures/summary.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));x=events.pivot(index='task',columns='method',values='WAPE_pct').reindex(columns=ORDER);x.rename(columns=ENGLISH).plot.bar(ax=ax,rot=0);ax.set(ylabel='Event-day WAPE %',xlabel='');ax.legend(fontsize=8,ncol=3);fig.tight_layout();fig.savefig('res/figures/events.png',dpi=160);plt.close(fig)
    stations=pd.read_csv('res/selected_stations.csv');dataset=pd.read_csv('res/dataset_summary.csv')
    text='''# 新站点独立Lite实验

本轮没有复用以往API回复、拟合权重或预测结果；模型配置保持不变。使用最新时段输入，重新生成训练、验证、测试阶段的角色概率并重新拟合外层规模。所有LLM结果来自本轮真实Lite API，不是聊天生成的替代数据。

## 主要比较

WAPE为绝对误差总和/真实客流总和，越低越好，不是分类准确率。Normal是两个正常天气任务WAPE的等权平均；Extreme是四个事件当天WAPE的等权平均，剔除9/29另列。每配置一次推断，不挑最好重复；未估计LLM重复波动。

'''
    display=summary.rename(index=NAMES).copy()
    for col in display:
        low=display[col].min();display[col]=[f'**{v:.2f}%**' if abs(v-low)<1e-9 else f'{v:.2f}%' for v in display[col]]
    text+=display.to_markdown()+'\n\n![汇总](res/figures/summary.png)\n\n## 分任务\n\n'
    text+=whole.pivot(index='method',columns='task',values='WAPE_pct').reindex(ORDER).rename(index=NAMES).round(2).to_markdown()
    text+='\n\n上表Extreme为含事件前后两天的整个5天窗口，不能代替下面的事件当天成绩。\n\n'
    text+=events.pivot(index='method',columns='task',values='WAPE_pct').reindex(ORDER).rename(index=NAMES).round(2).to_markdown()
    text+='\n\n![各事件](res/figures/events.png)\n\n事件当天实际总量相对训练期同星期均值的变化（描述性对照，不是天气因果效应）：\n\n'+observed.round(2).to_markdown(index=False)+'\n\n## 数据与范围\n\n'+stations[['station_complex_id','station_name','borough']].to_markdown(index=False)+'\n\n'+dataset.to_markdown(index=False)
    text+='''

站点选择只依据7/1—7/19的训练期客流和完整性，排除此前四站，按各区中位客流取一站。9/29的86 St少一个小时，当天四站中只计三站；所有方法同样排除。其他测试日均四站。原日期划分不变，训练和验证早于测试。新站点不等于新时间段；这些事件此前已探索过，partner的全网实验也可能包含本轮站点，不应称为整个项目从未见过的盲测。

## 方法未调整

12个固定画像，票种与原先一样；老人/残障仍为源数据合并标签。规则效用、日程、时段暴露上限3、角色份额10%—60%、星期活跃度、参数候选、提示词、few-shot检索条数、Lite模型、温度0.3和输出250 token均不变。日汇总规则作为对照；LLM三种方法统一使用最新时段输入。zero-shot是LLM不接收历史示例，并非整个ABM没有训练。

各方法用训练/验证数据完成原有拟合和选参，测试不参与选参。角色在四站共享同画像的概率，仅外层规模不同；因此站点不是独立的LLM行为样本。真实天气是事后观测；合成画像和时段、机场代表性、未知服务中断仍是限制。一次抽样不能排除LLM随机性。

## 成本与复现

'''+f"本轮API请求{len(calls):,}次，有效回复{len(byhash):,}条，对应{len(first_valid):,}个不同请求；计入预算{cost:,} token，硬上限5,000,000。{runtime_scope}耗时{runtime.seconds.sum()/60:.1f}分钟，包含已完成阶段的请求等待和拟合，但不含中断阶段的额外等待，不是完整墙钟时间。详见runtime.csv、budget.json；原始日志仅保留本地，不上传GitHub。\n\n"
    text+='''GitHub接收方运行`python reproduce_numeric.py`可核对公开数值结果，无需原始日志。仅本地持有原始日志时，运行`python run.py --offline`与`python report.py`可完成原始回复溯源核验。联网重新调用需密钥位于上一级key.txt。本轮自己的缓存会复用，缺失且offline时直接报错。

最终核验还发现续跑时18个请求已有有效重试回复却又被调用；已按每个请求最早的有效回复离线重新拟合全部结果，后来的重复回复不用于分析，全部费用仍计入。没有按分数选择回复。

网络中断曾导致47次无响应连接失败；连接恢复后，工程层允许按相同请求哈希续跑，有效回复不重抽。所有失败日志及保守预算均保留，模型、提示及参数不变，变更记录在protocol.json。

核验包括：冻结文件哈希、请求哈希、概率合法性、所有预测可追溯到本轮API、相同评分样本、角色份额约束和保存权重的数值回放。没有根据本轮结果更改模型配置。
'''
    lead=f"Normal平均最低为{NAMES[summary.Normal.idxmin()]}（{summary.Normal.min():.2f}%）；四个Extreme事件平均最低为{NAMES[summary.Extreme.idxmin()]}（{summary.Extreme.min():.2f}%）；排除9/29后最低为{NAMES[summary.Extreme_without_0929.idxmin()]}（{summary.Extreme_without_0929.min():.2f}%）。这是本轮单次推断结果，不代表稳定或显著优势。\n\n"
    text=text.replace('# 新站点独立Lite实验\n\n','# 新站点独立Lite实验\n\n'+lead)
    open('RESULTS.md','w').write(text)
    print(summary.round(2).to_string());print(budget)
if __name__=='__main__':main()
