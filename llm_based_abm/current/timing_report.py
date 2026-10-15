"""Verify and summarize the timing probe without network requests."""
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timing_api import ledger,charge,validate
from timing import hourly_weather,TASKS,STATIONS

def score(p):
    p=p.dropna(subset=['actual','predicted']);e=p.predicted-p.actual
    return dict(n=len(p),actual=p.actual.sum(),predicted=p.predicted.sum(),WAPE_pct=100*abs(e).sum()/p.actual.sum(),bias_pct=100*e.sum()/p.actual.sum())

def main():
    protocol=json.load(open('res/timing/protocol.json'))
    for f,h in protocol['files'].items():
        if hashlib.sha256(open(f,'rb').read()).hexdigest()!=h:raise ValueError('Protocol changed: '+f)
    calls=ledger();byhash={r['hash']:r for r in calls if r['status']=='valid'};tokens=sum(charge(r) for r in calls)
    if tokens>160000:raise ValueError('Over budget')
    for r in calls:
        digest=hashlib.sha256(json.dumps(r['request'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=r['hash']:raise ValueError('Request provenance mismatch')
        if r['status']=='valid':validate(r['parsed'],r['request']['route'],r['request']['view'].get('person'))
    p=pd.read_csv('res/timing/pro_predictions.csv',dtype={'station_complex_id':str})
    dec=pd.read_csv('res/timing/pro_decisions.csv',dtype={'station_complex_id':str})
    if p.duplicated(['task','method','arm','station_complex_id','date']).any():raise ValueError('Duplicate predictions')
    for _,r in p[p.method=='direct_few'].iterrows():
        if r.predicted!=byhash[r.request_hash]['parsed']['entries']:raise ValueError('Direct output mismatch')
    for _,r in dec.iterrows():
        np.testing.assert_allclose([r['p_'+a] for a in ['subway','bus','taxi','walk','stay_home']],byhash[r.request_hash]['parsed']['probabilities'],atol=1e-12)
    dec['prediction']=dec.effective_population*dec.p_subway
    keys=['task','method','arm','station_complex_id','date'];s=dec.groupby(keys).prediction.sum().reset_index().merge(p,on=keys)
    np.testing.assert_allclose(s.prediction,s.predicted,atol=1e-8)
    weights=dec.pivot(index=['task','method','station_complex_id','agent_id','date'],columns='arm',values='effective_population')
    np.testing.assert_allclose(weights.daily,weights.timing)
    metrics=pd.DataFrame([dict(task=t,method=m,arm=a,**score(r)) for (t,m,a),r in p.groupby(['task','method','arm'])]);metrics.to_csv('res/timing/pro_metrics.csv',index=False)
    rule=pd.read_csv('res/timing/rule_predictions.csv',dtype={'station_complex_id':str});dates=pd.read_csv('res/dates.csv');rule=rule.merge(dates[['task','date','event_day']],on=['task','date'])
    old=pd.read_csv('res/predictions.csv',dtype={'station_complex_id':str});b=rule[rule.method=='rule_daily'].merge(old[old.method=='rule_abm'],on=['task','station_complex_id','date'],suffixes=('_new','_old'))
    np.testing.assert_allclose(b.predicted_new,b.predicted_old,atol=1e-7)
    rm=[]
    for scope,part in [('window',rule),('event',rule[rule.event_day])]:
        for (t,m),r in part.groupby(['task','method']):rm.append(dict(scope=scope,task=t,method=m,**score(r)))
    rm=pd.DataFrame(rm);rm.to_csv('res/timing/rule_metrics.csv',index=False)
    for key,part in p.groupby(['task','method']):
        sets=[set(r.station_complex_id) for _,r in part.groupby('arm')]
        if sets[0]!=sets[1] or sets[0]!=set(STATIONS):raise ValueError('Unpaired stations')
    group=[]
    for (t,m,a,g),r in dec.groupby(['task','method','arm','group']):
        group.append(dict(task=t,method=m,arm=a,group=g,effective_population=r.effective_population.sum(),subway_probability=np.average(r.p_subway,weights=r.effective_population) if r.effective_population.sum()>0 else np.nan,stay_home_probability=np.average(r.p_stay_home,weights=r.effective_population) if r.effective_population.sum()>0 else np.nan))
    pd.DataFrame(group).to_csv('res/timing/group_choices.csv',index=False)
    budget=dict(model='doubao-seed-2-1-pro-260915',calls=len(calls),valid_calls=len(byhash),errors=sum(r['status']!='valid' for r in calls),charged_tokens=tokens,cap=160000,api_seconds_sum=sum(r['elapsed_seconds'] for r in calls),scope='New Pro probe only; cached prior Lite calibration excluded')
    json.dump(budget,open('res/timing/budget.json','w'),indent=2)
    json.dump(dict(protocol_hashes_match=True,probabilities_backed_by_API=True,fixed_weights_between_arms=True,direct_outputs_backed_by_API=True,rule_daily_matches_previous=True,equal_scoring_samples=True,new_tokens=tokens),open('res/timing/audit.json','w'),indent=2)
    names={'direct_few':'Direct few-shot','role_no_history':'Role zero-shot','role_with_history':'Role few-shot'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,task in zip(axes,TASKS):
        x=metrics[metrics.task==task].pivot(index='method',columns='arm',values='WAPE_pct').rename(index=names)
        x.plot.bar(ax=ax,color=['#9ba8b4','#398a88'],rot=15);ax.set(title=task+' | two stations, event day',ylabel='WAPE % (lower is better)',xlabel='');ax.legend(title='Pro input')
    fig.suptitle('Paired Pro probe: fixed prior Lite-calibrated role weights; one call per configuration',fontsize=10);fig.tight_layout();fig.savefig('res/timing/pro_comparison.png',dpi=160);plt.close(fig)
    weather=hourly_weather();fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
    for ax,date in zip(axes,['2023-08-13','2023-09-29']):ax.bar(range(24),weather[date],color='#527fb3');ax.set(title=date,ylabel='Hourly rain (mm)')
    axes[-1].set(xlabel='New York local hour',xticks=range(0,24,2));fig.tight_layout();fig.savefig('res/timing/hourly_rain.png',dpi=160);plt.close(fig)
    text='''# 加入降雨时段：小规模结果

主要观察：8/13凌晨降雨的Pro角色预测改善，9/29没有一致改善。规则模型也有升有降，不能说加入时段必然更好。

这次补上了小时降雨、角色出发时段及两者重叠的暴露程度。规则 ABM 完成六个原任务重新训练和验证；Pro 只做两个事件、两个站点的成对测试。

重要区别：规则两组分别重新校准；Pro两组沿用同一套之前Lite拟合的外层权重，只重问决策。不能把这次当作已经完整训练的Pro模型，也不能用Pro与旧Lite的差异证明时段效果。每个Pro配置只推断一次，输出随机性未充分评估。

## 天气与设定

8/13日雨量16.002mm，集中在凌晨03时；9/29日雨量207.010mm，多小时持续降雨。各站共享JFK天气，日期沿用既有测试集，是事后诊断而非新的盲测。

![小时雨量](res/timing/hourly_rain.png)

12个角色仍保留四类票种×三种活动类型。通勤类工作日7—9时、周末9—11时；灵活事务10—16时；休闲工作日17—21时、周末12—20时。均为合成日程，不是观测到的个人习惯；这里只模拟单趟出发，不含返程。规则的时段暴露倍数上限固定为3，没有用测试分数调这个上限。实际雨后积水和停运未提供，因此未虚构。

## 规则 ABM：WAPE %

下表是全部测试窗口（正常天气7天、事件窗口5天），四个原站点。两个版本使用相同训练、验证、测试日期；分别按验证误差选择参数。

'''
    text+=rm[rm.scope=='window'].pivot(index='task',columns='method',values='WAPE_pct').round(2).to_markdown()
    text+='\n\n只看事件当天：\n\n'+rm[rm.scope=='event'].pivot(index='task',columns='method',values='WAPE_pct').round(2).to_markdown()
    text+='''

9/29时段版验证集选出的雨量系数为0，即这一组事实上忽略雨量，不能把误差下降解释为成功识别雨的时段。10/7事件日反而从41.45%升到61.98%，说明固定日程、暴露函数以及少量验证天气仍有局限。

日汇总版已核对能复现上一版规则结果。时段版同时重新拟合了参数，因此该比较是完整方法对比，不是单独替换一个测试特征的因果效果。

## Pro：同模型、同权重，只改变时段天气输入

只评估Pelham Bay Park（360）和77 St / R（37）。daily只有日汇总，timing额外给24小时雨量；few-shot示例的天气也同步增加小时雨量。角色两组都有同一份假设日程。下面为两个站点的WAPE，不与四站主实验直接横比。

'''+metrics.pivot(index=['task','method'],columns='arm',values='WAPE_pct').round(2).to_markdown()
    text+='\n\n![Pro时段对照](res/timing/pro_comparison.png)\n\n'
    text+='''一个具体例子：8/13 general_0 的few-shot日汇总回复说“steady rain”，但实际雨在凌晨；加时段后改为“dry late morning”。这表明时段信息能纠正文字情景理解，不过该角色地铁概率从0.58变为0.56，并非每个角色都朝提高客流的方向变化。不能只凭理由文字判断预测正确。

看角色是否更少取消出行，可查group_choices.csv（按代表权重平均的地铁/不出行概率）；某组当天拟合规模为零时，群体加权概率留空，不伪造为零。具体理由和原始概率在pro_decisions.csv及压缩API日志。它们是模型选择，不是真实人的行为标签。两个站点共享同画像回复，因此不构成两个独立行为样本。

## 成本与复现

'''+f"新增Pro请求{budget['calls']}次，有效{budget['valid_calls']}次，失败{budget['errors']}次；计入预算{tokens:,} token，低于160,000上限。没有调用Lite或重新生成完整训练期角色回复。\n\n"
    text+='''四项时段测试通过：雨从夜间移到通勤时段能改变暴露；日程概率归一；小时雨量与原日量一致；目标客流不进入提示。离线检查同时核对原始API、两组权重一致、实际评分样本一致及日汇总规则结果复现。

运行方法与固定设定见[TIMING.md](TIMING.md)。下一步若正式比较，应在未见日期上补充事件，并重新校准Pro训练期；还需要检验日程假设、返程、局地雨量和服务中断。本次不自动扩展调用。
'''
    open('TIMING_RESULTS.md','w').write(text)
    print(metrics.to_string(index=False));print(rm[['scope','task','method','WAPE_pct']].to_string(index=False));print(budget)
if __name__=='__main__':main()
