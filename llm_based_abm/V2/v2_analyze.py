"""Run from llm_based_abm: python v2_analyze.py. No API calls."""
import json
import os
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = ['students', 'fair_fare', 'seniors_disability', 'general']
LABELS = {'profile': 'Calendar profile', 'weekly': 'Previous week', 'numeric': 'Dynamic numeric',
          'retrieval_numeric': 'Numeric retrieval', 'llm_zero': 'LLM zero-shot',
          'llm_retrieval': 'LLM retrieval', 'llm_retrieval_no_return': 'Retrieval: no return', 'llm_calibrated': 'LLM calibrated',
          'llm_hybrid': 'Calibrated hybrid', 'llm_feedback': 'LLM feedback',
          'llm_feedback_hybrid': 'Feedback hybrid'}
ORDER = list(LABELS)
COLORS = dict(zip(ORDER, plt.get_cmap('tab20').colors))


def scores(frame, actual='total_ridership', predicted='predicted'):
    valid = frame[[actual, predicted]].dropna()
    error = valid[predicted] - valid[actual]
    mass = valid[actual].sum()
    return {'n': len(valid), 'observed_sum': mass, 'predicted_sum': valid[predicted].sum(),
            'wape': error.abs().sum()/mass if mass else np.nan,
            'mae': error.abs().mean(), 'rmse': np.sqrt((error**2).mean()),
            'bias': error.sum()/mass if mass else np.nan}


def summarize(data, keys):
    rows = []
    for key, frame in data.groupby(keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(keys, key)), **scores(frame)})
    return pd.DataFrame(rows)


def seed_means(frame, keys):
    values = ['n', 'observed_sum', 'predicted_sum', 'wape', 'mae', 'rmse', 'bias']
    mean = frame.groupby(keys, as_index=False)[values].mean()
    mean['seed'] = 'mean'
    mean['repeat_count'] = frame.groupby(keys).seed.nunique().to_numpy()
    original = frame.copy()
    original['seed'] = original.seed.astype(str)
    original['repeat_count'] = 1
    return pd.concat([original, mean], ignore_index=True)


def finish_figure(name):
    plt.tight_layout()
    plt.savefig('results/figures/'+name, dpi=180, bbox_inches='tight')
    plt.close()


def make_figures(metrics, daily, station, group, data):
    weather = metrics[(metrics.seed == 'mean') & metrics.context.isin(['dry', 'light', 'moderate', 'extreme'])]
    pivot = weather.pivot(index='model', columns='context', values='wape').reindex(ORDER)
    pivot = pivot.reindex(columns=['dry', 'light', 'moderate', 'extreme']) * 100
    fig, ax = plt.subplots(figsize=(10, 7))
    values = pivot.to_numpy()
    im = ax.imshow(np.ma.masked_invalid(values), cmap='YlOrRd', aspect='auto')
    for i in range(len(pivot)):
        for j in range(4):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f'{values[i,j]:.1f}%', ha='center', va='center', color='black')
    ax.set_xticks(range(4), ['Dry', 'Light / recent rain', 'Moderate', 'Extreme'])
    ax.set_yticks(range(len(pivot)), [LABELS[x] for x in pivot.index])
    ax.set_title('Test station-hour WAPE by weather (mean across policy repeats)')
    fig.colorbar(im, ax=ax, label='WAPE (%)')
    finish_figure('weather_wape.png')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for model in ORDER:
        subset = daily[daily.model == model].groupby('date').wape.mean()
        ax = axes[1 if model.startswith('llm_') else 0]
        ax.plot(pd.to_datetime(subset.index), subset.values*100, label=LABELS[model], color=COLORS[model], marker='.', linewidth=1.4)
    for ax in axes:
        ax.set_ylabel('Daily WAPE (%)')
        ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.28))
        ax.grid(alpha=.2)
    fig.suptitle('Fixed models, daily forecast origins: test-day errors', y=1.04)
    axes[1].tick_params(axis='x', rotation=35)
    finish_figure('daily_wape.png')

    maps = ['numeric', 'llm_retrieval', 'llm_hybrid', 'llm_feedback_hybrid']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    mean_station = station[station.seed == 'mean']
    top = np.nanmax(mean_station[mean_station.model.isin(maps)].wape)*100
    for model, ax in zip(maps, axes.flat):
        rows = mean_station[mean_station.model == model]
        im = ax.scatter(rows.longitude, rows.latitude, c=rows.wape*100, cmap='YlOrRd', s=90, vmin=0, vmax=top, edgecolor='black', linewidth=.5)
        ax.set_title(LABELS[model]); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_aspect(1/np.cos(np.deg2rad(rows.latitude.mean())))
        ax.grid(alpha=.15)
        fig.colorbar(im, ax=ax, label='WAPE (%)', shrink=.75)
    fig.suptitle('Station errors at 12 selected stations; no background map', y=1.01)
    finish_figure('station_error_geography.png')

    event = data[data.timestamp.dt.strftime('%Y-%m-%d') == '2023-09-29']
    fig, ax = plt.subplots(figsize=(11, 5))
    real = event[event.model == 'numeric'].groupby('timestamp').total_ridership.sum()
    ax.plot(real.index, real, color='black', linewidth=2.5, label='Observed')
    for model in maps:
        hourly = event[event.model == model].groupby(['seed', 'timestamp']).predicted.sum().groupby('timestamp').mean()
        ax.plot(hourly.index, hourly, color=COLORS[model], label=LABELS[model], linewidth=1.7)
    ax.set_title('September 29: hourly demand summed over selected stations')
    ax.set_ylabel('Subway entries per hour'); ax.legend(ncol=2, fontsize=9); ax.grid(alpha=.2)
    ax.tick_params(axis='x', rotation=30)
    finish_figure('september29_hourly.png')

    subset = group[group.seed == 'mean'].pivot(index='model', columns='group', values='wape').reindex(ORDER)
    subset = subset.reindex(columns=GROUPS)*100
    ax = subset.rename(index=LABELS, columns={'students': 'Student fare', 'fair_fare': 'Fair Fare', 'seniors_disability': 'Senior / disability', 'general': 'Other fares'}).plot.bar(figsize=(12, 6), width=.8)
    ax.set_ylabel('Group WAPE (%)'); ax.set_xlabel(''); ax.set_title('Fare-group error: missing observations excluded')
    ax.legend(fontsize=8, ncol=4); ax.grid(axis='y', alpha=.2)
    plt.xticks(rotation=35, ha='right')
    finish_figure('group_wape.png')


def paired_bootstrap(data):
    records = []
    rng = np.random.default_rng(20261012)
    for context in ['all', 'ordinary']:
        selected = data if context == 'all' else data[data.weather_context != 'extreme']
        selected = selected.assign(ae=(selected.predicted-selected.total_ridership).abs())
        daily = selected.groupby(['model', 'seed', 'date']).agg(ae=('ae', 'sum'), observed=('total_ridership', 'sum')).reset_index()
        # Mean absolute errors across generated policies, not error of a seed ensemble.
        daily = daily.groupby(['model', 'date'])[['ae', 'observed']].mean().reset_index()
        base = daily[daily.model == 'numeric'].set_index('date')
        for model in ORDER:
            if not model.startswith('llm_'):
                continue
            candidate = daily[daily.model == model].set_index('date')
            pair = base[['ae', 'observed']].join(candidate[['ae', 'observed']], lsuffix='_base', rsuffix='_llm', how='inner').dropna()
            draws = rng.integers(0, len(pair), size=(2000, len(pair)))
            delta = pair.ae_llm.to_numpy()-pair.ae_base.to_numpy()
            denominator = pair.observed_base.to_numpy()
            bootstrap = delta[draws].sum(axis=1)/denominator[draws].sum(axis=1)
            lo, hi = np.quantile(bootstrap, [.025, .975])
            records.append({'model': model, 'context': context, 'comparison': 'LLM minus dynamic numeric',
                'policy_repeats': selected[selected.model == model].seed.nunique(), 'paired_days': len(pair),
                'bootstrap_samples': 2000, 'wape_delta': delta.sum()/denominator.sum(),
                'ci_low': lo, 'ci_high': hi, 'resampling_unit': 'calendar day, all stations retained'})
    return pd.DataFrame(records)


def report(data, metrics, groups, bootstrap):
    with open('results/calibration.json') as f:
        calibration = json.load(f)
    with open('results/numeric_fit.json') as f:
        numeric_fit = json.load(f)
    with open('results/reproducibility.json') as f:
        reproducibility = json.load(f)
    usage = {'calls': 0, 'usage_unknown_calls': 0, 'repair_calls': 0, 'budget_accounted_tokens': 0,
             'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for filename in ['results/api_calls.jsonl.gz', 'results/api_calls.jsonl']:
        if os.path.exists(filename):
            opener = gzip.open if filename.endswith('.gz') else open
            with opener(filename, 'rt') as f:
                for line in f:
                    call = json.loads(line)
                    usage['calls'] += 1
                    if call.get('usage_known') is False:
                        usage['usage_unknown_calls'] += 1
                    if 'FORMAT REPAIR' in call.get('request', {}).get('prompt', ''):
                        usage['repair_calls'] += 1
                    reported = call.get('usage', {})
                    usage['budget_accounted_tokens'] += call.get('budget_charge_tokens', reported.get('total_tokens', 0))
                    for name in ['input_tokens', 'output_tokens', 'total_tokens']:
                        usage[name] += reported.get(name, 0)
    pd.DataFrame([usage]).to_csv('results/api_usage_summary.csv', index=False)
    main = metrics[metrics.seed == 'mean'].pivot(index='model', columns='context', values='wape')
    numeric = main.loc['numeric', 'all']
    ordinary = main.loc['numeric', 'ordinary']
    llm = main.loc[[m for m in ORDER if m.startswith('llm_')]]
    changed = llm['all']-numeric
    best = changed.idxmin()
    best_all = main['all'].idxmin()
    conclusion = (f'测试结果的事后排序中整体误差最低的 LLM 方法为 `{best}`，比动态数值模型低 {-changed[best]*100:.2f} 个百分点。'
                  if changed[best] < -1e-10 else
                  '本轮没有 LLM 方法降低整体误差；应保留动态数值基线，并按天气和日期继续检查。')
    rows = [f'# 第二轮实验结果\n',
            f'本轮整体表现最好的是 `{best_all}`（WAPE {main.loc[best_all,"all"]*100:.2f}%）。统计校准后的 LLM 普通天气误差为 {main.loc["llm_calibrated","ordinary"]*100:.2f}%，低于 numeric 的 {main.loc["numeric","ordinary"]*100:.2f}%，但仍高于 weekly 的 {main.loc["weekly","ordinary"]*100:.2f}%。极端时段 weekly 为 {main.loc["weekly","extreme"]*100:.2f}%，LLM feedback 为 {main.loc["llm_feedback","extreme"]*100:.2f}%，numeric 为 {main.loc["numeric","extreme"]*100:.2f}%。没有一个方法在所有天气都最好。',
            f'\n本轮使用 {data.station_complex_id.nunique()} 个站点、{data.date.nunique()} 个测试日期，测试窗口为 {data.timestamp.min():%Y-%m-%d} 至 {data.timestamp.max():%Y-%m-%d}。{conclusion}若 LLM 只优于 numeric 而未优于 weekly，不能称为优于全部非 LLM 基线。',
            '\n这是旧数据上的重新划分，采用固定模型、逐日原点回测：每天产生新预测，但模型不每天重训。它不是项目级从未看过的新盲测；不能据此宣称新模型已在其他月份验证。事后最低误差只是测试后排序，不是测试前选定的赢家，也没有据此重新调参。',
            '\n## 预测口径',
            '\n每天午夜预测当天未来 24 小时，目标日真实客流不能进入提示或检索；前一个已完成测试日的真实客流允许成为次日已知历史。因此它不是在 9 月 15 日一次预测后续全部 16 天。当天实测天气作为已知条件，故这是给定天气的历史检验，不是获得真实天气预报后的部署效果。近期客流可及时取得也是回测假设。所有方法仅在共同测试站点小时比较。第二个策略重复使用 9 月 14 日完整一天热启动延期队列，但热启动日不参与最终测试评分；feedback / validation 指标保持冻结。',
            '\nWAPE 为绝对误差之和除以真实客流之和，越低越好。LLM 指标对两个策略生成重复分别评分再平均，不是先平均预测再评分；重复不是独立天气事件。',
            '\n## 主要误差\n',
            '| 方法 | 所有测试小时 | 普通天气 | 干天气 | 小雨/近期雨 | 中雨 | 极端雨 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for model in ORDER:
        row = main.loc[model]
        values = [f'{row[c]*100:.2f}%' if pd.notna(row.get(c)) else '无观测' for c in ['all', 'ordinary', 'dry', 'light', 'moderate', 'extreme']]
        rows.append('| '+model+' | '+' | '.join(values)+' |')
    overall = metrics[(metrics.seed == 'mean') & (metrics.context == 'all')].set_index('model')
    rows.extend(['\n纯 LLM 策略仍在减少需求：',
                 f'\n整体总量偏差 numeric 为 {overall.loc["numeric","bias"]*100:+.2f}%，zero-shot 为 {overall.loc["llm_zero","bias"]*100:+.2f}%，retrieval 为 {overall.loc["llm_retrieval","bias"]*100:+.2f}%，统计校准后为 {overall.loc["llm_calibrated","bias"]*100:+.2f}%。负值表示少预测。检索没有自动修复少预测；本轮明显改善来自校准后的量级恢复，不能把这部分全归功于 LLM 自发理解。',
                 f'\n上一周同小时基线在普通天气 WAPE 为 {main.loc["weekly","ordinary"]*100:.2f}%，但 extreme 为 {main.loc["weekly","extreme"]*100:.2f}%。它的整体最好不能证明它擅长极端天气；少数极端小时和多数普通小时需要分开判断。',
                 '\n本轮换了站点及时间划分，不能把首轮和第二轮误差直接作同条件升级幅度。'])
    development = pd.read_csv('results/development_metrics.csv')
    validation = development[(development.split == 'validation') & (development.context == 'all')].groupby('model').wape.mean()
    validation_all = validation.idxmin()
    validation_llm = validation[validation.index.str.startswith('llm_')].idxmin()
    rows.extend(['\n## 测试前的验证期选择\n',
                 '下面单独读取冻结的 development 指标，按验证期整体 WAPE 排序；不使用测试误差选择。若数值相同，列出的只是排序中第一个，不应宣称有优势。',
                 '\n| 范围 | 验证期最低模型 | 验证 WAPE | 该模型测试 WAPE |',
                 '|---|---|---:|---:|',
                 f'| 全部方法 | {validation_all} | {validation[validation_all]*100:.2f}% | {main.loc[validation_all,"all"]*100:.2f}% |',
                 f'| LLM 方法 | {validation_llm} | {validation[validation_llm]*100:.2f}% | {main.loc[validation_llm,"all"]*100:.2f}% |'])
    rows.extend(['\n普通天气仅排除 extreme 小时，并保留同一天内的其他小时；不是把所有暴雨日整天删除。light 也包括本小时已停雨但过去六小时仍有雨的情况。',
                 '\n`llm_retrieval_no_return` 免费重放相同 retrieval 策略，唯一变化是推迟需求返回率从 0.6 改为 0；它用于检查延迟回流这一模型假设，不代表真实乘客取消率为 100%，也没有重新生成 LLM 策略。',
                 '\n## 与数值基线的配对检查\n',
                 '按日将所有站点误差相加，再配对重采样 2,000 次。差值 = LLM WAPE − numeric WAPE，负值表示 LLM 更好。下表为普通天气；全部天气结果在 CSV 中。',
                 '\n| 方法 | 误差差值（百分点） | 95% 重采样区间（百分点） |',
                 '|---|---:|---:|'])
    for row in bootstrap[bootstrap.context == 'ordinary'].itertuples():
        rows.append(f'| {row.model} | {row.wape_delta*100:+.2f} | [{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}] |')
    rows.extend(['\n这里的区间只表达这些日期内误差波动。相邻日期可能相关；一天的多个站点、两个策略重复并不会增加独立事件数量。区间也未校正多个方法比较，不能据此宣称城市泛化或因果显著。',
                 '\n## 校准选择',
                 f'\n数值基线 Ridge alpha = {numeric_fit["selected_alpha"]}，只根据 feedback 选择。LLM 校准混合权重 = {calibration["selected_weight"]}；feedback 混合权重 = {calibration["feedback_selected_weight"]}，两者由 Sep12–14 validation 选择，未使用测试误差挑权重。0 表示完全采用 numeric，1 表示完全采用相应 LLM 输出。',
                 '\n若混合权重为 0，混合方法与 numeric 一致；这说明校准窗口未支持引入该 LLM 输出，不能把保留 numeric 的结果算成 LLM 的改善。数值校准的效果与真实 LLM feedback 的效果要分别比较。',
                 '\n## 群体与解释边界',
                 '\n学生、Fair Fare、老人/残障、其他票种都是票种代理标签，不是实测人格和完整人口属性。老人和残障仍无法分开。群体评分排除该组真实值缺失的行；站点总评分使用真实总量。预测误差低不证明“取消、换方式、延迟”的真实比例，也不能将假设行动图当作调查事实。',
                 '\n反馈 critic 输出 192 格活动倍率，而纯统计校准只拟合 12 个站点倍率，两者校准容量不同。反馈的区别不能单独归因于 LLM 推理，后续应加入同容量的纯统计对照。',
                 '\n检索输入的 `group_block_observed_over_numeric` 在群体有缺失时，分子只包含有观测小时，分母仍包括整个时间块，可能低估相似日比率。测试期两个 seed 共用的检索实例按唯一曝光计：学生部分缺失 796/2304（34.5%），Fair Fare 611/2304（26.5%），老人/残障 908/2304（39.4%），general 仅 1 个；相对于匹配观测小时口径，比率中位偏低分别为 4.2%、9.0%、8.2%。这是输入比率偏差，不能当作预测误差贡献或缺失真实值已知为零。本轮保留冻结模型与原预测，不据测试事后改提示；下一轮改为匹配小时分子/分母。',
                 '\n## API 与复现',
                 f'\n第二轮本地日志记录 {usage["calls"]} 次 API 调用，输入 {usage["input_tokens"]:,}、输出 {usage["output_tokens"]:,}，共 {usage["total_tokens"]:,} tokens。只统计 API 已返回 usage 的用量；另有 {usage["usage_unknown_calls"]} 次记录标记为用量未知，不能假设这些调用免费。冻结设计的预算为 2,000,000 tokens；最后因未知失败预留触发预算保护，操作上限扩至 2,100,000，仅完成原定修复，没有新增模型。此统计包括日志中的试跑与重试，不打印提示内容或密钥，不包含首轮另存的日志。',
                 f'\n日志中有 {usage["repair_calls"]} 次真实 API 格式修复调用，其中首次发现越界活动倍率 m=1.63。程序保留原始调用和用量，再向真实 API 请求符合约束的修复；没有手工裁剪或虚构策略。',
                 f'\n预算保守记账为 {usage["budget_accounted_tokens"]:,} tokens，包含无 usage 失败调用的预留额度；这个数字用于限制预算，不是已知实际计费量。',
                 f'\n从本目录运行 `python v2_analyze.py` 可免费重做分析，读取保存预测而不调用 API。`results/development_metrics.csv` 另存 feedback / validation，当前图表及结果表只报告 test。主流程另从源数据重建，并以 `test --offline` 重放：{reproducibility["rows"]:,} 行预测、{reproducibility["station_hours_per_method"]:,} 个共同站点小时、{reproducibility["output_model_samples"]} 个模型/样本组合，预测最大逐数差为 {reproducibility["max_absolute_prediction_difference"]}，校准选择保持一致，没有新增 API 请求；见 `results/reproducibility.json`。',
                 '\n数据表：`metrics_summary.csv` 保留每次重复和平均指标；`station_metrics.csv`、`group_metrics.csv`、`daily_metrics.csv`、`paired_bootstrap.csv` 保存细分结果。',
                 '\n## 下一步优先做什么\n',
                 '1. 修复群体检索比率的缺失对齐：分子和分母使用同一批有观测小时，并在新实验中重新冻结提示。',
                 '2. 加入和 critic 相同 192 格容量的纯统计校准，对比同信息、同容量的方法，判断收益究竟来自校准还是 LLM。',
                 '3. 补充学校日历、线路服务/停运及新的月份，先冻结方法再独立验证。暂不继续无条件增加 agent 或 API 次数。',
                 '\n## 图表\n',
                 '![天气误差](results/figures/weather_wape.png)\n',
                 '![每日误差](results/figures/daily_wape.png)\n',
                 '![站点误差](results/figures/station_error_geography.png)\n',
                 '![9月29日小时过程](results/figures/september29_hourly.png)\n',
                 '![群体误差](results/figures/group_wape.png)\n'])
    with open('RESULTS.md', 'w') as f:
        f.write('\n'.join(rows).rstrip()+'\n')
    print('V2 reported tokens:', usage['total_tokens'])
    print('Numeric overall / ordinary WAPE:', numeric, ordinary)


def main():
    os.makedirs('results/figures', exist_ok=True)
    data = pd.read_csv('results/predictions.csv.gz', dtype={'station_complex_id': str}, parse_dates=['timestamp'])
    data = data[data.split == 'test'].copy()
    data['date'] = data.timestamp.dt.strftime('%Y-%m-%d')
    metrics = []
    for context in ['all', 'ordinary', 'dry', 'light', 'moderate', 'extreme']:
        subset = data if context == 'all' else data[data.weather_context != 'extreme'] if context == 'ordinary' else data[data.weather_context == context]
        summary = summarize(subset, ['model', 'seed'])
        summary['context'] = context
        metrics.append(summary)
    metrics = seed_means(pd.concat(metrics, ignore_index=True), ['model', 'context'])
    metrics.to_csv('results/metrics_summary.csv', index=False)
    daily = summarize(data, ['model', 'seed', 'date'])
    daily.to_csv('results/daily_metrics.csv', index=False)
    station = summarize(data, ['model', 'seed', 'station_complex_id'])
    station = seed_means(station, ['model', 'station_complex_id'])
    coordinates = data[['station_complex_id', 'station_name', 'borough', 'latitude', 'longitude']].drop_duplicates('station_complex_id')
    station = station.merge(coordinates, on='station_complex_id')
    station.to_csv('results/station_metrics.csv', index=False)
    group_rows = []
    for (model, seed), subset in data.groupby(['model', 'seed']):
        for group in GROUPS:
            group_rows.append({'model': model, 'seed': seed, 'group': group, **scores(subset, group, 'pred_'+group)})
    group = seed_means(pd.DataFrame(group_rows), ['model', 'group'])
    group.to_csv('results/group_metrics.csv', index=False)
    bootstrap = paired_bootstrap(data)
    bootstrap.to_csv('results/paired_bootstrap.csv', index=False)
    make_figures(metrics, daily, station, group, data)
    report(data, metrics, group, bootstrap)
    print(metrics[(metrics.seed == 'mean') & (metrics.context.isin(['all', 'ordinary']))][['model', 'context', 'wape']].to_string(index=False))


if __name__ == '__main__':
    main()
