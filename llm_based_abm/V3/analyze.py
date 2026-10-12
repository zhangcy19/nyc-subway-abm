"""From V3: python analyze.py. Analyze saved personal choices; no API calls."""
import gzip
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = ['students', 'fair_fare', 'seniors_disability', 'general']
ACTIONS = ['subway', 'delay_subway', 'bus', 'taxi', 'walk', 'remote', 'cancel']
METHODS = ['no_change', 'rule', 'pro', 'lite']
LABELS = {'planned_profile': 'Planned hourly demand', 'no_change': 'No-change representatives',
          'rule': 'Rule decisions', 'pro': 'Pro personal decisions', 'lite': 'Lite personal decisions'}
COLORS = {'planned_profile': '#888888', 'no_change': '#0072B2', 'rule': '#E69F00', 'pro': '#009E73', 'lite': '#CC79A7'}
ACTION_COLORS = dict(zip(ACTIONS, ['#0072B2', '#56B4E9', '#E69F00', '#CC79A7', '#009E73', '#999999', '#D55E00']))


def summarize(frame, keys, actual='total_ridership', predicted='predicted'):
    rows = []
    for key, subset in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        valid = subset[[actual, predicted]].dropna()
        delta = valid[predicted]-valid[actual]
        mass = valid[actual].sum()
        rows.append({**dict(zip(keys, key)), 'n': len(valid), 'observed_sum': mass,
            'predicted_sum': valid[predicted].sum(), 'wape': delta.abs().sum()/mass if mass else np.nan,
            'bias': delta.sum()/mass if mass else np.nan, 'mae': delta.abs().mean(),
            'rmse': np.sqrt((delta**2).mean())})
    return pd.DataFrame(rows)


def timing_metrics(data):
    # Score predictions and actuals over the same observed-hour support before aggregation.
    data = data[data.total_ridership.notna()].copy()
    base = ['method', 'block', 'n_agents', 'sample']
    output = []
    for unit, frequency in [('daily', 'D'), ('four_hour', '4h'), ('hourly', 'h')]:
        frame = data.copy()
        frame['time_bin'] = frame.timestamp.dt.floor(frequency)
        bins = frame.groupby(base+['station_complex_id', 'time_bin'], as_index=False)[['predicted', 'total_ridership']].sum()
        result = summarize(bins, base)
        result['unit'] = unit
        output.append(result)
        pooled = bins.copy(); pooled['block'] = 'all_test'
        result = summarize(pooled, base); result['unit'] = unit
        output.append(result)
    return pd.concat(output, ignore_index=True)


def decision_predictions(panel, choices):
    output = []
    for key, frame in choices.groupby(['method', 'block', 'n_agents', 'sample']):
        selected = panel[panel.date.dt.strftime('%Y-%m-%d').isin(frame.date.unique()) & panel.station_complex_id.isin(frame.station_complex_id.unique())].copy()
        arrivals = frame[frame.action.isin(['subway', 'delay_subway'])].copy()
        arrivals['timestamp'] = pd.to_datetime(arrivals.actual_entry_timestamp)
        entry = arrivals.groupby(['station_complex_id', 'timestamp']).weight.sum().rename('predicted')
        selected = selected.merge(entry, on=['station_complex_id', 'timestamp'], how='left')
        selected['predicted'] = selected.predicted.fillna(0)
        for col, value in zip(['method', 'block', 'n_agents', 'sample'], key): selected[col] = value
        output.append(selected)
    return pd.concat(output, ignore_index=True)


def action_summary(data, keys):
    numerator = data.groupby(keys+['action'], dropna=False).agg(decisions=('agent_id', 'size'), mass=('weight', 'sum')).reset_index()
    denominator = data.groupby(keys, dropna=False).agg(total_decisions=('agent_id', 'size'), planned_mass=('weight', 'sum')).reset_index()
    result = numerator.merge(denominator, on=keys)
    result['weighted_share'] = result.mass/result.planned_mass
    result['unweighted_share'] = result.decisions/result.total_decisions
    return result


def paired_days(data):
    data = data[data.total_ridership.notna()].copy()
    day = data.groupby(['method', 'sample', 'station_complex_id', 'date'], as_index=False)[['predicted', 'total_ridership']].sum()
    day['ae'] = (day.predicted-day.total_ridership).abs()
    day = day.groupby(['method', 'sample', 'date'])[['ae', 'total_ridership']].sum().reset_index()
    rng = np.random.default_rng(20261012)
    rows = []
    for base in ['no_change', 'rule']:
        reference = day[day.method == base].groupby('date')[['ae', 'total_ridership']].mean()
        for method in ['pro', 'lite']:
            candidate = day[day.method == method].groupby('date')[['ae', 'total_ridership']].mean()
            pair = reference.join(candidate, lsuffix='_base', rsuffix='_candidate').dropna()
            difference = pair.ae_candidate.to_numpy()-pair.ae_base.to_numpy()
            denominator = pair.total_ridership_base.to_numpy()
            draws = rng.integers(0, len(pair), size=(2000, len(pair)))
            boot = difference[draws].sum(axis=1)/denominator[draws].sum(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append({'baseline': base, 'method': method, 'days': len(pair), 'bootstrap_samples': 2000,
                'daily_wape_difference': difference.sum()/denominator.sum(), 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows)


def save_figure(name):
    plt.tight_layout()
    plt.savefig('res/figures/'+name, dpi=180, bbox_inches='tight')
    plt.close()


def main_figures(predictions, decisions, actions, traits):
    predictions = predictions[predictions.total_ridership.notna()].copy()
    stations = predictions.station_complex_id.unique()
    fig, axes = plt.subplots(1, len(stations), figsize=(6*len(stations), 5), squeeze=False)
    for station, ax in zip(stations, axes.flat):
        frame = predictions[predictions.station_complex_id == station]
        observed = frame[frame.method == 'no_change'].groupby('date').total_ridership.sum()
        calendar = pd.date_range(pd.to_datetime(observed.index).min(), pd.to_datetime(observed.index).max())
        observed.index = pd.to_datetime(observed.index)
        observed = observed.reindex(calendar)
        ax.plot(observed.index, observed, 'k-o', label='Observed', linewidth=2)
        for method in METHODS:
            daily = frame[frame.method == method].groupby(['sample', 'date']).predicted.sum().groupby('date').mean()
            daily.index = pd.to_datetime(daily.index); daily = daily.reindex(calendar)
            ax.plot(daily.index, daily, label=LABELS[method], color=COLORS[method], marker='.', linewidth=1.7)
        ax.set_title(str(frame.station_name.iloc[0])); ax.set_ylabel('Daily subway entries')
        ax.tick_params(axis='x', rotation=40); ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.1), ncol=3, fontsize=9)
    save_figure('daily_station_predictions.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for group, ax in zip(GROUPS, axes.flat):
        x = actions[(actions.group == group) & (actions.method != 'no_change')]
        pivot = x.pivot_table(index=['block', 'method'], columns='action', values='weighted_share').reindex(columns=ACTIONS, fill_value=0).fillna(0)
        pivot.index = [b.replace('test_', '')+' / '+m for b, m in pivot.index]
        pivot.plot.bar(stacked=True, ax=ax, color=[ACTION_COLORS[a] for a in ACTIONS], legend=False)
        ax.set_title(group.replace('_', ' ')); ax.set_xlabel(''); ax.set_ylabel('Weighted planned-trip share')
        ax.tick_params(axis='x', rotation=35); ax.set_ylim(0, 1.05)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.06), ncol=7, fontsize=9)
    save_figure('group_actions.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feature, title in zip(axes, ['risk_band', 'type'], ['Hypothetical rain-risk traits', 'Hypothetical attendance / trip types']):
        subset = traits[traits.feature == feature]
        for method in ['rule', 'pro', 'lite']:
            rows = subset[subset.method == method].groupby('value')[['subway_mass', 'planned_mass']].sum()
            if feature == 'risk_band': rows = rows.reindex(['low', 'medium', 'high'])
            ax.plot(rows.index, rows.subway_mass/rows.planned_mass, '.-', color=COLORS[method], label=method)
        ax.set_title(title); ax.set_ylabel('Weighted subway share, including delayed subway')
        ax.tick_params(axis='x', rotation=35); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Descriptive synthetic trait patterns; not causal personality estimates', y=1.04)
    save_figure('trait_actions.png')


def matched_analysis():
    if not os.path.exists('res/matched_checks.csv.gz'): return None
    data = pd.read_csv('res/matched_checks.csv.gz', dtype={'station_complex_id': str})
    data['changed_action'] = data.action != data.reference_action
    data['changed_choice'] = data.changed_action | (data.delay_min != data.reference_delay_min)
    data['subway'] = data.action.isin(['subway', 'delay_subway'])
    data['reference_subway'] = data.reference_action.isin(['subway', 'delay_subway'])
    rows = []
    for (method, date), frame in data.groupby(['method', 'date']):
        mass = frame.weight.sum()
        rows.append({'method': method, 'date': date, 'n': len(frame), 'planned_mass': mass,
            'action_changed_fraction': frame.changed_action.mean(),
            'choice_changed_fraction': frame.changed_choice.mean(),
            'weighted_choice_changed_fraction': frame.loc[frame.changed_choice, 'weight'].sum()/mass,
            'weighted_subway_share': frame.loc[frame.subway, 'weight'].sum()/mass,
            'reference_weighted_subway_share': frame.loc[frame.reference_subway, 'weight'].sum()/mass})
    summary = pd.DataFrame(rows); summary.to_csv('res/matched_metrics.csv', index=False)
    transitions = data.groupby(['method', 'date', 'reference_action', 'action']).agg(decisions=('agent_id', 'size'), mass=('weight', 'sum')).reset_index()
    transitions.to_csv('res/matched_transitions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for date, ax in zip(sorted(summary.date.unique()), axes):
        frame = summary[summary.date == date].set_index('method')
        frame[['weighted_choice_changed_fraction', 'weighted_subway_share', 'reference_weighted_subway_share']].rename(columns={
            'weighted_choice_changed_fraction': 'Changed action or delay', 'weighted_subway_share': 'Probe subway share',
            'reference_weighted_subway_share': 'Reference subway share'}).plot.bar(ax=ax)
        ax.set_title(date+' (one station, matched personal inputs)'); ax.set_ylabel('Weighted share')
        ax.tick_params(axis='x', rotation=40); ax.set_ylim(0, 1.1)
        ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.14), ncol=3, fontsize=9)
    fig.suptitle('Single-day probes, not continuous alternative-agent histories', y=1.23)
    save_figure('matched_checks.png')
    return summary


def resolution_analysis(panel):
    if not os.path.exists('res/resolution_decisions.csv.gz'): return None
    data = pd.read_csv('res/resolution_decisions.csv.gz', dtype={'station_complex_id': str})
    predictions = decision_predictions(panel, data)
    metrics = timing_metrics(predictions)
    metrics = metrics[metrics.block != 'all_test']
    metrics.to_csv('res/resolution_metrics.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for unit, ax in zip(['daily', 'four_hour', 'hourly'], axes):
        for method in ['no_change', 'rule', 'pro']:
            frame = metrics[(metrics.unit == unit) & (metrics.method == method)].groupby('n_agents').wape.mean()
            ax.plot(frame.index, frame*100, '.-', color=COLORS[method], label=LABELS[method])
        ax.set_title(unit.replace('_', ' ')+' WAPE'); ax.set_xlabel('Representatives per station')
        ax.set_xticks([24, 48, 96]); ax.grid(alpha=.2)
    axes[0].set_ylabel('WAPE (%)'); axes[-1].legend(fontsize=8)
    fig.suptitle('One station, September 29; common initial synthetic history', y=1.06)
    save_figure('resolution_errors.png')
    return metrics


def api_records():
    records = []
    for filename in ['res/api_calls.jsonl.gz', 'res/api_calls.jsonl']:
        if os.path.exists(filename):
            opener = gzip.open if filename.endswith('.gz') else open
            with opener(filename, 'rt') as f:
                records.extend(json.loads(line) for line in f)
    return records


def choice_audit(choice, view):
    payload = choice if isinstance(choice, dict) else {}
    action, delay = payload.get('action'), payload.get('delay_min')
    structure = isinstance(action, str) and isinstance(delay, int) and not isinstance(delay, bool) and isinstance(payload.get('reason'), str) and bool(payload.get('reason', '').strip())
    allowed_action = structure and action in view.get('available_actions', [])
    limit = view.get('person', {}).get('max_delay_min', 0)
    allowed_delays = [60, 120] if limit >= 120 else [60] if limit >= 60 else []
    allowed_delay = structure and (delay in allowed_delays if action == 'delay_subway' else delay == 0)
    return {'json_object': isinstance(choice, dict), 'structure_valid': structure,
            'action_allowed': allowed_action, 'delay_allowed': allowed_delay,
            'all_valid': allowed_action and allowed_delay}


def response_quality(decisions):
    calls = api_records()
    rows = []
    for call in calls:
        request = call.get('request', {})
        # First-attempt completed responses are assessed before repairs or execution filtering.
        if request.get('repair_attempt', 0) == 0 and call.get('status') == 'completed':
            rows.append({'provider': call['provider'], 'stage': 'first_completed_response',
                **choice_audit(call.get('parsed'), request.get('view', {}))})
    by_hash = {call['hash']: call for call in calls}
    for row in decisions[decisions.method.isin(['pro', 'lite'])].to_dict('records'):
        call = by_hash[row['request_hash']]
        rows.append({'provider': row['method'], 'stage': 'executed_main_choice',
            **choice_audit(row, call['request']['view'])})
    audit = pd.DataFrame(rows)
    summary = audit.groupby(['provider', 'stage']).agg(responses=('all_valid', 'size'),
        json_objects=('json_object', 'sum'), structures_valid=('structure_valid', 'sum'),
        actions_allowed=('action_allowed', 'sum'), delays_allowed=('delay_allowed', 'sum'),
        all_valid=('all_valid', 'sum')).reset_index()
    summary['valid_fraction'] = summary.all_valid/summary.responses
    summary.to_csv('res/response_quality.csv', index=False)
    return summary


def usage_summary():
    rows = []
    for call in api_records():
        usage = call.get('usage', {})
        rows.append({'provider': call['provider'], 'calls': 1,
            'unknown_usage_calls': int(call.get('status') == 'request_error' and not usage),
            'retry_attempt_calls': int(call.get('request', {}).get('repair_attempt', 0) > 0),
            'format_repair_calls': int('format_repair' in call.get('request', {}).get('prompt', '')),
            'incomplete_responses': int(call.get('status') == 'incomplete'),
            'input_tokens': usage.get('input_tokens', 0), 'output_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'budget_accounted_tokens': call.get('budget_charge_tokens', usage.get('total_tokens', 0))})
    result = pd.DataFrame(rows).groupby('provider', as_index=False).sum()
    result.to_csv('res/api_usage_summary.csv', index=False)
    return result


def write_report(metrics, predictions, decisions, matched, resolution, usage, paired, quality):
    with open('res/manifest.json') as f: manifest = json.load(f)
    with open('res/rule_selection.json') as f: rule = json.load(f)
    coverage = pd.read_csv('res/coverage.csv', dtype={'station_complex_id': str})
    incomplete = coverage[coverage.observed_hours < 24]
    main = metrics[(metrics.block == 'all_test') & (metrics.unit == 'daily')].groupby('method').wape.mean()
    best = main.reindex(METHODS).idxmin()
    lines = ['# V3 个人选择实验结果',
        f'本轮站点日总误差最低的是 `{best}`，WAPE {main[best]*100:.2f}%。主实验为 {predictions.station_complex_id.nunique()} 站、{predictions.date.nunique()} 天、每站 {manifest["main_n_agents_per_station"]} 个代表人物。',
        '每个 LLM 请求只替一个人选择一次出行方式，Python 按权重加总到站客流。它看不到站点总客流、权重或整体倍率。Rule、Pro、Lite 使用同一批人物、计划、环境和行动约束；规则仅在 Sep8–10 feedback 选择参数，测试后不调参。',
        '## 主要误差',
        '| 方法 | 站点日总 WAPE | 4 小时 WAPE | 逐小时 WAPE | 日总偏差 |',
        '|---|---:|---:|---:|---:|']
    for method in ['planned_profile']+METHODS:
        values = metrics[(metrics.method == method) & (metrics.block == 'all_test')].groupby('unit')[['wape', 'bias']].mean()
        lines.append(f'| {method} | {values.loc["daily","wape"]*100:.2f}% | {values.loc["four_hour","wape"]*100:.2f}% | {values.loc["hourly","wape"]*100:.2f}% | {values.loc["daily","bias"]*100:+.2f}% |')
    episodes = metrics[metrics.unit == 'daily'].groupby(['method', 'block']).wape.mean()
    lines.extend(['按窗口看日总误差：', '| 方法 | 普通窗口 Sep22–24 | 暴雨窗口 Sep28–30 |', '|---|---:|---:|'])
    for method in METHODS:
        lines.append(f'| {method} | {episodes.loc[(method,"test_ordinary")]*100:.2f}% | {episodes.loc[(method,"test_storm")]*100:.2f}% |')
    lines.append('这两个窗口各含三天，不等于逐小时的普通/极端天气分类。Pro 在普通窗口较好，但暴雨窗口未超过规则；不能只选一种窗口宣称整体优势。')
    lines.extend(['`planned_profile` 是小时计划量；`no_change` 让所有代表按计划乘地铁。完整网格的总量相同，但一个代表只在一个时刻出现，小时误差会受离散影响，不能直接算作决策错误。日总优先检查计划规模。',
        '指标和日图只加总有真实观测的共同小时，缺失不按零处理。群体指标也使用该组可观测小时。'])
    for r in incomplete.itertuples():
        lines.append(f'站点 {r.station_complex_id} / {r.date} 只有 {r.observed_hours}/24 小时观测，因此是部分日汇总；见 coverage.csv。')
    lines.extend(['| 日总配对比较 | WAPE 差（百分点） | 按日重采样区间 |', '|---|---:|---:|'])
    for r in paired.itertuples():
        lines.append(f'| {r.method} − {r.baseline} | {r.daily_wape_difference*100:+.2f} | [{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}] |')
    lines.extend(['## 个人动作、记忆与代表人数',
        '人物权重代表合成出行机会，不是调查人口数。人格、目的、费用和记忆都是假设；票种只作群体标签，老人/残障不能从真实数据拆开。图中的动作/人格差异和 reason 不能证明真实乘客行为或人格因果效应。',
        'subway 与 delay_subway 都是地铁选择；延期只在到站时计一次。窗口外到站和缺失观测小时的质量另列在 mass_audit.csv。',
        '配对检查只取一站的 Sep22、Sep29，保留同个人主选择前状态，比较无记忆、few-shot、重复请求。它们不是连续六天的替代学习轨迹；记忆效应须与重复采样的变化一起看。'])
    if matched is not None:
        lines.extend(['| 检查 | 日期 | 选择改变（加权） | 地铁份额变化（百分点） |', '|---|---|---:|---:|'])
        for r in matched.itertuples():
            lines.append(f'| {r.method} | {r.date} | {r.weighted_choice_changed_fraction*100:.1f}% | {(r.weighted_subway_share-r.reference_weighted_subway_share)*100:+.1f} |')
        probes = matched[matched.date == '2023-09-29'].set_index('method')
        facts = []
        for provider in ['pro', 'lite']:
            memory = probes.loc[provider+'_no_memory']; repeat = probes.loc[provider+'_repeat']
            facts.append(f'{provider} 地铁份额下降 {(memory.reference_weighted_subway_share-memory.weighted_subway_share)*100:.1f} 个百分点，选择变化 {memory.weighted_choice_changed_fraction*100:.1f}%（重复请求变化 {repeat.weighted_choice_changed_fraction*100:.1f}%）')
        lines.append('Sep29 去掉记忆后，'+'；'.join(facts)+'。这说明本次输入对记忆敏感，不等于真实乘客学习已验证。')
    if resolution is not None:
        lines.append('24/48/96 人数探针是一站 Sep29 的独立单日 episode，共用初始合成记忆，不是主三天窗口重跑。no_change 的小时误差可看离散程度；增加代表也会改变采样。结果见 resolution_metrics.csv 和下图。')
        probe = resolution.set_index(['method', 'n_agents', 'unit'])
        lines.append(f'no_change 的逐小时误差从 24 人的 {probe.loc[("no_change",24,"hourly"),"wape"]*100:.2f}% 降到 96 人的 {probe.loc[("no_change",96,"hourly"),"wape"]*100:.2f}%，而日总误差不变。Pro 的日总误差却从 {probe.loc[("pro",24,"daily"),"wape"]*100:.2f}% 到 {probe.loc[("pro",96,"daily"),"wape"]*100:.2f}%；更多代表主要减轻时间离散，不能保证提高日总预测。')
    lines.extend(['## 首次响应与最终执行',
        '| 模型 | 首次完成响应数 | 首次结构合法 | 首次行动及延期合法 | 主测试最终执行合法 |',
        '|---|---:|---:|---:|---:|'])
    for provider in ['pro', 'lite']:
        rows = quality[quality.provider == provider].set_index('stage')
        first = rows.loc['first_completed_response']; executed = rows.loc['executed_main_choice']
        lines.append(f'| {provider} | {int(first.responses)} | {int(first.structures_valid)}/{int(first.responses)} | {int(first.all_valid)}/{int(first.responses)} | {int(executed.all_valid)}/{int(executed.responses)} |')
    lines.extend(['首次指日志 repair_attempt=0 且 status=completed 的原始返回，按该请求 available_actions 和 delay 上限检查；未完成响应、网络失败另计。最终执行是修复/重试后的主测试选择，两种分母不同。最终都合法不意味着原模型首次都合法。日志汇总包含 pilot 等额外请求。',
        '## 用量与可信度',
        '| 模型 | 调用 | 已报告 tokens | 未知失败 | 不完整返回 | 重试轮次 | 格式修复 | 保守记账 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in usage.itertuples():
        lines.append(f'| {r.provider} | {r.calls} | {r.total_tokens:,} | {r.unknown_usage_calls} | {r.incomplete_responses} | {r.retry_attempt_calls} | {r.format_repair_calls} | {r.budget_accounted_tokens:,} |')
    audit = pd.read_csv('res/mass_audit.csv')
    lines.append(f'到站质量守恒的最大差为 {audit.mass_balance_difference.abs().max():.2g}，属于浮点舍入；缺失源观测小时的到站质量单独记录，不参加评分。')
    lines.extend([f'冻结预算记录：{json.dumps(manifest["caps"])}。未知失败按预留记账，不能视为免费；重试轮次既可能是网络重试，也可能格式修复。日志包含试跑、feedback、测试和额外探针，不是只有主测试。',
        '这是实测天气条件下的旧数据回测，不是天气预报部署或新盲测。普通、暴雨各三天独立重置记忆；日期少且共享天气，按日 bootstrap 不代表独立极端事件置信保证。V1/V2 与本轮任务和指标不同，不能直接作升级前后比较。',
        '## 人物理由示例'])
    examples = decisions[decisions.method.isin(['pro', 'lite'])].sort_values(['method', 'date', 'agent_id']).groupby(['method', 'group', 'weather_context']).head(1)
    examples.to_csv('res/reason_examples.csv', index=False)
    for provider in ['pro', 'lite']:
        for r in examples[examples.method == provider].head(2).itertuples():
            text = str(r.reason).replace('\n', ' ')
            lines.append(f'- {r.method} / {r.group} / {r.type} / {r.weather_context} → {r.action}，延期 {r.delay_min} 分钟：{text}')
    lines.extend(['示例按模型、群体、天气取排序第一条，未挑精彩案例；完整原文见 reason_examples.csv。',
        '另有一条 Lite 理由提及未提供的 forecast，而输入明确没有未来天气预报。菜单与延期约束合法不保证理由信息忠实；这个实例需要审阅，不能当成真实预报或全体语义准确率。',
        '从 V3 执行 `python analyze.py` 免费重做分析。feedback_metrics.csv 单独保留，不混入测试；metrics_summary.csv、action_shares.csv、trait_actions.csv、paired_bootstrap.csv、response_quality.csv 保存详细结果。',
        '## 图表', '![站点日总](res/figures/daily_station_predictions.png)',
        '![群体行动](res/figures/group_actions.png)', '![人格描述](res/figures/trait_actions.png)'])
    if resolution is not None: lines.append('![代表人数](res/figures/resolution_errors.png)')
    if matched is not None: lines.append('![个人配对检查](res/figures/matched_checks.png)')
    rendered = lines[0]
    for previous, line in zip(lines, lines[1:]):
        separator = '\n' if previous.startswith('|') and line.startswith('|') else '\n\n'
        rendered += separator+line
    with open('RESULTS.md', 'w') as f: f.write(rendered+'\n')


def main():
    os.makedirs('res/figures', exist_ok=True)
    panel = pd.read_csv('res/panel.csv.gz', dtype={'station_complex_id': str}, parse_dates=['timestamp', 'date'])
    predictions = pd.read_csv('res/predictions.csv.gz', dtype={'station_complex_id': str}, parse_dates=['timestamp', 'date'])
    predictions = predictions[predictions.block.isin(['test_ordinary', 'test_storm'])].copy()
    decisions = pd.read_csv('res/decisions.csv.gz', dtype={'station_complex_id': str}, parse_dates=['planned_timestamp', 'actual_entry_timestamp'])
    decisions = decisions[decisions.block.isin(['test_ordinary', 'test_storm'])].copy()
    planned = predictions[predictions.method == 'no_change'].copy()
    planned['method'] = 'planned_profile'; planned['predicted'] = planned.planned_total
    metrics = timing_metrics(pd.concat([predictions, planned], ignore_index=True))
    metrics.to_csv('res/metrics_summary.csv', index=False)
    coverage = predictions[predictions.method == 'no_change'].drop_duplicates(['station_complex_id', 'timestamp'])
    coverage = coverage.groupby(['block', 'station_complex_id', 'date']).agg(grid_hours=('timestamp', 'size'), observed_hours=('total_ridership', 'count')).reset_index()
    coverage['source_missing_hours'] = coverage.grid_hours-coverage.observed_hours
    coverage['complete_day'] = coverage.observed_hours == 24
    coverage.to_csv('res/coverage.csv', index=False)
    daily = predictions[predictions.total_ridership.notna()].groupby(['method', 'block', 'n_agents', 'sample', 'station_complex_id', 'date'], as_index=False)[['predicted', 'total_ridership']].sum()
    daily['absolute_error'] = (daily.predicted-daily.total_ridership).abs()
    daily.to_csv('res/daily_metrics.csv', index=False)
    group_rows = []
    for group in GROUPS:
        frame = predictions.dropna(subset=[group]).copy()
        frame['predicted'] = frame['pred_'+group]; frame['total_ridership'] = frame[group]
        frame = frame.groupby(['method', 'block', 'n_agents', 'sample', 'station_complex_id', 'date'], as_index=False)[['predicted', 'total_ridership']].sum()
        result = summarize(frame, ['method', 'block', 'n_agents', 'sample'])
        result['group'] = group; group_rows.append(result)
    pd.concat(group_rows, ignore_index=True).to_csv('res/group_metrics.csv', index=False)
    actions = action_summary(decisions, ['method', 'block', 'sample', 'group'])
    actions.to_csv('res/action_shares.csv', index=False)
    trait_rows = []
    decisions['risk_band'] = pd.cut(decisions.risk_aversion, [0, .35, .6, 1], labels=['low', 'medium', 'high']).astype(str)
    decisions['subway_mass'] = decisions.weight*decisions.action.isin(['subway', 'delay_subway'])
    for feature in ['risk_band', 'type', 'subway_habit', 'travel_budget', 'mobility_support']:
        rows = decisions.groupby(['method', 'block', feature], as_index=False).agg(decisions=('agent_id', 'size'), planned_mass=('weight', 'sum'), subway_mass=('subway_mass', 'sum'))
        rows = rows.rename(columns={feature: 'value'}); rows['feature'] = feature
        rows['weighted_subway_share'] = rows.subway_mass/rows.planned_mass
        trait_rows.append(rows)
    traits = pd.concat(trait_rows, ignore_index=True); traits.to_csv('res/trait_actions.csv', index=False)
    audit = []
    for (method, block), frame in decisions.groupby(['method', 'block']):
        selected = predictions[(predictions.method == method) & (predictions.block == block)]
        entered = frame.action.isin(['subway', 'delay_subway'])
        supported = pd.MultiIndex.from_frame(frame[['station_complex_id', 'actual_entry_timestamp']]).isin(pd.MultiIndex.from_frame(selected[['station_complex_id', 'timestamp']]))
        observed_support = pd.MultiIndex.from_frame(frame[['station_complex_id', 'actual_entry_timestamp']]).isin(pd.MultiIndex.from_frame(selected[selected.total_ridership.notna()][['station_complex_id', 'timestamp']]))
        audit.append({'method': method, 'block': block, 'planned_mass': frame.weight.sum(),
            'subway_mass': frame.loc[entered, 'weight'].sum(), 'other_action_mass': frame.loc[~entered, 'weight'].sum(),
            'predicted_in_window': selected.predicted.sum(), 'subway_outside_support': frame.loc[entered & ~supported, 'weight'].sum(),
            'subway_at_source_missing_hour': frame.loc[entered & supported & ~observed_support, 'weight'].sum(),
            'predicted_on_observed_hours': selected.loc[selected.total_ridership.notna(), 'predicted'].sum(),
            'mass_balance_difference': frame.loc[entered & supported, 'weight'].sum()-selected.predicted.sum()})
    pd.DataFrame(audit).to_csv('res/mass_audit.csv', index=False)
    main_figures(predictions, decisions, actions, traits)
    matched = matched_analysis(); resolution = resolution_analysis(panel)
    paired = paired_days(predictions); paired.to_csv('res/paired_bootstrap.csv', index=False)
    usage = usage_summary()
    write_report(metrics, predictions, decisions, matched, resolution, usage, paired, response_quality(decisions))
    if os.path.exists('res/feedback_decisions.csv.gz'):
        feedback = pd.read_csv('res/feedback_decisions.csv.gz', dtype={'station_complex_id': str})
        timing_metrics(decision_predictions(panel, feedback)).query("block != 'all_test'").to_csv('res/feedback_metrics.csv', index=False)
    print(metrics[(metrics.block == 'all_test') & (metrics.unit == 'daily')][['method', 'wape', 'bias']].to_string(index=False))


if __name__ == '__main__': main()
