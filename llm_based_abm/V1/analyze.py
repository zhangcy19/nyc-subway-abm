"""Analyze the held-out experiment. Run from the llm_based_abm folder.

    python analyze.py --extended

Input: results/predictions.csv.gz (one row per model, seed, station, hour and group).
Outputs: summary CSVs and static scientific figures in results/figures.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


CONFIG = ['model', 'shot', 'n_agents', 'seed']
GROUP_LABELS = {
    'students': 'Student fare',
    'fair_fare': 'Fair Fare',
    'seniors_disability': 'Senior / disability fare',
    'general': 'Other fares',
}


def error_summary(frame):
    observed = frame['observed_ridership'].to_numpy(float)
    predicted = frame['predicted_ridership'].to_numpy(float)
    error = predicted - observed
    denominator = observed.sum()
    return {
        'n_observations': len(frame),
        'observed_sum': denominator,
        'predicted_sum': predicted.sum(),
        'wape': np.abs(error).sum() / denominator if denominator else np.nan,
        'mae': np.abs(error).mean(),
        'rmse': np.sqrt(np.square(error).mean()),
        'bias_fraction': error.sum() / denominator if denominator else np.nan,
    }


def summarize(frame, keys):
    records = []
    for values, subset in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        records.append({**dict(zip(keys, values)), **error_summary(subset)})
    return pd.DataFrame(records)


def label(row):
    if row['model'] == 'llm_abm':
        return f"LLM ABM {row['shot']} ({int(row['n_agents'])} agents)"
    if row['model'] == 'llm_direct':
        return f"LLM direct {row['shot']}"
    return row['model'].replace('_', ' ').capitalize()


def save_figure(filename):
    plt.tight_layout()
    plt.savefig('results/figures/' + filename, dpi=180, bbox_inches='tight')
    plt.close()


def primary_configs(frame):
    # The main time-series panel uses 10 agents; all counts remain in metrics.
    return frame[(frame['model'] != 'llm_abm') | (frame['n_agents'] == 10)].copy()


def analyze(filename):
    os.makedirs('results/figures', exist_ok=True)
    data = pd.read_csv(filename, dtype={'station_complex_id': str})
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['shot'] = data['shot'].fillna('none')
    data['n_agents'] = data['n_agents'].fillna(0).astype(int)
    data = data[(data['timestamp'] >= '2023-09-22') &
                (data['timestamp'] < '2023-10-01')].copy()
    data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')

    # Sum predictions across fare groups; use the independently recorded true total.
    station_keys = CONFIG + ['timestamp', 'station_complex_id']
    totals = data.groupby(station_keys, dropna=False).agg(
        predicted_ridership=('predicted_ridership', 'sum'),
        observed_ridership=('total_observed', 'first'),
        rain_mm=('rain_mm', 'first'),
        weather_context=('weather_context', 'first'),
        borough=('borough', 'first'),
        latitude=('latitude', 'first'),
        longitude=('longitude', 'first'),
    ).reset_index()
    totals['date'] = totals['timestamp'].dt.strftime('%Y-%m-%d')
    observed_totals = totals[totals['observed_ridership'].notna()].copy()
    observed_groups = data[data['observed_ridership'].notna()].copy()
    total_metrics = summarize(observed_totals, CONFIG)
    total_metrics['group'] = 'total'
    group_metrics = summarize(observed_groups, CONFIG + ['group'])
    metrics = pd.concat([total_metrics, group_metrics], ignore_index=True)
    metrics.to_csv('results/metrics.csv', index=False)

    daily = observed_totals.groupby(CONFIG + ['date'], dropna=False).agg(
        observed_ridership=('observed_ridership', 'sum'),
        predicted_ridership=('predicted_ridership', 'sum'),
        n_station_hours=('station_complex_id', 'size'),
    ).reset_index()
    daily.to_csv('results/daily_totals.csv', index=False)

    # Context labels are those available to the model, not post-hoc causal categories.
    weather_metrics = summarize(observed_totals, CONFIG + ['weather_context'])
    weather_metrics.to_csv('results/weather_metrics.csv', index=False)
    station_metrics = summarize(observed_totals, CONFIG + ['station_complex_id'])
    coordinates = totals[['station_complex_id', 'borough', 'latitude', 'longitude']].drop_duplicates('station_complex_id')
    station_metrics = station_metrics.merge(coordinates, on='station_complex_id', how='left')
    station_metrics.to_csv('results/station_metrics.csv', index=False)

    count_metrics = total_metrics[total_metrics['model'] == 'llm_abm'].copy()
    count_metrics.to_csv('results/agent_count_metrics.csv', index=False)
    paired = count_metrics.pivot(index=['shot', 'seed'], columns='n_agents', values='wape')
    paired.columns = [f'wape_{int(count)}_agents' for count in paired.columns]
    for count in [5, 20]:
        column = f'wape_{count}_agents'
        if column in paired.columns and 'wape_10_agents' in paired.columns:
            paired[f'delta_{count}_vs_10'] = paired[column] - paired['wape_10_agents']
    paired.reset_index().to_csv('results/agent_count_paired.csv', index=False)

    config_keys = ['model', 'shot', 'n_agents']
    accuracy = total_metrics.groupby(config_keys, dropna=False).agg(
        wape_mean=('wape', 'mean'), wape_min=('wape', 'min'),
        wape_max=('wape', 'max'), seeds=('seed', 'nunique'),
    ).reset_index()
    accuracy['label'] = accuracy.apply(label, axis=1)
    accuracy = accuracy.sort_values('wape_mean')
    plt.figure(figsize=(9, max(4, .45 * len(accuracy))))
    positions = np.arange(len(accuracy))
    plt.barh(positions, accuracy['wape_mean'] * 100, color='#4477AA')
    plt.errorbar(accuracy['wape_mean'] * 100, positions,
                 xerr=np.vstack([(accuracy['wape_mean'] - accuracy['wape_min']) * 100,
                                  (accuracy['wape_max'] - accuracy['wape_mean']) * 100]),
                 fmt='none', ecolor='black', capsize=3)
    plt.yticks(positions, accuracy['label'])
    plt.gca().invert_yaxis()
    plt.xlabel('Total ridership WAPE (%) — lower is better')
    plt.title('Held-out hours at 12 training-selected stations: 22–30 September 2023\nBars: seed mean; whiskers: seed range')
    save_figure('accuracy_by_configuration.png')

    hourly = observed_totals.groupby(CONFIG + ['timestamp'], dropna=False)[
        ['observed_ridership', 'predicted_ridership']].sum().reset_index()
    hourly.to_csv('results/hourly_totals.csv', index=False)
    event = primary_configs(hourly)
    event = event[event['timestamp'].dt.strftime('%Y-%m-%d') == '2023-09-29']
    plt.figure(figsize=(10, 5))
    truth = event.groupby('timestamp')['observed_ridership'].first()
    plt.plot(truth.index.hour, truth.values, color='black', linewidth=2.5, label='Observed')
    for values, frame in event.groupby(config_keys, dropna=False):
        average = frame.groupby('timestamp')['predicted_ridership'].mean()
        plt.plot(average.index.hour, average.values, label=label(dict(zip(config_keys, values))))
    plt.xlabel('Local hour (America/New_York)')
    plt.ylabel('Ridership across selected stations')
    plt.title('29 September 2023: observed and predicted hourly ridership')
    plt.legend(fontsize=8)
    save_figure('september29_hourly.png')

    group_accuracy = primary_configs(group_metrics)
    group_accuracy['label'] = group_accuracy.apply(label, axis=1)
    group_pivot = group_accuracy.groupby(['group', 'label'])['wape'].mean().unstack() * 100
    group_pivot.index = [GROUP_LABELS.get(value, value) for value in group_pivot.index]
    group_pivot.plot.bar(figsize=(10, 5), width=.8)
    plt.xlabel('Fare category (not verified personal demographics)')
    plt.ylabel('Group ridership WAPE (%)')
    plt.title('Prediction error by observed fare category')
    plt.xticks(rotation=15, ha='right')
    plt.legend(fontsize=8)
    save_figure('group_accuracy.png')

    weather_accuracy = primary_configs(weather_metrics)
    weather_accuracy['label'] = weather_accuracy.apply(label, axis=1)
    weather_pivot = weather_accuracy.groupby(['weather_context', 'label'])['wape'].mean().unstack() * 100
    weather_pivot = weather_pivot.reindex(['dry', 'light', 'moderate', 'extreme'])
    weather_pivot.plot.bar(figsize=(10, 5), width=.8)
    plt.xlabel('Weather context available to the model')
    plt.ylabel('Total ridership WAPE (%)')
    plt.title('Prediction error by weather context — 10-agent main comparison')
    plt.xticks(rotation=0)
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    save_figure('weather_accuracy.png')

    group_daily = observed_groups.groupby(CONFIG + ['group', 'date'])[
        ['observed_ridership', 'predicted_ridership', 'baseline_demand']].sum().reset_index()
    group_daily.to_csv('results/group_daily_totals.csv', index=False)
    effects = primary_configs(group_daily[group_daily['date'] == '2023-09-29']).copy()
    effects['observed_ratio'] = effects['observed_ridership'] / effects['baseline_demand'].replace(0, np.nan)
    effects['predicted_ratio'] = effects['predicted_ridership'] / effects['baseline_demand'].replace(0, np.nan)
    effects['label'] = effects.apply(label, axis=1)
    effect_pivot = effects.groupby(['group', 'label'])['predicted_ratio'].mean().unstack()
    effect_pivot['Observed / baseline'] = effects.groupby('group')['observed_ratio'].first()
    effect_pivot.index = [GROUP_LABELS.get(value, value) for value in effect_pivot.index]
    effect_pivot.plot.bar(figsize=(10, 5), width=.8)
    plt.axhline(1, color='gray', linewidth=1, linestyle='--')
    plt.xlabel('Fare category')
    plt.ylabel('29 September ridership / baseline demand')
    plt.title('Event-day group response relative to the shared demand baseline')
    plt.xticks(rotation=15, ha='right')
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    save_figure('group_event_response.png')

    if len(count_metrics):
        plt.figure(figsize=(7, 4))
        for shot, frame in count_metrics.groupby('shot'):
            color = {'zero': '#4477AA', 'few': '#CC6677'}[shot]
            for seed, run in frame.groupby('seed'):
                run = run.sort_values('n_agents')
                plt.plot(run['n_agents'], run['wape'] * 100, alpha=.3, linewidth=1, color=color)
            mean = frame.groupby('n_agents')['wape'].mean()
            plt.plot(mean.index, mean.values * 100, marker='o', linewidth=2, label=shot, color=color)
        plt.xlabel('Number of representative agents')
        plt.ylabel('Total ridership WAPE (%)')
        plt.title('Agent-count sensitivity\nThin lines: individual seeds; bold lines: means')
        plt.xticks(sorted(count_metrics['n_agents'].unique()))
        plt.legend(title='Prompt setting')
        save_figure('agent_count_sensitivity.png')

    geography = primary_configs(station_metrics)
    panels = list(geography.groupby(config_keys, dropna=False))
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False)
    maximum = geography.groupby(config_keys + ['station_complex_id'])['wape'].mean().max() * 100
    for axis, (values, frame) in zip(axes.flat, panels):
        points = frame.groupby('station_complex_id').agg(
            longitude=('longitude', 'first'), latitude=('latitude', 'first'), wape=('wape', 'mean'))
        scatter = axis.scatter(points['longitude'], points['latitude'], c=points['wape'] * 100,
                               cmap='viridis', vmin=0, vmax=maximum, s=65, edgecolor='black', linewidth=.4)
        axis.set_title(label(dict(zip(config_keys, values))), fontsize=10)
        axis.set_xlabel('Longitude')
        axis.set_ylabel('Latitude')
        axis.xaxis.set_major_locator(MaxNLocator(3))
        axis.tick_params(axis='x', labelsize=8)
        axis.set_aspect(1 / np.cos(np.deg2rad(40.7)))
        figure.colorbar(scatter, ax=axis, label='Station WAPE (%)', shrink=.8)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.suptitle('Spatial distribution of prediction errors — selected stations only', y=1.01)
    save_figure('station_error_geography.png')
    print(f'Analyzed {len(data):,} prediction rows; outputs saved in results/.')


def analyze_sensitivity(prediction_filename):
    """Additional full-run outputs; return-rate alternatives never enter main metrics."""
    main = pd.read_csv(prediction_filename, dtype={'station_complex_id': str})
    alternatives = pd.read_csv('results/return_rate_sensitivity.csv.gz', dtype={'station_complex_id': str})
    main = main[(main['model'] == 'llm_abm') & (main['n_agents'] == 10)]
    sensitivity = pd.concat([main, alternatives], ignore_index=True)
    sensitivity['timestamp'] = pd.to_datetime(sensitivity['timestamp'])
    sensitivity = sensitivity[(sensitivity['timestamp'] >= '2023-09-22') &
                              (sensitivity['timestamp'] < '2023-10-01')].copy()
    keys = CONFIG + ['return_rate']
    totals = sensitivity.groupby(keys + ['timestamp', 'station_complex_id'], dropna=False).agg(
        predicted_ridership=('predicted_ridership', 'sum'),
        observed_ridership=('total_observed', 'first'),
        weather_context=('weather_context', 'first'),
    ).reset_index()
    totals = totals[totals['observed_ridership'].notna()]
    metrics = summarize(totals, keys)
    metrics['group'] = 'total'
    group_metrics = summarize(sensitivity[sensitivity['observed_ridership'].notna()], keys + ['group'])
    pd.concat([metrics, group_metrics], ignore_index=True).to_csv('results/return_rate_metrics.csv', index=False)
    summarize(totals, keys + ['weather_context']).to_csv('results/return_rate_weather_metrics.csv', index=False)
    plt.figure(figsize=(7, 4))
    for shot, frame in metrics.groupby('shot'):
        mean = frame.groupby('return_rate')['wape'].mean()
        plt.plot(mean.index, mean.values * 100, marker='o', label=shot,
                 color={'zero': '#4477AA', 'few': '#CC6677'}[shot])
    plt.xlabel('Assumed fraction of postponed demand returning one hour later')
    plt.ylabel('Total ridership WAPE (%)')
    plt.title('Sensitivity to the assumed return rate — 10 agents')
    plt.xticks(sorted(metrics['return_rate'].unique()))
    plt.legend(title='Prompt setting')
    save_figure('return_rate_sensitivity.png')

    actions = pd.read_csv('results/actions.csv.gz', dtype={'station_complex_id': str})
    actions['timestamp'] = pd.to_datetime(actions['timestamp'])
    context = sensitivity[['timestamp', 'station_complex_id', 'group', 'weather_context', 'is_peak']].drop_duplicates(
        ['timestamp', 'station_complex_id', 'group'])
    actions = actions.merge(context, on=['timestamp', 'station_complex_id', 'group'], how='inner')
    # Action mass concerns new planned demand; returning postponed riders are excluded.
    action_keys = ['shot', 'n_agents', 'seed', 'group', 'weather_context', 'is_peak']
    shares = actions.groupby(action_keys + ['action'], dropna=False)['mass'].sum().reset_index()
    shares['planned_mass'] = shares.groupby(action_keys, dropna=False)['mass'].transform('sum')
    shares['weighted_action_share'] = shares['mass'] / shares['planned_mass'].replace(0, np.nan)
    shares.to_csv('results/action_shares.csv', index=False)

    grouped = actions.groupby(['shot', 'n_agents', 'seed', 'group', 'weather_context', 'action'])['mass'].sum().reset_index()
    grouped['weighted_action_share'] = grouped['mass'] / grouped.groupby(
        ['shot', 'n_agents', 'seed', 'group', 'weather_context'])['mass'].transform('sum').replace(0, np.nan)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    action_order = ['continue', 'postpone', 'other_mode', 'cancel']
    for axis, group in zip(axes.flat, GROUP_LABELS):
        subset = grouped[(grouped['n_agents'] == 10) & (grouped['group'] == group)]
        pivot = subset.groupby(['weather_context', 'shot', 'action'])['weighted_action_share'].mean().unstack('action')
        order = [(weather, shot) for weather in ['dry', 'light', 'moderate', 'extreme']
                 for shot in ['zero', 'few'] if (weather, shot) in pivot.index]
        pivot = pivot.reindex(order)
        pivot = pivot.reindex(columns=action_order)
        pivot.index = [weather + ' / ' + shot for weather, shot in pivot.index]
        pivot.plot.bar(stacked=True, ax=axis, color=['#4477AA', '#66CCEE', '#CCBB44', '#EE6677'])
        axis.set_title(GROUP_LABELS[group])
        axis.set_ylabel('Demand-weighted action share')
        axis.set_xlabel('Weather context / prompt setting')
        axis.tick_params(axis='x', rotation=45)
        axis.set_ylim(0, 1)
        axis.legend(fontsize=7)
    figure.suptitle('Simulated actions under weather contexts — 10 agents\nModel assumptions; no observed individual decisions or causal effect estimates')
    save_figure('simulated_action_shares.png')
    print('Saved return-rate sensitivity and demand-weighted action summaries.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='results/predictions.csv.gz')
    parser.add_argument('--extended', action='store_true', help='Analyze full-run return-rate sensitivity and actions')
    args = parser.parse_args()
    analyze(args.predictions)
    if args.extended:
        analyze_sensitivity(args.predictions)
