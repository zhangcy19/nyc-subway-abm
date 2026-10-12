"""Run from llm_based_abm: python prepare_experiment.py. No API calls."""
import os
import numpy as np
import pandas as pd

os.makedirs('experiment_data', exist_ok=True)
source = pd.read_csv('../../data/station_hourly.csv.gz', dtype={'station_complex_id': str}, parse_dates=['timestamp'])
weather = pd.read_csv('../../data/weather_hourly.csv', parse_dates=['timestamp']).sort_values('timestamp')
weather['rain_6h_mm'] = weather.rain_mm.rolling(6, min_periods=1).sum()
weather['rain_24h_mm'] = weather.rain_mm.rolling(24, min_periods=1).sum()
weather['weather_context'] = np.select(
    [weather.rain_6h_mm >= 30, (weather.rain_mm >= 5) | (weather.rain_6h_mm >= 10),
     (weather.rain_mm > .1) | (weather.rain_6h_mm > .1)],
    ['extreme', 'moderate', 'light'], default='dry')
weather['split'] = np.select([weather.timestamp < '2023-09-01', weather.timestamp < '2023-09-22'], ['train', 'validation'], default='test')
weather['weekday'] = weather.timestamp.dt.dayofweek
weather['hour'] = weather.timestamp.dt.hour
weather['is_peak'] = (weather.weekday < 5) & (weather.hour.between(7, 9) | weather.hour.between(16, 19))
weather.to_csv('experiment_data/scenarios.csv', index=False)

aug = source[source.timestamp < '2023-09-01']
coverage = aug.groupby(['borough', 'station_complex_id']).agg(august_rows=('timestamp', 'size'), august_ridership=('total_ridership', 'sum')).reset_index()
chosen = []
for borough in ['Bronx', 'Brooklyn', 'Manhattan', 'Queens']:
    candidates = coverage[(coverage.borough == borough) & (coverage.august_rows >= .98 * 744)].sort_values(['august_ridership', 'station_complex_id'], ascending=[False, True])
    chosen.append(candidates.iloc[[0, len(candidates)//3, 2*len(candidates)//3]])
selection = pd.concat(chosen)
stations = pd.read_csv('../../data/stations.csv', dtype={'station_complex_id': str}).merge(selection, on=['station_complex_id', 'borough'])
stations.to_csv('experiment_data/stations.csv', index=False)
x = source[source.station_complex_id.isin(stations.station_complex_id)].copy()
groups = {
    'students': ['metrocard_students'],
    'fair_fare': ['metrocard_fair_fare'],
    'seniors_disability': ['metrocard_seniors_disability', 'omny_seniors_disability'],
    'general': ['metrocard_full_fare', 'metrocard_unlimited_30day', 'metrocard_unlimited_7day', 'metrocard_other', 'omny_full_fare', 'omny_other']}
for group, channels in groups.items():
    x[group] = x[channels].sum(axis=1, min_count=1)
panel = x.melt(id_vars=['timestamp', 'station_complex_id', 'station_name', 'borough', 'latitude', 'longitude', 'total_ridership'], value_vars=list(groups), var_name='group', value_name='observed_ridership')
panel = panel.rename(columns={'total_ridership': 'total_observed'}).merge(weather, on='timestamp', how='left', suffixes=('', '_weather'))
panel['observation_available'] = panel.observed_ridership.notna()
dry = panel[(panel.split == 'train') & (panel.weather_context == 'dry')]
# Explicit allocation assumption only: absent fare cells contribute no baseline mass.
# Original observations remain NaN and are excluded from group-error evaluation.
dry = dry.copy()
dry['observed_ridership'] = dry.observed_ridership.fillna(0)
keys = ['station_complex_id', 'group', 'weekday', 'hour']
profile = dry.groupby(keys).observed_ridership.agg(['mean', 'count']).reset_index()
fallback = dry.groupby(['station_complex_id', 'group', 'hour']).observed_ridership.mean().rename('hour_mean').reset_index()
grid = pd.MultiIndex.from_product([stations.station_complex_id, list(groups), range(7), range(24)], names=keys).to_frame(index=False)
profile = grid.merge(profile, on=keys, how='left').merge(fallback, on=['station_complex_id', 'group', 'hour'], how='left')
group_mean = dry.groupby(['station_complex_id', 'group']).observed_ridership.mean().rename('group_mean').reset_index()
profile = profile.merge(group_mean, on=['station_complex_id', 'group'], how='left')
profile['hour_mean'] = profile.hour_mean.fillna(profile.group_mean).fillna(0)
profile['count'] = profile['count'].fillna(0)
profile['baseline_aug'] = (profile['mean'].fillna(0)*profile['count'] + 2*profile.hour_mean)/(profile['count']+2)
panel = panel.merge(profile[keys+['baseline_aug']], on=keys, how='left')

# The total profile uses genuine station totals, independently of incomplete fare cells.
total_dry = x.merge(weather[['timestamp','split','weekday','hour','weather_context']], on='timestamp')
total_dry = total_dry[(total_dry.split == 'train') & (total_dry.weather_context == 'dry')]
total_profile = total_dry.groupby(['station_complex_id','weekday','hour']).total_ridership.agg(['mean','count']).reset_index()
total_fallback = total_dry.groupby(['station_complex_id','hour']).total_ridership.mean().rename('hour_mean').reset_index()
total_grid = pd.MultiIndex.from_product([stations.station_complex_id, range(7), range(24)], names=['station_complex_id','weekday','hour']).to_frame(index=False)
total_profile = total_grid.merge(total_profile,on=['station_complex_id','weekday','hour'],how='left').merge(total_fallback,on=['station_complex_id','hour'],how='left')
total_profile['count'] = total_profile['count'].fillna(0)
total_profile['station_baseline_aug'] = (total_profile['mean'].fillna(0)*total_profile['count']+2*total_profile.hour_mean)/(total_profile['count']+2)
panel = panel.merge(total_profile[['station_complex_id','weekday','hour','station_baseline_aug']],on=['station_complex_id','weekday','hour'],how='left')

# September school-term/season correction uses validation dry hours only, never test.
v = panel[(panel.split == 'validation') & (panel.weather_context == 'dry')].copy()
v['observed_ridership'] = v.observed_ridership.fillna(0)
pooled = v.groupby('group').agg(observed=('observed_ridership','sum'),expected=('baseline_aug','sum'))
pooled['pooled_scale'] = pooled.observed/pooled.expected
calibration = v.groupby(['station_complex_id','group']).agg(observed=('observed_ridership','sum'),expected=('baseline_aug','sum'),n=('timestamp','size')).reset_index().merge(pooled[['pooled_scale']],on='group')
prior = profile.groupby(['station_complex_id','group']).baseline_aug.mean().rename('prior_hourly').reset_index()
calibration = calibration.merge(prior,on=['station_complex_id','group'])
calibration['season_scale'] = (calibration.observed+24*calibration.prior_hourly*calibration.pooled_scale)/(calibration.expected+24*calibration.prior_hourly)
panel = panel.merge(calibration[['station_complex_id','group','season_scale']],on=['station_complex_id','group'],how='left')
panel['baseline_demand'] = panel.baseline_aug * np.where(panel.split == 'train', 1, panel.season_scale)
total_validation = panel[(panel.split == 'validation') & (panel.weather_context == 'dry')].drop_duplicates(['timestamp','station_complex_id'])
total_scale = total_validation.groupby('station_complex_id').agg(observed=('total_observed','sum'),expected=('station_baseline_aug','sum')).reset_index()
pooled_total_scale = total_scale.observed.sum()/total_scale.expected.sum()
total_prior = total_profile.groupby('station_complex_id').station_baseline_aug.mean().rename('prior_hourly').reset_index()
total_scale = total_scale.merge(total_prior,on='station_complex_id')
total_scale['station_season_scale'] = (total_scale.observed+24*total_scale.prior_hourly*pooled_total_scale)/(total_scale.expected+24*total_scale.prior_hourly)
panel = panel.merge(total_scale[['station_complex_id','station_season_scale']],on='station_complex_id')
panel['station_baseline_demand'] = panel.station_baseline_aug*np.where(panel.split == 'train',1,panel.station_season_scale)
calibration.to_csv('experiment_data/calibration.csv',index=False)
profile[keys+['baseline_aug','count']].to_csv('experiment_data/baseline_hourly.csv',index=False)
panel = panel.sort_values(['timestamp','station_complex_id','group'])
panel.to_csv('experiment_data/panel.csv',index=False,float_format='%.6f')
summary = panel.groupby(['split','group']).agg(rows=('timestamp','size'),observed_rows=('observed_ridership','count'),observed_total=('observed_ridership','sum')).reset_index()
summary.to_csv('experiment_data/coverage.csv',index=False)
print(stations[['station_complex_id','station_name','borough','august_rows']].to_string(index=False))
print(summary.to_string(index=False))
print('Panel rows:',len(panel),'Station-hours:',len(x),'Missing weather rows:',panel.rain_mm.isna().sum())
