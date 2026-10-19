"""Independently reconcile every subway-hour CSV row against gauge observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['rain_mm', 'temp_c', 'wind_ms']
meta_file = ROOT / 'subway_weather_metadata.json'
meta = json.loads(meta_file.read_text(encoding='utf-8'))
stations = pd.read_csv(ROOT.parent / 'stations.csv', dtype={'station_complex_id': str})
codes = [s['station'] for s in meta['weather_sources']]
expected_hours = pd.date_range('2023-07-01', '2023-11-01', freq='h', inclusive='left')
expected_timestamps = expected_hours.strftime('%Y-%m-%d %H:%M:%S').tolist()
gauge_frames = [pd.read_csv(ROOT / s['filename']) for s in meta['weather_sources']]
assert all(g.timestamp.tolist() == expected_timestamps for g in gauge_frames)
gauge_values = np.stack([g[FIELDS].to_numpy(float) for g in gauge_frames], axis=1)  # hour, gauge, variable
gauge_lat = np.array([s['latitude'] for s in meta['weather_sources']])
gauge_lon = np.array([s['longitude'] for s in meta['weather_sources']])
catalog = {s['station_complex_id']: s for s in json.loads((ROOT / 'subway_weather_weights.json').read_text(encoding='utf-8'))}
counts = {v: dict(top3_full=0, top3_partial=0, fallback=0, missing=0) for v in FIELDS}
source_counts = {v: dict.fromkeys(codes, 0) for v in FIELDS}
examples = {}
rows_checked = complete = station_count = 0
max_error = 0.0

for station_index, df in enumerate(pd.read_csv(ROOT / meta['output'], dtype={'station_complex_id': str}, chunksize=2952)):
    s = stations.iloc[station_index]
    assert df.columns.tolist() == meta['columns']
    assert len(df) == len(expected_hours)
    assert df.timestamp.tolist() == expected_timestamps
    for column in ['station_complex_id', 'station_name', 'borough', 'latitude', 'longitude']:
        assert df[column].eq(s[column]).all(), f'Metadata mismatch: {s.station_complex_id} / {column}'
    assert df.weather_available.dtype == bool

    # Use atan2 great-circle distance independently of the producer's asin form.
    lat, lon = np.radians(s.latitude), np.radians(s.longitude)
    a = np.sin((np.radians(gauge_lat)-lat)/2)**2 + np.cos(lat)*np.cos(np.radians(gauge_lat))*np.sin((np.radians(gauge_lon)-lon)/2)**2
    a = np.clip(a, 0, 1)
    distances = 2*6371.0088*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    ranked = np.argsort(distances, kind='stable')
    nearest3 = ranked[:3]
    saved = catalog[s.station_complex_id]
    assert saved['nearest_three'] == [codes[i] for i in nearest3]
    np.testing.assert_allclose([g['distance_km'] for g in saved['all_candidates']], distances[ranked], rtol=1e-11, atol=1e-10)

    expected_values = np.full((2952,3), np.nan)
    encoded = []
    for v, field in enumerate(FIELDS):
        source = gauge_values[:,:,v]
        observed = np.isfinite(source)
        selected = np.zeros_like(observed)
        selected[:,nearest3] = observed[:,nearest3]
        n = selected.sum(axis=1)
        fallback = (n == 0) & observed.any(axis=1)
        missing = (n == 0) & ~observed.any(axis=1)
        chosen_fallback = ranked[np.argmax(observed[:,ranked], axis=1)]
        selected[np.flatnonzero(fallback), chosen_fallback[fallback]] = True

        zero_distance = distances <= 1e-9
        use_zero_only = (selected & zero_distance[None,:]).any(axis=1)
        selected[use_zero_only] &= zero_distance
        raw_weights = selected / np.where(zero_distance, 1, distances**2)[None,:]
        raw_weights[use_zero_only] = selected[use_zero_only].astype(float)
        denominator = raw_weights.sum(axis=1)
        weights = np.divide(raw_weights, denominator[:,None], out=np.zeros_like(raw_weights), where=denominator[:,None] > 0)
        predicted = np.sum(np.where(selected, source, 0)*weights, axis=1)
        predicted[missing] = np.nan
        expected_values[:,v] = predicted
        actual = df[field].to_numpy(float)
        np.testing.assert_allclose(actual, predicted, rtol=0, atol=5.0001e-7, equal_nan=True,
                                   err_msg=f'{s.station_complex_id} / {field}')
        finite = np.isfinite(predicted)
        if finite.any():
            max_error = max(max_error, float(np.max(np.abs(actual[finite]-predicted[finite]))))
            low = np.min(np.where(selected, source, np.inf),axis=1)
            high = np.max(np.where(selected, source, -np.inf),axis=1)
            assert ((actual[finite] >= low[finite]-5.0001e-7) & (actual[finite] <= high[finite]+5.0001e-7)).all()
        counts[field]['top3_full'] += int((n == 3).sum())
        counts[field]['top3_partial'] += int(((n > 0) & (n < 3)).sum())
        counts[field]['fallback'] += int(fallback.sum())
        counts[field]['missing'] += int(missing.sum())
        for g, code in enumerate(codes):
            source_counts[field][code] += int(selected[:,g].sum())

        entries = []
        # Only a handful of source combinations occur for one target station.
        cache = {}
        for h in range(2952):
            used = tuple(int(i) for i in ranked if selected[h,i])
            method = 'NONE' if missing[h] else ('NEAREST' if fallback[h] else 'IDW')
            key = (method, used)
            if key not in cache:
                cache[key] = f'{field}={method}' + (':'+'|'.join(codes[i] for i in used) if used else '')
            entries.append(cache[key])
        encoded.append(entries)
        for method, mask in [('top3_full',n == 3),('top3_partial',(n > 0)&(n < 3)),('fallback',fallback),('missing',missing)]:
            key = f'{field}/{method}'
            if key not in examples and mask.any():
                h = int(np.flatnonzero(mask)[0])
                examples[key] = {'station_complex_id':s.station_complex_id,'timestamp':expected_timestamps[h],
                    'value':None if missing[h] else float(actual[h]), 'nearest_three':[codes[i] for i in nearest3],
                    'used':[{'station':codes[i],'distance_km':float(distances[i]),'weight':float(weights[h,i]),
                             'source_value':float(source[h,i])} for i in ranked if selected[h,i]]}
    provenance = [';'.join(parts) for parts in zip(*encoded)]
    assert df.weather_station.tolist() == provenance, f'Source attribution mismatch: {s.station_complex_id}'
    available = np.isfinite(expected_values).all(axis=1)
    np.testing.assert_array_equal(df.weather_available, available)
    assert (df.rain_mm.dropna() >= 0).all() and (df.wind_ms.dropna() >= 0).all()
    rows_checked += len(df)
    complete += int(available.sum())
    station_count += 1
    if station_count % 100 == 0:
        print(f'Validated {station_count}/424 stations, {rows_checked} rows', flush=True)

assert station_count == 424 and rows_checked == 1251648
assert counts == meta['counts_by_variable']
assert source_counts == meta['contributor_counts_by_variable']
assert complete == meta['complete_rows']
with (ROOT / meta['output']).open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
report = {'status':'passed', 'rows_checked':rows_checked,'stations_checked':station_count,
    'unique_key':'station_complex_id,timestamp', 'duplicate_keys':0,
    'complete_rows':complete,'incomplete_rows':rows_checked-complete,
    'max_absolute_error_before_output_rounding':max_error,
    'checks':['exact schema and station metadata','full continuous hourly grid per station',
        'independent distances and fixed nearest-three ranking','all output values recomputed from original gauge CSVs',
        'variable-wise missingness, fallback and no-source behavior','actual contributors and complete-case flags',
        'convex combination bounds and nonnegative rain/wind'],
    'counts_by_variable':counts,'examples':examples,'output_sha256':digest}
(ROOT / 'subway_weather_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
meta['verification'] = 'passed: all 1,251,648 rows reconciled against independent distances and gauge observations'
meta['output_sha256'] = digest
meta_file.write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','rows_checked','complete_rows','incomplete_rows','max_absolute_error_before_output_rounding','counts_by_variable']},ensure_ascii=False))
