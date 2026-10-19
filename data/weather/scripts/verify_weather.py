"""Validate exported CSVs against their source observations and write a QA report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'station_inventory.json').read_text(encoding='utf-8'))
headers = manifest['columns']
reference_header = pd.read_csv(ROOT.parent / 'weather_hourly.csv', nrows=0).columns.tolist()
assert headers == reference_header, 'Output schema differs from the reference file'
expected = pd.date_range('2023-07-01', '2023-11-01', freq='h', inclusive='left')
raw_rain = pd.concat([pd.read_csv(p) for p in (ROOT / 'sources').glob('*_20230701_20231101_precip.csv')])
raw_metar = pd.concat([pd.read_csv(p) for p in (ROOT / 'sources').glob('*_20230701_20231101_routine_metar.csv')])
raw_rain['valid'] = pd.to_datetime(raw_rain['valid'])
raw_metar['valid'] = pd.to_datetime(raw_metar['valid'])
report = {'status': 'passed', 'expected_rows_per_station': len(expected), 'stations': []}
for info in manifest['stations']:
    if not info['exported']:
        assert not (ROOT / info['filename']).exists(), 'Pending station must not masquerade as downloaded'
        continue
    station = info['station']
    file = ROOT / info['filename']
    df = pd.read_csv(file, parse_dates=['timestamp'])
    assert df.columns.tolist() == headers
    assert len(df) == len(expected)
    assert pd.DatetimeIndex(df.timestamp).equals(expected)
    assert df.weather_station.eq(station).all()
    assert df.weather_available.dtype == bool
    values = df[['rain_mm', 'temp_c', 'wind_ms']]
    assert np.isfinite(values.to_numpy()[values.notna().to_numpy()]).all()
    assert (df.rain_mm.dropna() >= 0).all()
    assert (df.wind_ms.dropna() >= 0).all()
    assert df.weather_available.equals(values.notna().all(axis=1))
    assert int(df.weather_available.sum()) == info['available_hours']

    # Independent source reconciliation: mask only documented PNO hours,
    # otherwise every exported rainfall value must match the IEM source.
    rr = raw_rain.loc[raw_rain.station.eq(station)].drop_duplicates('valid').set_index('valid')
    source_rain = rr.precip_in.reindex(expected).astype(float) * 25.4
    oo = raw_metar.loc[raw_metar.station.eq(station)].sort_values('valid', kind='stable')
    flagged = oo.loc[oo.metar.fillna('').str.contains(r'\bPNO\b')].valid.dt.floor('h').unique()
    source_rain.loc[source_rain.index.isin(flagged)] = np.nan
    np.testing.assert_allclose(df.rain_mm, source_rain, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert df.loc[df.timestamp.isin(flagged), 'rain_mm'].isna().all()

    # Check each selected routine report, using alternative unit expressions.
    for output, source, conversion in [
        ('temp_c', 'tmpf', lambda x: (x - 32) / 1.8),
        ('wind_ms', 'sknt', lambda x: x * 0.5144444444444445),
    ]:
        numeric = pd.to_numeric(oo[source], errors='coerce')
        observations = pd.DataFrame({'hour': oo.valid.dt.floor('h'), 'value': numeric}).dropna(subset=['value'])
        picked = observations.drop_duplicates('hour', keep='last').set_index('hour').value.reindex(expected)
        np.testing.assert_allclose(df[output], conversion(picked), rtol=1e-12, atol=1e-12, equal_nan=True)

    info['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    info['validation'] = 'passed_source_reconciliation'
    if station == 'JRB':
        info['rain_quality_note'] = 'Large PNO outage and very low remaining reported rainfall: review this gauge before using it for rainfall IDW; do not treat its missing hours as dry.'
    events = []
    for date in ['2023-08-13', '2023-09-29', '2023-10-07']:
        sub = df.loc[df.timestamp.dt.strftime('%Y-%m-%d').eq(date)]
        events.append({'date': date, 'rain_observed_hours': int(sub.rain_mm.notna().sum()),
                       'observed_partial_sum_mm': None if sub.rain_mm.notna().sum() == 0 else float(sub.rain_mm.sum())})
    entry = {'station': station, 'rows': len(df), 'complete_hours': int(df.weather_available.sum()),
             'missing_by_variable': {k: int(df[k].isna().sum()) for k in values.columns},
             'pno_hours': len(flagged), 'event_day_coverage': events, 'sha256': info['sha256']}
    report['stations'].append(entry)
    print(json.dumps(entry, ensure_ascii=False))

assert len(report['stations']) == 6
assert len(manifest['stations']) == 10
(ROOT / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
(ROOT / 'station_inventory.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
