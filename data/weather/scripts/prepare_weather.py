"""Fetch documented IEM observations; prepare six-column hourly weather tables.

This script only writes raw sources and a JSON intermediate. export_weather.mjs
authors the requested CSV files with Artifact Tool. No experiment inputs are replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "sources"
TZ = "America/New_York"
HEADER = ["timestamp", "rain_mm", "temp_c", "wind_ms", "weather_available", "weather_station"]
STATIONS = [
    ("BRON", "布朗克斯", "NYS_MESONET"),
    ("NYC", "中央公园", "NY_ASOS"),
    ("LGA", "拉瓜迪亚机场", "NY_ASOS"),
    ("MANH", "曼哈顿", "NYS_MESONET"),
    ("JRB", "曼哈顿市中心直升机场", "NY_ASOS"),
    ("BKLN", "布鲁克林", "NYS_MESONET"),
    ("JFK", "肯尼迪机场", "NY_ASOS"),
    ("TEB", "泰特伯勒机场", "NJ_ASOS"),
    ("STAT", "斯塔滕岛", "NYS_MESONET"),
    ("EWR", "纽瓦克自由国际机场", "NJ_ASOS"),
]


def fetch(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                req = Request(url, headers={"User-Agent": "NYC-subway-weather-research/1.0"})
                with urlopen(req, timeout=60) as response:
                    payload = response.read()
                if not payload or payload.lstrip().lower().startswith(b"<!doctype"):
                    raise ValueError(f"Unexpected response for {url}")
                path.write_bytes(payload)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    payload = path.read_bytes()
    return {"file": str(path.relative_to(ROOT)), "url": url,
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def longest_gap(series: pd.Series) -> int:
    missing = series.isna()
    return int(missing.groupby((~missing).cumsum()).sum().max()) if len(series) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--end-exclusive", default="2023-11-01")
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz=TZ)
    end = pd.Timestamp(args.end_exclusive, tz=TZ)
    if end <= start:
        raise ValueError("End must be later than start")
    index = pd.date_range(start, end, freq="h", inclusive="left").tz_localize(None)
    if index.has_duplicates:
        raise ValueError("This six-column naive-time format cannot represent repeated DST hours")
    utc_start = start.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    utc_end = end.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}"
    sources = []
    sources.append(fetch("https://mesonet.agron.iastate.edu/geojson/network/NY_ASOS.geojson", RAW / "NY_ASOS_metadata.json"))
    sources.append(fetch("https://mesonet.agron.iastate.edu/geojson/network/NJ_ASOS.geojson", RAW / "NJ_ASOS_metadata.json"))
    nys_url = "https://nysgeohub.ny.gov/arcgis/rest/services/Weather/NYS_Mesonet/FeatureServer/0/query?" + urlencode(
        {"where": "1=1", "outFields": "*", "returnGeometry": "true", "outSR": 4326, "f": "geojson"})
    sources.append(fetch(nys_url, RAW / "NYS_MESONET_metadata.json"))

    meta = {}
    for network in ["NY_ASOS", "NJ_ASOS"]:
        for item in json.loads((RAW / f"{network}_metadata.json").read_text())['features']:
            lon, lat = item['geometry']['coordinates'][:2]
            meta[item['id']] = {"longitude": lon, "latitude": lat,
                                "source_name": item['properties'].get('sname'),
                                "archive_begin": item['properties'].get('archive_begin')}
    for item in json.loads((RAW / "NYS_MESONET_metadata.json").read_text())['features']:
        p = item['properties']
        meta[p['code']] = {"longitude": p['lon'], "latitude": p['lat'], "source_name": p['station_na']}

    rains, obs = [], []
    for network in ["NY_ASOS", "NJ_ASOS"]:
        ids = [s for s, _, n in STATIONS if n == network]
        common = dict(network=network, station=ids, sts=utc_start, ets=utc_end, tz=TZ)
        rain_path = RAW / f"{network}_{tag}_precip.csv"
        rain_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/hourlyprecip.py?" + urlencode(common | {"lalo": 1}, doseq=True)
        sources.append(fetch(rain_url, rain_path))
        rains.append(pd.read_csv(rain_path))
        print(f"Downloaded hourly precipitation: {network}", flush=True)
        # Keep regular hourly METARs; avoid storm-dependent sampling from specials.
        obs_path = RAW / f"{network}_{tag}_routine_metar.csv"
        obs_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urlencode(
            common | {"data": ["tmpf", "sknt", "p01i", "metar"], "report_type": 3,
                      "format": "onlycomma", "latlon": "yes", "elev": "yes",
                      "missing": "M", "trace": "0.0001", "direct": "no"}, doseq=True)
        time.sleep(1.1)
        sources.append(fetch(obs_url, obs_path))
        obs.append(pd.read_csv(obs_path))
        print(f"Downloaded regular METARs: {network}", flush=True)

    rain = pd.concat(rains, ignore_index=True)
    observation = pd.concat(obs, ignore_index=True)
    rain['valid'] = pd.to_datetime(rain.valid)
    observation['valid'] = pd.to_datetime(observation.valid)
    tables, quality, qc_events = {}, [], []
    for station, name, network in STATIONS:
        summary = {"station": station, "name": name, "network": network, **meta[station],
                   "expected_hours": len(index), "filename": f"weather_hourly_{station}.csv"}
        if network == "NYS_MESONET":
            summary.update(status="pending_official_data_request", exported=False,
                           request_url="https://www.nysmesonet.org/weather/requestdata")
            quality.append(summary)
            continue
        rr = rain[rain.station == station].copy()
        rr['rain_mm'] = pd.to_numeric(rr.precip_in, errors='coerce') * 25.4
        duplicates = rr.duplicated('valid', keep=False)
        if rr[duplicates].groupby('valid').rain_mm.nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting precipitation duplicates for {station}")
        duplicate_count = int(rr.duplicated('valid').sum())
        rr = rr.drop_duplicates('valid').set_index('valid')
        oo = observation[observation.station == station].copy()
        oo['temp_c'] = (pd.to_numeric(oo.tmpf, errors='coerce') - 32) * 5 / 9
        oo['wind_ms'] = pd.to_numeric(oo.sknt, errors='coerce') * (1852 / 3600)
        oo['hour'] = oo.valid.dt.floor('h')
        # PNO explicitly means that precipitation amount is unavailable. IEM's
        # parsed numeric field can still be zero, so do not interpret it as dry.
        # Conservatively mask a clock hour if any routine report flags PNO.
        oo['precipitation_unavailable'] = oo.metar.fillna('').str.contains(r'\bPNO\b', regex=True)
        pno = oo.groupby('hour').precipitation_unavailable.any().reindex(index, fill_value=False)
        # Corrected routine reports at an identical time can occur; use last valid
        # value per variable in the hour, recording the report counts in QA.
        routine_counts = oo.groupby('hour').size().reindex(index, fill_value=0)
        hourly = oo.sort_values('valid', kind='stable').groupby('hour')[['temp_c', 'wind_ms']].last().reindex(index)
        out = pd.DataFrame(index=index)
        out['rain_mm'] = rr.rain_mm.reindex(index)
        source_rain_missing = int(out.rain_mm.isna().sum())
        for hour in index[pno]:
            raw_value = out.at[hour, 'rain_mm']
            qc_events.append({"station": station, "timestamp": hour.strftime('%Y-%m-%d %H:%M:%S'),
                              "variable": "rain_mm", "reason": "METAR_PNO",
                              "source_value_mm": None if pd.isna(raw_value) else float(raw_value),
                              "action": "set_missing"})
        out.loc[pno, 'rain_mm'] = np.nan
        out[['temp_c', 'wind_ms']] = hourly
        if (out.rain_mm.dropna() < 0).any() or (out.wind_ms.dropna() < 0).any():
            raise ValueError(f"Negative rain or wind for {station}")
        if not np.isfinite(out.to_numpy()[pd.notna(out.to_numpy())]).all():
            raise ValueError(f"Nonfinite observations for {station}")
        out['weather_available'] = out[['rain_mm', 'temp_c', 'wind_ms']].notna().all(axis=1)
        out['weather_station'] = station
        out.index.name = 'timestamp'
        out = out.reset_index()
        out['timestamp'] = out.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S')
        # JSON null preserves missing observations for the CSV export.
        rows = json.loads(out[HEADER].to_json(orient='values', double_precision=15))
        tables[station] = {"headers": HEADER, "rows": rows}
        summary.update(status="downloaded_with_explicit_missing_values", exported=False,
                       available_hours=int(out.weather_available.sum()),
                       available_fraction=float(out.weather_available.mean()),
                       source_rain_rows=len(rr), identical_rain_duplicates_removed=duplicate_count,
                       source_rain_missing_hours=source_rain_missing,
                       precipitation_pno_hours=int(pno.sum()),
                       routine_metar_rows=len(oo), hours_multiple_routine_reports=int((routine_counts > 1).sum()),
                       missing_by_variable={k: int(out[k].isna().sum()) for k in ['rain_mm', 'temp_c', 'wind_ms']},
                       longest_missing_hours={k: longest_gap(out[k]) for k in ['rain_mm', 'temp_c', 'wind_ms']},
                       rain_sum_mm=float(out.rain_mm.sum()), rain_max_hour_mm=float(out.rain_mm.max()))
        quality.append(summary)
        print(f"Prepared {station}: {summary['available_hours']}/{len(index)} complete hours; missing {summary['missing_by_variable']}", flush=True)

    manifest = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "start_inclusive": str(start), "end_exclusive": str(end), "timezone": TZ,
                "columns": HEADER, "requested_aliases": {"EWER": "EWR"},
                "processing": {
                    "rain": "IEM processed hourly precipitation, inch * 25.4; retain provider hour label; trace 0.0001 inch = 0.00254 mm",
                    "temperature": "Last nonmissing routine METAR temperature within the clock hour; (degF - 32) * 5/9; not a time-weighted hourly mean",
                    "wind": "Last nonmissing routine METAR sustained speed within the clock hour; knots * 1852/3600; not gust or a time-weighted hourly mean",
                    "availability": "True only when rain_mm, temp_c, wind_ms are all observed; partial observations retained",
                    "missing": "CSV empty fields, no zero fill, interpolation, cross-station substitution or extrapolation",
                    "quality_control": "Any routine METAR PNO in a clock hour sets rain_mm missing; PNO means precipitation amount unavailable. Other unflagged observations are not independently certified.",
                    "time": "New York local clock hour; underlying METAR precipitation reset is not necessarily at HH:00",
                    "scope": "Gauge observations only; no subway IDW or changes to earlier experiment inputs"},
                "stations": quality, "sources": sources,
                "documentation": ["https://mesonet.agron.iastate.edu/cgi-bin/request/hourlyprecip.py?help",
                                  "https://mesonet.agron.iastate.edu/request/download.phtml",
                                  "https://www.weather.gov/media/zse/metar_decode_key.pdf",
                                  "https://www.nysmesonet.org/weather/requestdata"]}
    (ROOT / "prepared_tables.json").write_text(json.dumps(tables, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (ROOT / "station_inventory.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (ROOT / "quality_control_events.json").write_text(json.dumps(qc_events, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
