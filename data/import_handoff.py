"""Run from data/: python import_handoff.py. Requires the local 5011 handoff folder."""
from contextlib import ExitStack
import gzip
import os
import shutil
import pandas as pd


def copy_csv(source, destination):
    with open(source) as incoming, open(destination, 'w') as outgoing:
        outgoing.write(incoming.read())


os.makedirs('supplementary', exist_ok=True)

# Keep the verified August–September input. New months use seven-day chunks
# so each upload stays small; load_data.py presents one panel to model callers.
os.makedirs('extensions', exist_ok=True)
for month in ['07', '10']:
    folder = 'jul' if month == '07' else 'oct'
    with open('../../5011/数据附件合集/model_input_2023_' + folder + '/station_hourly.csv', 'rb') as source:
        header = next(source)
        with ExitStack() as stack:
            outputs = {}
            for line in source:
                first_day = ((int(line[8:10]) - 1) // 7) * 7 + 1
                if first_day not in outputs:
                    destination = stack.enter_context(open('extensions/station_hourly_2023_' + month + '_' + str(first_day).zfill(2) + '.csv.gz', 'wb'))
                    compressed = stack.enter_context(gzip.GzipFile(filename='', fileobj=destination, mode='wb', mtime=0))
                    compressed.write(header)
                    outputs[first_day] = compressed
                outputs[first_day].write(line)

copy_csv('../../5011/数据附件合集/model_input_2023_jul_oct/stations.csv', 'stations.csv')

# Retain the original weather text precision for the published experiments.
# Even tiny rounding changes can perturb V2's fitted numerical coefficients.
with open('weather_hourly.csv') as source:
    historical_weather = {line.split(',')[0]: line for line in source
                          if '2023-08-01' <= line.split(',')[0] < '2023-10-01'}
with open('../../5011/数据附件合集/model_input_2023_jul_oct/weather_hourly.csv') as source:
    with open('weather_hourly.csv', 'w') as destination:
        for line in source:
            destination.write(historical_weather.get(line.split(',')[0], line))

copy_csv('../../5011/数据附件合集/network_static_gtfs_current/network_edges.csv',
                'supplementary/network_edges_current.csv')
copy_csv('../../5011/数据附件合集/supplementary_sources_2026-10-12/mta_subway_stations_complexes_current.csv',
                'supplementary/station_complexes_current.csv')

availability = pd.read_csv('../../5011/数据附件合集/supplementary_sources_2026-10-12/mta_accessible_platform_availability.csv')
availability = availability[(availability.month >= '2023-07-01') & (availability.month < '2023-11-01')]
availability.to_csv('supplementary/accessible_platform_availability_2023_jul_oct.csv', index=False)

for table in ['age_B01001', 'school_enrollment_B14001', 'disability_B18101']:
    copy_csv('../../5011/数据附件合集/supplementary_sources_2026-10-12/acs_2023_5yr_detailed/acs_2023_5yr_nyc_tract_' + table + '.csv',
                    'supplementary/acs_2023_nyc_tract_' + table + '.csv')

# Retain original operational records, including repeated trip-stop observations.
for table in ['trips', 'stop_times']:
    with open('../../5011/数据附件合集/service_proxy_2023-09-29/subwaydatanyc_2023-09-29_' + table + '.csv', 'rb') as source:
        with open('supplementary/service_2023_09_29_' + table + '.csv.gz', 'wb') as destination:
            with gzip.GzipFile(filename='', fileobj=destination, mode='wb', mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)

print('Imported July–October inputs and eight supplementary CSV files. See README.md for scope and joins.')
