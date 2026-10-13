"""Shared station-hour inputs; start is inclusive and end is exclusive."""
import glob
import os
import pandas as pd


def load_station_hourly(start='2023-07-01', end='2023-11-01'):
    """Return the same 22-column panel for any window within July–October 2023."""
    frames = []
    filenames = [os.path.join(os.path.dirname(__file__), 'station_hourly.csv.gz')]
    filenames += sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'extensions', '*.csv.gz')))
    for filename in filenames:
        frame = pd.read_csv(filename,
                            dtype={'station_complex_id': str}, parse_dates=['timestamp'])
        frames.append(frame[(frame.timestamp >= start) & (frame.timestamp < end)])
    return pd.concat(frames, ignore_index=True).sort_values(
        ['timestamp', 'station_complex_id']).reset_index(drop=True)
