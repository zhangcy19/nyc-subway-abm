"""Reproduce the latest daily-versus-hourly-weather comparison without API calls.
Run from llm_based_abm/current with Python 3.11.
"""
import subprocess
import sys

for args in [
    ['-m','unittest','test_experiment.py','test_timing.py'],
    ['timing.py','rules'],
    ['timing.py','pro','--offline'],
    ['timing_report.py'],
]:
    print('\nRunning: python '+' '.join(args),flush=True)
    subprocess.run([sys.executable]+args,check=True)
print('\nDone. Read TIMING_RESULTS.md and res/timing/. No API requests were made.')
