"""Offline integration using frozen numerical decisions, never simulated LLM scores."""
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import run

class RunnerTest(unittest.TestCase):
    def test_interrupt_cancels_queued_requests(self):
        started=threading.Event();finished=[]
        def call(job):
            started.set();time.sleep(.1);finished.append(job)
        def interrupt(futures):
            started.wait(timeout=1)
            raise KeyboardInterrupt
        with patch.object(run,'as_completed',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):run.progress_map(call,list(range(20)),1,'interrupt test')
        self.assertEqual(finished,[0])

    def test_baseline_recalculation(self):
        ref=pd.read_csv('results/shared/predictions.csv',dtype={'station_complex_id':str})
        sid=ref.station_complex_id.iloc[0]
        with tempfile.TemporaryDirectory() as folder, patch.object(run.api,'request',side_effect=RuntimeError('Baselines must never call API')):
            run.execute(dict(station_ids=[sid]),folder,baseline=True)
            got=pd.read_csv(folder+'/predictions.csv',dtype={'station_complex_id':str})
            merged=got.merge(ref,on=['task','method','station_complex_id','date'],suffixes=('_new','_old'),validate='one_to_one')
            self.assertEqual(len(got),136)
            self.assertTrue(np.allclose(merged.predicted_new,merged.predicted_old,rtol=1e-6,atol=.01))

    def test_isolated_export_and_resume(self):
        policies=pd.read_csv('results/doubao_lite_reference/policies.csv')
        ref=pd.read_csv('results/doubao_lite_reference/predictions.csv',dtype={'station_complex_id':str})
        sid=ref.station_complex_id.iloc[0]
        def policy(d,w,phase,method,*args):
            z=policies[(policies.task==phase)&(policies.method==method)]
            return {(r.date,r.agent_id):dict(probabilities=[r[k] for k in ['p_subway','p_bus','p_taxi','p_walk','p_stay_home']],hash='',reason='reference numeric replay') for _,r in z.iterrows()}
        def direct(p,w,phase,*args):
            return ref[(ref.task==phase)&ref.method.eq('direct_few')&ref.station_complex_id.eq(sid)].drop(columns=['task','method']).to_dict('records')
        with tempfile.TemporaryDirectory() as folder:
            c=dict(station_ids=[sid],workers=1)
            with patch.object(run,'get_policy',side_effect=policy),patch.object(run,'direct',side_effect=direct),patch.object(run.api,'ledger',return_value=[]),patch.object(run.api,'request',side_effect=RuntimeError('No network in regression')):
                run.execute(c,folder,offline=True)
                got=pd.read_csv(folder+'/predictions.csv',dtype={'station_complex_id':str})
                merged=got.merge(ref,on=['task','method','station_complex_id','date'],suffixes=('_new','_old'),validate='one_to_one')
                self.assertEqual(len(got),102)
                self.assertTrue(np.allclose(merged.predicted_new,merged.predicted_old,rtol=1e-6,atol=.01))
                with patch.object(run,'get_policy',side_effect=RuntimeError('Already complete')),patch.object(run,'direct',side_effect=RuntimeError('Already complete')):
                    run.execute(c,folder,offline=True)
                self.assertEqual(len(pd.read_csv(folder+'/comparison/predictions.csv')),238)

if __name__=='__main__':unittest.main()
