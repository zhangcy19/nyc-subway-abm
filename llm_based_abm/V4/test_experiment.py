import unittest
import numpy as np
import pandas as pd
from experiment import apply_policy, direct_view, load_panel, spread_direct, weather_view
from model import population, validate_choice, PolicyModel


class ExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel=load_panel()
        cls.station=cls.panel.station_complex_id.iloc[0]
        cls.date=pd.Timestamp('2023-09-29')
        cls.rows=cls.panel[(cls.panel.station_complex_id==cls.station)&(cls.panel.date==cls.date)].sort_values('hour')
        cls.people=population([cls.station],8)

    def test_population_includes_flexible_and_fixed_types(self):
        for group in ['students','fair_fare','seniors_disability','general']:
            people=[p for p in self.people if p['group']==group]
            self.assertTrue(any(p['cancel_allowed'] for p in people))
            self.assertTrue(any(p['attendance_required'] for p in people))
        self.assertTrue(any(p['mobility_support'] for p in self.people))

    def test_target_counts_never_enter_llm_views(self):
        before=direct_view(self.panel,self.rows,True)
        changed=self.panel.copy()
        changed.loc[(changed.station_complex_id==self.station)&(changed.date==self.date),'total_ridership']=1e9
        after=direct_view(changed,self.rows,True)
        self.assertEqual(before,after)

    def test_mass_and_no_change_hourly_resolution(self):
        choice={'action':'subway','delay_min':0,'reason':'I take subway.'}
        pred,_,audit=apply_policy(self.rows,self.people,{'choices':[[choice]*6 for p in self.people],'hashes':['']*8},'no_change')
        np.testing.assert_allclose(pred,self.rows.planned_total,rtol=1e-12)
        self.assertAlmostEqual(audit['mass_error'],0,places=8)

    def test_delay_preserves_mass_including_next_day(self):
        choices=[[{'action':'delay_subway','delay_min':60,'reason':'I wait.'}]*6 for p in self.people]
        pred,_,audit=apply_policy(self.rows,self.people,{'choices':choices,'hashes':['']*8},'delay')
        self.assertGreater(audit['spillover_next_day'],0)
        self.assertAlmostEqual(audit['mass_error'],0,places=8)

    def test_constraint_validation(self):
        fixed=next(p for p in self.people if not p['cancel_allowed'])
        with self.assertRaises(ValueError):validate_choice({'action':'cancel','delay_min':0,'reason':'I cancel.'},fixed)

    def test_nontravel_has_no_experienced_delay(self):
        agent=list(PolicyModel([self.station],8).agents)[0]
        for action in ['cancel','remote']:
            agent.observe_policy([{'action':action}]*6,weather_view(self.rows))
            self.assertEqual(agent.memory[-1]['experienced_extra_min'],0)

    def test_direct_blocks_conserve_each_prediction(self):
        values=[100,200,300,400,500,600]
        pred=spread_direct(self.rows,values)
        np.testing.assert_allclose(pred.reshape(6,4).sum(axis=1),values)
        self.assertEqual(len(weather_view(self.rows)),6)


if __name__=='__main__':unittest.main()
