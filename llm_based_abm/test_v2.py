import unittest
import numpy as np
import pandas as pd
from v2 import GROUPS, DynamicModel, build_prompt, load, validate


class DailyForecastTests(unittest.TestCase):
    def row(self, hour=0):
        return {'station_complex_id':'1','alias':'s0','block':hour//6,
                **{'base_'+g:100. for g in GROUPS}}

    def policies(self, vector):
        return {'s0|'+g+'|'+str(b):{'m':1.,'p':vector}
                for g in GROUPS for b in range(4)}

    def test_mass_and_dry_adjustment(self):
        model=DynamicModel(['1'])
        p=self.policies([1,0,0,0])
        for cell in p.values(): cell['m']=1.2
        result=model.advance(pd.Timestamp('2023-09-08'),[self.row()],p)
        self.assertAlmostEqual(sum(result.values()),480.)

    def test_delay_across_midnight_and_gap(self):
        model=DynamicModel(['1'])
        date=pd.Timestamp('2023-09-08 23:00')
        model.advance(date,[self.row(23)],self.policies([0,1,0,0]))
        result=model.advance(date+pd.Timedelta(hours=1),[self.row()],self.policies([1,0,0,0]))
        self.assertAlmostEqual(sum(result.values()),640.)
        model.advance(date+pd.Timedelta(hours=2),[self.row(1)],self.policies([0,1,0,0]))
        result=model.advance(date+pd.Timedelta(hours=4),[self.row(3)],self.policies([1,0,0,0]))
        self.assertAlmostEqual(sum(result.values()),400.)

    def test_probability_and_id_validation(self):
        with self.assertRaises(ValueError):
            validate({'cells':[{'id':'a','m':1,'p':[1,1,0,0]}]},['a'])
        with self.assertRaises(ValueError):
            validate({'cells':[{'id':'a','m':1,'p':[1,0,0,0]}]*2},['a'])

    def test_forecast_day_labels_never_enter_prompt(self):
        panel=load()
        target=panel[(panel.date=='2023-09-15') & (panel.borough=='Queens')]
        prompt,cells=build_prompt(panel,target,'retrieval',0)
        altered=panel.copy()
        mask=altered.date>='2023-09-15'
        altered.loc[mask,['total_ridership']+GROUPS]=999999
        target2=altered[(altered.date=='2023-09-15') & (altered.borough=='Queens')]
        prompt2,cells2=build_prompt(altered,target2,'retrieval',0)
        self.assertEqual(prompt,prompt2)
        self.assertEqual(cells,cells2)

    def test_group_baselines_sum_to_total(self):
        panel=load()
        difference=panel[['base_'+g for g in GROUPS]].sum(axis=1)-panel.numeric
        self.assertLess(np.abs(difference).max(),1e-7)


if __name__=='__main__':
    unittest.main()
