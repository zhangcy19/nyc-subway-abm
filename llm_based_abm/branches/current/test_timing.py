import unittest
import numpy as np
from timing import schedule,exposure,hourly_weather,person_view,PEOPLE
import pandas as pd
class TimingTests(unittest.TestCase):
    def test_shift_rain(self):
        r={'weekday':1};p=PEOPLE[0];night=np.zeros(24);night[3]=20;morning=np.zeros(24);morning[8]=20
        self.assertEqual(exposure(p,r,night),0)
        self.assertEqual(exposure(p,r,morning),3)
        self.assertAlmostEqual(exposure(p,r,np.ones(24)),1)
        self.assertEqual(exposure(p,r,np.zeros(24)),0)
    def test_schedules(self):
        for p in PEOPLE:
            for day in range(7):self.assertAlmostEqual(schedule(p,day).sum(),1)
    def test_weather_totals(self):
        weather=hourly_weather();daily=pd.read_csv('res/weather_daily.csv')
        for _,r in daily.iterrows():self.assertAlmostEqual(weather[r.date].sum(),r.rain_mm)
    def test_input_isolation(self):
        weather=hourly_weather();panel=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});part=panel[panel.task=='Extreme_0813'];row=part[part.event_day].iloc[0];train=part[part.split=='train']
        for few in [False,True]:
            view=person_view(PEOPLE[0],row,train,weather,few,True);changed=row.copy();changed['total_ridership']=-999;changed['students']=-999
            self.assertEqual(view,person_view(PEOPLE[0],changed,train,weather,few,True))
            self.assertEqual('group_history' in view,few)
        a=person_view(PEOPLE[0],row,train,weather,False,False);b=person_view(PEOPLE[0],row,train,weather,False,True)
        del b['target']['hourly_rain_mm_00_to_23'];self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
