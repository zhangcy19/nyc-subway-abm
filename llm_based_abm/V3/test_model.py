import copy
import unittest
import numpy as np
import pandas as pd
from model import population, TravelerModel, make_plan, validate_choice, rule_choice, GROUPS
from run import aggregate


def panel():
    time=pd.date_range('2023-09-29',periods=48,freq='h')
    x=pd.DataFrame({'timestamp':time,'date':time.normalize(),'hour':time.hour,'station_complex_id':'s',
        'total_ridership':100.,'planned_total':100.,'rain_mm':0.,'rain_6h_mm':0.,'temp_c':20.,'wind_ms':2.,'weather_context':'dry'})
    for g in GROUPS:x[g]=25.;x['planned_'+g]=25.
    return x


class TestTraveler(unittest.TestCase):
    def test_population_weights_and_constraints(self):
        for n in [24,48,96]:
            people=population(['s'],n)
            self.assertEqual(len(people),n)
            for g in GROUPS:self.assertAlmostEqual(sum(p['within_group_weight'] for p in people if p['group']==g),1)
            self.assertEqual(len({p['agent_id'] for p in people}),n)
            self.assertEqual(len({p['type'] for p in people}),6)

    def test_single_person_information(self):
        x=panel();model=TravelerModel(population(['s']))
        agent=next(iter(model.agents));plan=make_plan(x,agent.persona,pd.Timestamp('2023-09-29'))
        view=agent.perceive(plan)
        self.assertNotIn('weight',view['trip']);self.assertNotIn('station_complex_id',str(view))
        self.assertNotIn('total_ridership',str(view));self.assertNotIn('2023-09-29',str(view))
        self.assertEqual(len(view['past_own_experiences']),3)
        altered=x.copy();altered['total_ridership']=99999
        self.assertEqual(view,agent.perceive(make_plan(altered,agent.persona,pd.Timestamp('2023-09-29'))))

    def test_action_rejections(self):
        p=population(['s'])[0]
        with self.assertRaises(ValueError):validate_choice({'action':'cancel','delay_min':0,'reason':'skip'},p)
        with self.assertRaises(ValueError):validate_choice({'action':'subway','delay_min':60,'reason':'late'},p)
        with self.assertRaises(ValueError):validate_choice({'action':'subway','delay_min':0,'reason':''},p)

    def test_individual_history_and_midnight_delay(self):
        x=panel();model=TravelerModel(population(['s']))
        agent=next(a for a in model.agents if a.persona['max_delay_min']>=60)
        plan=make_plan(x,agent.persona,pd.Timestamp('2023-09-29'))
        plan['planned_timestamp']=pd.Timestamp('2023-09-29 23:00');plan['planned_hour']=23
        original=copy.deepcopy(agent.memory)
        r=agent.execute(plan,{'action':'delay_subway','delay_min':60,'reason':'wait'},'pro','test',24,0)
        self.assertEqual(r['actual_entry_timestamp'],pd.Timestamp('2023-09-30'))
        self.assertEqual(agent.memory[-1]['action'],'delay_subway');self.assertEqual(len(agent.memory),3)
        another=next(a for a in model.agents if a is not agent)
        self.assertEqual(another.memory,original)

    def test_weight_conservation_and_shared_execution(self):
        x=panel();model=TravelerModel(population(['s']))
        date=pd.Timestamp('2023-09-29')
        choices={a.persona['agent_id']:{'action':'subway','delay_min':0,'reason':'usual trip'} for a in model.agents}
        records=pd.DataFrame(model.advance(x,date,choices,'no_change','test',24,0))
        predicted=aggregate(x,records)
        self.assertAlmostEqual(records.weight.sum(),2400.)
        self.assertAlmostEqual(predicted.predicted.sum(),2400.)
        for g in GROUPS:self.assertAlmostEqual(predicted['pred_'+g].sum(),600.)
        self.assertTrue((predicted.predicted>=0).all())


if __name__=='__main__':unittest.main()
