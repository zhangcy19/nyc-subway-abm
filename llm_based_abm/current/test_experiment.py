import unittest
import numpy as np
import pandas as pd
from api import validate
from experiment import examples,role_view
from model import profiles,rule_prob,fit_population,predict,MIN_SHARE,MAX_SHARE

class DesignTests(unittest.TestCase):
    def test_splits_and_events(self):
        d=pd.read_csv('res/dates.csv')
        for _,p in d.groupby('task'):
            self.assertLess(p[p.split=='train'].date.max(),p[p.split=='validation'].date.min())
            self.assertLess(p[p.split=='validation'].date.max(),p[p.split=='test'].date.min())
            self.assertEqual(p.date.nunique(),len(p))
        for task,part in d.groupby('task'):
            if task.startswith('Extreme'):
                self.assertEqual(part[(part.split=='test')&part.event_day].shape[0],1)
            if task.endswith('Summer') or task in ['Extreme_0729','Extreme_0813']:
                self.assertTrue((part.date<'2023-09-07').all())
            else:self.assertTrue((part.date>='2023-09-07').all())
        # The October storm is evaluated after the September record storm was observed.
        last=d[(d.task=='Extreme_1007')&(d.split=='train')]
        self.assertIn('2023-09-29',set(last.date))

    def test_history_excludes_target(self):
        p=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});train=p[(p.task=='Normal_Summer')&(p.split=='train')];r=train.iloc[-1]
        retrieved=examples(train,r,8)
        self.assertTrue((retrieved.date<r.date).all())
        view=role_view(profiles()[0],r,train,False)
        self.assertNotIn('group_history',view)
        self.assertNotIn('total_ridership',str(view))
        # Changing target and future labels cannot affect a history prompt.
        changed=train.copy();changed.loc[changed.date>=r.date,'students']=999999
        self.assertEqual(role_view(profiles()[0],r,train,True),role_view(profiles()[0],r,changed,True))

    def test_action_constraints(self):
        person=profiles()[0]
        with self.assertRaises(ValueError):validate({'probabilities':[.5,0,.5,0,0],'reason':'test'},'role',person)
        with self.assertRaises(ValueError):validate({'probabilities':[1,1,0,0,0],'reason':'test'},'role',person)
        p=pd.read_csv('res/dates.csv').iloc[0]
        for person in profiles():
            probs=rule_prob(person,p)
            self.assertAlmostEqual(sum(probs),1)
            self.assertTrue((probs>=0).all())

    def test_population_recovers_known_synthetic_counts(self):
        people=profiles();dates=['a','b','c','d'];policy={}
        x=np.array([[.1,.4,.8],[.8,.2,.1],[.3,.9,.4],[.6,.5,.2]])
        coefficients=np.array([10,20,30]);frame=[]
        for j,d in enumerate(dates):
            for person in people:policy[(d,person['agent_id'])]={'probabilities':[x[j,person['kind']],0,0,0,1-x[j,person['kind']]],'reason':'synthetic test'}
            frame.append(dict(station_complex_id='s',date=d,weekday=0,total_ridership=float(4*x[j]@coefficients),**{g:float(x[j]@coefficients) for g in ['students','fair_fare','seniors_disability','general']}))
        frame=pd.DataFrame(frame);weights=fit_population(frame,policy,people,0)
        rows,_=predict(frame,weights,policy,people)
        np.testing.assert_allclose([r['predicted'] for r in rows],frame.total_ridership,atol=1e-3)

    def test_strong_weekly_cycle_without_weather(self):
        people=profiles();policy={};rows=[]
        for j in range(28):
            day=j%7;date=f'd{j:02}';amount=100. if day<5 else 30. if day==5 else 10.
            for person in people:policy[(date,person['agent_id'])]={'probabilities':[.5,0,0,0,.5],'reason':'same choices every day'}
            rows.append(dict(station_complex_id='s',date=date,weekday=day,total_ridership=4*amount,**{g:amount for g in ['students','fair_fare','seniors_disability','general']}))
        frame=pd.DataFrame(rows);weights=fit_population(frame.iloc[:21],policy,people,0,0)
        held=frame.iloc[21:].copy();held['total_ridership']=-999 # target labels cannot enter prediction
        prediction,_=predict(held,weights,policy,people)
        np.testing.assert_allclose([r['predicted'] for r in prediction],frame.iloc[21:].total_ridership,rtol=1e-5)

    def test_dominant_persona_is_bounded(self):
        people=profiles();policy={};rows=[]
        for j in range(14):
            date=f'd{j:02}';probs=[.2+.04*j,.3,.7-.03*j]
            for person in people:
                v=probs[person['kind']];policy[(date,person['agent_id'])]={'probabilities':[v,0,0,0,1-v],'reason':'synthetic'}
            rows.append(dict(station_complex_id='s',date=date,weekday=0,total_ridership=400*probs[0],**{g:100*probs[0] for g in ['students','fair_fare','seniors_disability','general']}))
        weights=pd.DataFrame(fit_population(pd.DataFrame(rows),policy,people,0,0))
        shares=weights.effective_population/weights.groupby('group').effective_population.transform('sum')
        self.assertLessEqual(shares.max(),MAX_SHARE+1e-7)
        self.assertGreaterEqual(shares.min(),MIN_SHARE-1e-7)

if __name__=='__main__':unittest.main()
