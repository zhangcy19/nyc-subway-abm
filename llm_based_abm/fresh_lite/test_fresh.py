import unittest,json,hashlib
import pandas as pd
from api import reused,MODEL,CAP,ledger,request
from timing import person_view,hourly_weather
from experiment import PEOPLE,SHRINK,CALENDAR_PRIOR,RAIN
from model import MIN_SHARE,MAX_SHARE
class FreshTests(unittest.TestCase):
    def test_isolated_and_fixed(self):
        self.assertEqual(reused(),{});self.assertEqual(CAP,5000000)
        self.assertEqual(MODEL,'doubao-seed-2-1-lite-260915')
        self.assertEqual((MIN_SHARE,MAX_SHARE),(.1,.6));self.assertEqual(len(PEOPLE),12)
        self.assertEqual(SHRINK,[0.,.1,1.,10.]);self.assertEqual(CALENDAR_PRIOR,[0.,1.,3.]);self.assertEqual(RAIN,[0.,.25,.5,1.,2.])
    def test_split(self):
        p=pd.read_csv('res/panel.csv',dtype={'station_complex_id':str});self.assertTrue(set(p.station_complex_id).isdisjoint({'360','37','228','269'}))
        for _,r in p.groupby('task'):
            self.assertLess(r[r.split=='train'].date.max(),r[r.split=='validation'].date.min());self.assertLess(r[r.split=='validation'].date.max(),r[r.split=='test'].date.min())
    def test_hidden_target(self):
        p=pd.read_csv('res/panel.csv');part=p[p.task=='Extreme_0929'];train=part[part.split=='train'];r=part[part.event_day].iloc[0];w=hourly_weather()
        for few in [False,True]:
            original=person_view(PEOPLE[0],r,train,w,few,True);r2=r.copy();r2['total_ridership']=-999;r2['students']=-999
            self.assertEqual(original,person_view(PEOPLE[0],r2,train,w,few,True));self.assertEqual('group_history' in original,few)
    def test_earliest_valid_attempt_is_reused(self):
        if not ledger():self.skipTest("Raw API evidence is local-only; public numeric replay cannot verify attempt ordering")
        groups={}
        for row in ledger():
            if row['status']!='valid':continue
            descriptor=row['request'].copy();descriptor.pop('attempt')
            groups.setdefault(json.dumps(descriptor,sort_keys=True),[]).append(row)
        before=len(ledger())
        for rows in groups.values():
            if len(rows)<2:continue
            first=rows[0];d=first['request']
            result=request(d['route'],d['view'],d['view'].get('person'),d['replicate'],offline=True)
            self.assertEqual(result['hash'],first['hash'])
        self.assertEqual(before,len(ledger()))
    def test_exact_core_code(self):
        with open('res/protocol.json') as source:frozen=json.load(source)['files']
        for f in ['model.py','experiment.py','timing.py']:
            with open(f,'rb') as source:self.assertEqual(hashlib.sha256(source.read()).hexdigest(),frozen[f])
if __name__=='__main__':unittest.main()
