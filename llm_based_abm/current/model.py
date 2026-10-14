"""Mesa traveler representatives. Population coefficients are fitted, not daily plans."""
import numpy as np
import mesa
from scipy.optimize import nnls
from scipy.special import softmax
from prepare import GROUPS

ACTIONS=['subway','bus','taxi','walk','stay_home']

def profiles():
    people=[]
    for group in GROUPS:
        for kind in range(3):
            identity={'students':'student','fair_fare':'income-qualified discounted-fare adult',
                      'seniors_disability':['older adult, independently mobile','adult with mobility limitations','older adult, independently mobile'][kind],
                      'general':'adult, general fare'}[group]
            people.append(dict(agent_id=f'{group}_{kind}',group=group,identity=identity,
                routine=['regular attendance','flexible errands or appointments','discretionary social or shopping'][kind],
                flexibility=['low','medium','high'][kind],weather_caution=['low','medium','high'][kind],
                budget_usd=[6,15,25][kind] if group!='fair_fare' else [6,6,15][kind],
                mobility_limited=(group=='seniors_disability' and kind==1),
                usual_mode='subway',access_walk_minutes=[5,10,15][kind],kind=kind))
    return people

def rule_prob(person,weather,rain_scale=1.):
    k=person['kind'];weekend=weather['weekday']>=5;holiday=bool(weather['holiday'])
    # Utilities include not making a trip: no trip is assumed to be pre-planned.
    drive=[2.8,1.9,1.4][k]
    if k==0 and (weekend or holiday):drive-=2.0
    if k==2 and weekend:drive+=.5
    if person['group']=='students' and not weather['school_term']:drive-=1.0 if k==0 else .2
    severity=np.log1p(weather['rain_mm']/10)+.5*np.log1p(weather['peak_mm']/5)
    caution=[.4,.8,1.2][k];limited=person['mobility_limited']
    costs=np.array([3.,3.,20.,0.]);minutes=np.array([25+person['access_walk_minutes'],45,25,65])
    exposures=np.array([.35+person['access_walk_minutes']/30,1.1,.2,2.])
    utilities=drive-costs/person['budget_usd']-minutes/60-rain_scale*severity*caution*exposures
    utilities[0]+=.8
    if limited:utilities[3]=-100;utilities[1]-=.4;utilities[0]-=.2
    utilities[costs>person['budget_usd']]=-100
    return softmax(np.r_[utilities,0.])

def fit_population(frame,policy,people,shrink):
    """Nonnegative effective representative counts; three coefficients per station/fare group."""
    weights=[]
    for station,part in frame.groupby('station_complex_id'):
        for group in GROUPS:
            ids=[p['agent_id'] for p in people if p['group']==group]
            valid=part[part[group].notna()]
            x=np.array([[policy[(d,a)]['probabilities'][0] for a in ids] for d in valid.date])
            y=valid[group].to_numpy(float);scale=max(1.,y.mean())
            prior=np.full(3,1/max(x.sum(axis=1).mean(),.01))
            matrix=np.vstack([x,np.sqrt(shrink)*np.eye(3)])
            target=np.r_[y/scale,np.sqrt(shrink)*prior]
            coeff,_=nnls(matrix,target)
            for a,v in zip(ids,coeff*scale):weights.append(dict(station_complex_id=station,group=group,agent_id=a,effective_population=float(v)))
    return weights

class Traveler(mesa.Agent):
    def __init__(self,model,person,weight):
        super().__init__(model);self.person=person;self.weight=weight
    def step(self):
        p=self.model.policy[(self.model.date,self.person['agent_id'])]
        self.model.entries+=self.weight*p['probabilities'][0]
        self.model.records.append(dict(station_complex_id=self.model.station,date=self.model.date,
            agent_id=self.person['agent_id'],group=self.person['group'],effective_population=self.weight,
            **{f'p_{a}':float(v) for a,v in zip(ACTIONS,p['probabilities'])},reason=p['reason'],request_hash=p.get('hash','')))

class DailyModel(mesa.Model):
    def __init__(self,station,date,weights,policy,people):
        super().__init__(seed=5011);self.station=station;self.date=date;self.policy=policy;self.entries=0.;self.records=[]
        lookup={p['agent_id']:p for p in people}
        for row in weights:
            if row['station_complex_id']==station:Traveler(self,lookup[row['agent_id']],row['effective_population'])
    def step(self):self.agents.do('step')

def predict(frame,weights,policy,people):
    predictions=[];decisions=[]
    for _,r in frame.iterrows():
        model=DailyModel(r.station_complex_id,r.date,weights,policy,people);model.step()
        predictions.append(dict(station_complex_id=r.station_complex_id,date=r.date,actual=r.total_ridership,predicted=model.entries))
        decisions.extend(model.records)
    return predictions,decisions
