"""Mesa traveler representatives. Population coefficients are fitted, not daily plans."""
import numpy as np
import mesa
from scipy.optimize import minimize, LinearConstraint
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

MIN_SHARE=0.10
MAX_SHARE=0.60

def bounded_counts(x,y,shrink):
    """Fit three nonnegative counts with 10–60% shares, not demographic estimates."""
    prior=np.full(3,1/max(x.sum(axis=1).mean(),.01))
    matrix=np.vstack([x,np.sqrt(shrink)*np.eye(3)])
    target=np.r_[y,np.sqrt(shrink)*prior]
    constraints=LinearConstraint(np.vstack([np.eye(3)-MIN_SHARE,MAX_SHARE-np.eye(3)]),0,np.inf)
    def objective(c):
        error=matrix@c-target
        return float(error@error),2*matrix.T@error
    fitted=minimize(objective,prior,jac=True,method='SLSQP',bounds=[(0,None)]*3,
                    constraints=constraints,options={'ftol':1e-10,'maxiter':200})
    if not fitted.success:raise RuntimeError('Representative-weight fit failed: '+fitted.message)
    return fitted.x

def fit_population(frame,policy,people,shrink,calendar_prior=1.):
    """Jointly fit fixed base counts and a seven-day activity schedule from TRAIN only.

    Activity multipliers change potential journey opportunities, not the LLM probabilities.
    Their training-frequency-weighted mean is one, separating scale from weekly shape.
    calendar_prior shrinks each weekday towards one pseudo-observation of the overall mean.
    """
    weights=[]
    for station,part in frame.groupby('station_complex_id'):
        for group in GROUPS:
            ids=[p['agent_id'] for p in people if p['group']==group]
            valid=part[part[group].notna()]
            x=np.array([[policy[(d,a)]['probabilities'][0] for a in ids] for d in valid.date])
            y=valid[group].to_numpy(float);scale=max(1.,y.mean());target=y/scale
            dow=valid.weekday.to_numpy(int);activity=np.ones(7)
            if not y.any():coeff=np.zeros(3)
            else:
                # Alternating constrained count fitting and smoothed weekday residual fitting.
                for _ in range(8):
                    coeff=bounded_counts(x*activity[dow,None],target,shrink)
                    base=x@coeff
                    for day in range(7):
                        take=dow==day
                        denominator=base[take].sum()+calendar_prior*base.mean()
                        activity[day]=(target[take].sum()+calendar_prior*target.mean())/denominator if take.any() and denominator>0 else 1.
                    activity/=activity[dow].mean()
                coeff=bounded_counts(x*activity[dow,None],target,shrink)
            for a,v in zip(ids,coeff*scale):
                weights.append(dict(station_complex_id=station,group=group,agent_id=a,effective_population=float(v),
                    **{f'weekday_{day}':float(activity[day]) for day in range(7)}))
    return weights

class Traveler(mesa.Agent):
    def __init__(self,model,person,weight,activity):
        super().__init__(model);self.person=person;self.base_weight=weight;self.activity=activity;self.weight=weight*activity
    def step(self):
        p=self.model.policy[(self.model.date,self.person['agent_id'])]
        self.model.entries+=self.weight*p['probabilities'][0]
        self.model.records.append(dict(station_complex_id=self.model.station,date=self.model.date,
            agent_id=self.person['agent_id'],group=self.person['group'],effective_population=self.weight,
            base_population=self.base_weight,weekday_multiplier=self.activity,
            **{f'p_{a}':float(v) for a,v in zip(ACTIONS,p['probabilities'])},reason=p['reason'],request_hash=p.get('hash','')))

class DailyModel(mesa.Model):
    def __init__(self,station,date,weekday,weights,policy,people):
        super().__init__(seed=5011);self.station=station;self.date=date;self.policy=policy;self.entries=0.;self.records=[]
        lookup={p['agent_id']:p for p in people}
        for row in weights:
            if row['station_complex_id']==station:Traveler(self,lookup[row['agent_id']],row['effective_population'],row[f'weekday_{weekday}'])
    def step(self):self.agents.do('step')

def predict(frame,weights,policy,people):
    predictions=[];decisions=[]
    for _,r in frame.iterrows():
        model=DailyModel(r.station_complex_id,r.date,int(r.weekday),weights,policy,people);model.step()
        predictions.append(dict(station_complex_id=r.station_complex_id,date=r.date,actual=r.total_ridership,predicted=model.entries))
        decisions.extend(model.records)
    return predictions,decisions
