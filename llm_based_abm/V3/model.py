"""A persistent synthetic traveler chooses one action for one planned trip."""
import copy
import mesa
import numpy as np
import pandas as pd

GROUPS = ['students', 'fair_fare', 'seniors_disability', 'general']
ACTIONS = ['subway', 'delay_subway', 'bus', 'taxi', 'walk', 'remote', 'cancel']


def population(stations, n=24):
    # Four fare strata; equally represented hypothetical types within each stratum.
    # Resolutions approximate the same 24-profile pool per group, not new NYC demographics.
    k = n//4
    positions = np.floor((np.arange(k)+.5)*24/k).astype(int)
    people = []
    purposes = {
        'students': ['class', 'class', 'library', 'social visit', 'class', 'study appointment'],
        'fair_fare': ['work', 'appointment', 'errand', 'social visit', 'work', 'appointment'],
        'seniors_disability': ['appointment', 'work', 'errand', 'social visit', 'appointment', 'errand'],
        'general': ['work', 'work', 'errand', 'social visit', 'work', 'appointment']}
    for s, station in enumerate(stations):
        for g, group in enumerate(GROUPS):
            for j in positions:
                kind = int(j//4)
                flexible = [0, 60, 120, 120, 60, 120][kind]
                support = group == 'seniors_disability' and kind in [2, 4, 5]
                remote = kind == 1 and group in ['general', 'fair_fare']
                identity = 'student' if group == 'students' else 'adult traveler'
                if group == 'seniors_disability':
                    identity = 'adult with mobility support needs' if support else 'older adult traveling independently'
                people.append({
                    'agent_id': str(station)+'-'+group+'-'+str(j), 'station_complex_id': str(station),
                    'group': group, 'identity': identity, 'type': ['fixed_attendance', 'adaptable_attendance', 'flexible_errand',
                        'optional_visit', 'fixed_shift', 'flexible_appointment'][kind],
                    'purpose': purposes[group][kind], 'attendance_required': kind in [0, 1, 4],
                    'max_delay_min': flexible, 'cancel_allowed': kind in [2, 3, 5],
                    'remote_allowed': remote, 'mobility_support': support,
                    'risk_aversion': round(.15+.7*((int(j)*7+s*3+g)%24+.5)/24, 3),
                    'subway_habit': round(.45+.5*((int(j)*5+g*3)%24+.5)/24, 3),
                    'travel_budget': [6, 25, 6, 25, 6, 15][kind],
                    'access_walk_min': int(5+(int(j)*3+s)%10),
                    'usual_hour_quantile': (int(j)+.5)/24,
                    'within_group_weight': 1/k,
                    'trait_source': 'hypothetical balanced design; not observed NYC demographics'})
    return people


def allowed_actions(person):
    actions = ['subway', 'bus']
    if person['max_delay_min'] >= 60:
        actions.append('delay_subway')
    if person['travel_budget'] >= 20:
        actions.append('taxi')
    if not person['mobility_support']:
        actions.append('walk')
    if person['remote_allowed']:
        actions.append('remote')
    if person['cancel_allowed']:
        actions.append('cancel')
    return actions


def validate_choice(choice, person):
    if choice.get('action') not in allowed_actions(person):
        raise ValueError('Action violates the supplied individual availability')
    delay = choice.get('delay_min')
    allowed = [60, 120] if person['max_delay_min'] >= 120 else [60]
    if choice['action'] == 'delay_subway':
        if not isinstance(delay, int) or isinstance(delay, bool) or delay not in allowed:
            raise ValueError('Delayed subway must use an allowed 60/120 minute delay')
    elif delay != 0:
        raise ValueError('Non-delay action requires delay_min=0')
    if not isinstance(choice.get('reason'), str) or not choice['reason'].strip():
        raise ValueError('A brief individual reason is required')
    return {key: choice[key] for key in ['action', 'delay_min', 'reason']}


def severity(environment):
    return min(3., environment['rain_mm']/5+environment['rain_6h_mm']/30)


def rule_choice(person, environment, memory, rain_scale=1., memory_on=True):
    # Transparent utility assumptions; only rain_scale is selected on feedback days.
    wet = severity(environment)
    bad_memory = 0.
    if memory_on and memory:
        bad_memory = np.mean([max(0, x['experienced_extra_min'])/60 for x in memory])
    risk = person['risk_aversion']
    scores = {}
    for action in allowed_actions(person):
        if action == 'remote':
            scores[action] = -.6
        elif action == 'cancel':
            scores[action] = -.5 if person['type'] == 'optional_visit' else -1.4
        else:
            mode = 'subway' if action == 'delay_subway' else action
            cost = {'subway': 3., 'bus': 3., 'taxi': 20., 'walk': 0.}[mode]
            minutes = {'subway': 25+person['access_walk_min'], 'bus': 45., 'taxi': 25., 'walk': 65.}[mode]
            exposure = {'subway': .4+person['access_walk_min']/20, 'bus': 1.1, 'taxi': .25, 'walk': 2.5}[mode]
            score = -cost/person['travel_budget']-minutes/90-rain_scale*wet*risk*exposure
            if mode == 'subway':
                score += person['subway_habit']-.4*bad_memory
            if action == 'delay_subway':
                # No known rain forecast: waiting has a cost and only a modest option benefit.
                score += .3*wet*risk-.35
            scores[action] = score
    action = max(scores, key=scores.get)
    return {'action': action, 'delay_min': 60 if action == 'delay_subway' else 0,
        'reason': 'Highest assumed utility given my attendance, budget, accessibility, rain and previous experience.'}


def make_plan(panel, person, date):
    rows = panel[(panel.station_complex_id == person['station_complex_id']) & (panel.date == date)].sort_values('hour')
    mass = rows['planned_'+person['group']].to_numpy()
    cumulative = np.cumsum(mass)
    if cumulative[-1] <= 0:
        return None
    index = int(np.searchsorted(cumulative, person['usual_hour_quantile']*cumulative[-1]))
    row = rows.iloc[index]
    return {'planned_timestamp': row.timestamp, 'planned_hour': int(row.hour),
        'weight': float(cumulative[-1]*person['within_group_weight']), 'date': date,
        'weather': {'rain_mm': round(float(row.rain_mm),2), 'rain_6h_mm': round(float(row.rain_6h_mm),2),
            'temp_c': round(float(row.temp_c),1), 'wind_ms': round(float(row.wind_ms),1),
            'weather_context': row.weather_context}}


def initial_memory():
    # Common synthetic seed experiences, completed within training period.
    return [{'date': date, 'action': 'subway', 'delay_min': 0, 'experienced_extra_min': extra,
        'outcome': 'completed; synthetic seed history, not an observed individual trajectory'}
        for date, extra in [('2023-09-05', 0), ('2023-09-06', 5), ('2023-09-07', 0)]]


class Traveler(mesa.Agent):
    def __init__(self, model, persona):
        super().__init__(model)
        self.persona = persona
        self.memory = initial_memory()
        self.last_choice = None

    def perceive(self, plan, memory_on=True):
        p = self.persona
        return {'person': {k:p[k] for k in ['identity','type','purpose','attendance_required','max_delay_min',
            'cancel_allowed','remote_allowed','mobility_support','risk_aversion','subway_habit',
            'travel_budget','access_walk_min']},
            'trip': {'departure_hour': plan['planned_hour'], 'weekday': int(plan['date'].dayofweek), 'purpose': p['purpose'],
                'usual_mode': 'subway', 'one_way_trip': True},
            'current_weather': plan['weather'], 'service': 'closure unknown; no verified disruption supplied',
            'available_actions': allowed_actions(p),
            'scenario_costs': {'subway':3,'bus':3,'taxi':20,'walk':0},
            'scenario_travel_min': {'subway':25+p['access_walk_min'],'bus':45,'taxi':25,'walk':65},
            'past_own_experiences': [{k:v for k,v in x.items() if k != 'date'} for x in copy.deepcopy(self.memory[-3:])] if memory_on else [],
            'output': {'action':'one available action','delay_min':'0 or permitted 60/120',
                'reason':'one short first-person explanation, at most 20 words'}}

    def execute(self, plan, choice, method, block, n_agents, sample, memory_on=True):
        choice = validate_choice(choice, self.persona)
        subway = choice['action'] in ['subway','delay_subway']
        arrival = plan['planned_timestamp']+pd.Timedelta(minutes=choice['delay_min']) if subway else pd.NaT
        extra = round(severity(plan['weather'])*8,1) if choice['action'] in ['subway','delay_subway','bus','taxi','walk'] else 0.
        outcome = {'date': str(plan['date'].date()), 'action':choice['action'], 'delay_min':choice['delay_min'],
            'experienced_extra_min': extra, 'outcome':'synthetic travel outcome; rain delay is an assumption'}
        record = {**{k:self.persona[k] for k in ['agent_id','station_complex_id','group','type','risk_aversion',
                    'subway_habit','travel_budget','mobility_support','max_delay_min']},
            'method':method,'block':block,'n_agents':n_agents,'sample':sample,
            'date':str(plan['date'].date()),'planned_timestamp':plan['planned_timestamp'],
            'actual_entry_timestamp':arrival,'weight':plan['weight'],**plan['weather'],**choice,
            'memory_length':len(self.memory[-3:]) if memory_on else 0,'experienced_extra_min':extra}
        self.memory.append(outcome)
        self.memory = self.memory[-3:]
        self.last_choice = choice
        return record


class TravelerModel(mesa.Model):
    def __init__(self, personas):
        super().__init__(seed=20261012)
        for persona in personas:
            Traveler(self,persona)

    def advance(self, panel, date, decisions, method, block, n_agents, sample, memory_on=True):
        records=[]
        for agent in self.agents:
            plan=make_plan(panel,agent.persona,date)
            if plan is not None:
                records.append(agent.execute(plan,decisions[agent.persona['agent_id']],method,block,n_agents,sample,memory_on))
        return records
