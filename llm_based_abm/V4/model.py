"""Mesa representatives choose policies for mutually exclusive departure periods."""
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
        s = int(station) % 24  # Stable traits when station subsets are reordered.
        for g, group in enumerate(GROUPS):
            small={'students':[2,14],'fair_fare':[10,18],'seniors_disability':[2,22],'general':[6,14]}
            middle={'students':[2,6,14,22],'fair_fare':[6,10,18,22],'seniors_disability':[2,14,18,22],'general':[6,10,14,18]}
            sampled=small[group] if n==8 else middle[group] if n==16 else positions
            for j in sampled:
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



class Representative(mesa.Agent):
    def __init__(self, model, persona):
        super().__init__(model)
        self.persona = persona
        self.memory = [dict(action='subway', experienced_extra_min=0, source='synthetic initial commute')]

    def observe_policy(self, choices, weather):
        # One synthetic memory summary per evaluation episode, not six real trips.
        index = min(5, int(self.persona['usual_hour_quantile'] * 6))
        self.memory.append(dict(action=choices[index]['action'],
            experienced_extra_min=(0 if choices[index]['action'] in ('cancel', 'remote')
                else round(severity(weather[index]) * 8, 1)),
            source='synthetic policy episode; not observed personal history'))
        self.memory = self.memory[-3:]


class PolicyModel(mesa.Model):
    def __init__(self, stations, n=8):
        super().__init__(seed=20261013)
        for person in population(stations, n):
            Representative(self, person)
