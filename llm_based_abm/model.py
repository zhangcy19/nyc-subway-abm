"""Weighted cohort agents; one agent represents demand mass, not an identified person."""
import mesa
import numpy as np

GROUPS = ['students', 'fair_fare', 'seniors_disability', 'general']
CONTEXTS = [weather + '_' + peak for weather in ['dry', 'light', 'moderate', 'extreme']
            for peak in ['offpeak', 'peak']]


def make_personas(n_agents):
    counts = dict.fromkeys(GROUPS, 1)
    for i in range(n_agents - 4):
        counts[['general', 'general', 'students', 'general', 'fair_fare', 'general', 'seniors_disability'][i % 7]] += 1
    agents = []
    for group in GROUPS:
        for j in range(counts[group]):
            # Midpoint quadrature: the same hypothetical uniform flexibility distribution
            # is represented at every resolution; mean flexibility is always .5.
            flexibility = (j + .5) / counts[group]
            agents.append({'id': 'a' + str(len(agents)), 'group': group,
                           'profile': 'balanced', 'flexibility_position': flexibility,
                           'description': 'Hypothetical departure flexibility: 0=fixed appointment, 1=fully flexible. Risk preference balanced; no measured individual attributes.',
                           'within_group_weight': 1 / counts[group]})
    return agents


class CohortAgent(mesa.Agent):
    def __init__(self, model, station, persona, policy):
        super().__init__(model)
        self.station, self.persona, self.policy = station, persona, policy
        self.pending = 0.0

    def step(self):
        key = (self.station, self.persona['group'])
        row = self.model.rows.get(key)
        if row is None:
            self.pending = 0
            return
        if self.model.reset_pending:
            self.pending = 0
        context = row['weather_context'] + ('_peak' if row['is_peak'] else '_offpeak')
        probabilities = self.policy[context]
        mass = row['baseline_demand'] * self.persona['within_group_weight']
        # Pending departures return one hour later; the fixed return rate is a scenario assumption.
        current = mass * probabilities[0] + self.pending * self.model.return_rate
        self.model.outputs[key] = self.model.outputs.get(key, 0) + current
        self.pending = mass * probabilities[1]
        for action, probability in zip(['continue', 'postpone', 'other_mode', 'cancel'], probabilities):
            self.model.action_mass[(key, action)] = self.model.action_mass.get((key, action), 0) + mass * probability


class SubwayModel(mesa.Model):
    def __init__(self, stations, personas, policies, return_rate=0.6, seed=0):
        super().__init__(seed=seed)
        self.return_rate = return_rate
        self.previous_timestamp = None
        self.rows, self.outputs, self.action_mass = {}, {}, {}
        self.reset_pending = True
        for station in stations:
            for persona in personas:
                CohortAgent(self, station, persona, policies[persona['id']])

    def step(self):
        self.agents.do('step')

    def advance(self, timestamp, rows):
        self.reset_pending = self.previous_timestamp is None or (timestamp - self.previous_timestamp).total_seconds() != 3600
        self.rows = {(r['station_complex_id'], r['group']): r for r in rows}
        self.outputs, self.action_mass = {}, {}
        self.step()
        self.previous_timestamp = timestamp
        return self.outputs, self.action_mass


def validate_policy(payload, personas):
    agents = {item['id']: item for item in payload['agents']}
    if set(agents) != {p['id'] for p in personas}:
        raise ValueError('Policy must contain exactly the requested agents')
    policies = {}
    for persona in personas:
        policy = agents[persona['id']]['policies']
        if set(policy) != set(CONTEXTS):
            raise ValueError('Policy context keys differ from experiment definition')
        policies[persona['id']] = {}
        for context in CONTEXTS:
            values = np.asarray(policy[context], dtype=float)
            if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or abs(values.sum() - 1) > .02:
                raise ValueError('Expected four valid probabilities summing to one')
            policies[persona['id']][context] = (values / values.sum()).tolist()
            # The shared dry baseline is the counterfactual planned subway demand.
            if context.startswith('dry_'):
                policies[persona['id']][context] = [1., 0., 0., 0.]
    return policies
