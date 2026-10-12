"""Offline checks for weighted mass accounting and one-hour postponement."""
import unittest
from datetime import datetime, timedelta
import numpy as np
from model import GROUPS, CONTEXTS, SubwayModel, make_personas, validate_policy


def fixture(n_agents, vector=(.6, .2, .1, .1)):
    personas = make_personas(n_agents)
    policy = {p['id']: {c: list(vector) for c in CONTEXTS} for p in personas}
    rows = [{'station_complex_id': 's1', 'group': group, 'baseline_demand': mass,
             'weather_context': 'moderate', 'is_peak': False}
            for group, mass in zip(GROUPS, [10., 20., 30., 40.])]
    return SubwayModel(['s1'], personas, policy), rows


class WeightedCohortTests(unittest.TestCase):
    def test_agent_resolution_keeps_group_flexibility_mean_and_unit_weight(self):
        for count in [5, 10, 20]:
            personas = make_personas(count)
            self.assertEqual(len(personas), count)
            for group in GROUPS:
                cohort = [p for p in personas if p['group'] == group]
                self.assertAlmostEqual(sum(p['within_group_weight'] for p in cohort), 1.)
                self.assertAlmostEqual(sum(p['flexibility_position']*p['within_group_weight'] for p in cohort), .5)

    def test_same_policy_preserves_group_mass_across_agent_counts(self):
        for count in [5, 10, 20]:
            with self.subTest(count=count):
                model, rows = fixture(count)
                output, actions = model.advance(datetime(2023, 9, 22), rows)
                for row in rows:
                    key = ('s1', row['group'])
                    self.assertAlmostEqual(output[key], .6*row['baseline_demand'])
                    self.assertAlmostEqual(sum(actions[(key, a)] for a in ['continue','postpone','other_mode','cancel']), row['baseline_demand'])
                self.assertAlmostEqual(sum(output.values()), 60.)

    def test_postponed_mass_returns_exactly_one_hour_later(self):
        model, rows = fixture(10)
        start = datetime(2023, 9, 22)
        model.advance(start, rows)
        output, actions = model.advance(start+timedelta(hours=1), rows)
        self.assertAlmostEqual(sum(output.values()), 60.+20.*.6)
        # Action counts concern newly planned trips; returning pending mass is separate.
        self.assertAlmostEqual(sum(actions.values()), 100.)

    def test_time_gap_discards_stale_postponements(self):
        model, rows = fixture(10)
        start = datetime(2023, 9, 22)
        model.advance(start, rows)
        output, _ = model.advance(start+timedelta(hours=2), rows)
        self.assertAlmostEqual(sum(output.values()), 60.)

    def test_missing_group_discards_its_pending_mass_only(self):
        model, rows = fixture(10)
        start = datetime(2023, 9, 22)
        model.advance(start, rows)
        output, _ = model.advance(start+timedelta(hours=1), rows[1:])
        self.assertNotIn(('s1','students'), output)
        output, _ = model.advance(start+timedelta(hours=2), rows)
        self.assertAlmostEqual(output[('s1','students')], 6.)
        self.assertAlmostEqual(output[('s1','general')], 24.+8.*.6)

    def test_policy_validation_enforces_dry_counterfactual(self):
        personas = make_personas(5)
        payload = {'agents': [{'id': p['id'], 'policies': {c:[.6,.2,.1,.1] for c in CONTEXTS}} for p in personas]}
        policies = validate_policy(payload, personas)
        for p in personas:
            self.assertEqual(policies[p['id']]['dry_peak'], [1.,0.,0.,0.])
        payload['agents'][0]['policies']['moderate_peak'] = [np.nan,.2,.1,.1]
        with self.assertRaises(ValueError):
            validate_policy(payload, personas)


if __name__ == '__main__':
    unittest.main()
