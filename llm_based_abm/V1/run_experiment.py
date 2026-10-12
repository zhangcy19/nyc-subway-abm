"""Run from llm_based_abm. Pro API budget capped at 200,000 reported tokens."""
import argparse
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.optimize import least_squares
from model import GROUPS, CONTEXTS, make_personas, validate_policy, SubwayModel

MODEL = 'doubao-seed-2-1-pro-260915'
BUDGET = 200000
TEMPERATURE = .7
os.makedirs('results', exist_ok=True)

SYSTEM = '''You build a retrospective subway travel decision experiment. Return only valid JSON.
No dates, named stations, historical storm facts or real demographic counts are supplied.
The agent profiles are hypothetical weighted cohort representatives, not real people.
Forecast behavior conditional on the supplied weather and planned normal subway demand.
Do not invent station shutdowns or service suspension. An extreme-rain warning does not imply closure.
Students means observed student fare proxy; fair_fare means discounted-fare proxy;
seniors_disability is a jointly labeled reduced-fare proxy that cannot be separated.
A fixed appointment need not mean employment; avoid treating all seniors as immobile.
Dry baseline is already calibrated to the current season. Do not change baseline demand for school-term effects.
In dry contexts everyone follows their baseline planned subway departure, [1,0,0,0].'''

WEATHER = {
    'dry': {'rain_hour_mm': 0, 'rain_6h_mm': 0, 'description': 'no recent measurable rain'},
    'light': {'rain_hour_mm': .5, 'rain_6h_mm': 3, 'description': 'light rain or recently wet access'},
    'moderate': {'rain_hour_mm': 5, 'rain_6h_mm': 15, 'description': 'heavy hourly rain or accumulated moderate rainfall'},
    'extreme': {'rain_hour_mm': 12, 'rain_6h_mm': 60, 'description': 'extreme recent rainfall; access flooding risk, closure unknown'}}


def load_panel():
    panel = pd.read_csv('experiment_data/panel.csv', dtype={'station_complex_id': str}, parse_dates=['timestamp'])
    for source, total in [('baseline_demand', 'station_baseline_demand'), ('baseline_aug', 'station_baseline_aug')]:
        denominator = panel.groupby(['timestamp', 'station_complex_id'])[source].transform('sum')
        panel[source] = np.where(denominator > 0, panel[source] / denominator * panel[total], 0)
    return panel


def examples(panel):
    train = panel[panel.split == 'train'].copy()
    # Match the allocation convention; these ratios are not observed individual decisions.
    train['observed_ridership'] = train.observed_ridership.fillna(0)
    train['context'] = train.weather_context + np.where(train.is_peak, '_peak', '_offpeak')
    x = train.groupby(['context', 'group']).agg(observed=('observed_ridership', 'sum'), baseline=('baseline_demand', 'sum'), hours=('timestamp', 'size')).reset_index()
    x['observed_ratio'] = x.observed / x.baseline.replace(0, np.nan)
    return x[x.context.str.startswith(('light_', 'moderate_', 'extreme_'))][['context', 'group', 'observed_ratio', 'hours']].round({'observed_ratio': 3}).to_dict('records')


def cached_calls():
    if not os.path.exists('results/api_calls.jsonl'):
        return []
    with open('results/api_calls.jsonl') as f:
        return [json.loads(line) for line in f]


def request_policy(prompt, max_tokens):
    request = {'model': MODEL, 'system': SYSTEM, 'prompt': prompt, 'temperature': TEMPERATURE, 'max_output_tokens': max_tokens}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    calls = cached_calls()
    for call in calls:
        if call['request_hash'] == digest and call['status'] == 'completed':
            return call['parsed'], call
    used = sum(call['usage'].get('total_tokens', 0) for call in calls)
    reserve = len(SYSTEM + prompt) // 2 + max_tokens + 1000
    if used + reserve > BUDGET:
        raise RuntimeError(f'Budget guard: used {used}, reserved {reserve}, cap {BUDGET}')
    with open('../key.txt') as f:
        key = f.read().strip().strip('\"\'')
    client = OpenAI(base_url='https://ark.cn-beijing.volces.com/api/v3', api_key=key, timeout=180, max_retries=0)
    started = time.time()
    response = client.responses.create(model=MODEL,
        input=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
        temperature=TEMPERATURE, max_output_tokens=max_tokens,
        extra_body={'thinking': {'type': 'disabled'}})
    raw = response.output_text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    record = {'request_hash': digest, 'request': request, 'response_id': response.id,
              'status': response.status, 'raw_response': response.output_text,
              'usage': response.usage.model_dump(), 'elapsed_seconds': round(time.time()-started, 2)}
    try:
        record['parsed'] = json.loads(raw)
    except json.JSONDecodeError:
        record['parsed'] = None
    with open('results/api_calls.jsonl', 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print('API tokens', record['usage']['total_tokens'], 'cumulative', used + record['usage']['total_tokens'], flush=True)
    if response.status != 'completed' or record['parsed'] is None:
        raise ValueError('Incomplete or invalid JSON response; recorded for inspection, no synthetic replacement')
    return record['parsed'], record


def get_policy(shot, n_agents, seed, few_examples):
    personas = make_personas(n_agents)
    content = {'task': 'For each agent, provide one probability vector per weather/peak context. Vector order: continue subway now, postpone exactly one hour, use another mode, cancel this trip. Each sums to1. Other mode and cancel leave the modeled station; no measured destinations available. Peak means weekday7-9 or16-19; offpeak means all other times.',
               'weather_scenarios': WEATHER, 'required_contexts': CONTEXTS,
               'personas': personas, 'independent_sample_index': seed,
               'output_schema': {'agents': [{'id': 'a0', 'policies': {'dry_offpeak': [1,0,0,0], 'extreme_peak': [.5,.15,.2,.15]}, 'reason': 'brief behavioral rationale, <=20 words'}]}}
    if shot == 'few':
        content['training_examples'] = few_examples
        content['example_caveat'] = 'August observed/baseline aggregate ratios, not measured individual actions; observational, not causal. Same prior for all agent counts. Some ratios exceed1 from temporal redistribution or baseline error.'
    payload, call = request_policy(json.dumps(content), 900 + 450*n_agents)
    policies = validate_policy(payload, personas)
    return personas, policies, call['request_hash']


def get_direct(shot, seed, few_examples):
    content = {'task': 'Direct aggregate weather-conditioned prediction comparator. Give multipliers on shared season-calibrated station baseline for each context. Dry must equal1; wet may be0 to1.5. No individual agents or agent memory.',
               'weather_scenarios': WEATHER, 'required_contexts': CONTEXTS, 'independent_sample_index': seed,
               'output_schema': {'multipliers': {context: 1 for context in CONTEXTS}}}
    if shot == 'few':
        content['training_examples'] = few_examples
        content['example_caveat'] = 'Observed aggregate ratios are observational and not individual decisions.'
    payload, call = request_policy(json.dumps(content), 1200)
    multiplier = payload['multipliers']
    if set(multiplier) != set(CONTEXTS) or any(not np.isfinite(v) or v < 0 or v > 1.5 for v in multiplier.values()):
        raise ValueError('Invalid direct prediction multiplier')
    for context in CONTEXTS:
        if context.startswith('dry_'):
            multiplier[context] = 1.
    return multiplier, call['request_hash']


def fit_rule(panel):
    params = {}
    for group in GROUPS:
        v = panel[(panel.split == 'validation') & (panel.group == group) & panel.observation_available]
        features = np.column_stack([np.log1p(v.rain_6h_mm), np.log1p(v.rain_mm)])
        base, y = v.baseline_demand.to_numpy(), v.observed_ridership.to_numpy()
        fit = least_squares(lambda beta: (base*np.exp(-features@beta)-y)/np.sqrt(base+1), [.02,.02], bounds=(0,1))
        params[group] = fit.x.tolist()
    return params


def prediction_frame(test, values, model, shot='none', n_agents=0, seed=0, return_rate=0):
    columns = ['timestamp', 'station_complex_id', 'group', 'observed_ridership',
               'baseline_demand', 'baseline_aug', 'total_observed', 'rain_mm',
               'weather_context', 'is_peak', 'borough', 'latitude', 'longitude']
    result = test[columns].copy()
    result['predicted_ridership'] = values
    result['model'], result['shot'], result['n_agents'], result['seed'], result['return_rate'] = model, shot, n_agents, seed, return_rate
    return result


def simulate(test, personas, policies, return_rate, seed):
    model = SubwayModel(test.station_complex_id.unique(), personas, policies, return_rate=return_rate, seed=seed)
    rows, actions = [], []
    for timestamp, frame in test.groupby('timestamp', sort=True):
        output, action = model.advance(timestamp, frame.to_dict('records'))
        for row in frame.to_dict('records'):
            row['predicted_ridership'] = output[(row['station_complex_id'], row['group'])]
            rows.append(row)
        for ((station, group), name), mass in action.items():
            actions.append({'timestamp': timestamp, 'station_complex_id': station, 'group': group, 'action': name, 'mass': mass})
    return pd.DataFrame(rows), pd.DataFrame(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Use only already cached policy requests')
    args = parser.parse_args()
    global BUDGET
    if args.offline:
        BUDGET = 0
    panel = load_panel()
    few = examples(panel)
    with open('results/few_shot_examples.json', 'w') as f:
        json.dump(few, f, indent=2)
    rule = fit_rule(panel)
    with open('results/rule_parameters.json', 'w') as f:
        json.dump(rule, f, indent=2)
    test = panel[panel.split == 'test'].copy()
    context = test.weather_context + np.where(test.is_peak, '_peak', '_offpeak')
    output = [prediction_frame(test, test.baseline_demand, 'seasonal_baseline'),
              prediction_frame(test, test.baseline_aug, 'august_baseline')]
    rule_values = []
    for row in test.itertuples():
        factor = np.exp(-np.dot(rule[row.group], [np.log1p(row.rain_6h_mm), np.log1p(row.rain_mm)]))
        rule_values.append(row.baseline_demand * factor)
    output.append(prediction_frame(test, rule_values, 'rule_based'))
    policies_saved, actions_saved, sensitivity = [], [], []
    seeds = [0] if args.pilot else [0,1,2]
    shots = ['zero'] if args.pilot else ['zero','few']
    counts = [5] if args.pilot else [5,10,20]
    for shot in shots:
        for seed in seeds:
            direct, request_hash = get_direct(shot, seed, few)
            output.append(prediction_frame(test, test.baseline_demand*context.map(direct), 'llm_direct', shot, 0, seed))
            for n_agents in counts:
                print('Policy', shot, n_agents, seed, flush=True)
                personas, policies, request_hash = get_policy(shot, n_agents, seed, few)
                policies_saved.append({'shot': shot, 'n_agents': n_agents, 'seed': seed, 'personas': personas, 'policies': policies, 'request_hash': request_hash})
                predictions, actions = simulate(test, personas, policies, .6, seed)
                output.append(prediction_frame(test, predictions.predicted_ridership.to_numpy(), 'llm_abm', shot, n_agents, seed, .6))
                actions['shot'], actions['n_agents'], actions['seed'] = shot, n_agents, seed
                actions_saved.append(actions)
                if n_agents == 10:
                    for rate in [0.,1.]:
                        pred, _ = simulate(test, personas, policies, rate, seed)
                        sensitivity.append(prediction_frame(test, pred.predicted_ridership.to_numpy(), 'llm_abm', shot, n_agents, seed, rate))
                # Save each completed configuration before moving to another paid request.
                pd.concat(output, ignore_index=True).to_csv('results/predictions.csv.gz', index=False)
                with open('results/policies.json', 'w') as f:
                    json.dump(policies_saved, f, indent=2)
    pd.concat(actions_saved, ignore_index=True).to_csv('results/actions.csv.gz', index=False)
    if sensitivity:
        pd.concat(sensitivity, ignore_index=True).to_csv('results/return_rate_sensitivity.csv.gz', index=False)
    print('Completed configurations', len(output), 'Reported API tokens', sum(x['usage'].get('total_tokens',0) for x in cached_calls()), flush=True)


if __name__ == '__main__':
    main()
