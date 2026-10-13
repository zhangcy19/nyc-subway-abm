"""Run from llm_based_abm. Daily-origin forecasts, cached Pro calls and Mesa cohorts."""
import argparse
import hashlib
import gzip
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import mesa
import numpy as np
import pandas as pd
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
GROUPS = ['students', 'fair_fare', 'seniors_disability', 'general']

MODEL = 'doubao-seed-2-1-pro-260915'
CAP = 2100000
API_LOCK = threading.Lock()
RATE_LOCK = threading.Lock()
LAST_START = 0.
RESERVED = 0
SYSTEM = '''Return valid JSON only. You forecast daily subway demand conditional on supplied weather.
Stations and dates are anonymized. No unreported shutdown or event is known. Do not invent one.
Four groups are fare proxies, not measured individual demographics. Seniors/disability are combined.
The numerical baseline already incorporates calendar and recent observed travel, including school-term changes.
You may adjust planned activity above or below baseline, then choose continuing subway, postponing one hour,
switching mode, or cancelling. These actions are hypotheses, not inferred individual truth.
Use modest corrections when evidence is weak. Dry weather does not imply unchanged demand.
Vectors have four nonnegative probabilities summing to one, in the action order above.
Output exactly every supplied cell id. m is activity multiplier (0.5 to 1.5); p is four probabilities.
Do not emit absolute counts. No actual ridership for the forecast day is available.'''


def split_dates(x):
    return np.select([x < '2023-09-08', x < '2023-09-12', x < '2023-09-15'],
                     ['train', 'feedback', 'validation'], default='test')


def prepare():
    os.makedirs('results', exist_ok=True)
    source = pd.read_csv('../../data/station_hourly.csv.gz', dtype={'station_complex_id': str}, parse_dates=['timestamp'])
    weather = pd.read_csv('../../data/weather_hourly.csv', parse_dates=['timestamp']).sort_values('timestamp')
    # Keep the published V2 experiment on its original August–September window.
    source = source[(source.timestamp >= '2023-08-01') & (source.timestamp < '2023-10-01')]
    weather = weather[(weather.timestamp >= '2023-08-01') & (weather.timestamp < '2023-10-01')].copy()
    weather['rain_6h_mm'] = weather.rain_mm.rolling(6, min_periods=1).sum()
    weather['rain_24h_mm'] = weather.rain_mm.rolling(24, min_periods=1).sum()
    weather['weather_context'] = np.select([weather.rain_6h_mm >= 30,
        (weather.rain_mm >= 5) | (weather.rain_6h_mm >= 10),
        (weather.rain_mm > .1) | (weather.rain_6h_mm > .1)], ['extreme', 'moderate', 'light'], default='dry')
    previous = pd.read_csv('../V1/experiment_data/stations.csv', dtype={'station_complex_id': str})
    train_source = source[source.timestamp < '2023-09-08']
    coverage = train_source.groupby(['borough', 'station_complex_id']).agg(n=('timestamp', 'size'), volume=('total_ridership', 'sum')).reset_index()
    chosen = []
    for borough in ['Bronx', 'Brooklyn', 'Manhattan', 'Queens']:
        candidates = coverage[(coverage.borough == borough) & (coverage.n >= .98*38*24)
            & ~coverage.station_complex_id.isin(previous.station_complex_id)].sort_values(['volume', 'station_complex_id'], ascending=[False, True])
        chosen.append(candidates.iloc[[0, len(candidates)//3, 2*len(candidates)//3]])
    chosen = pd.concat(chosen)
    stations = source[['station_complex_id', 'station_name', 'borough', 'latitude', 'longitude']].drop_duplicates('station_complex_id').merge(chosen)
    stations['alias'] = ['s'+str(i) for i in range(len(stations))]
    stations.to_csv('results/stations.csv', index=False)
    x = source[source.station_complex_id.isin(stations.station_complex_id)].copy()
    channels = {'students': ['metrocard_students'], 'fair_fare': ['metrocard_fair_fare'],
        'seniors_disability': ['metrocard_seniors_disability', 'omny_seniors_disability'],
        'general': ['metrocard_full_fare', 'metrocard_unlimited_30day', 'metrocard_unlimited_7day', 'metrocard_other', 'omny_full_fare', 'omny_other']}
    for group, cols in channels.items():
        x[group] = x[cols].sum(axis=1, min_count=1)
    x = x.drop(columns=['rain_mm', 'temp_c', 'wind_ms', 'weather_available', 'weather_station']).merge(weather, on='timestamp')
    x = x.merge(stations[['station_complex_id', 'alias']], on='station_complex_id')
    x['date'] = x.timestamp.dt.normalize()
    x['weekday'], x['hour'] = x.timestamp.dt.dayofweek, x.timestamp.dt.hour
    x['block'] = x.hour // 6
    x['split'] = split_dates(x.timestamp)
    keys = ['station_complex_id', 'weekday', 'hour']
    train = x[x.split == 'train'].copy()
    # September observations provide school-term shape; August supplies a shrinkage prior.
    train['weight'] = np.where(train.timestamp >= '2023-09-01', 3., 1.)
    for col in ['total_ridership'] + GROUPS:
        train['weighted'] = train[col].fillna(0)*train.weight
        profile = (train.groupby(keys).weighted.sum()/train.groupby(keys).weight.sum()).rename('profile_'+col)
        x = x.merge(profile, on=keys)
    indexed = x.set_index(['station_complex_id', 'timestamp']).total_ridership
    for lag in [24, 168]:
        lookup = pd.MultiIndex.from_arrays([x.station_complex_id, x.timestamp-pd.Timedelta(hours=lag)])
        x['lag'+str(lag)] = indexed.reindex(lookup).to_numpy()
    # Every lag comes from a completed day at the daily forecast origin.
    history_ratio = x.total_ridership / x.profile_total_ridership.clip(lower=1)
    daily = x.assign(ratio=history_ratio).groupby(['station_complex_id', 'date']).ratio.mean()
    lookup = pd.MultiIndex.from_arrays([x.station_complex_id, x.date-pd.Timedelta(days=1)])
    x['recent_ratio'] = daily.reindex(lookup).to_numpy()
    numeric = ['log_profile', 'log_lag24', 'log_lag168', 'recent_ratio', 'log_rain', 'log_rain6', 'temp_c', 'wind_ms']
    x['log_profile'] = np.log1p(x.profile_total_ridership)
    x['log_lag24'] = np.log1p(x.lag24.fillna(x.profile_total_ridership))
    x['log_lag168'] = np.log1p(x.lag168.fillna(x.profile_total_ridership))
    x['recent_ratio'] = x.recent_ratio.fillna(1).clip(.2, 3)
    x['log_rain'], x['log_rain6'] = np.log1p(x.rain_mm), np.log1p(x.rain_6h_mm)
    transform = ColumnTransformer([('number', StandardScaler(), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore'), ['station_complex_id', 'weekday', 'hour'])])
    train_mask = (x.split == 'train') & (x.timestamp >= '2023-08-08')
    feedback = x.split == 'feedback'
    models, scores = [], []
    for alpha in [10., 100., 1000.]:
        fit = make_pipeline(transform, Ridge(alpha=alpha))
        fit.fit(x[train_mask], np.log1p(x.loc[train_mask, 'total_ridership']))
        pred = np.maximum(np.expm1(fit.predict(x)), 0)
        score = np.abs(pred[feedback]-x.loc[feedback, 'total_ridership']).sum()/x.loc[feedback, 'total_ridership'].sum()
        scores.append({'alpha': alpha, 'feedback_wape': score})
        models.append(pred)
    best = int(np.argmin([s['feedback_wape'] for s in scores]))
    x['numeric'] = models[best]
    x['weekly'] = x.lag168.fillna(x.profile_total_ridership)
    raw_group = x[['profile_'+g for g in GROUPS]].to_numpy()
    group_sum = raw_group.sum(axis=1)
    for i, group in enumerate(GROUPS):
        x['base_'+group] = x.numeric * np.divide(raw_group[:, i], group_sum, out=np.zeros(len(x)), where=group_sum>0)
    x = x.sort_values(['timestamp', 'station_complex_id'])
    x.to_csv('results/panel.csv.gz', index=False)
    with open('results/numeric_fit.json', 'w') as f:
        json.dump({'candidates': scores, 'selected_alpha': scores[best]['alpha'],
                   'selected_on': 'feedback only', 'forecast_origin': 'daily midnight'}, f, indent=2)
    print('Selected 12 new stations, numeric fit', scores, flush=True)


def load():
    return pd.read_csv('results/panel.csv.gz', dtype={'station_complex_id': str}, parse_dates=['timestamp', 'date'])


def ledger():
    calls=[]
    for name in ['results/api_calls.jsonl.gz','results/api_calls.jsonl']:
        if os.path.exists(name):
            opener=gzip.open if name.endswith('.gz') else open
            with opener(name,'rt') as f:
                calls.extend(json.loads(line) for line in f)
    return calls


def request(prompt, offline=False):
    global RESERVED, LAST_START
    content = {'model': MODEL, 'system': SYSTEM, 'prompt': prompt, 'temperature': .3, 'max_output_tokens': 6500}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    with API_LOCK:
        calls = ledger()
    for call in calls:
        if call['hash'] == digest and call['status'] == 'completed' and call['parsed'] is not None:
            return call['parsed'], digest
    if offline:
        raise RuntimeError('Missing cached request '+digest)
    reserve = len(SYSTEM+prompt)//2 + 7500
    with API_LOCK:
        used = sum(c.get('budget_charge_tokens',c['usage'].get('total_tokens', 0)) for c in ledger())
        if used + RESERVED + reserve > CAP:
            raise RuntimeError('Budget guard: '+str(used)+' tokens used')
        RESERVED += reserve
    with open('../key.txt') as f:
        key = f.read().strip().strip('\"\'')
    client = OpenAI(base_url='https://ark.cn-beijing.volces.com/api/v3', api_key=key, timeout=180, max_retries=0)
    with RATE_LOCK:
        pause=max(0,12-(time.monotonic()-LAST_START))
        time.sleep(pause)
        LAST_START=time.monotonic()
    started = time.time()
    try:
        response = client.responses.create(model=MODEL, input=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
            temperature=.3, max_output_tokens=6500, extra_body={'thinking': {'type': 'disabled'}})
    except Exception as error:
        failure={'hash':digest,'request':content,'status':'request_error','parsed':None,'usage':{},
            'usage_known':False,'budget_charge_tokens':reserve,'error_type':type(error).__name__,
            'error_code':getattr(error,'code',None),'http_status':getattr(error,'status_code',None)}
        with API_LOCK:
            with open('results/api_calls.jsonl','a') as f:
                f.write(json.dumps(failure)+'\n')
            RESERVED -= reserve
        raise
    raw = response.output_text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    record = {'hash': digest, 'request': content, 'response_id': response.id, 'status': response.status,
        'parsed': parsed, 'raw': response.output_text, 'usage': response.usage.model_dump(), 'elapsed_seconds': time.time()-started}
    with API_LOCK:
        with open('results/api_calls.jsonl', 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        RESERVED -= reserve
        completed = ledger()
        print('Pro call',len(completed),'tokens',record['usage']['total_tokens'],'total',sum(c['usage'].get('total_tokens',0) for c in completed), flush=True)
    if parsed is None or response.status != 'completed':
        raise ValueError('API response incomplete; saved, no fabricated replacement')
    return parsed, digest


def analogs(panel, target):
    # Retrieval bank is frozen at Sep 7 even during later test days.
    train = panel[panel.split == 'train']
    features = ['rain_mm', 'rain_6h_mm', 'temp_c', 'wind_ms']
    summaries = train.groupby('date')[features].mean()
    summaries['weekday'] = summaries.index.dayofweek
    target_weather = target[features].mean()
    scale = np.array([3, 10, 8, 4])
    distance = ((summaries[features].to_numpy()-target_weather.to_numpy())/scale)**2
    distance = distance.sum(axis=1) + (summaries.weekday.to_numpy()!=target.weekday.iloc[0])*2
    chosen = summaries.iloc[np.argsort(distance)[:3]].index
    weights = 1/(distance[np.argsort(distance)[:3]]+.2)
    return chosen, weights/weights.sum()


def build_prompt(panel, target, mode, seed):
    origin = target.date.iloc[0]
    dates, weights = analogs(panel, target)
    weather = target.drop_duplicates('hour')[['hour', 'rain_mm', 'rain_6h_mm', 'temp_c', 'wind_ms']].round(2).to_numpy().tolist()
    cells, stations = [], []
    for station, rows in target.groupby('station_complex_id'):
        alias = rows.alias.iloc[0]
        yesterday = panel[(panel.station_complex_id == station) & (panel.date == origin-pd.Timedelta(days=1))]
        history = panel[(panel.station_complex_id == station) & (panel.date < origin)]
        entry = {'station': alias, 'weekday': int(rows.weekday.iloc[0]),
            'previous_7_daily_totals': history.groupby('date').total_ridership.sum().tail(7).round().tolist(),
            'previous_day_block_totals': yesterday.groupby('block').total_ridership.sum().round().to_dict(),
            'normal_group_block_mass': {}, 'training_missing_group_fraction': {}, 'similar_days': []}
        for group in GROUPS:
            entry['normal_group_block_mass'][group] = rows.groupby('block')['base_'+group].sum().reindex(range(4),fill_value=0).round(1).tolist()
            entry['training_missing_group_fraction'][group] = round(float(history[history.split == 'train'][group].isna().mean()), 3)
            for block in range(4):
                cells.append(alias+'|'+group+'|'+str(block))
        if mode == 'retrieval':
            for date, weight in zip(dates, weights):
                h = history[history.date == date]
                entry['similar_days'].append({'weekday': int(date.dayofweek), 'weight': round(float(weight),3),
                    'weather': h[ ['rain_mm', 'rain_6h_mm', 'temp_c']].mean().round(2).to_dict(),
                    'group_block_observed_over_numeric': {g: (h.groupby('block')[g].sum(min_count=1)/h.groupby('block')['base_'+g].sum().clip(lower=1)).reindex(range(4)).round(3).fillna(1).tolist() for g in GROUPS}})
        stations.append(entry)
    prompt = {'task': 'Predict this anonymous future day in four six-hour blocks (0=00-05,1=06-11,2=12-17,3=18-23). m controls planned demand relative to supplied numerical baseline; p controls actions. Moderate rain may increase subway use instead of walking; no fixed monotonicity requirement. Distinguish baseline bias from weather disruption.',
        'sample_index': seed, 'weather_columns':['hour','rain_mm','rain_6h_mm','temp_c','wind_ms'],
        'weather_by_hour': weather, 'block_array_order':[0,1,2,3], 'stations': stations,
        'required_cells': cells, 'output_schema': {'cells': [{'id': cells[0], 'm': 1.0, 'p': [.97,.01,.01,.01]}],
        'station_reasons': {s['station']: 'one brief explanation, maximum 25 words' for s in stations}},
        'caveats': 'Numeric baseline is a forecast, not a guaranteed no-rain counterfactual. Ratios can exceed1. Missing group cells are not observed zeros. Similar-day ratios are observational, not true individual decisions.'}
    return json.dumps(prompt, allow_nan=False, separators=(',',':')), cells


def validate(payload, required):
    records = payload['cells']
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(required):
        raise ValueError('Cell ids differ from requested station/group/block grid')
    policies = {}
    for cell in records:
        m, p = float(cell['m']), np.asarray(cell['p'], dtype=float)
        if not np.isfinite(m) or not .5 <= m <= 1.5 or p.shape != (4,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or abs(p.sum()-1)>.02:
            raise ValueError('Invalid activity multiplier or action vector')
        policies[cell['id']] = {'m': m, 'p': (p/p.sum()).tolist()}
    return policies


def validated_response(prompt, required, offline=False, critic=False):
    original=prompt
    for attempt in range(3):
        try:
            payload,digest=request(prompt,offline)
        except (RateLimitError,APIConnectionError,APITimeoutError):
            if offline or attempt==2:
                raise
            time.sleep(15*(attempt+1))
            continue
        try:
            policies=validate(payload,required)
            if critic and any(not .75<=v['m']<=1.25 or v['p']!=[1.,0.,0.,0.] for v in policies.values()):
                raise ValueError('Critic correction must be bounded .75 to1.25 and p exactly [1,0,0,0]')
            return payload,policies,digest
        except ValueError as error:
            if attempt==2:
                raise
            # Ask the API to correct its real response; never substitute a synthetic policy.
            prompt=original+'\nFORMAT REPAIR: '+str(error)+'\nPrevious real response:\n'+json.dumps(payload)+'\nRegenerate the full required cell grid, respecting every numerical bound. Repair attempt '+str(attempt+1)


class DynamicCohort(mesa.Agent):
    def __init__(self, model, station, group):
        super().__init__(model)
        self.station, self.group, self.pending = station, group, 0.

    def step(self):
        row = self.model.rows.get(self.station)
        if row is None or self.model.reset:
            self.pending = 0.
        if row is None:
            return
        policy = self.model.policies[row['alias']+'|'+self.group+'|'+str(row['block'])]
        mass = row['base_'+self.group]*policy['m']
        current = mass*policy['p'][0]+self.pending*self.model.return_rate
        self.model.output[(self.station, self.group)] = current
        self.pending = mass*policy['p'][1]


class DynamicModel(mesa.Model):
    def __init__(self, stations, return_rate=.6):
        super().__init__(seed=0)
        self.return_rate=return_rate
        self.previous, self.rows, self.output, self.policies, self.reset = None, {}, {}, {}, True
        for station in stations:
            for group in GROUPS:
                DynamicCohort(self, station, group)

    def advance(self, timestamp, rows, policies):
        self.reset = self.previous is None or timestamp-self.previous != pd.Timedelta(hours=1)
        self.rows = {r['station_complex_id']: r for r in rows}
        self.policies, self.output = policies, {}
        self.agents.do('step')
        self.previous = timestamp
        return self.output


def run_policies(panel, stage, offline=False):
    targets = panel[panel.split.isin(['feedback', 'validation'] if stage == 'development' else ['test'])]
    if stage == 'pilot':
        targets = panel[panel.date == '2023-09-08']
    samples = [0,1] if stage == 'test' else [0]
    with open('results/policies.json') as f:
        saved = json.load(f)
    def one_batch(item):
        key, batch, mode, seed = item
        prompt, cells = build_prompt(panel, batch, mode, seed)
        payload, policies, digest = validated_response(prompt, cells, offline)
        return key, {'policies': policies, 'hash': digest, 'reasons': payload.get('station_reasons', {})}
    work=[]
    for mode in ['zero', 'retrieval']:
        for seed in samples:
            sample_targets=targets
            if stage=='test' and seed==1:
                # Warm up the second policy sample for a complete day before test.
                sample_targets=panel[panel.date>='2023-09-14']
            for date, day in sample_targets.groupby('date'):
                for borough, batch in day.groupby('borough'):
                    key = mode+'|'+str(seed)+'|'+str(date.date())+'|'+borough
                    if key in saved:
                        continue
                    work.append((key,batch,mode,seed))
    errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one_batch,item) for item in work]
        for future in as_completed(futures):
            try:
                key,item=future.result()
            except Exception as error:
                errors.append(type(error).__name__)
                continue
            saved[key]=item
            with open('results/policies.json','w') as f:
                json.dump(saved,f)
    if errors:
        raise RuntimeError(str(len(errors))+' batches need resume; completed requests are cached. Error types: '+str(sorted(set(errors))))
    return saved


def frames(panel, model, predicted, group_values=None, seed=0):
    result = panel[['timestamp','date','station_complex_id','station_name','borough','latitude','longitude',
        'split','weather_context','total_ridership']+GROUPS].copy()
    result['model'], result['seed'], result['predicted'] = model, seed, predicted
    if group_values is None:
        for group in GROUPS:
            share = panel['base_'+group]/panel.numeric.clip(lower=1e-8)
            result['pred_'+group] = result.predicted.to_numpy()*share.to_numpy()
    else:
        for i, group in enumerate(GROUPS):
            result['pred_'+group] = group_values[:,i]
    return result


def simulate(panel, saved, mode, seed, correction=None, return_rate=.6):
    evaluation = panel[panel.split != 'train'].copy()
    model = DynamicModel(evaluation.station_complex_id.unique(),return_rate)
    records = []
    for timestamp, batch in evaluation.groupby('timestamp', sort=True):
        policies = {}
        available = True
        for borough in batch.borough.unique():
            key = mode+'|'+str(seed)+'|'+str(timestamp.date())+'|'+borough
            if key not in saved:
                available = False
                break
            policies.update(saved[key]['policies'])
        if not available:
            continue
        if correction is not None and timestamp >= pd.Timestamp('2023-09-12'):
            policies = {key: {'m': value['m']*correction[key], 'p': value['p']}
                        for key,value in policies.items()}
        output = model.advance(timestamp, batch.to_dict('records'), policies)
        for r in batch.to_dict('records'):
            records.append({'timestamp': timestamp, 'station_complex_id': r['station_complex_id'],
                **{'pred_'+g: output[(r['station_complex_id'],g)] for g in GROUPS}})
    pred = evaluation.merge(pd.DataFrame(records), on=['timestamp','station_complex_id'])
    values = pred[['pred_'+g for g in GROUPS]].to_numpy()
    return frames(pred, 'llm_'+mode, values.sum(axis=1), values, seed)


def fit_feedback(panel, offline=False):
    with open('results/policies.json') as f:
        saved=json.load(f)
    predicted=simulate(panel,saved,'retrieval',0)
    feedback=predicted[predicted.split=='feedback'].copy()
    feedback=feedback.merge(panel[['timestamp','station_complex_id','alias','block','numeric']],on=['timestamp','station_complex_id'])
    correction={}
    for borough,batch in feedback.groupby('borough'):
        summaries=[]
        required=[]
        for (station,block), rows in batch.groupby(['alias','block']):
            for group in GROUPS:
                observed=rows[rows[group].notna()]
                cell=station+'|'+group+'|'+str(block)
                required.append(cell)
                actual=float(observed[group].sum())
                forecast=float(observed['pred_'+group].sum())
                summaries.append({'id':cell,'observed_hours':len(observed),
                    'actual_group_mass':round(actual,1),'predicted_group_mass':round(forecast,1),
                    'actual_over_predicted':round(actual/max(forecast,1),3),
                    'absolute_error':round(float(np.abs(observed[group]-observed['pred_'+group]).sum()),1)})
        prompt={'task':'You are a calibration critic. These are errors on four COMPLETED feedback days. Propose a shrinkage activity correction for each cell. Use m in [0.75,1.25], p exactly [1,0,0,0]. Sparse counts and missing hours require more shrinkage toward1. The correction will multiply a future daily agent activity policy, leaving action probabilities unchanged. Do not claim true personality or action identification. There are no future actual outcomes.',
            'feedback':summaries,'required_cells':required,'output_schema':{'cells':[{'id':required[0],'m':1,'p':[1,0,0,0]}],
                'station_reasons':{'s0':'brief calibration reasoning'}}}
        payload,calibrated,digest=validated_response(json.dumps(prompt),required,offline,critic=True)
        for key,value in calibrated.items():
            if not .75<=value['m']<=1.25 or value['p']!=[1.,0.,0.,0.]:
                raise ValueError('Critic must return bounded activity correction only')
            correction[key]=value['m']
    with open('results/feedback_correction.json','w') as f:
        json.dump(correction,f,indent=2)
    return correction


def retrieval_numeric(panel):
    result = panel[panel.split!='train'].copy()
    factors = np.ones(len(result))
    for date, day in result.groupby('date'):
        dates, weights = analogs(panel, day)
        for station, rows in day.groupby('station_complex_id'):
            for block in range(4):
                ratios = []
                for analogous in dates:
                    h = panel[(panel.date == analogous) & (panel.station_complex_id == station) & (panel.block == block)]
                    ratios.append(h.total_ridership.sum()/max(h.numeric.sum(),1))
                factor = np.clip(np.dot(ratios,weights),.5,1.5)
                mask = (result.date == date) & (result.station_complex_id == station) & (result.block==block)
                factors[np.flatnonzero(mask)] = factor
    return frames(result, 'retrieval_numeric', result.numeric.to_numpy()*factors)


def wape(frame):
    return np.abs(frame.predicted-frame.total_ridership).sum()/frame.total_ridership.sum()


def evaluate(panel, final=False):
    with open('results/policies.json') as f:
        saved = json.load(f)
    evaluated = panel[panel.split!='train']
    baselines = [frames(evaluated,'numeric',evaluated.numeric),frames(evaluated,'profile',evaluated.profile_total_ridership),
        frames(evaluated,'weekly',evaluated.weekly),retrieval_numeric(panel)]
    raw = [simulate(panel,saved,mode,seed) for mode in ['zero','retrieval'] for seed in ([0,1] if final else [0])]
    with open('results/feedback_correction.json') as f:
        correction=json.load(f)
    feedback_raw=[]
    for seed in ([0,1] if final else [0]):
        adjusted=simulate(panel,saved,'retrieval',seed,correction)
        adjusted['model']='llm_feedback'
        feedback_raw.append(adjusted)
    no_delay=[]
    for seed in ([0,1] if final else [0]):
        ablation=simulate(panel,saved,'retrieval',seed,return_rate=0.)
        ablation['model']='llm_retrieval_no_return'
        no_delay.append(ablation)
    output = baselines+raw+feedback_raw+no_delay
    calibration = {}
    reference = raw[1 if not final else 2]
    feedback = reference[reference.split=='feedback']
    # A small station-total correction is identifiable; action decomposition is not.
    prior = feedback.groupby('station_complex_id').agg(y=('total_ridership','sum'),p=('predicted','sum'))
    scales = ((prior.y+prior.p*.5)/(prior.p*1.5)).clip(.75,1.25).to_dict()
    validation = reference[reference.split=='validation'].copy()
    vbase = evaluated.set_index(['timestamp','station_complex_id']).numeric
    bases = vbase.reindex(pd.MultiIndex.from_frame(validation[['timestamp','station_complex_id']])).to_numpy()
    adjusted = validation.predicted.to_numpy()*validation.station_complex_id.map(scales).to_numpy()
    grid = []
    for weight in [0.,.1,.25,.5,.75,1.]:
        score = np.abs(bases*(1-weight)+adjusted*weight-validation.total_ridership.to_numpy()).sum()/validation.total_ridership.sum()
        grid.append({'weight': weight, 'validation_wape': score})
    weight = min(grid,key=lambda r:r['validation_wape'])['weight']
    calibration = {'station_scales': scales, 'blend_candidates': grid, 'selected_weight': weight,
        'scale_fit_period': 'Sep8-11', 'weight_selection_period': 'Sep12-14',
        'test_used_for_selection': False}
    for seed in ([0,1] if final else [0]):
        llm = raw[1 if not final else 2+seed]
        idx = pd.MultiIndex.from_frame(llm[['timestamp','station_complex_id']])
        numeric = vbase.reindex(idx).to_numpy()
        adjusted = llm.predicted.to_numpy()*llm.station_complex_id.map(scales).to_numpy()
        groups = llm[['pred_'+g for g in GROUPS]].to_numpy()*llm.station_complex_id.map(scales).to_numpy()[:,None]
        base_groups = evaluated.set_index(['timestamp','station_complex_id'])[['base_'+g for g in GROUPS]].reindex(idx).to_numpy()
        output.append(frames(llm,'llm_calibrated',adjusted,groups,seed))
        output.append(frames(llm,'llm_hybrid',(1-weight)*numeric+weight*adjusted,(1-weight)*base_groups+weight*groups,seed))
    feedback_validation=feedback_raw[0].query("split=='validation'")
    fbidx=pd.MultiIndex.from_frame(feedback_validation[['timestamp','station_complex_id']])
    fbbase=vbase.reindex(fbidx).to_numpy()
    fbgrid=[]
    for fbweight in [0.,.1,.25,.5,.75,1.]:
        score=np.abs((1-fbweight)*fbbase+fbweight*feedback_validation.predicted.to_numpy()-feedback_validation.total_ridership.to_numpy()).sum()/feedback_validation.total_ridership.sum()
        fbgrid.append({'weight':fbweight,'validation_wape':score})
    fbweight=min(fbgrid,key=lambda r:r['validation_wape'])['weight']
    calibration['feedback_blend_candidates']=fbgrid
    calibration['feedback_selected_weight']=fbweight
    for llm in feedback_raw:
        idx=pd.MultiIndex.from_frame(llm[['timestamp','station_complex_id']])
        numeric=vbase.reindex(idx).to_numpy()
        base_groups=evaluated.set_index(['timestamp','station_complex_id'])[['base_'+g for g in GROUPS]].reindex(idx).to_numpy()
        output.append(frames(llm,'llm_feedback_hybrid',(1-fbweight)*numeric+fbweight*llm.predicted.to_numpy(),
            (1-fbweight)*base_groups+fbweight*llm[['pred_'+g for g in GROUPS]].to_numpy(),int(llm.seed.iloc[0])))
    with open('results/calibration.json', 'w') as f:
        json.dump(calibration,f,indent=2)
    if not final:
        with open('results/selection.json','w') as f:
            json.dump(calibration,f,indent=2)
    else:
        with open('results/selection.json') as f:
            frozen=json.load(f)
        if frozen != calibration:
            raise ValueError('Validation selections changed after test; inspect before continuing')
    combined = pd.concat(output,ignore_index=True)
    if final:
        combined=combined[combined.split=='test']
    # All method/repeat comparisons use their common timestamp/station support.
    counts = combined.groupby(['timestamp','station_complex_id']).model.size()
    support = counts[counts == len(output)].index
    common = pd.MultiIndex.from_frame(combined[['timestamp','station_complex_id']]).isin(support)
    combined = combined[common]
    combined.to_csv('results/predictions.csv.gz',index=False)
    metrics = []
    for (model,seed,split), frame in combined.groupby(['model','seed','split']):
        for context in ['all','ordinary','dry','light','moderate','extreme']:
            f = frame if context=='all' else frame[frame.weather_context!='extreme'] if context=='ordinary' else frame[frame.weather_context==context]
            if len(f):
                metrics.append({'model':model,'seed':seed,'split':split,'context':context,'n':len(f),
                    'wape':wape(f),'rmse':np.sqrt(((f.predicted-f.total_ridership)**2).mean()),
                    'bias':(f.predicted.sum()-f.total_ridership.sum())/f.total_ridership.sum()})
    pd.DataFrame(metrics).to_csv('results/metrics.csv',index=False)
    if not final:
        pd.DataFrame(metrics).to_csv('results/development_metrics.csv',index=False)
    print('Validation blend selection',grid,'selected',weight,flush=True)
    print(pd.DataFrame(metrics).query("context=='all'")[['model','seed','split','wape']].to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','pilot','development','test','evaluate'])
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare()
        if not os.path.exists('results/policies.json'):
            with open('results/policies.json','w') as f: json.dump({},f)
        return
    panel=load()
    if args.stage in ['pilot','development','test']:
        run_policies(panel,args.stage,args.offline)
    if args.stage=='development':
        fit_feedback(panel,args.offline)
    if args.stage in ['development','test','evaluate']:
        evaluate(panel,final=args.stage in ['test','evaluate'])


if __name__=='__main__':
    main()
