"""Configurable OpenAI-compatible API; run-local cache and token ceiling."""
import gzip
import hashlib
import json
import os
import threading
import time
import numpy as np
from openai import OpenAI

MODEL='doubao-seed-2-1-lite-260915'
CAP=5000000
BASE_URL=""
API_MODE="chat"
API_KEY=""
EXTRA_BODY={}
TEMPERATURE=.3
MAX_OUTPUT=250
BATCH_OUTPUT=700
LOGDIR="res"
LOCK=threading.Lock();RATE=threading.Lock();RESERVED=0;LAST=0.;LEDGER=None; REUSED=None; HITS=set(); NETWORK_FAILED=False
ROLE_SYSTEM='''You are the hypothetical NYC traveler described. Consider a random day with the supplied calendar and weather.
No trip is pre-booked or assumed. Give your probability of each mutually exclusive DAILY outcome for ONE potential journey:
subway, bus, taxi, walk, stay_home. stay_home includes remote activity and no need to travel.
Consider whether you need to travel at all, then the mode. Probabilities must sum to 1, each between 0 and 1.
Taxi is unavailable if budget is below its scenario cost; walking is unavailable if mobility_limited is true.
Known weather is retrospective, not a forecast. Do not invent station closures or personal travel observations.
Historical group entry indices, when supplied, are aggregate observations, NOT your past choices or action probabilities.
Return JSON {"probabilities":[five numbers in the stated order],"reason":"one brief first-person decision summary, at most 20 words"}.
Do not provide a chain of thought, crowd counts, population weights, or a daily baseline.'''
DIRECT_SYSTEM='''Forecast daily subway entries for one anonymous NYC station from the supplied observed training examples.
Predict the target directly, with no supplied baseline. Examples are the only observed demand data available to you.
Weather is assumed known retrospectively. The calendar and weather are available for the target; target demand is hidden.
No verified service closures are supplied. Do not invent closures or additional history.
Return JSON {"entries": nonnegative number, "reason":"brief explanation, at most 20 words"}.'''

def ledger():
    global LEDGER
    if LEDGER is None:
        LEDGER=[]
        for f in [LOGDIR+'/api_calls.jsonl.gz',LOGDIR+'/api_calls.jsonl']:
            if os.path.exists(f):
                with (gzip.open(f,'rt') if f.endswith('.gz') else open(f)) as source:LEDGER.extend(json.loads(line) for line in source)
    return LEDGER

def reused():
    return {}  # This experiment never reads another experiment's replies.

def charge(r):return r.get('charged_tokens',r.get('usage',{}).get('total_tokens',0))

def validate(payload,route,person=None):
    if not isinstance(payload,dict) or not isinstance(payload.get('reason'),str) or not payload['reason'].strip():raise ValueError('Need brief reason')
    if route=='batch':
        records=payload.get('predictions',[])
        if not records or any(not isinstance(r,dict) or not isinstance(r.get('date'),str) for r in records):raise ValueError('Need dated predictions')
        for r in records:validate(dict(entries=r.get('entries'),reason='batch'), 'direct')
    elif route=='direct':
        v=payload['entries']
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:raise ValueError('Need finite nonnegative entries')
    else:
        p=payload['probabilities']
        if len(p)!=5 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 or v>1 for v in p):raise ValueError('Need five probabilities in [0,1]')
        if abs(sum(p)-1)>.015:raise ValueError('Probabilities must sum to 1')
        if person['budget_usd']<20 and p[2]>0:raise ValueError('Taxi unavailable under budget')
        if person['mobility_limited'] and p[3]>0:raise ValueError('Walking unavailable')
        payload={**payload,'probabilities':(np.array(p)/sum(p)).tolist()}
    return payload

def request(route,view,person=None,replicate=0,offline=False):
    global RESERVED,LAST,NETWORK_FAILED
    if NETWORK_FAILED:raise RuntimeError("Network circuit open; resume after connectivity recovers")
    system=BATCH_SYSTEM if route=='batch' else DIRECT_SYSTEM if route=='direct' else ROLE_SYSTEM
    # Reuse the earliest valid response across ALL format/transport attempts.
    hashes=set()
    for a in range(3):
        d=dict(model=MODEL,route=route,system=system,view=view,replicate=replicate,attempt=a,endpoint=BASE_URL,api_mode=API_MODE,extra_body=EXTRA_BODY,temperature=TEMPERATURE,max_output_tokens=(BATCH_OUTPUT if route=="batch" else MAX_OUTPUT))
        hashes.add(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    with LOCK:
        first=next((r for r in ledger() if r['hash'] in hashes and r['status']=='valid'),None)
    if first:return {**first['parsed'],'hash':first['hash']}
    for attempt in range(3):
        descriptor=dict(model=MODEL,route=route,system=system,view=view,replicate=replicate,attempt=attempt,endpoint=BASE_URL,api_mode=API_MODE,extra_body=EXTRA_BODY,temperature=TEMPERATURE,max_output_tokens=(BATCH_OUTPUT if route=="batch" else MAX_OUTPUT))
        digest=hashlib.sha256(json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with LOCK:
            rows=ledger();matching=[r for r in rows if r['hash']==digest]
            cached=next((r for r in matching if r['status']=='valid'),matching[-1] if matching else None)
            if cached is None and digest in reused():
                cached=reused()[digest]
                if digest not in HITS:
                    with open(LOGDIR+'/reused_hashes.txt','a') as f:f.write(digest+'\n')
                    HITS.add(digest)
        if cached:
            if cached['status']=='valid':return {**cached['parsed'],'hash':digest}
            if not cached.get('error','').startswith(('APIConnectionError:', 'APITimeoutError:')):continue
        if offline:raise RuntimeError('Missing real API response '+digest)
        prompt=json.dumps(view,separators=(',',':'),allow_nan=False)
        reserve=len((system+prompt).encode())+(BATCH_OUTPUT if route=="batch" else MAX_OUTPUT)+100
        with LOCK:
            if sum(charge(r) for r in rows)+RESERVED+reserve>CAP:raise RuntimeError('Configured API token budget reached')
            RESERVED+=reserve
        with RATE:
            time.sleep(max(0,.18-(time.monotonic()-LAST)));LAST=time.monotonic()
        client=OpenAI(api_key=API_KEY,base_url=BASE_URL,timeout=180,max_retries=0)
        start=time.perf_counter();row={'hash':digest,'request':descriptor}
        try:
            messages=[{'role':'system','content':system},{'role':'user','content':prompt}]
            limit=BATCH_OUTPUT if route=='batch' else MAX_OUTPUT
            if API_MODE=='responses':
                response=client.responses.create(model=MODEL,input=messages,temperature=TEMPERATURE,max_output_tokens=limit,extra_body=EXTRA_BODY)
                raw=response.output_text
                complete=response.status=='completed'
            else:
                response=client.chat.completions.create(model=MODEL,messages=messages,temperature=TEMPERATURE,max_tokens=limit,extra_body=EXTRA_BODY)
                raw=response.choices[0].message.content or ''
                complete=response.choices[0].finish_reason=='stop'
            usage=response.usage.model_dump() if response.usage else {}
            row.update(raw=raw,usage=usage,response_id=response.id)
            if not usage.get('total_tokens'):row['charged_tokens']=reserve
            raw=raw.strip()
            if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
            if not complete:raise ValueError('Incomplete output')
            row['parsed']=validate(json.loads(raw),route,person)
            if route=='batch':
                expected=[r['date'] for r in view['targets']]
                got=[r['date'] for r in row['parsed']['predictions']]
                if sorted(expected)!=sorted(got):raise ValueError('Batch target dates differ')
            row['status']='valid'
        except Exception as error:
            if type(error).__name__ in ['APIConnectionError','APITimeoutError']:NETWORK_FAILED=True
            row.update(status='error',error=type(error).__name__+': '+str(error)[:250])
            if 'usage' not in row:row['charged_tokens']=reserve
        finally:
            client.close();row['elapsed_seconds']=round(time.perf_counter()-start,4)
            with LOCK:
                rows.append(row)
                with open(LOGDIR+'/api_calls.jsonl','a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                RESERVED-=reserve
                if len(rows)%50==0:print('Real calls',len(rows),'charged tokens',sum(charge(r) for r in rows),flush=True)
        if row['status']=='valid':return {**row['parsed'],'hash':digest}
        if NETWORK_FAILED:raise RuntimeError('Transport interruption; response absent, safe to resume the same request later')
        time.sleep(1)
    raise RuntimeError('Three unsuccessful API attempts; see ledger')

BATCH_SYSTEM = DIRECT_SYSTEM.replace('Predict the target directly', 'Predict every target date directly').replace('Return JSON {"entries": nonnegative number, "reason":"brief explanation, at most 20 words"}.', 'Return JSON {"predictions":[{"date":"YYYY-MM-DD","entries":nonnegative number}],"reason":"brief explanation, at most 20 words"}. Return exactly one record per target date. No target demand is supplied. Do not treat your own predictions as observations.')
