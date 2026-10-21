"""Real Lite requests, resumable cache, and a strict 2M-token conservative budget."""
import gzip
import hashlib
import json
import os
import threading
import time
import numpy as np
from openai import OpenAI

MODEL='doubao-seed-2-1-lite-260915'
CAP=2000000
LOCK=threading.Lock();RATE=threading.Lock();RESERVED=0;LAST=0.;LEDGER=None; REUSED=None; HITS=set()
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
        for f in ['res/api_calls.jsonl.gz','res/api_calls.jsonl']:
            if os.path.exists(f):
                with (gzip.open(f,'rt') if f.endswith('.gz') else open(f)) as source:LEDGER.extend(json.loads(line) for line in source)
    return LEDGER

def reused():
    global REUSED
    if REUSED is None:
        with gzip.open('res/reused_api.jsonl.gz','rt') as f:
            REUSED={r['hash']:r for r in map(json.loads,f) if r['status']=='valid'}
    return REUSED

def charge(r):return r.get('charged_tokens',r.get('usage',{}).get('total_tokens',0))

def validate(payload,route,person=None):
    if not isinstance(payload,dict) or not isinstance(payload.get('reason'),str) or not payload['reason'].strip():raise ValueError('Need brief reason')
    if route=='direct':
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
    global RESERVED,LAST
    system=DIRECT_SYSTEM if route=='direct' else ROLE_SYSTEM
    for attempt in range(3):
        descriptor=dict(model=MODEL,route=route,system=system,view=view,replicate=replicate,attempt=attempt,temperature=.3,max_output_tokens=250)
        digest=hashlib.sha256(json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with LOCK:
            rows=ledger();cached=next((r for r in rows if r['hash']==digest),None)
            if cached is None and digest in reused():
                cached=reused()[digest]
                if digest not in HITS:
                    with open('res/reused_hashes.txt','a') as f:f.write(digest+'\n')
                    HITS.add(digest)
        if cached:
            if cached['status']=='valid':return {**cached['parsed'],'hash':digest}
            continue
        if offline:raise RuntimeError('Missing real API response '+digest)
        prompt=json.dumps(view,separators=(',',':'),allow_nan=False)
        reserve=(len((system+prompt).encode())+1)//2+350
        with LOCK:
            if sum(charge(r) for r in rows)+RESERVED+reserve>CAP:raise RuntimeError('2M API budget reached')
            RESERVED+=reserve
        with RATE:
            time.sleep(max(0,.18-(time.monotonic()-LAST)));LAST=time.monotonic()
        with open('../../key.txt') as f:key=f.read().strip().strip('\"\'')
        client=OpenAI(api_key=key,base_url='https://ark.cn-beijing.volces.com/api/v3',timeout=90,max_retries=0)
        start=time.perf_counter();row={'hash':digest,'request':descriptor}
        try:
            response=client.responses.create(model=MODEL,input=[{'role':'system','content':system},{'role':'user','content':prompt}],temperature=.3,max_output_tokens=250,extra_body={'thinking':{'type':'disabled'}})
            row.update(raw=response.output_text,usage=response.usage.model_dump(),response_id=response.id)
            raw=response.output_text.strip()
            if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
            if response.status!='completed':raise ValueError('Incomplete output')
            row['parsed']=validate(json.loads(raw),route,person);row['status']='valid'
        except Exception as error:
            row.update(status='error',error=type(error).__name__+': '+str(error)[:250])
            if 'usage' not in row:row['charged_tokens']=reserve
        finally:
            client.close();row['elapsed_seconds']=round(time.perf_counter()-start,4)
            with LOCK:
                rows.append(row)
                with open('res/api_calls.jsonl','a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                RESERVED-=reserve
                if len(rows)%50==0:print('Real calls',len(rows),'charged tokens',sum(charge(r) for r in rows),flush=True)
        if row['status']=='valid':return {**row['parsed'],'hash':digest}
        time.sleep(1)
    raise RuntimeError('Three unsuccessful API attempts; see ledger')
