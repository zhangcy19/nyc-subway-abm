# Mainline handoff for Codex agents and collaborators

This is the supported entry point for comparing different LLM backends on the **same frozen 152-station NYC subway experiment**. Run commands from this directory. Do not use `../branches/` as the production entry point.

## Quick start

```bash
cd nyc-subway-abm/llm_based_abm/mainline
conda activate dev
# On another machine, install ../requirements.txt in your own Python environment.
python verify.py
python run.py preflight --config configs/qwen_local.json
# Start only after checking the endpoint, exact model ID and token cap:
python run.py llm --config configs/qwen_local.json
```

For local Qwen, the teammate must first serve the model through an **OpenAI-compatible `/v1/chat/completions` endpoint**. The provided configuration assumes `http://localhost:8000/v1` and served model ID `Qwen/Qwen2.5-1.5B-Instruct`. Set `model` to the exact ID advertised by that server. GPU deployment itself is outside this runner. The local adapter was checked against a mock server; real Qwen output quality and hardware compatibility have not been tested here.

For Doubao, export `ARK_API_KEY` in your terminal and use `configs/doubao_lite.json`. Do not put keys in config files. For another provider, copy `configs/other_openai_compatible.json`, fill in the model and endpoint, export `LLM_API_KEY`, and use a new `run_id`. Select `api_mode: "chat"` or `"responses"`; `extra_body` is passed through for provider-specific options. Disable reasoning for the main comparison if supported, and record any exception.

The config controls `run_id`, `model`, `base_url`, `api_mode`, `key_env`, `extra_body`, `temperature`, both output limits, `workers`, and `token_cap`. Optional `station_ids` selects a small pilot; optional `methods` selects from `direct_few`, `role_no_history`, `role_with_history`. Otherwise all 152 stations and all three methods run. Use separate IDs for pilot and full runs. Never rename a model's existing result folder to represent another model.

`preflight` makes **no API calls**. For the full protocol it currently estimates roughly **6.8–12.1 million tokens**, including output allowance but not retries; actual tokenizer usage varies. The template's 5-million cap is deliberately a stopping ceiling, not a promise that a full run fits. Agree on a budget before increasing it. A local endpoint still uses this cap as a work limit; local API billing is zero but GPU time is not free.

## Files and execution order

```text
mainline/
  README.md                   handoff and protocol
  configs/                    endpoint/model templates; no keys
  inputs/                     frozen daily data, date splits, donors and parameters
  run.py                      preflight -> fit/predict -> export/report
  api.py                      backend adapter, validation, retries, cache, token limit
  role_history.py             accepted original role prompt/history construction
  design.py                   direct-LLM normal history and held-event batches
  experiment.py               calendar/weather features and weekday mean helpers
  model.py                    12 profiles, rule utilities, population fitting, Mesa aggregation
  timing.py                   shared hourly rain and profile departure schedules
  prepare.py                  fare-group and event constants
  verify.py                   offline numerical regression against accepted results
  test_api.py                 local mock-server adapter test
  results/
    shared/                   LLM-independent baselines, computed once
    doubao_lite_reference/     accepted saved results and numeric role policies
    <run_id>/                 one backend/configuration's new results
```

The source weather is `../../data/weather_hourly.csv`, shared across stations. This mainline does **not** use station-specific `subway_weather_hourly.csv`. Daily station input is already aggregated in `inputs/daily.csv`; downloading or rereading hourly ridership is unnecessary. `inputs/dates.csv` is the authoritative exact date list; filtering means the date ranges below are not necessarily complete calendars.

## Frozen experiment

- Normal Summer: training July 10–26, validation August 1–7, test August 17–23, 2023; dry/light-rain days retained after storm-window exclusion.
- Normal School: training September 12–October 4, validation October 10–16, test October 23–29, 2023; same filtering.
- Extreme: July 29, August 13, September 29, October 7, 2023. Each held event contains its center and two days before/after. The other three events provide references. Different event intensities are retained; do not call all four “severe rainstorms”.
- Extreme populations come from the corresponding Summer/School Normal training fit. **No population is fitted to the held Extreme event.** Later reference events and later seasonal Normal data are allowed: this is retrospective event generalization, not a strictly chronological operational forecast.
- The 152 evaluation stations exclude original donor stations `256` and `385`. Role few-shot histories remain based on these two frozen donors (`inputs/role_donors.csv`), preserving the accepted result. Shared probabilities do not imply station-specific behavior.
- Twelve hypothetical profiles = four fare groups × three routines. Groups are students, Fair Fares, seniors/disability combined, general. Source fare records cannot separate seniors from disability; do not claim otherwise. Three routines are regular attendance, flexible errands, discretionary activity. Parameters are assumptions, not measured people.
- Each role produces probabilities for subway/bus/taxi/walk/stay-home plus a short reason. Mesa aggregates profile subway probabilities using fitted station/profile coefficients and calendar adjustment. Coefficients are effective represented demand, **not identifiable real demographic proportions**.
- Role zero-shot: profile, mode scenario, target calendar and 24-hour weather, no demand history. Role few-shot Normal: up to four earlier training fare-group indices. Role few-shot Extreme: the other three events' five-day group histories, normalized by station/season Normal weekday means, then averaged over donors. This original protocol does not add the later experimental Normal-reference augmentation to Extreme.
- Direct few-shot Normal: up to eight station-specific earlier training examples. Extreme: four seasonal Normal examples plus the other three events, with the five held dates predicted in one API request. Historical rainfall is compacted to six-hour sums; target rainfall is hourly. This differs intentionally from the role history interface.
- New backends generate their own decisions and fit their own station populations on Normal training rows. Shrinkage/calendar/rain/Ridge settings are frozen in `inputs/frozen_parameters.csv`, originally selected using 12 development stations. This is a **fixed-configuration backend comparison**, not equally tuned best-possible performance for every model.

## Baselines and reference reproduction

`results/shared/predictions.csv` already contains weekday mean, Ridge, rule daily and historical rule timing for all 152 stations. LLM runs **do not recompute them**. Use rule daily as the primary rule comparator; timing remains available as a documented historical ablation.

To independently recompute baselines without touching the published reference:

```bash
python run.py baselines --output results/shared_recheck
```

`verify.py` recalculates four ABM variants on three stations using published numeric decisions, checks all accepted headline scores, and checks role held-event exclusion. It calls no API. To check both transport protocols using a localhost mock:

```bash
python -m unittest test_api -v
```

Accepted reference, score = `100 * (1 - WAPE)`:

| Method | Normal | Extreme center |
|---|---:|---:|
| Weekday mean | 91.71 | 79.20 |
| Ridge | 91.15 | 85.79 |
| Rule daily | 92.07 | 84.41 |
| Historical rule timing | 91.69 | 58.08 |
| Direct LLM few-shot | 92.38 | 84.34 |
| Role LLM zero-shot | 87.71 | 78.86 |
| Role LLM few-shot | 91.74 | 86.29 |

These are saved Doubao results, produced by original cached role decisions plus station refits and existing direct predictions. They are **not** a newly sampled repeat. Previously inspected test stations and performance-based development/stopping decisions mean they are not an untouched blind test. Do not promise another model will beat the baselines, drop bad stations, or tune on held-out scores to force that outcome.

## Outputs and safe resumption

Every new `results/<run_id>/` contains:

| File | Meaning |
|---|---|
| `manifest.json` | model/endpoint/settings and input/code hashes; contains no API key |
| `predictions.csv` | `task, method, station_complex_id, date, actual, predicted` |
| `populations.csv` | Normal-fitted profile coefficients, separately by method and station |
| `decisions.csv` | profile-level numerical decisions and represented journeys; no narrative logs |
| `runtime.csv` | per task/method elapsed seconds and newly consumed tokens in that attempt |
| `usage.json` | cumulative ledger tokens, calls, errors and API seconds across resumes |
| `prediction_details.csv` | errors, regime and event-day indicators |
| `task_scores.csv`, `station_scores.csv` | task and station-level scores |
| `summary.csv` | Normal and Extreme-center headline scores |
| `window_summary.csv` | includes all five days per Extreme event |
| `comparison/` | same reports joining this model to the fixed shared baselines |
| `private/api_calls.jsonl` | local request/reply cache, brief reasons and usage; never publish |

Station IDs are strings; dates use `YYYY-MM-DD`, New York local calendar. Prediction unit is **entries per station per day**. Score can be negative; it is a transformed error, not classification accuracy. Each task first sums absolute errors and actual entries across stations/dates; Normal averages its two task scores equally, Extreme averages four center-day scores equally. Recovery-day performance must be reported separately rather than silently changing denominators.

Resume with the identical command and config. Completed blocks are skipped only when station/date coverage matches. Interrupted blocks use the same run's API cache. Changing model, endpoint, station scope, prompts, data or generation settings requires a new `run_id`; worker count and budget may change when resuming. No cached response from another backend is imported. Three unsuccessful response attempts or a transport outage stop the run; invalid Qwen output is not silently replaced with rules or another LLM.

Price is deliberately not hard-coded. Compute cost from actual input/output usage using the provider's current price and cache discounts; record the price date. `usage.json` includes conservative estimated charges for requests without reported usage. Runtime after a resume may include cache speedup; do not compare it to a completely cold run without qualification. For local Qwen record GPU, quantization, serving software, concurrency and model revision in an accompanying note.

## Instructions for another Codex agent

1. Read this file, inspect the config and run `verify.py` before changing models.
2. Preserve the frozen protocol. Modify only backend/generation configuration for the first comparison. Identify any unavoidable provider limitation explicitly.
3. Run preflight, then a separately named small pilot if authorized. The full run requires its own budget authorization; do not infer it from a README command.
4. Report failures and complete coverage. Do not choose prompts/stations based on test-score improvement or overwrite the reference results.
5. Do not send test outcomes as input. Actuals are retained locally for evaluation; request builders select only authorized training/reference labels.
6. Publish only code and explicitly selected numerical exports. New result folders are ignored by default. Keep secrets, raw prompts/replies, private caches and PPT files out of Git. Push only when the user requests it.

An additional offline integration test, `python -m unittest test_runner -v`, checks one station across all six tasks, separate outputs, baseline joins and restart behavior using the saved numerical reference.
