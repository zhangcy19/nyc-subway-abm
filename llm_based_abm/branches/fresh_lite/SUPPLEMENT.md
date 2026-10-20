# Four-station presentation supplement

The five-method overview remains the original frozen experiment: weekday mean, Ridge, rule ABM with rainfall timing, direct few-shot LLM, and role few-shot LLM. No main scores were replaced after observing supplemental results.

## Added experiments

1. **Rainfall input ablation.** Remove only the 24-hour rainfall array from target and historical examples. Keep the same profiles, earlier examples, fitted station weights and weekly factors. Run all original test dates. This tests input information at inference, not a separately trained daily-only model.
2. **Repeated event-day calls.** Two more samples for both timing conditions, both LLM routes, and all four event days. The original sample is retained as sample 0. All three samples are reported; none is selected by performance. Temperature stays 0.3; replicate is a cache identifier, not an API random seed.
3. **Future scenario.** Hypothetical Friday, 23 October 2026: 80 mm daily rain, 20 mm/h peak, concentrated in the morning and early afternoon. Compare with zero rain on the same day. Use the original September-event model's fitted weights. The 2023 demand scale is held fixed; this is not a current weather forecast or a validated 2026 ridership forecast.
4. **Planning agent.** A separate Lite conversation receives four-station numeric outputs, group outcomes and assumptions. It proposes conditional information, accessibility and school-travel support. The presentation summarizes the response and flags weak student predictions. No service changes are executed.

## Main findings

- Original Normal WAPE: direct LLM 10.30%; original Extreme WAPE: role LLM 23.53%.
- First-sample Extreme timing comparison: role 29.23% daily-only vs 23.53% timed; direct 27.71% vs 25.88%.
- Across three event-day samples, the role timing advantage does not persist. See `res/supplement/repeated_metrics.csv`. Timing is useful context, but these results do not establish a robust improvement.
- Student demand remains weak: 7 observed student entries versus about 278 predicted on October 7. A good total forecast does not establish reliable subgroup forecasts.
- Future scenario: four-station demand falls by about 38.1% versus the matched dry control. This is a modeled stress response under fixed historical scale.

## Outputs and rerunning

Run commands from `llm_based_abm/fresh_lite/` using the dev environment:

- `python supplement.py preflight`: estimate scope without API calls.
- `python supplement.py ablation` and `python supplement.py scenario`: real calls or same-supplement cache reuse.
- `python plan_scenario.py`: independent planning response; saves locally outside the code repository.
- `python repeat_ablation.py`: two additional samples, within the same supplemental cap.
- `python analyze_supplement.py`: group metrics and presentation evidence.
- `python web_export.py`: all-network synthetic OD/route export from published probabilities, with no API calls.
- `python audit_supplement.py`: matched inputs, equal scoring samples, four-station weight consistency, flow conservation and budget checks. Requires local API evidence for the ledger audit.

The supplemental API client has a separate **1,000,000-token hard cap**, including estimated charges for requests without reported usage. Never delete the ledger to reset this cap. Do not run two API scripts simultaneously. The existing 5M-token experiment ledger is unchanged. Final supplemental charge: **817,805 tokens**, 823 attempts, 3 invalid-format/validation attempts included.

Public results omit raw prompts, original response text and service metadata. The local presentation and web handoff include selected brief decision summaries. Raw API logs stay local; no API key is distributed. Presentation files are outside the repository and are not uploaded.

## Web scope

All 424 source stations have coordinates. 418 have sufficient complete training days for weights; six unavailable predictions remain missing, not zero. Nine additional origins lack a usable current graph route, leaving 409 origins with exported synthetic OD. Station IDs and route references are documented in `web_data/README.md`.

This is an all-network **visualization transfer**, not a completed 424-station evaluation. The same numerical fitting code reproduces the four study stations exactly, but transferred LLM probabilities and synthetic destinations have not been validated at other stations. GTFS complex edges are schematic and are not verified accessible paths.
