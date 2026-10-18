# Subway scenario animation data

Run from `llm_based_abm/fresh_lite/`: `python web_export.py`. Uses the published numerical scenario probabilities; no key or API call is required. These outputs support a web visualization, not an all-station accuracy claim.

## What is real and what is simulated

- Real inputs: MTA station coordinates, current GTFS connections, and 2023 training entries.
- Model output: daily entry totals under matched **dry** and **storm** scenarios on **2026-10-23**. The storm is hypothetical (80 mm, 20 mm/h peak), not an official forecast. Demand scale remains at 2023 levels.
- All-network output transfers the four-station role probabilities, refitting station weights to their 2023 training entries with fixed hyperparameters. Only the four study stations have been comparatively evaluated.
- OD destinations are synthetic gravity allocations. Destination attractiveness uses training entry volume as an explicit proxy, not observed exits. Up to 12 reachable destinations per origin are retained and normalized. Destination shares stay fixed between weather scenarios; only origin volume changes.
- Routes use shortest scheduled-time connections after collapsing GTFS platforms to complexes. Transfers inside a complex are implicit. Coordinates join stations with straight segments: these are schematic routes, not track geometry. Accessible paths are unknown; do not label them wheelchair-safe.
- Individuals, ages and destinations are illustrative. Their mode is sampled with a fixed quantile from the role probabilities. The same quantile is used for both scenarios. A changed mode or cancelled trip need not imply a changed subway route. No station closures are assumed.

## Files and interface

All station IDs are **strings**, coordinates are **WGS84 longitude, latitude**, times are **America/New_York**. Probabilities range from 0 to 1. Fractional trips are expected journey counts, not unique people.

| File | Key / format | Use |
|---|---|---|
| `stations.csv` | `station_complex_id` | Names, boroughs and coordinates; includes all 424 source stations |
| `station_flows.csv` | `scenario, station_complex_id` | `predicted`: daily expected entries; `fit_available=false` means missing, never zero; provenance records the simulation assumption |
| `od_flows.csv` | `scenario, origin, destination` | `expected_trips`, `destination_probability`, `route_id`, `travel_seconds`; animate volume changes |
| `routes.json` | array, unique `route_id` | Ordered `stations`, `[lon,lat]` coordinates, cumulative `elapsed_seconds`; interpolate points over segment time |
| `network.geojson` | GeoJSON FeatureCollection | Directed schematic links, route label, scheduled seconds; accessibility unknown |
| `individuals.json` | `individual_id, scenario` | Synthetic age, group-role ID, probabilities, mode, departure hour, route ID; null route means no subway animation |
| `coverage.csv` | `station_complex_id` | Reachable and exported destination counts; do not silently fabricate paths for unroutable origins |
| `network_population_weights.csv` | `station_complex_id, agent_id` | Fixed fitted scale and weekday factors, for numeric reproducibility |
| `validation.json` | object | Coverage and OD conservation checks |

Join `od_flows.route_id` to `routes.route_id`; join station IDs to `stations`. For playback, keep a configurable visual scale (e.g. one dot per 100 expected trips) and preserve the same scale in both scenarios. Do not render one dot as a verified person. A `stay_home` decision is no moving trip; bus/taxi/walk lack validated route geometry and should be shown as a mode change rather than drawing a subway path.

The local handoff additionally includes short model decision summaries, allowed by the user for presentation. Raw API logs and provider metadata are not included in the public repository. No API key is needed to display these exports.
