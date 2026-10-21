# Branch experiments and development history

The supported multi-model entry point is `../mainline/README.md`. Do not run these scripts to reproduce the accepted mainline result.

| Directory | Purpose |
|---|---|
| `full_network/side_studies.py` | Rainfall daily/hourly ablation and baseball-event context |
| `full_network/role_prompt_diagnosis.py` | Role probability and prompt diagnosis |
| `full_network/station_role_branch.py` | Mets–Willets Point station-specific role experiment |
| `full_network/rollback_*.py` | Development/replay that produced the accepted 152-station reference |
| `event_holdout/` | Original two-donor event-holdout role policies |
| `spatial_weather/` | Station-specific weather experiment |
| `fresh_lite/`, `current/` | Earlier experiment pipelines; dependencies of historical scripts |

Historical directories retain their internal layout. Run a historical script from its own directory. Large scratch outputs and all raw API logs remain local; some historical commands consequently require local artifacts. The mainline has no runtime dependency on these directories. Its frozen numeric policies and prepared input tables are included in Git.

Do not merge side-study prompts, station-specific probabilities, revised rule timing, or new rain summaries into mainline without creating a new protocol and result folder.
