# NYC Subway: LLM-based Agent-Based Modeling

Daily station-entry prediction under normal weather and four rain-event windows in 2023.

Start with **[the experiment handoff](llm_based_abm/mainline/README.md)** for the supported 152-station pipeline, model configurations, commands, input/output formats, and instructions for another Codex agent. It supports separately stored LLM backends (including local Qwen through an OpenAI-compatible server) and reuses fixed numerical baselines.

- [`data/`](data/README.md): source data and data documentation.
- [`llm_based_abm/mainline/`](llm_based_abm/mainline/README.md): current experiment and accepted numerical reference.
- [`llm_based_abm/branches/`](llm_based_abm/branches/README.md): side studies and historical development.
- [`llm_based_abm/plots.ipynb`](llm_based_abm/plots.ipynb): figures and their saved source data in `figures/`.
- [`llm_based_abm/web_data/`](llm_based_abm/web_data/README.md): Doubao Lite decisions and synthetic web-animation data.

Raw API requests/replies, credentials, local workspace settings and presentation files are not included. Selected short decision summaries from synthetic traveler profiles are included in the plotting and web datasets. The shared probability protocol uses two reference stations and fitted effective demand weights; see the handoff for interpretation and limitations.

`rule_based_abm/` remains the collaborator's separate workspace; the mainline does not modify it.
