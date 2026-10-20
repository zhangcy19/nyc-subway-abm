# NYC subway LLM / ABM experiments

- **Mainline:** [`mainline/README.md`](mainline/README.md). Frozen 152-station protocol, shared baselines, isolated model outputs, local Qwen / OpenAI-compatible APIs.
- **Side studies and earlier development:** [`branches/README.md`](branches/README.md). Rain timing, sports-event context, station-specific roles, spatial weather and historical pipelines.
- **Dependencies:** `requirements.txt`.

New model comparisons should start in `mainline/`. The accepted Doubao numerical reference and all required prepared inputs are shipped there. No private API cache is needed to run another model.
