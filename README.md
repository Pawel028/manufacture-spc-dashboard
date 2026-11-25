# Process Monitoring Dashboard (SPC)

Live Statistical Process Control dashboard: control charts, run-rule alarms, process capability and
an exception log, on a simulated feed of seven process measurements.

**Live demo:** _add your URL after deploying_

## What it does
- **Simulated live feed:** a new batch every two seconds; stop/start at any time. The feed ends after
  the last batch.
- **Control charts** per parameter with control limits, specification limits and a distribution view.
- **Three run rules:** point beyond the control limits, 2 of 3 beyond 2 sigma, 8 in a row on one side
  of the mean. Each rule has its own marker.
- **Capability (Cp, Cpk, Ppk)** for every parameter, updating as batches arrive; hover for % outside spec.
- **Exception log** of the latest alarms across all parameters.
- **Specification Settings tab (working):** set USL/LSL and control limits per parameter, reset to
  baseline, and see everything update.

## Method notes
- Control limits come from the first 100 batches (Phase I). Sigma is estimated from the average moving
  range (MRbar / 1.128). Baseline points beyond 3 sigma are removed once as special causes before the
  limits are finalised.
- Specification limits are **illustrative defaults** (baseline mean +/- 4 sigma). Replace them with real
  specifications.
- Cpk uses within-batch sigma; Ppk uses overall sigma. A large gap between them indicates instability.
- Several parameters are quantised or skewed, so run rules can fire on a genuine shift in the mean
  (for example Diameter after the baseline period).

## Credit and licence
Adapted from Plotly's open-source
[dash-manufacture-spc-dashboard](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-manufacture-spc-dashboard)
sample app (MIT). The dataset is a demonstration dataset bundled with that app; treat it as
illustrative, not real production data.

**My changes:** moving-range control limits from a cleaned baseline (the original used mean +/- 3 sd of all
data and labelled mean +/- 1 sd as "specification limits"), run rules 2 and 4, capability indices,
exception log, working settings tab (the original's Update button was disabled), windowed sparklines
(fixes duplicated points), simulation stops at the end of the data, rebranded UI, removed the
Mantine dependency, updated for Dash 4 / pandas 3, Docker and Render config.

## Run locally
```
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:8050
```
Optional environment variables: `PORTFOLIO_URL` (shows a "Back to portfolio" button),
`SPC_INTERVAL_MS` (feed speed, default 2000).
