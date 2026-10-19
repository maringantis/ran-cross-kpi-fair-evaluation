# Cross-KPI predictive maintenance in carrier-grade RAN: fair-evaluation experiments

Code and synthetic-data experiments for:

> S. C. Maringanti and T. Kundu, "Fair Evaluation of Cross-KPI Predictive Maintenance in Carrier-Grade RAN: Calibration, Feature Representation, and Matched Baselines," IEEE HONET 2026.

**All data are synthetic.** No operator telemetry, real failure records, vendor measurements, or production metrics are used.

## Important scope note
The paper's Tables I to III come from the authors' original harness, which is not in this repository. This repository contains a **vectorized reimplementation** of the benchmark (same degradation functions, 17/17/16 mode split, 2% node-level persistent false-alarm calibration, 30 seeds) and the analyses reported in the paper's "Additional analyses" subsection: it reproduces the main results approximately (lead times within 0.4 days, miss rates within 0.8 points), adds CUSUM and EWMA baselines, paired tests, a common-subset lead-time check, sensor-noise and missing-telemetry probes, an unseen-failure-mode probe, and the overlapping-signature variant.

## What is here
| Path | Purpose |
|---|---|
| `src/bench37.py` | Generator, calibration, detectors (threshold, CUSUM, EWMA, trajectory matchers), stress and open-set probes |
| `src/analyze37.py` | Tables, bootstrap CIs, paired tests, McNemar, common-subset lead time |
| `src/overlap37.py` | Variant in which each degrading KPI appears in two modes |
| `results/` | Seed-level and node-level outputs and the summary tables |

## Run
```bash
pip install -r requirements.txt
cd src
python3 bench37.py 30      # about 1 minute, writes ../results/per_seed.json and node_level.csv
python3 analyze37.py
python3 overlap37.py
```
Seeds are fixed, so repeated runs give identical output.

## Notes
- Portions of the code and text were produced with the assistance of a generative AI tool (Anthropic Claude). The authors reviewed and verified the work.

## License
MIT, see `LICENSE`.
