# STATUS

Updated: 2026-11-23 (Phase 1 submission day)

| Milestone | Status | Evidence |
|---|---|---|
| M0 setup, split | Passed | `ml/splits/pill.json`, `results/dataset_stats.json`, split tests |
| G1 vision model + parity | Passed | ONNX parity `results/parity_pill_bank*_onnx.csv`; device golden parity `results/device_summary.json` |
| Calibration | Done | `results/calibration/thresholds_pill_bank*.json` (val_good + calib only) |
| G2 app, Tier 1 | Done | Camera + replay pipeline, quality gate, item finder, decision core, Room log + CSV, overlay, advisory queue, validator, templates; 21 unit tests pass |
| Llama on ExecuTorch on device | Done | `.pte` exported (`results/llama/`), advisory test `results/advisory_summary.json` |
| Device benchmarks | Done | `results/device/`, `results/device_summary.json`, `results/detection_eval_device.json` |
| Synthetic test matrix | Done (17/20 pass, failures explained) | `results/testcases.csv`, report appendix D |
| Report | Draft complete | `report/report.md`, tables from `bench/make_tables.py` |
| Real-item test | Open | Only camera quality-gate checks; no counted real-tablet set |
| Power measurement | Open | Phone was charging in all runs |
| GitHub repo, demo video, submission form | Human | See checklist below |

## Human checklist for submission

1. Create the GitHub repository and push `master` (no model weights, images or `.pte` are tracked).
2. Attach the APK to a GitHub release (it contains the ONNX vision models; Llama files are pushed separately).
3. Record the demo video (outline in the main plan, section 11): airplane mode on, tray with good and defective
   tablets or the replay set, per-item overlay, REJECT advice from Llama, History tab, CSV export.
4. Ask the mentor to skim the report and score 15 advisory outputs (optional before Phase 1).
5. Fill in the submission form with repo link, video link and report; save the confirmation.

## Open issues

- I6b (dark object on dark tray) and I8 (similar-colour wrong product reported as CLUSTER) are known limitations.
- Advisory validator rule for invented re-inspection outcomes was added after review; not re-measured on a fresh seed.
- Phone `stay_on_while_plugged_in` was set to 7 for benchmarking; reset with
  `adb shell settings put global stay_on_while_plugged_in 0` (or 2 for USB only) when done.
