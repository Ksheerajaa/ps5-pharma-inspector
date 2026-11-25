# PS5 Pharma Inspector: Implementation Plan (master spec for the coding agent)

Project: on-device defect detection and quality-control assistant for tablets and capsules, for the Bharat AI-SoC Challenge 2026-27, Problem Statement 5.
Owner: Ksheeraja (solo). Device: Nothing Phone (3a), Android, Arm CPU. Written Tue 10 Nov 2026.
Hard dates: code freeze Wed 18 Nov. Submit Fri 20 Nov. Official Phase 1 deadline Mon 23 Nov 2026 (buffer only).

---

## 0. Kickoff prompt (paste this into the agent, then attach or place this file at the repo root)

> You are the lead engineer on the project described in PS5_IMPLEMENTATION_PLAN.md at the repo root. Read the whole file before doing anything. Follow section 12 (working rules) exactly. Start with section 13 (preflight questions): ask me them in ONE message and wait. Then execute the milestones in section 7 in order. At the end of every milestone: run its acceptance tests, update docs/STATUS.md, commit, and stop at any "HUMAN CHECKPOINT" to tell me exactly what I must do. Never invent a measurement, a dataset statistic, an API signature or a licence term; verify it from the source or mark it TODO. If a gate fails, use the fallback ladder in the plan and record the decision in docs/DECISIONS.md. Do not widen scope. If you are about to miss a date, apply the cut list in section 8 instead of asking.

---

## 1. Mission and non-negotiables

Build an Android app that inspects one tablet or capsule at a time using the phone camera, entirely offline, and:

1. **Detects** surface defects and anomalies (vision model, runs every inspection).
2. **Classifies** defect type and **severity**.
3. **Alerts** the operator with a colour-coded verdict (PASS, RE-INSPECT, REJECT) and **actionable advice** written by **Llama 3.2 1B Instruct running on-device through ExecuTorch**.
4. **Logs** every inspection (SQLite, CSV export), with the log sentence written by Llama.
5. **Runs at the edge**: no cloud, no network permission in the app manifest.

Non-negotiables (from the organisers' mentoring video and the challenge page):

- Llama on ExecuTorch is mandatory in every submission. The vision model may use another runtime if ExecuTorch export fails, but the two pipelines must be documented separately (this is inferred from a Raspberry Pi slide; confirm with tech-queries and record the answer in docs/DECISIONS.md).
- Give Llama structured findings (type, severity, location, confidence, verdict), never pixels.
- Suggested evaluation lens: detection quality (misses versus false alarms), measured latency and FPS on the device, advice grounded in the findings, operator UX (clear alerts, easy log), rigour (both runtimes documented).
- Deliverables: runnable documented source code on GitHub, working app on the target device, demo video (detection, alerts, logging), technical report (design, runtimes, results, limits).
- Entries must be original work. Every dataset, model and library licence must be respected and cited.
- Use public datasets only (MVTec AD is an organiser-suggested starter dataset). Do not train on photos taken with the phone. Real tablets or capsules are used only for an honest final test.

---

## 2. Facts and assumptions

Verified (as of 10 Nov 2026):

- Phone SoC: Snapdragon 7s Gen 3 (4x Cortex-A720 plus 4x Cortex-A520). RAM is 8 or 12 GB depending on variant: read the real value from the device.
- ExecuTorch Android: AAR `org.pytorch:executorch-android` on Maven Central. XNNPACK is the CPU backend to use. Docs exist for release 1.5, so pin the latest stable release at the start and use that same release for the Python exporter and the AAR.
- Current Llama export entry point: `python -m executorch.extension.llm.export.export_llm --config <yaml> +base.<overrides>` (config fields below). An older equivalent is `python3 -m examples.models.llama.export_llama ...`. Use whichever form the README of your pinned release documents.
- Llama 3.2 1B checkpoint files needed for export: `consolidated.00.pth`, `params.json`, plus `tokenizer.model` for the app. The checkpoint is gated on Hugging Face (the human must accept the licence).
- MVTec AD licence: CC BY-NC-SA 4.0 (non-commercial, share-alike). Cite Bergmann et al. and attribute images used in figures. Do not redistribute the images in the repo.
- anomalib supports EfficientAd and exports to Torch, ONNX and OpenVINO. EfficientAd trains with batch size 1.

Assumptions to verify and record in docs/STATUS.md before relying on them:

- Defect folder names for MVTec `pill` (colour, combined, contamination, crack, faulty_imprint, pill_type, scratch) and `capsule` (crack, faulty_imprint, poke, scratch, squeeze). Read them from disk.
- Whether EfficientAD exports through `torch.export` to ExecuTorch. Unknown; gate G1 decides.
- Exact `LlmModule` Kotlin signatures for the pinned release. Read them from the release Javadoc and the LlamaDemo app in the `meta-pytorch/executorch-examples` repository. Do not guess.

---

## 3. Who does what

Agent can do: all code, scripts, notebooks, config, tests, docs, report drafting, figure and table generation, repo hygiene.
Agent cannot do (HUMAN CHECKPOINTS): accept the Llama licence on Hugging Face and download weights, accept MVTec terms and download the dataset, run GPU training in Colab or Kaggle if there is no local GPU, connect the phone and enable USB debugging, test with real tablets or capsules, record the demo video, attend the doubt-clearing session, submit the form, talk to the faculty mentor.

---

## 4. Architecture

```
                     EVERY INSPECTION (models stay loaded)
 CameraX ─> Guide-box crop ─> Resize 256x256 ─> Vision model ─> Decision core
 (locked     (matches the      float32 NCHW     anomaly map     (plain Kotlin, no AI)
 focus/exp)   on-screen box)                     + score        features, severity,
                                                                verdict, confidence
                                                        │
                    ┌───────── PASS ────────────────────┤ (still logged, template text)
                    │                                   │ RE-INSPECT or REJECT
                    │                                   v
                    │                         Defect-type CNN (crop of top blob)
                    │                                   │
                    │                                   v
                    │              Prompt builder (JSON-ish record + SOP hint)
                    │                                   │
                    │                                   v
                    │              Llama 3.2 1B, 4-bit, ExecuTorch, XNNPACK
                    │                                   │
                    │                                   v
                    │              Output validator ── fails ──> template text
                    v                                   v
              Operator UI (verdict at once, advice streams in) + Room log + CSV export
```

Design rules:

- Llama never decides the verdict. It explains, recommends and writes the log sentence. If its output fails validation, show template text and mark `advisory_source = template`.
- Verdict appears immediately; advice streams in afterwards on a background thread.
- One product selector (Tablet or Capsule) switches the vision model, the type classifier, the thresholds and the SOP hints. Separate models per product (MVTec trains one anomaly model per category).
- Inspection is per item (capture button or auto-capture when the item is stable), plus a throttled live preview that shows the heatmap. FPS is measured in the continuous vision-only loop; per-item latency is measured end to end.
- The app manifest has NO `INTERNET` permission. That is the offline proof; say so in the report.

Runtimes (document as two pipelines in the report):

| Pipeline | Model | Runtime | Backend |
| --- | --- | --- | --- |
| Advisory | Llama 3.2 1B Instruct, 4-bit | ExecuTorch | XNNPACK, Arm CPU |
| Vision | EfficientAD small plus type CNN | ExecuTorch if gate G1 passes, else ONNX Runtime Mobile or LiteRT (decided in docs/DECISIONS.md) | CPU |

---

## 5. Repository layout

```
/
  README.md                 build + run + reproduce, 1 page
  LICENSE                   own code (human chooses; default Apache-2.0 until told otherwise)
  NOTICE                    Llama 3.2 notice line + other required notices
  THIRD_PARTY.md            every dataset, model, library: name, version, licence, URL
  VERSIONS.md               pinned ExecuTorch, torch, anomalib, Android Gradle, Kotlin, etc.
  docs/
    STATUS.md               milestone table, gates, open issues (agent updates after each milestone)
    DECISIONS.md            every fallback or deviation, with date and reason
    real_items_protocol.md  how the human runs the real-item test
    demo_script.md
  ml/
    requirements.txt
    configs/                category configs, split seeds, thresholds schema
    data/                   (gitignored) MVTec AD, optional external sets
    scripts/
      00_prepare_splits.py  seeded splits, writes ml/splits/*.json
      01_train_effad.py     anomalib EfficientAd, per category
      02_export_vision.py   wrapper nn.Module, export to .pte (or ONNX fallback)
      03_parity_test.py     PyTorch vs exported runtime
      04_train_type_cnn.py  5-fold CV, then final model on all defect images
      05_export_type_cnn.py
      06_calibrate.py       thresholds, severity cut points, writes thresholds_<cat>.json
      07_eval_detection.py  AUROC, F1, miss rate, false-alarm rate, confusion matrix
      08_eval_end_to_end.py out-of-fold type + severity + verdict tables
      export_llama.sh       Llama export, reads a YAML config
    colab/
      train_and_export.ipynb  thin notebook that calls the scripts above
  android/
    app/ ...                Kotlin, Jetpack Compose, CameraX, Room, ExecuTorch AAR
    app/src/main/assets/
      config/thresholds_tablet.json, thresholds_capsule.json, sop.yaml, prompts/system.txt, prompts/user_template.txt
      models/ (small vision .pte files; never the Llama .pte)
      golden/ (5 test images per product + expected scores)
  advisory_eval/
    cases_50.json           50 structured records covering all verdicts and types
    rubric.md
  bench/
    adb_meminfo.sh, pull_logs.sh
    make_tables.py          reads CSVs in results/, writes tables and figures
  results/                  raw CSVs and JSONs (committed, small)
  report/
    report.md, figures/, build.sh
```

Large files (.pte, datasets, checkpoints) are gitignored. Commit scripts, hashes and versions so anyone can rebuild. Decide at M5 whether to attach the Llama .pte as a release asset with the licence copy.

---

## 6. Component specifications

### 6.1 Data and split protocol (no leakage, honest numbers)

Primary: MVTec AD `pill` and `capsule`. Pill first; capsule only after pill passes gate G3 (see section 7 and the cut list).

Seeded split (seed 42, written to `ml/splits/<category>.json`, never changed after M1):

- `train_good`: 90% of `train/good`. Anomaly model trains only on this.
- `val_good`: the other 10% of `train/good`. Never trained on.
- Test set (good plus defective) is split 50/50, stratified by defect type: `calib` and `eval`.
- Thresholds, severity cut points and sensitivity defaults come from `val_good` plus `calib` only. All reported detection metrics come from `eval`, touched once per final configuration.
- Defect-type classifier: 5-fold stratified cross-validation over all defective test images for reporting (out-of-fold predictions). The deployed classifier is then trained on all defective images, so end-to-end type accuracy in the report uses out-of-fold predictions only. State this in the report.
- Optional external check (only if time allows, M6): one CC BY Roboflow tablet set for a domain-shift test. Verify its licence page and that the images are original. Never train on it for the headline numbers.

Print and save dataset statistics from disk (image counts per split and defect class) into `results/dataset_stats.json`. Do not type counts from memory into the report.

### 6.2 Vision model and fallback ladder

Primary: EfficientAD small via anomalib, trained per category on `train_good`. Budget the training steps to the GPU time available; record steps and the AUROC cost if reduced.

Export wrapper (`02_export_vision.py`):

- Input: RGB float tensor 1x3x256x256, values in 0..1. Put ImageNet normalisation inside the wrapper so the app only scales bytes to 0..1.
- Output: anomaly map (1x1xHxW, normalised using the quantile constants from training, stored as buffers) and a scalar score (max of the smoothed map).
- Export with `torch.export.export(wrapper.eval(), (example,))`, then `to_edge_transform_and_lower` with the XNNPACK partitioner, then `to_executorch()`.

Ladder (gate G1, time-box ONE day):

- A. ExecuTorch .pte matches PyTorch (score within 2% relative, map correlation above 0.99, AUROC on `eval` within 1 point). Use it.
- B. If an op fails, rewrite the op in the wrapper (replace unsupported resize or pooling with equivalents) once. Retest.
- C. If still failing by end of day: export ONNX, run with ONNX Runtime Mobile (or LiteRT) on Android. Log in docs/DECISIONS.md. Keep the Llama path on ExecuTorch.
- D. If EfficientAD itself will not train or export in time: "mini-PatchCore": ResNet-18 layer2 and layer3 features, patch embeddings from `train_good`, random-projection coreset (a few thousand patches), nearest-neighbour distance as a matmul inside the exported graph. Plain CNN plus a matmul exports reliably. Record the AUROC difference.

Optional 8-bit quantisation of the vision model is cut (see section 8).

### 6.3 Defect-type classifier

- Per product (pill and capsule), classes read from the folder names, plus no "good" class (it only runs on flagged items).
- Input: crop of the largest anomalous blob from the anomaly map, expanded 25%, minimum 96 px, resized to 128x128.
- Model: MobileNetV3-small, ImageNet-pretrained, fine-tuned 15 epochs with strong augmentation (flips, rotation, colour jitter small, blur). Class-weighted loss for rare types.
- Report per-class precision, recall, macro-F1, confusion matrix, from the 5-fold out-of-fold predictions. If a class has fewer than about 10 images, merge it into "other" and say so.
- Export to the same runtime as the vision model. Add a golden-image test.

### 6.4 Decision core (plain Kotlin, unit-tested)

Features from the anomaly map: `peak` (max), `area_pct` (share of the inspected crop above pixel threshold `t_pix`), `bbox` and `centroid` of the largest blob, `location_word` (centre, upper-left and so on, from the centroid), `blob_count`.

Thresholds (from `06_calibrate.py`, stored in `thresholds_<product>.json`, with the date and split seed):

- `t_reject`: score threshold giving at most 5% false alarms on good images in `val_good` plus `calib`.
- `t_pass`: score threshold giving at most 5% misses on defective images in `calib`.
- If `t_pass >= t_reject` (overlapping distributions), set both to the F1-optimal threshold and disable RE-INSPECT; say so in the report.
- `t_pix`: pixel threshold from `calib` masks (maximise pixel F1), then fixed.
- `sev_cuts` (a1, a2): quantile-match predicted `area_pct` to ground-truth mask-area tertiles on `calib`.

Rules:

- `severity`: `area_pct < a1` Minor; `a1 <= area_pct < a2` Major; `>= a2` Critical. A rules file `sop.yaml` can bump one level for types flagged as safety-relevant (crack, contamination), marked "illustrative".
- `verdict`: score below `t_pass` PASS; between RE-INSPECT; at or above `t_reject` or severity Critical REJECT.
- `confidence`: logistic of the distance from the nearest threshold, scaled on `calib`; shown as a percentage, defined in the report.
- A user sensitivity slider (-2..+2) shifts both thresholds by a calibrated step. Log the value used.

Reference severity for evaluation: derived from MVTec ground-truth mask area using the same tertile cut points. State this in the report: it is derived, not annotated by an expert.

### 6.5 Llama on device (ExecuTorch)

Export (human downloads weights; agent writes the script and config). Verified config fields from the ExecuTorch docs:

```yaml
# ml/configs/llama3_2_1b_xnnpack.yaml  (field names verified for export_llm; check your pinned release)
base:
  model_class: llama3_2
  checkpoint: <path>/consolidated.00.pth
  params: <path>/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True
quantization:
  qmode: 8da4w
  group_size: 32          # 64 also appears in Arm's guide; try 32 first, 64 if memory is tight
  embedding_quantize: 4,32
backend:
  xnnpack:
    enabled: True
    extended_ops: True
# output file name and max sequence length: check the pinned release docs for the field names
```

Command: `python -m executorch.extension.llm.export.export_llm --config ml/configs/llama3_2_1b_xnnpack.yaml`. A working older form from Arm's guide for the same model: `python3 -m examples.models.llama.export_llama --checkpoint ... --params ... -kv --use_sdpa_with_kv_cache -X --xnnpack-extended-ops -qmode 8da4w --group_size 64 -d fp32 --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001, 128006, 128007]}' --embedding-quantize 4,32 --output_name="llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte" --max_seq_length 1024 --max_context_length 1024`. Keep context at 1024 tokens: prompts here are short.

Smoke test order (gate G0): laptop run, then `adb push` the .pte and `tokenizer.model` to the app's external files directory, then Meta's LlamaDemo from `meta-pytorch/executorch-examples` (find its path in the repo) to measure tokens per second and RAM before writing app code. If quality is poor, try Meta's pre-quantised SpinQuant weights if the pinned release supports them.

App integration rules:

- Use `LlmModule` from the pinned AAR. Copy signatures from the Javadoc and the LlamaDemo source.
- Load on first use, on a dedicated single thread. Setting "keep Llama loaded" (default on if memory allows; the benchmark compares on and off).
- Every inspection is a fresh conversation: reset or stop the context between calls (find the supported call in your release). A KV cache carried across items is a bug.
- Check the runner does not add a second BOS token when your prompt already starts with `<|begin_of_text|>`.
- Temperature 0.2, max new tokens about 110, stop on end-of-turn.

### 6.6 Prompt, output format, validation

Files: `prompts/system.txt`, `prompts/user_template.txt`, `sop.yaml`. Use the Llama 3 chat template.

System prompt (starting point, tune on the 50-case set):

```
You are a quality-control assistant for tablet and capsule inspection on a production line.
You receive one inspection record. The verdict was already decided by the inspection system. Never change it.
Reply in exactly four lines, labelled ACTION, REASON, CHECK, LOG.
Use only the facts in the record and the SOP hint. Do not invent numbers, batch details, causes or regulations.
Each line has at most 20 words. Plain English, no markdown.
```

User message: one `key: value` line per field (product, verdict, defect_type, severity, area_pct, location, confidence_pct, recent_rejects_last_20, lot_id if entered) plus `SOP_HINT:` from `sop.yaml` for that defect type. `sop.yaml` text is short, illustrative engineering guidance, labelled "not GMP or regulatory advice".

Validator (Kotlin, unit-tested, runs on every output):

1. Exactly the four labels, in order, each non-empty, within the word limit.
2. ACTION agrees with the verdict (keyword sets: REJECT: reject, remove, segregate, quarantine; RE-INSPECT: re-inspect, recheck, inspect again; PASS: release, continue, pass). A keyword from a conflicting verdict set fails it.
3. Every number in the output appears in the record (after rounding). Any invented number fails it.
4. Fail any step: use the template for that verdict and defect type, set `advisory_source = template`.

Stream tokens to the UI as they arrive; show the validated text at the end. LOG line goes into the database as `log_text`.

### 6.7 Android app

Stack: Kotlin, Jetpack Compose, CameraX, Room, Coroutines, ExecuTorch AAR (pinned), minSdk 29 or higher (needed for thermal status), no network permission.

Screens:

1. **Inspect**: preview with a guide box, product selector, optional lot ID field, big colour-coded verdict banner (green PASS, amber RE-INSPECT, red REJECT), result card (type, severity, location, confidence, heatmap overlay toggle), streaming advice, Capture button, auto-capture toggle (item stable for about 1 s), sensitivity slider, offline indicator.
2. **History**: list with verdict filters, detail view (thumbnail, heatmap, all fields, advice, timings), CSV export through the share sheet.
3. **Replay**: runs MVTec test images pushed to `files/replay/` through the exact same pipeline (needed for reproducible demos and benchmarks; the dataset is not shipped in the APK).
4. **Benchmark** (long-press the title): runs the protocol in 6.9, writes CSVs to the app files directory. Also holds the Real-test mode (a ground-truth tag per item: good, damaged-simulated).
5. **Advisory test**: runs the 50 cases from `cases_50.json` on device and writes a CSV.
6. **About**: "Built with Llama", licence texts, model versions, dataset credits, the "illustrative advice" disclaimer.

Threading and memory:

- CameraX analyzer with keep-only-latest. One single-thread executor for vision, one for Llama. In Capture mode, nothing else runs while Llama generates; in live preview, throttle vision and pause it during generation.
- Vision models stay loaded for the app's lifetime. Llama policy per the setting above. Handle `onTrimMemory` by unloading Llama.
- Preprocessing identical to training (resize, crop, scaling). The golden-image test asserts that a fixed image gives the same score on the phone as in Python within 2% relative.

Models: small vision .pte files in assets (copied to files on first run); Llama .pte and tokenizer pushed via adb to `/sdcard/Android/data/<package>/files/models/` (document the exact commands in the README).

### 6.8 Log schema (Room)

`inspections`: id, ts_utc, session_id, product, item_no, lot_id, score, threshold_pass, threshold_reject, sensitivity, verdict, severity, defect_type, type_conf, area_pct, bbox, confidence, advisory_action, advisory_reason, advisory_check, log_text, advisory_source (llm or template), gt_tag (nullable, real-test mode), vision_ms, classify_ms, llm_load_ms, llm_ttft_ms, llm_tokens, llm_total_ms, thumb_path, heatmap_path, model_versions.

### 6.9 Benchmark harness and protocol

State on every result: phone model, RAM, Android version, ExecuTorch version, backend, thread count, start battery %, start thermal status, brightness, airplane mode on.

| Metric | Measurement | Output |
| --- | --- | --- |
| Vision latency | preprocess, inference, postprocess timed separately; 20 warm-up then 200 timed | median, p95, n |
| FPS | continuous vision-only loop for 60 s | mean FPS, items per minute |
| Llama | load time, time to first token, decode tokens per second, 80-token total, 30 prompts from the 50 cases | median, p95 |
| End to end | capture to verdict; capture to finished advice | median, p95 |
| Peak RAM | `Debug.getMemoryInfo` total PSS in app plus `bench/adb_meminfo.sh` sampling every 5 s at idle, vision loaded, both loaded, during generation | MB per state |
| Battery and thermal | 30-minute loop of replay image, vision, decision, Llama on flagged, log; log battery %, battery temperature, thermal status every 30 s; screen on, fixed brightness, data off | % per 30 min, peak temperature, throttle point |
| Sizes | each .pte and the release APK | MB |

Report medians and p95 with n, never a best run. Raw CSVs go to `results/`. `bench/make_tables.py` generates every table and figure in the report from those CSVs.

### 6.10 Evaluation scripts and metrics

- Detection (on `eval` only): image AUROC, F1 at the chosen operating point, **miss rate (false-negative rate)** and **false-alarm rate (false-positive rate)** as counts and percentages, confusion matrix. Also pixel AUROC if cheap.
- Verdict table: for PASS, RE-INSPECT, REJECT counts split by truly good and truly defective.
- Type: per-class table and confusion matrix from out-of-fold predictions.
- Severity: agreement with the mask-derived reference, as a confusion matrix.
- Advisory: on the 50 cases on device, auto-checks (format valid, verdict-consistent, grounded numbers, template fallback rate) plus the mentor's manual rubric on a sample of 15.
- Parity: PyTorch versus exported runtime on the golden images and on the full `eval` AUROC.

---

## 7. Milestones, dates and acceptance tests

Dates assume work starts the evening of Tue 10 Nov. At the start, audit what already exists (section 13) and skip milestones that already pass their tests.

| ID | Window | Deliverable | Acceptance test (must pass to leave the milestone) |
| --- | --- | --- | --- |
| M0 | Tue 10 to Wed 11 Nov | Repo scaffold, VERSIONS.md, licence files, split script, Colab notebook, Llama export script, preflight done | `pytest` passes for split script; THIRD_PARTY.md has every dependency so far |
| G0 | Wed 11 Nov | Llama 3.2 1B answers on the phone through ExecuTorch (demo app or own app) | Written tokens per second, TTFT, RAM in `results/llama_smoke.csv` |
| M1 / G1 | Thu 12 Nov | Vision model trained, exported, parity checked, loaded in the app | Golden image score matches within 2%; decision in DECISIONS.md |
| M2 / G2 | Fri 13 to Sat 14 Nov | Full loop on the phone: camera or replay, vision, decision core, Room log, History screen, Llama advice with template fallback | Replay 100 items with both models loaded, no crash; peak RAM recorded; decision-core unit tests pass |
| M3 | Sun 15 to Mon 16 Nov | Type classifier trained and exported, calibration done and thresholds shipped, advisory prompt tuned on the 50 cases, validator, CSV export, About screen | Advisory test CSV produced on device; `07_eval_detection.py` and `08_eval_end_to_end.py` produce tables from `eval` |
| M4 / G3 | Tue 17 Nov | Real-item test and full benchmarks | `results/` has all CSVs from 6.9; `make_tables.py` builds every report table |
| M5 | Wed 18 Nov | **Code freeze.** Tag v1.0, release APK, README tested from a fresh clone, report v1 complete | Fresh-clone build works; release manifest has no INTERNET permission |
| M6 | Thu 19 Nov | Final report, demo video recorded, mentor skim | Report cites every dataset, model and licence; video shows detection, alerts, logging, airplane mode |
| M7 | Fri 20 Nov | Submit through the form, save confirmation | Confirmation screenshot saved; repo reachable logged out |
| Buffer | Sat 21 to Mon 23 Nov | No new work | None |

Gate failures: use the ladder or the cut list the same day. Record in DECISIONS.md.

---

## 8. Cut list (apply in this order if behind)

1. Capsule category (ship pill only; the report says so).
2. The external Roboflow domain-shift check.
3. The learned type classifier: replace with rule-based type from colour and shape statistics and label it "indicative".
4. Live-preview heatmap at video rate: show the heatmap only after capture.
5. Auto-capture.
6. The 30-minute run drops to 15 minutes (state it).
7. Never cut: Llama on ExecuTorch, logging, offline operation, honest benchmarks, licence notices, the report.

---

## 9. Licences and attribution (agent maintains from day one)

- `THIRD_PARTY.md`: for every item, name, version, licence, URL, where used.
- Llama 3.2: ship a copy of the Llama 3.2 Community Licence and the exact NOTICE text it requires, show "Built with Llama" on the About screen, in the README and in the report, and follow the Acceptable Use Policy. Never commit the .pte or checkpoint.
- MVTec AD: CC BY-NC-SA 4.0. Cite the paper; credit under every figure that uses its images; do not redistribute images.
- EfficientAD downloads an auxiliary image set during training: record its name and licence from the anomalib source.
- Any external Roboflow or Kaggle set: licence from its own page, verify originality, cite by name. One set is suspected to derive from MVTec (Macau University capsule set); treat it as NC-SA unless proven otherwise or skip it.
- The owner's earlier RF PCB paper may be cited as prior work. Do not copy text, figures or code from it.

---

## 10. Report (report/report.md, written section by section as features finish)

1. Summary: problem, approach, headline results.
2. Problem and use case, and what the system does not claim (not a regulated device; advice is illustrative).
3. Requirements traceability table: each PS5 requirement, feature, evidence, section.
4. System architecture: the diagram, responsibilities, why Llama never decides.
5. Data: MVTec AD pill and capsule, licences, split protocol, leakage rules.
6. Vision pipeline: EfficientAD training, calibration, severity rule, type classifier.
7. Advisory pipeline: Llama choice, quantisation, prompt, validator, template fallback.
8. Runtimes table: model, runtime, backend, device, latency, RAM, one row per pipeline (both runtimes documented separately).
9. Results: detection (miss rate and false-alarm rate first), type, severity, advisory rubric.
10. Benchmarks: all tables from section 6.9 with conditions.
11. Real-item validation: fixture, items, counts, simulated-defect disclosure, failure cases with pictures.
12. Limitations: lab-to-phone gap, small defect counts, thermals, advice safety, licence constraints.
13. Future work: Qualcomm or Vulkan backend, LoRA on advice, more categories, operator feedback.
14. Reproducibility: pinned versions, commands, how each table is regenerated.
15. References and licences.

Every number in the report must come from a file in `results/` produced by a script. No hand-typed measurements.

---

## 11. Demo video outline (length: confirm with organisers; plan for about 3 minutes)

1. Title card and one-line problem (10 s).
2. Airplane mode on, app opens, About screen showing "Built with Llama" and no network permission (15 s).
3. Live inspection of a good item: PASS, heatmap, log entry (25 s).
4. Defective items (replay of MVTec images and, if available, the simulated-damaged real items): REJECT and RE-INSPECT, advice streaming in, severity and location shown (60 s).
5. History screen and CSV export (20 s).
6. Benchmark screen with latency, FPS, RAM, temperature numbers (20 s).
7. Architecture slide with the two pipelines and runtimes (20 s).
8. Close with limits and credits (10 s).

---

## 12. Agent working rules

1. Work milestone by milestone. After each: run its acceptance tests, update `docs/STATUS.md` (table of milestones, gates passed, open issues, what the human must do next), commit with a clear message, then continue.
2. Never fabricate. Dataset counts, metrics, timings, API signatures and licence terms come from files, command output or the pinned source. Unknown means TODO plus a note in STATUS.md.
3. Pin versions in `VERSIONS.md` on day one. Use one ExecuTorch release for the exporter and the AAR.
4. Small, testable units. Unit tests for the decision core, validator, prompt builder and calibration. Python tests for splits.
5. No large files in git. Add `.gitignore` for data, checkpoints, `.pte`, build outputs.
6. Reproducibility: every table and figure comes from a script reading `results/`.
7. When blocked on a HUMAN CHECKPOINT, write the exact instructions (commands, links, expected file names and paths) and stop that branch; continue on work that does not depend on it.
8. Scope discipline: no new features outside this file. If something would improve the project but is not listed, add it to a "Later" list in STATUS.md instead of building it.
9. Keep the human informed: at each stop, say what passed, what failed, what you decided, and what is needed next, in under 15 lines.

---

## 13. Preflight (agent asks these in ONE message at the start)

1. Which gates are already done (Llama on the phone, vision training, any app code)? Where is the existing code?
2. Phone: exact RAM variant, Android version, USB debugging working, `adb devices` shows it?
3. Machine: OS, Android Studio installed, Python version, any local NVIDIA GPU? If none, are Colab or Kaggle GPUs available?
4. MVTec AD downloaded (pill and capsule)? Path?
5. Llama 3.2 1B Instruct licence accepted on Hugging Face and weights downloaded? Path?
6. Faculty mentor and college nomination done? Has the doubt-clearing session happened, and did it change any rule or date? Has the submission form arrived, and what fields does it ask for?
7. Own-code licence preference (default Apache-2.0)?
8. Do you have tablets or capsules for the real-item test (items you own; never use medicines belonging to others)?

## 14. Human checklist (for Ksheeraja)

- [ ] Answer the preflight questions.
- [ ] Accept the Llama 3.2 licence on Hugging Face and download weights.
- [ ] Download MVTec AD (accept terms) and, only if you want it, one CC BY tablet set.
- [ ] Run the Colab notebook for training and export; copy outputs back.
- [ ] Connect the phone; push models with the commands in the README.
- [ ] Email tech-queries (subject "PS5"): may the vision model use another runtime on Android; demo video length; report length and format; any scoring weights.
- [ ] Real-item test following `docs/real_items_protocol.md` (items you own, never ingest them, label damaged items as simulated).
- [ ] Record the demo from `docs/demo_script.md`, in airplane mode.
- [ ] Ask your mentor to skim the report on Thu 19 Nov and to score 15 advisory outputs.
- [ ] Submit by Fri 20 Nov; save the confirmation.
