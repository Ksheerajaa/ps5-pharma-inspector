# PS5 Pharma Inspector: Addendum v2 (read together with PS5_IMPLEMENTATION_PLAN.md)

Written Tue 17 Nov 2026. Nothing is built yet. Official Phase 1 deadline: Mon 23 Nov 2026 (the human must confirm the exact date, time zone and form fields; see section 9).

**Precedence.** This addendum SUPERSEDES: section 4 (architecture), section 6.2 (vision model), section 6.7 (screens, additions only), section 7 (milestones and dates), section 8 (cut list). Everything else in the main plan still applies (non-negotiables, data split protocol, decision core, Llama export, prompt and validator, log schema with the additions below, benchmarks, licences, report, working rules). If the two files disagree, this file wins.

## 0. Updated kickoff prompt (replaces section 0 of the main plan)

> You are the lead engineer. Read PS5_IMPLEMENTATION_PLAN.md and PS5_ADDENDUM_v2.md in full; where they disagree, the addendum wins. The official deliverable video (2026-11-13, Charitha Balivada, 52 min) confirms the requirements and evaluation lens below — treat these as authoritative.
>
> KEY CLARIFICATIONS FROM THE OFFICIAL VIDEO FAQ:
> 1. **Vision runtime does NOT need ExecuTorch** (FAQ 1 Q2). ONNX Runtime, TFLite, HailoRT are all acceptable for vision. Only the Llama advisory part must use ExecuTorch. This resolves the tech-queries question — our fallback ladder is explicitly sanctioned.
> 2. **Pass/fail CAN be rule-based** (FAQ 3 Q4). Rules decide the verdict; Llama explains and advises. It must be genuinely integrated. Our decision-core design is correct.
> 3. **No Llama fine-tuning needed** (FAQ 1 Q4). Pretrained Llama 3.2 1B or 3B Instruct with well-designed structured prompts is enough.
> 4. **Severity = Low/Medium/High from type, size, location. Confidence ≠ severity** (FAQ 3 Q3).
> 5. **No FPS target published** (FAQ 4 Q1). Report FPS, per-image and end-to-end latency, justify for the use case. No tokens/s target for Llama either (FAQ 4 Q6).
> 6. **Benchmark must include**: precision, recall, F1, FPS, memory, power, plus Llama TTFT, tokens/s and total response time (FAQ 4 Q2).
> 7. **JSON hand-off** between vision and advisory: defect, severity, confidence, box. Test each interface alone; integrate early (FAQ 4 Q5).
> 8. **No conveyor belt needed** (FAQ 3 Q1). Static jig with trigger capture, live camera or recorded video works.
> 9. **No dataset supplied** (FAQ 3 Q2). MVTec AD, NEU and KolektorSDD are optional. Mixing your own camera images is fine.
> 10. Hardware targets include Raspberry Pi 5 (CPU or with Hailo). We build for Android (Nothing Phone 3a) — decision in docs/DECISIONS.md. CPU-only is scored equally (FAQ 2 Q2).
> 11. The report MUST keep vision pipeline and advisory pipeline documentation completely separate.
> 12. **Phase timeline**: submit 23 Nov 2026 → shortlist mid-Dec → Phase 2 PoC 5 Feb 2027 → finals 15-16 Feb 2027.
> 13. **Suggested to have by 23 Nov**: chosen app/dataset/defect classes, working detector with P/R, Llama on ExecuTorch on device, vision→Llama hand-off with local log, offline run with FPS/latency/memory, GitHub repo, demo video, report draft.
>
> Nothing exists yet and the deadline is approaching, so follow the tiers and schedule in the addendum strictly: build Tier 1 first and keep the app always in a demonstrable state. Ask the preflight questions (main plan section 13, plus addendum section 9) in ONE message and wait. Then execute. After each milestone run its tests, update docs/STATUS.md, commit, and stop at HUMAN CHECKPOINTS with exact instructions. Never invent measurements, dataset statistics, API signatures or licence terms. If behind, apply the cut order in addendum section 7 without asking.

## 1. What is refined, and why

The first plan was a one-item-at-a-time inspector. Real lines singulate items (feeders, lanes, trays, blister packs) and image them at an inspection station; a pile of overlapping tablets cannot be inspected from above. The refined concept is a **tray station**:

> The camera sees a tray or lane holding several separated tablets or capsules. The app checks image quality, finds each item, inspects each item on its own, shows a verdict per item on the image, logs one row per item, and summarises the batch. Anything it cannot inspect safely (empty frame, blurred frame, touching items, partial items, foreign objects) gets an explicit status instead of a guess.

This is the way to "handle all test cases": the system never silently guesses. Every input maps to a defined output state (section 5 lists 27 cases).

Explicitly **out of scope** (say so in the report, do not build): continuous belt motion at line speed (needs strobe or line-scan hardware), overlapping or stacked piles, defects on the underside, transparent blister packs, products other than loose tablets and capsules, and rejection hardware (the app alerts and logs; it does not drive an air-jet).

Operating mode is **stop-and-shoot**: a tray arrives or is swapped, the scene becomes stable, the app inspects it once, and it will not inspect the same tray again until the scene has changed. This also prevents duplicate counting.

## 2. Revised architecture (replaces main plan section 4)

```
CameraX frame
  -> Stability trigger (frame difference low for ~0.5 s AND scene changed since last inspection)
  -> Quality gate (blur, exposure, glare)            -- fail -> BAD_IMAGE (no verdicts)
  -> Item finder (OpenCV)                            -- 0 items -> NO_ITEMS
        per blob: PASS-THROUGH, CLUSTER, PARTIAL, UNKNOWN_OBJECT, or OK
  -> For each OK item (sequential, batch size 1):
        crop + scale-normalise to 256x256 -> vision model -> anomaly map + score
        -> decision core (verdict, severity, confidence, location)
        -> defect type (rule-based in Tier 1, CNN in Tier 2) for flagged items
  -> Per-item overlay on the camera image (boxes, ids, colours) + one Room row per item
  -> Advisory queue (bounded, priority REJECT > RE-INSPECT) -> Llama 3.2 1B (ExecuTorch)
        -> validator -> text, or template fallback
  -> Batch summary (counts and rates; optional Llama "summarise this batch" on demand)
```

Verdicts per item: PASS, RE-INSPECT, REJECT. Item statuses that are not verdicts: CLUSTER, PARTIAL, UNKNOWN_OBJECT. Frame statuses: NO_ITEMS, BAD_IMAGE (reason: blur, dark, bright, glare). Any item that is not OK counts as "needs operator attention" in the batch summary; none of them may ever be shown as PASS.

### 2.1 Quality gate (plain Kotlin or OpenCV, unit-tested)

- Blur: variance of the Laplacian on the frame (or each crop). Threshold is a constant frozen before the real-item test; derive it from the variance of MVTec replay images and of those same images with synthetic Gaussian blur. Never tune it on real-item photos.
- Exposure: mean luma inside an allowed band; fraction of clipped pixels below a limit.
- Glare: saturated connected region larger than a set fraction of the frame.
- On failure: no verdicts, status BAD_IMAGE with the reason, message "re-capture: <reason>", one log row for the frame.

### 2.2 Item finder (OpenCV for Android, Apache-2.0; verify the Maven package and version when adding it)

1. Grayscale, light blur, Otsu threshold; choose polarity automatically from border pixels (background is the majority colour at the border). Morphological open then close. Connected components with statistics.
2. **Setup step** (button "Set reference item"): place one known-good item; store its area, aspect ratio and the item-to-crop scale. These live in app settings and in each log row's `reference_id`.
3. Classify each blob against the reference:
   - area within a set band around the reference (default plus or minus 35%) and solidity high: **OK**
   - area clearly above the band or solidity low (touching or overlapping): **CLUSTER**, not inspected
   - blob touches the frame border: **PARTIAL**, not inspected
   - area, aspect ratio or colour far from the reference, or wrong product mode: **UNKNOWN_OBJECT**, not inspected, flagged for the operator
4. Crop each OK item with a margin so the item's share of the 256x256 crop matches the MVTec training images of that category. Compute that target share from the training images at calibration time (`06_calibrate.py` writes `target_item_fraction`). If MVTec capsule orientation is not consistent, normalise by `minAreaRect` long axis; read the orientation from the dataset first.
5. Cap at 12 items per frame; extra items get status TOO_MANY with a warning "tray too full".
6. The fixture should use a matte background similar in tone to the MVTec background so crops look like training images.

### 2.3 Concurrency, throughput and the advisory queue

- Vision runs on one thread, item by item (export batch size 1). Per-frame time grows with the item count, so the benchmark reports latency per item and per frame at 1, 4 and 9 items.
- Llama never blocks vision or the UI. A single worker with a **bounded queue of 5**, priority REJECT then RE-INSPECT. Items that do not get an LLM slot receive template text with `advisory_source = template`, `advisory_reason = queue_full`. A generation that runs longer than a timeout (set from the measured 80-token time, doubled) is cancelled, template text is used, and the verdict is unaffected.
- PASS items never call Llama (template log line).
- "Summarise batch" button: sends aggregated counts and rates (never raw rows) to Llama on demand, for the inspection log summary requested in the brief. It uses the same validator rules (numbers must appear in the input).
- Throughput claim for the report: items per minute at one station, computed from measured per-item vision latency at 1, 4 and 9 items, plus advisory throughput measured separately. Do not claim conveyor line speed.

### 2.4 Log schema additions

Add to `inspections`: `frame_id`, `item_idx`, `n_items_in_frame`, `status` (OK, NO_ITEMS, BAD_IMAGE, CLUSTER, PARTIAL, UNKNOWN_OBJECT, TOO_MANY), `status_reason`, `advisory_reason`, `reference_id`, `queue_wait_ms`. Frame-level events (NO_ITEMS, BAD_IMAGE) get one row with `item_idx` null.

### 2.5 UI additions

- Per-item boxes on the preview (green, amber, red, grey for not inspected) with item numbers; tapping an item opens its card (type, severity, location, confidence, advice).
- Batch banner: "9 items: 7 PASS, 1 RE-INSPECT, 1 REJECT" plus "2 need attention" when statuses are not OK.
- Frame-status banner for NO_ITEMS and BAD_IMAGE with the exact reason and what to do.
- "Set reference item" button and "Summarise batch" button; stability auto-trigger toggle; manual Capture still available.

## 3. Revised vision model (replaces main plan section 6.2)

Time does not allow a long training run plus an export gamble. **Primary: PatchCore-lite**, which needs no gradient training and exports as a plain CNN plus a matrix operation.

- Backbone: ResNet-18, ImageNet-pretrained (torchvision weights; record the licence), features from layer2 and layer3, local 3x3 average aggregation, upsample layer3 to layer2 size, concatenate, reduce with a fixed random projection to 128 dimensions.
- Memory bank: patch embeddings of `train_good` (the split in main plan 6.1), reduced by greedy coreset sampling to a few thousand patches. Choose the bank size from a measured accuracy-versus-latency curve on the phone (try a few sizes; record them). Do not assume a size.
- Wrapper `nn.Module`: input 1x3x256x256 in 0..1 (ImageNet normalisation inside), output anomaly map (nearest-neighbour distance per patch via a distance matrix operation, upsampled and smoothed with a fixed Gaussian convolution) and scalar score (map maximum). Bank and normalisation constants stored as buffers so the exported file is self-contained.
- Export with `torch.export` then XNNPACK lowering to a `.pte`. Parity test against PyTorch as in the main plan (score within 2% relative, AUROC on `eval` within 1 point).
- Gate G1 (Wed 18 Nov, half a day): `.pte` loads on the phone, golden image matches, latency recorded. If the ExecuTorch export fails by midday, use ONNX Runtime Mobile or LiteRT for vision and record it in docs/DECISIONS.md; Llama stays on ExecuTorch regardless. Email tech-queries tonight to confirm that other runtimes are acceptable for the vision model.
- Stretch (only if everything in Tier 1 and 2 is done): EfficientAD small via anomalib for a comparison row. Not required.
- The model has not been measured on this phone. All accuracy and latency figures come from the scripts, never from this document.

## 4. Defect type and severity under time pressure

- **Tier 1 (rule-based, labelled "indicative" in the UI and report):** from the anomaly blob: elongated and thin leans "crack or scratch", compact with strong colour deviation leans "contamination or colour fault", large area leans "deformation". Features: blob aspect ratio, area share, mean colour difference in Lab against the item's own median colour. Output a type label plus the feature values that justified it.
- **Tier 2:** MobileNetV3-small classifier trained per main plan 6.3 (5-fold out-of-fold reporting), replacing the rules where it exists. Report both if both exist.
- Severity: main plan 6.4. Because tertile severity is relative to the dataset, use fixed absolute area cut points instead if the human prefers (decide at M3 and note it in DECISIONS.md).

## 5. Test-case matrix (the system must produce these defined outputs)

Implement as a table-driven suite. Frames are generated by `tools/make_test_frames.py` from MVTec test images (tiling into grids on a matched background, Gaussian blur, gamma changes, saturated blobs, offsets to make items touch, crops at the border, synthetic shapes for foreign objects). Mark every generated frame "synthetic" in the report. The suite writes `results/testcases.csv` (case id, input, expected, observed, pass or fail), and the report includes it as a table.

| ID | Scenario | How produced | Expected output |
| --- | --- | --- | --- |
| F1 | Empty tray | plain background frame | NO_ITEMS, no verdict, hint "place items" |
| F2 | Blurred frame | Gaussian blur above the gate | BAD_IMAGE (blur), no verdicts |
| F3 | Too dark / too bright | gamma shifts | BAD_IMAGE (dark or bright) |
| F4 | Glare patch | saturated blob over the tray | BAD_IMAGE (glare) |
| F5 | Camera permission denied | instrumented test | blocking screen with instructions; Replay still works |
| F6 | Llama files missing | rename model | vision verdicts work; advice is template with banner "advisory model unavailable" |
| F7 | Vision model missing | rename model | clear error, no inspection |
| I1 | One good item | golden image | verdict per calibrated model (golden expected score within tolerance) |
| I2 | One defective item | golden image | RE-INSPECT or REJECT per calibrated model |
| I3 | 2x2 and 3x3 mixed trays | tiled MVTec images | correct item count; per-item verdicts agree with single-item runs; report the agreement rate and investigate any flip |
| I4 | Touching items | offset tiles until blobs merge | CLUSTER, not inspected, counted as "needs attention" |
| I5 | Item cut by frame edge | shifted tile | PARTIAL, not inspected |
| I6 | Foreign object | synthetic shape | UNKNOWN_OBJECT, never PASS |
| I7 | More than 12 items | large grid | first 12 inspected, rest TOO_MANY, warning shown |
| I8 | Wrong product mode | capsule image in tablet mode | UNKNOWN_OBJECT |
| I9 | Mixed tray with one cluster and several good items | composite | good items inspected, cluster flagged, batch banner shows attention count |
| I10 | Defect near threshold | calibration images | RE-INSPECT band behaves as calibrated |
| A1 | Llama output malformed | injected string | validator rejects; template; `advisory_source = template` |
| A2 | Llama too slow | forced delay | timeout, template, verdict unchanged |
| A3 | Advisory queue full | 9 flagged items at once | first 5 queued by priority, rest template with `queue_full` |
| A4 | Invented number in output | injected string | validator rejects |
| A5 | Output contradicts verdict | injected string | validator rejects |
| S1 | Stress replay | frames at maximum rate for 5 minutes | no crash, bounded queue, flat memory trend, drops logged |
| S2 | Thermal status moderate or worse | log and UI check during the long run | warning shown, event in CSV |
| S3 | Airplane mode | manual | full flow works, no network permission present |
| S4 | App killed mid-inspection | instrumented test | database consistent on restart, no half rows |
| D1 | Same tray twice | repeat the same frame | exactly one set of rows until the scene changes |

Real-item tests (human, main plan `docs/real_items_protocol.md`) cover I1, I2, I3 and the fixture lighting on actual tablets or capsules, with the final gate constants frozen first.

## 6. Tiers

- **Tier 1 (must exist by Thu 19 Nov night; the app must always run):** Llama 3.2 1B on ExecuTorch with validator and template fallback; PatchCore-lite on pill; quality gate; item finder with single and multi-item loop and cluster, partial, unknown handling; decision core; rule-based type; Room log with CSV export; per-item overlay; offline (no network permission); replay mode; core benchmarks; README; report v1.
- **Tier 2 (Fri 20 to Sat 21):** test-case suite and mosaics; advisory queue limits and batch summary; type CNN; 50-case advisory test; 30-minute battery and thermal run; real-item test; polished UI.
- **Tier 3 (only if ahead):** capsule category, EfficientAD comparison, external Roboflow check.

## 7. Schedule and cut order (replaces main plan sections 7 and 8)

| Day | Work | Gate |
| --- | --- | --- |
| Tue 17 Nov (today) | Preflight answers; repo scaffold, VERSIONS.md, licence files; HUMAN: accept Llama licence and download weights, download MVTec pill; Colab: build the PatchCore-lite bank for pill; Llama export and phone smoke test | G0: Llama answers on the phone, numbers recorded |
| Wed 18 Nov | Vision export (half-day box), parity test; Android skeleton: CameraX, quality gate, item finder, vision per crop; decision core and calibration | G1: vision model runs on the phone and matches Python |
| Thu 19 Nov | Room log, History, per-item overlay, Llama integration, validator, template fallback, advisory queue; calibrated thresholds shipped; report sections written as features land | G2: Tier 1 complete and demonstrable (tag v0.9) |
| Fri 20 Nov | Test-case suite and mosaic generator; replay and stress mode; Tier 2 items; 50-case advisory run on device | Test matrix runs, results/testcases.csv produced |
| Sat 21 Nov | Benchmarks (30-minute run, or 15 minutes if behind), real-item test, **code freeze at 22:00 IST** | G3: results CSVs complete; release APK builds from a fresh clone |
| Sun 22 Nov | Final report, demo video, mentor skim, **submit by 18:00 IST** | Confirmation saved |
| Mon 23 Nov | Buffer only; confirm the submission is received | |

Cut order if behind (apply without asking): Tier 3 items; type CNN (keep rules); batch summary via Llama; 30-minute run shortened to 15; real-item test reduced to what fits (state the count); mosaics reduced to 2x2 only. Never cut: Llama on ExecuTorch, logging, offline operation, the quality gate and cluster handling, honest benchmarks, licence notices, the report.

## 8. Additional report content

- Section "Scope and operating mode": tray station, stop-and-shoot, the out-of-scope list in section 1.
- Table: the 27 test cases with observed results (from `results/testcases.csv`).
- Throughput table: items per minute at 1, 4 and 9 items per frame; advisory queue behaviour under load; the statement that line-speed conveyor inspection needs strobe or line-scan hardware and is future work.
- Runtimes table (both pipelines documented separately), as in the main plan.
- Section "Hardware choice": why Android was chosen over Raspberry Pi 5 (official video lists both as valid targets); acknowledge the Hailo AI HAT+ NPU option for vision acceleration and explain why it was not used (no hardware available).
- Section "Alternative approaches considered": YOLO / EfficientDet for detection (from official video); why anomaly detection (PatchCore-lite) was preferred for the few-defect-sample pharma setting.
- Section "Future work" additions: Raspberry Pi 5 port (with and without Hailo), NEU Surface Defect and KolektorSDD datasets for multi-domain generalisation testing, YOLO/EfficientDet comparison.

## 9. Extra preflight questions (add to main plan section 13)

1. Has the submission form arrived? What exactly does it ask (repo link, video link, report upload), and what is the exact deadline time and time zone?
2. Did the doubt-clearing session happen? Did it confirm whether vision may use a runtime other than ExecuTorch, the demo video length and the report format?
3. Has anything been built or downloaded yet (Llama weights, MVTec, Android Studio set up, Colab access)?
4. What background colour and surface will the real-item fixture use? (Match the MVTec background tone.)
