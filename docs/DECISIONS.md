# Decisions

All fallback and deviation decisions are recorded here with date and reason.

## 2026-11-22

- Android app foundation: use AGP 9.5.2 / Gradle 9.9.2, built-in Kotlin with Compose compiler 2.4.21, CameraX 1.7.0, Room 2.8.5, OpenCV 4.13.0, ONNX Runtime Android 1.23.2, and ExecuTorch 1.5.1. Versions were checked against published metadata/POMs; Gradle distribution checksum was verified.
- Compose BOM 2026.11.01 required compile SDK 37.1+, but only Android platform 37.0 is installed. Pin BOM 2026.06.00, whose Compose UI AAR metadata requires compile SDK 35, instead of installing an additional SDK platform.
- The first Android screen is intentionally a camera-preview shell. It must not issue PASS/RE-INSPECT/REJECT until vision inference and thresholds derived from the fixed calibration protocol are available. Llama access is accepted, but the checkpoint and tokenizer are not downloaded yet.
- Updated 2026-11-22 after rereading the modified addendum: only Llama must run on ExecuTorch; a vision runtime such as ONNX Runtime or TFLite is acceptable. Keep the two runtime pipelines documented separately.
- Apply the video FAQ: decision rules may set verdicts; Llama only advises; no fine-tuning is required; confidence is not severity; report precision, recall, F1, FPS, memory, power, and Llama TTFT/tokens/s/response time; use and test a structured JSON vision-to-advisory interface; static jig/trigger capture is sufficient.
- Severity labels will use Low/Medium/High and depend on type, size, and location, per the updated FAQ. Do not reuse the main plan's Minor/Major/Critical labels without reconciling implementation and tests.
- Official submission date from the updated addendum is 2026-11-23; exact submission time/time zone remain unconfirmed because the form has not arrived.
- Pill archive supplied in the workspace and extracted for the required pill-only Tier 1 work. Capsule archive retained but not extracted, per the cut order.
- Test `calib`/`eval` split is stratified independently by each class. Counts are generated into `results/dataset_stats.json` from the extracted files.
- The fixture background should be matte near-black/dark charcoal, based on visual inspection of MVTec pill train images. The exact RGB target is still to be derived from image pixels.
- ExecuTorch Python/AAR 1.5.1 is pinned. The 1.5.1 Android source JAR and Meta's `llm/android/LlamaDemo` `ChatViewModel.kt` were inspected: `LlmModule` has a constructor taking model type/path/tokenizer/temperature/data files; `generate(prompt, config, callback)`, `resetContext()`, `stop()`, `load()`, and `close()` are present; `LlmGenerationConfig` exposes maxNewTokens, seqLen, temperature, numBos and numEos builders. App integration has not yet been compiled.
- Candidate memory banks of 256, 1024, and 4096 patches are experimental only. No final bank size is selected until phone latency and accuracy are measured.
- The 256-bank vision `.pte` exported, but the ExecuTorch Windows runtime also fails to execute a minimal XNNPACK Add program with `tensor does not have a device`. Since the updated FAQ explicitly allows another vision runtime, select ONNX Runtime 1.23.2 for vision rather than treating a Windows runtime failure as a model-operator failure. Keep Llama on ExecuTorch 1.5.1.
- ONNX export and ONNX Runtime parity passed on five calibration images using the scripted 2% score/0.99 map-correlation checks (`results/parity_pill_bank256_onnx.csv`). This is not phone parity, does not include `eval`, and does not select a final bank size. G1 remains blocked pending device testing.
- ResNet-18 ImageNet weight/data-use terms are unverified. Keep derived checkpoints and `.pte` artifacts local until reviewed; do not make a release claim.
- Hardware target: Android (Nothing Phone 3a) chosen over Raspberry Pi 5. Rationale: the phone has a built-in camera, higher-performance SoC (Snapdragon 7s Gen 3), more RAM, and is the hardware available to us. The official video (2026-11-13, Charitha Balivada) lists both RPi 5 and Arm-based Android devices as valid targets; CPU-only is scored equally if execution is good (FAQ 2 Q2). We document this choice explicitly.
- Vision runtime: does NOT need ExecuTorch. Official FAQ 1 Q2 confirms "ONNX Runtime, TFLite, HailoRT or ExecuTorch are all fine for vision. Only the Llama part is tied to ExecuTorch." This resolves the tech-queries question from the main plan. Our fallback ladder (ExecuTorch → ONNX Runtime Mobile) is explicitly sanctioned.
- Hailo AI HAT+ / HAT+ 2 NPU: not used (FAQ 1 Q3: "all are optional"). We do not have Hailo hardware. Vision runs on the phone CPU via ExecuTorch or fallback runtime. Documented in the report under "alternative approaches considered".
- Llama fine-tuning: not required. FAQ 1 Q4 confirms "A pretrained Llama 3.2 1B or 3B Instruct with well-designed structured prompts is enough to start." Our structured prompt approach is correct.
- Pass/fail rules: confirmed rule-based is acceptable. FAQ 3 Q4: "Yes, and it is sensible. Rules decide; Llama explains and advises. It must still be genuinely integrated." This validates our decision-core design.
- Severity definition: FAQ 3 Q3 confirms "Set Low, Medium, High from type, size and location, and document it. Confidence is not severity." Our tertile-based severity from mask area is consistent with this.
- JSON hand-off between modules: FAQ 4 Q5 recommends "Agree one JSON hand-off: defect, severity, confidence, box. Test each interface alone; integrate early."
- No prescribed FPS target or tokens/s target: FAQ 4 Q1 and Q6 confirm we report what we measure and justify it. No minimum threshold.
- Phase timeline confirmed: Phase 1 submission 23 Nov 2026, shortlist mid-Dec 2026, Phase 2 PoC 5 Feb 2027, finals 15-16 Feb 2027.
- Additional datasets noted from official video: NEU Surface Defect (steel, 6 classes) and KolektorSDD (surface cracks). FAQ 3 Q2 says "None is supplied. MVTec AD, NEU and KolektorSDD are optional. Mixing in your own images is fine; test on your camera." Added to report "future work".
- Alternative vision models noted: YOLO / EfficientDet are listed as optional detection models (FAQ 1 Q3). Our approach (anomaly detection via PatchCore-lite) is better suited to the few-defect-sample pharma setting. Mentioned in report "alternative approaches".

## 2026-11-19

- Starting from no existing code or downloaded assets.
- Own-code licence preference: Apache-2.0.
- Addendum v2 takes precedence over conflicting implementation-plan text.
- Vision baseline: PatchCore-lite for the pill category.
- Target mode: tray station, stop-and-shoot.
- Human-dependent assets are not fabricated or downloaded without licence acceptance.

## 2026-11-22 (afternoon)

- Calibration (`ml/scripts/06_calibrate.py`) uses `val_good` + `calib` only, per plan 6.4: t_reject = 95th percentile of good scores (≤5% false alarms), t_pass = 5th percentile of defective calib scores (≤5% misses), t_pix maximises pixel F1 on calib masks, severity cuts = tertiles of predicted defect area on calib defects, confidence scale = median distance of calib scores to the nearest threshold.
- Calib AUROC: bank 256 = 0.894, 1024 = 0.946, 4096 = 0.949. Bank 4096 selected as default because it has the best calib AUROC and measured phone inference is ~50 ms/item (median, replay), which is acceptable for stop-and-shoot. Banks 256/1024 remain in the APK for the latency comparison.
- Severity uses Low/Medium/High from calibrated defect-area tertiles (size), bumped one level for crack/scratch and contamination/colour types (type, illustrative SOP). Location is reported but does not change severity; documented as a limitation. Severity is only assigned to flagged items (score ≥ t_pass); t_pix (3.02) > t_pass (2.91) for bank 4096, so PASS items have no defect pixels.
- Frame gate constants from MVTec pill train_good at 256 px: p99 luma 189–198 → DARK if p99 < 95; p1 luma 6–17 → BRIGHT if p1 > 70; saturated fraction 0 → GLARE if > 0.5% ≥ 250. Blur per crop: Laplacian variance sharp min 36.1 vs Gaussian σ=3 max 1.98 → threshold 8.46 (geometric mean). Frozen before any real-item test.
- DARK is evaluated after the item finder: an empty dark fixture is legitimately dark and must be NO_ITEMS (test F1); a dark frame containing items is BAD_IMAGE(dark).
- Item finder rule order: PARTIAL (touches edge) → colour distance > 60 (UNKNOWN_OBJECT) → area > 1.35× ref or solidity < 0.90 (CLUSTER) → area < 0.65× ref or aspect deviation > 35% (UNKNOWN_OBJECT). Colour precedes cluster so a different product (capsule, I8) is UNKNOWN rather than CLUSTER. Changed after the first synthetic test-case run; constants were not changed.
- Known limitation: a dark foreign object on the dark fixture is not segmented by Otsu and is therefore invisible (test I6b). It is never marked PASS, but it is also not flagged.
- Item crop: rotate long axis to horizontal (MVTec orientation), square crop so the item's long side is 86.7% of the crop (median in MVTec train), INTER_AREA resize to 256. Training used PIL bilinear-antialias resize; the device/PyTorch difference is measured by the golden test.
- Llama export config verified against executorch 1.5.1 `LlmConfig`: 8da4w, group 32, 4-bit embeddings, fp32 (docs recommend fp32 for XNNPACK), max_seq_length 1024. Runner called with numBos=0 because the prompt writes `<|begin_of_text|>` once (unit-tested).
- Advisory: bounded priority queue (5, REJECT before RE-INSPECT), PASS never calls Llama, timeout = 2× measured median generation time (default 30 s until measured), validator per plan 6.6; every fallback is logged with `advisory_source=template` and a reason.

## 2026-11-22 (evening)

- Llama export: the default HQQ scale search (`quantization.use_hqq: true` in executorch 1.5.1) crashed twice with a native access violation inside torchao `round_det` while quantizing (logs `results/llama/export_log*.txt`), on the 7.2 GB RAM build laptop. Set `use_hqq: false`, so 8da4w uses affine per-group (group 32) min/max scales computed in one pass. Same bit-widths and group size; HQQ usually gives slightly lower quantization error, so advisory quality numbers are for the affine variant. The validator and template fallback stay in place either way.
- Phone reports its USB cable as AC power, so `stay_on_while_plugged_in` was set to 7 (any charger) for benchmarking; reset at the end.
- ExecuTorch 1.5.1 on Windows looks for a resource named `flatc` but ships `executorch/data/bin/flatc.exe`, so XNNPACK serialization failed at the last export step. Fixed without code changes by setting `FLATC_EXECUTABLE` to the shipped `flatc.exe` (24.3.25); `ml/scripts/export_llama.ps1` sets it.
- Final-configuration device run (`results/device/replay_20261122_212513_bank4096.csv`): eval metrics identical to the pre-fix run (finder rule-order and dark-gate changes do not affect single-item MVTec frames). Tray verdicts agree with single-image verdicts for 12/13 items; the one disagreement is a near-threshold item (tray 3.23 vs single 3.45, t_reject 3.39).
- Test case I8 (capsule) still fails: the capsule's mean colour is within the colour tolerance, so it is caught by the area rule (2.0x reference) and reported CLUSTER. It is never PASS and is flagged for attention, but it is not identified as a wrong product. Not tuned further, to avoid fitting the rules to the test set; documented as a limitation.
- Golden parity (device whole-image path vs PyTorch): median relative score difference 1.4%, max 3.4% on 5 images. Explains I10.1 (device 3.64 vs torch band score just under t_reject): near-threshold items can move one verdict band between preprocessing paths; RE-INSPECT exists for this band.
- Vision latency on device (200 timed iterations, 4 threads): inference median 38/39/51 ms for banks 256/1024/4096; post-processing (anomaly analysis, typing, heatmap) median ~40 ms, comparable to inference. Full-pipeline replay throughput 9.5 frames/s single-item frames (bank 4096).

## 2026-11-23

- Advisory prompt v1 (seed 2026, 50 cases): 26/50 format-valid. TTFT median 1.05 s, decode 18 tok/s, total median 5.0 s (p95 12 s, max 37.7 s, up to 590 tokens although maxNewTokens = 110). Used as the diagnostic set.
- Prompt v2: ACTION pre-filled with the verdict verb from the decision core (Reject / Re-inspect / Release + item number), one worked example, 15-word target, removal of the "verdict was already decided" sentence the model kept paraphrasing. Generation is stopped by the app once a LOG line is complete or maxNewTokens is reached; only the first ACTION..LOG block is used.
- v2 reported on unseen seed 2027: 50/50 format-valid; TTFT median 2.09 s (longer prompt), total median 4.99 s, p95 5.32 s, decode 17 tok/s, PSS ~1.07 GB with Llama loaded. Manual review of all 50: 47 consistent with the record; 3 RE-INSPECT answers invented the re-inspection result ("found no issues").
- Post-hoc validator rule: RE-INSPECT answers that claim an outcome (found / no issues / cleared / passed / ...) are rejected and fall back to the template. Added after seeing the seed-2027 answers; it catches exactly those 3 and was not re-measured on a fresh seed. Reported as such.
- Long run 15 min (vision + advisory, replay, phone charging): 3041 frames; throughput fell from ~4.0 to ~3.2 frames/s; battery temperature 31 -> 42 C; Android thermal status reached 3 (severe) after ~11 min. Advisory queue stayed at its cap (5) because replay feeds frames far faster than Llama explains them; bounded queue and REJECT-first priority held. Battery stayed at 100% (charging), so no drain figure.

## 2026-11-25

- Glare gate changed after the first real-camera test: a white paracetamol tablet's own surface saturates on the phone camera (MVTec pills never do), so the whole-frame saturated-pixel rule flagged every real tablet as glare. Glare is now measured only outside items that match the reference (OK blobs, box padded by 10%). Glare on the tray, or glare merged with an item (which then fails the shape/colour checks), is still caught. Threshold unchanged (0.5% of the 256-px gate image).

- Second product (paracetamol) taught from 16 phone photos of 2 real tablets (`paracetamol_data/`, not in Git), `ml/scripts/07_new_product.py`: tablets cropped exactly like the app, labelled per tablet (good scored face; ink, broken halves, halves rejoined as defects; one blurred photo excluded; one photo skipped because the tablets touched). Training uses augmented good views from 6 photos only (205 crops); 2 good views (32 crops with augmentation) are held out. Held-out good scores 2.00 to 3.48, defect scores 7.70 to 11.88, so t_pass = 3.48 (highest held-out good), t_reject = 7.70 (lowest defect), RE-INSPECT between. One physical good tablet only: this is a proof of concept, not a measured accuracy. Blur gate is effectively off for this product (a plain white tablet has almost no texture). The app loads extra products from `files/products/<name>/` and switches on the Bench tab.
- Paracetamol t_pass raised from 3.48 to 5.0 after 19 live phone-camera captures of the good tablet scored 3.51 to 4.40 (all RE-INSPECT). The live CameraX frame differs from the WhatsApp-compressed training photos. Defect scores start at 7.70, so t_reject is unchanged.
