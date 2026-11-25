# PS5 Pharma Inspector

Offline tablet-inspection station on an Android phone. Built with Llama.

- **Vision:** PatchCore-lite (ResNet-18 + coreset memory bank) on ONNX Runtime Android: finds every tablet on a
  dark tray, scores it, and marks the defect area.
- **Decision:** calibrated rules give PASS / RE-INSPECT / REJECT, severity (Low/Medium/High) and confidence.
- **Advisory:** Llama 3.2 1B Instruct on ExecuTorch 1.5.1 (XNNPACK) explains flagged items in four lines
  (ACTION, REASON, CHECK, LOG). It gets a JSON record, never pixels, and cannot change the verdict.
- **Offline:** the app has no `INTERNET` permission. Every inspection is logged locally (Room) and exportable
  as CSV.

Results, method and limitations: [report/report.md](report/report.md). Decision log: [docs/DECISIONS.md](docs/DECISIONS.md).

## Repository layout

| Path | Contents |
|---|---|
| `android/` | Kotlin + Compose app (camera, replay, history, bench, about) |
| `ml/scripts/` | Split, training, ONNX export, calibration, Llama export |
| `ml/configs/llama3_2_1b_xnnpack.yaml` | ExecuTorch Llama export config |
| `tools/` | Synthetic test frames, device result scoring and summaries |
| `bench/make_tables.py` | Regenerates every table in the report from `results/` |
| `results/` | All measured results (device CSVs in `results/device/`) |

## Requirements

- Windows 11 (tested), Python 3.10, PyTorch 2.12.1, ExecuTorch 1.5.1, ONNX Runtime 1.23.2 (see `VERSIONS.md`)
- Android Studio with SDK platform 37; an arm64 Android phone (tested: Nothing Phone (3a), Android 17)
- MVTec AD `pill` (download and accept the licence yourself) and access to `meta-llama/Llama-3.2-1B-Instruct`

## 1. Build and install the app

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
cd android
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

The calibrated thresholds (`android/app/src/main/assets/config/`) are in Git. The vision models
(`assets/vision/patchcore_pill_{256,1024,4096}.onnx`, 11-13 MB each, derived from MVTec AD) are not: build
them with step 3 before building the APK, or install the prebuilt APK from the GitHub release, which
contains them.

## 2. Llama model on the phone

1. Download the Meta checkpoint (`original/` folder) to `ml/checkpoints/llama-3.2-1b-instruct/`:
   ```powershell
   python -c "from huggingface_hub import login, snapshot_download; login(); snapshot_download('meta-llama/Llama-3.2-1B-Instruct', allow_patterns=['original/*'], local_dir='ml/checkpoints/llama-3.2-1b-instruct')"
   ```
2. Export (about 40 minutes and 7+ GB RAM/pagefile on a laptop CPU):
   ```powershell
   powershell -File ml\scripts\export_llama.ps1    # Windows: sets FLATC_EXECUTABLE
   # Linux/macOS: sh ml/scripts/export_llama.sh
   ```
   The `.pte` (about 820 MB) is written to the current directory; move it to `ml/checkpoints/llama_pte/`.
3. Push to the app's folder (no root needed):
   ```powershell
   $d = "/sdcard/Android/data/com.example.ps5/files/models"
   adb push ml\checkpoints\llama_pte\llama3_2_1b_instruct_8da4w_g32_emb4_1024.pte "$d/llama.pte"
   adb push ml\checkpoints\llama-3.2-1b-instruct\original\tokenizer.model "$d/tokenizer.model"
   ```
   Without these files the app still works and uses template advice (shown as "template" in the log).

## 3. Reproduce the vision model and calibration

```powershell
python ml/scripts/00_prepare_splits.py --data-root ml/data --output ml/splits/pill.json --category pill --seed 42
python ml/scripts/01_train_patchcore_lite.py --data-root ml/data --split ml/splits/pill.json --output-dir ml/checkpoints/pill --bank-sizes 256,1024,4096
python ml/scripts/02_export_vision.py --checkpoint ml/checkpoints/pill/patchcore_pill_bank_4096.pt --output ml/checkpoints/pill/patchcore_pill_4096.onnx
python ml/scripts/03_parity_test.py --checkpoint ml/checkpoints/pill/patchcore_pill_bank_4096.pt --onnx ml/checkpoints/pill/patchcore_pill_4096.onnx --split ml/splits/pill.json --data-root ml/data --output results/parity_pill_bank4096_onnx.csv
python ml/scripts/06_calibrate.py --data-root ml/data --split ml/splits/pill.json --checkpoints ml/checkpoints/pill/patchcore_pill_bank_256.pt ml/checkpoints/pill/patchcore_pill_bank_1024.pt ml/checkpoints/pill/patchcore_pill_bank_4096.pt --output-dir results/calibration
python -m pytest -q
```

Copy the ONNX files to `android/app/src/main/assets/vision/` and `results/calibration/thresholds_pill_bank*.json`
to `android/app/src/main/assets/config/`. The same steps are in `ml/colab/train_and_export.ipynb`.

## 4. Use the app

- **Inspect:** place tablets on a matte dark tray in even light, phone fixed above. Tap *Set reference item*
  with one good tablet in view, then *Capture* or switch on *Auto* (captures when the scene is still).
- **Replay:** runs images from `/sdcard/Android/data/com.example.ps5/files/replay/` through the same pipeline;
  *Run all → CSV* writes `files/results/replay_*.csv`.
- **History:** every item with verdict, severity, advice and its source (llm / template + reason); *Export CSV*.
- **Bench:** vision latency, golden parity, FPS, memory, 50-case advisory test, 15-minute long run.

## 5. Reproduce the device results

```powershell
python tools/make_test_frames.py --data-root ml/data --split ml/splits/pill.json --calib-scores results/calibration/calib_scores_pill_bank4096.csv --thresholds results/calibration/thresholds_pill_bank4096.json --capsule capsule.tar.xz --out replay/testcases
# push replay/ (eval images, testcases, golden) to files/replay/, run Replay > Run all and the Bench buttons, then:
adb pull /sdcard/Android/data/com.example.ps5/files/results/. results/device/
python tools/eval_device.py --replay results/device/<replay csv> --out results
python tools/summarize_device.py
python bench/make_tables.py
```

## Licences

Code: Apache-2.0. MVTec AD: CC BY-NC-SA 4.0 (non-commercial; images not included). Third-party components:
[THIRD_PARTY.md](THIRD_PARTY.md).

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights
Reserved. Use of the model must follow the Llama 3.2 Acceptable Use Policy. Model weights and the `.pte` are not
distributed in this repository.

Advice produced by the app is illustrative engineering guidance, not GMP or regulatory advice.
