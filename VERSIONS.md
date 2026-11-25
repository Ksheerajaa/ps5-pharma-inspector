# Pinned versions

| Component | Pinned version | Source |
|---|---|---|
| ExecuTorch exporter and Android AAR | 1.5.1 | Latest shared Maven Central artifact for `org.pytorch:executorch-android`; Python package version available |
| ONNX | 1.20.1 | Active Python environment; used for static export |
| ONNX Runtime Python | 1.23.2 | Active Python environment |
| ONNX Runtime Android | 1.23.2 | Maven Central artifact, verified from POM |
| Android Gradle Plugin | 9.5.2 | Google Maven stable artifact |
| Gradle | 9.9.2 | Official Gradle distribution, checksum verified |
| Kotlin Compose plugin | 2.4.21 | Maven Central stable artifact; AGP built-in Kotlin |
| KSP | 2.3.14 | Maven Central stable artifact; Room compiler |
| Compose BOM | 2026.06.00 | Google Maven stable artifact; 2026.11 requires compile SDK 37.1+ |
| AndroidX Activity Compose | 1.13.0 | Google Maven stable artifact |
| AndroidX Lifecycle Runtime Compose | 2.11.1 | Google Maven stable artifact |
| CameraX | 1.7.0 | Google Maven stable artifacts |
| Room | 2.8.5 | Google Maven stable artifacts |
| OpenCV Android | 4.13.0 | Maven Central AAR; POM reports Apache-2.0 |
| PyTorch | 2.12.1+cu126 | Active Python environment |
| torchvision | 0.27.1+cu126 | Active Python environment |
| anomalib | TODO | Stretch only |
| Android Studio | Rabbit 2, 2026.2.2 Patch 1 | Installed Windows installer |
| Android SDK Platform | 37.0 | Installed SDK platform |
| Android Build Tools | 36.0.0 | Installed SDK build tools |

Workspace observations:
- OS: Windows 11
- Python: 3.10.0 in the active environment (user-reported 3.13 differs)
- Local GPU: NVIDIA GeForce RTX 2050, 610.91 driver, 4096 MiB
