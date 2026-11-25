# Third-party licences

| Component | Version | Licence | URL | Where used |
|---|---|---|---|---|
| MVTec AD | `pill` and `capsule` archives as downloaded by the user (archive version not recorded) | CC BY-NC-SA 4.0 | https://www.mvtec.com/company/research/datasets/mvtec-ad | Pill/capsule training and evaluation |
| ExecuTorch Android | 1.5.1 | BSD 3-Clause | https://github.com/pytorch/executorch/blob/main/LICENSE | Llama advisory runtime |
| ExecuTorch Python | 1.5.1 | BSD 3-Clause | https://github.com/pytorch/executorch/blob/main/LICENSE | Model export |
| PyTorch | 2.12.1 | BSD 3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE | PatchCore-lite |
| torchvision | 0.27.1 | BSD 3-Clause | https://github.com/pytorch/vision/blob/main/LICENSE | ResNet-18 backbone and preprocessing |
| Pillow | 12.2.0 | MIT-CMU | https://github.com/python-pillow/Pillow/blob/12.2.0/LICENSE | Reading MVTec images for preprocessing |
| pytest | 8.2.2 | MIT | https://github.com/pytest-dev/pytest/blob/8.2.2/LICENSE | Python unit tests |
| ResNet-18 ImageNet-1K weights | `ResNet18_Weights.IMAGENET1K_V1` | TODO: verify weight/data-use terms | https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html | PatchCore-lite backbone |
| ONNX | 1.20.1 | Apache-2.0 | https://github.com/onnx/onnx/blob/v1.20.1/LICENSE | Portable vision model format |
| ONNX Runtime Python/Android | 1.23.2 | MIT | https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.pom | Vision parity and Android inference |
| Android Gradle Plugin | 9.5.2 | Apache-2.0 | https://developer.android.com/build/releases/gradle-plugin | Android build |
| Gradle | 9.9.2 | Apache-2.0 | https://gradle.org/licensing/ | Android build |
| Kotlin Compose plugin | 2.4.21 | Apache-2.0 | https://github.com/JetBrains/kotlin/blob/master/license/README.md | Compose compiler |
| KSP | 2.3.14 | Apache-2.0 | https://github.com/google/ksp/blob/main/LICENSE | Room code generation |
| Jetpack Compose BOM/UI/Material 3 | 2026.06.00 | Apache-2.0 | https://developer.android.com/jetpack/compose | Operator UI |
| AndroidX Activity Compose | 1.13.0 | Apache-2.0 | https://developer.android.com/jetpack/androidx/releases/activity | Activity integration |
| AndroidX Lifecycle Runtime Compose | 2.11.1 | Apache-2.0 | https://developer.android.com/jetpack/androidx/releases/lifecycle | Lifecycle-aware camera binding |
| AndroidX CameraX | 1.7.0 | Apache-2.0 | https://developer.android.com/training/camerax | Camera preview and capture |
| Room | 2.8.5 | Apache-2.0 | https://developer.android.com/jetpack/androidx/releases/room | Inspection log |
| OpenCV Android | 4.13.0 | Apache-2.0 | https://repo.maven.apache.org/maven2/org/opencv/opencv/4.13.0/opencv-4.13.0.pom | Item finder |
| Llama 3.2 1B Instruct | `meta-llama/Llama-3.2-1B-Instruct` `original/` checkpoint, downloaded 2026-11-22; exported `.pte` sha256 in `results/llama/pte_sha256.txt` | Llama 3.2 Community Licence + Acceptable Use Policy (copies in `licenses/`; attribution in `NOTICE`) | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct | On-device advisory |

| huggingface_hub | as installed | Apache-2.0 | https://github.com/huggingface/huggingface_hub/blob/main/LICENSE | Downloading the Llama checkpoint |
| FlatBuffers `flatc` (shipped in ExecuTorch) | 24.3.25 | Apache-2.0 | https://github.com/google/flatbuffers/blob/master/LICENSE | `.pte` serialization during export |

| Outfit font | Google Fonts (variable) | SIL Open Font License 1.1 (`report/fonts/OFL-Outfit.txt`) | https://fonts.google.com/specimen/Outfit | Report headings |
| Arimo font | Google Fonts v36 | Apache-2.0 | https://fonts.google.com/specimen/Arimo | Report body text and charts |
| Matplotlib | 3.10.8 | Matplotlib licence (PSF-based) | https://matplotlib.org/stable/project/license.html | Report charts |

No third-party licence text or terms are invented in this file. The Llama licence and Acceptable Use Policy in `licenses/` are unmodified copies downloaded from the model repository.
