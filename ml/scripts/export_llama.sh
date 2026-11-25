#!/usr/bin/env sh
set -eu

python -m executorch.extension.llm.export.export_llm \
  --config ml/configs/llama3_2_1b_xnnpack.yaml
