# Windows wrapper for the Llama export (run from the repository root).
# executorch 1.5.1 ships flatc.exe but looks up "flatc"; point it at the shipped binary.
$flatc = python -c "import importlib.resources as r; print(r.files('executorch.data.bin').joinpath('flatc.exe'))"
$env:FLATC_EXECUTABLE = $flatc
python -m executorch.extension.llm.export.export_llm --config ml/configs/llama3_2_1b_xnnpack.yaml
