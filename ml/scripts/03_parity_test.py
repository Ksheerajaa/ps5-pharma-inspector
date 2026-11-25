from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.patchcore_lite import PatchCoreLite


def load_image(path: Path) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    import onnxruntime as ort

    split = json.loads(args.split.read_text(encoding="utf-8"))
    category_root = args.data_root / split["category"]
    paths = [category_root / relative for relative in split["calib"][: args.limit]]
    if not paths:
        raise RuntimeError("The calibration split contains no images")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bank = checkpoint["state_dict"]["bank"]
    pytorch_model = PatchCoreLite(bank=bank)
    pytorch_model.load_state_dict(checkpoint["state_dict"])
    pytorch_model.eval()
    onnx_model = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "input",
                "score_torch",
                "score_onnxruntime",
                "score_relative_error",
                "map_correlation",
                "pass",
            ],
        )
        writer.writeheader()
        for path in paths:
            sample = load_image(path)
            with torch.inference_mode():
                torch_map, torch_score = pytorch_model(sample)
                onnx_map, onnx_score = onnx_model.run(
                    ["anomaly_map", "score"], {"image": sample.numpy()}
                )
            torch_map = torch_map.detach().cpu().reshape(-1).float()
            torch_score = torch_score.detach().cpu().reshape(-1)[0].float()
            onnx_map = torch.from_numpy(onnx_map).reshape(-1).float()
            onnx_score = torch.from_numpy(onnx_score).reshape(-1)[0].float()
            relative_error = (torch_score - onnx_score).abs() / torch_score.abs().clamp_min(1e-8)
            correlation = torch.corrcoef(torch.stack((torch_map, onnx_map)))[0, 1]
            passed = bool(relative_error <= 0.02 and correlation >= 0.99)
            result = {
                "input": path.as_posix(),
                "score_torch": float(torch_score),
                "score_onnxruntime": float(onnx_score),
                "score_relative_error": float(relative_error),
                "map_correlation": float(correlation),
                "pass": passed,
            }
            writer.writerow(result)
            if not passed:
                failures.append(path)

    if failures:
        raise SystemExit(f"Parity failed for {len(failures)} of {len(paths)} calibration images")
    print(f"Parity passed for {len(paths)} calibration images; results saved to {args.output}")


if __name__ == "__main__":
    main()
