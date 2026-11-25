from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.patchcore_lite import PatchCoreLite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bank = checkpoint["state_dict"]["bank"]
    model = PatchCoreLite(bank=bank)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    sample = torch.zeros((1, 3, 256, 256), dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        str(args.output),
        input_names=["image"],
        output_names=["anomaly_map", "score"],
        opset_version=18,
        dynamo=False,
    )
    onnx.checker.check_model(str(args.output))
    print(f"saved and checked {args.output}")


if __name__ == "__main__":
    main()
