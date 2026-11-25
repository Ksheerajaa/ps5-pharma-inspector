from __future__ import annotations

import argparse
import os
import shutil
import sys
import sysconfig
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.patchcore_lite import PatchCoreLite


def expose_flatc() -> None:
    scripts_dir = Path(sysconfig.get_path("scripts"))
    executable = "flatc.exe" if os.name == "nt" else "flatc"
    if (scripts_dir / executable).is_file() and shutil.which("flatc") is None:
        os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expose_flatc()
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bank = checkpoint["state_dict"]["bank"]
    model = PatchCoreLite(bank=bank)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    sample = torch.zeros((1, 3, 256, 256), dtype=torch.float32)

    exported = torch.export.export(model, (sample,))
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    )
    executorch_program = edge_program.to_executorch()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(executorch_program.buffer)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
