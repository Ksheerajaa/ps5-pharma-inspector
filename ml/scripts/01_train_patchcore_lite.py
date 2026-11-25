from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.patchcore_lite import PatchCoreLite, greedy_coreset


class GoodImages(Dataset):
    def __init__(self, root: Path) -> None:
        self.paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = sorted(set(int(part) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bank sizes must be comma-separated positive integers") from error
    if not sizes or sizes[0] <= 0:
        raise argparse.ArgumentTypeError("bank sizes must be positive")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank-sizes", type=parse_sizes, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    category_root = args.data_root / split["category"]
    train_paths = [category_root / relative for relative in split["train_good"]]
    dataset = GoodImages(category_root / "train" / "good")
    expected = {path.resolve() for path in train_paths}
    dataset.paths = [path for path in dataset.paths if path.resolve() in expected]
    if not dataset.paths:
        raise RuntimeError("No train_good images found; verify --data-root and split file")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreLite().to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    embeddings = []
    with torch.inference_mode():
        for images in loader:
            embedded = model.embed(images.to(device))
            embeddings.append(embedded.flatten(2).transpose(1, 2).reshape(-1, 128))
    features = torch.cat(embeddings, dim=0)

    if args.bank_sizes[-1] > features.shape[0]:
        raise ValueError(
            f"largest requested bank size ({args.bank_sizes[-1]}) exceeds "
            f"available training patches ({features.shape[0]})"
        )
    selected = greedy_coreset(features, args.bank_sizes[-1], seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for size in args.bank_sizes:
        bank = selected[:size].detach().cpu()
        checkpoint_model = copy.deepcopy(model).cpu()
        checkpoint_model.bank = bank
        checkpoint_model.bank_sq_norm = bank.square().sum(dim=1)
        output = args.output_dir / f"patchcore_pill_bank_{size}.pt"
        torch.save(
            {
                "state_dict": checkpoint_model.state_dict(),
                "bank_size": size,
                "projection_seed": args.seed,
                "image_size": 256,
                "feature_dim": 128,
                "train_image_count": len(dataset),
                "train_patch_count": int(features.shape[0]),
            },
            output,
        )
        print(f"saved {output}")


if __name__ == "__main__":
    main()
