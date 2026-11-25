from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _class_images(category_root: Path, relative_dir: str) -> list[str]:
    root = category_root / relative_dir
    if not root.is_dir():
        return []
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        str(path.relative_to(category_root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def prepare_split(data_root: Path, category: str, seed: int) -> dict:
    category_root = data_root / category
    rng = random.Random(seed)

    train_good = _class_images(category_root, "train/good")
    test_good = _class_images(category_root, "test/good")
    test_defective: dict[str, list[str]] = {}

    test_root = category_root / "test"
    if test_root.is_dir():
        for child in sorted(path for path in test_root.iterdir() if path.is_dir()):
            if child.name == "good":
                continue
            test_defective[child.name] = _class_images(category_root, f"test/{child.name}")

    rng.shuffle(train_good)
    val_count = max(1, int(len(train_good) * 0.10)) if train_good else 0
    val_good = sorted(train_good[:val_count])
    train_good = sorted(train_good[val_count:])

    calib: list[str] = []
    eval_images: list[str] = []
    class_splits: dict[str, dict[str, list[str]]] = {}
    test_classes = {"good": test_good, **test_defective}
    for class_name, images in sorted(test_classes.items()):
        shuffled = list(images)
        rng.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        class_calib = sorted(shuffled[:midpoint])
        class_eval = sorted(shuffled[midpoint:])
        class_splits[class_name] = {"calib": class_calib, "eval": class_eval}
        calib.extend(class_calib)
        eval_images.extend(class_eval)

    good_leftovers = sorted(test_good)
    defective_images = sorted(
        image
        for defect_name, images in test_defective.items()
        for image in images
    )

    return {
        "category": category,
        "seed": seed,
        "train_good": sorted(train_good),
        "val_good": sorted(val_good),
        "test_good": good_leftovers,
        "test_defective": test_defective,
        "calib": sorted(calib),
        "eval": sorted(eval_images),
        "class_splits": class_splits,
        "stats": {
            "train_good": len(train_good),
            "val_good": len(val_good),
            "test_good": len(good_leftovers),
            "defective": len(defective_images),
            "calib": len(calib),
            "eval": len(eval_images),
            "test_by_class": {
                name: {
                    "total": len(images),
                    "calib": len(class_splits[name]["calib"]),
                    "eval": len(class_splits[name]["eval"]),
                }
                for name, images in sorted(test_classes.items())
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-output", type=Path, default=Path("results/dataset_stats.json"))
    args = parser.parse_args()

    split = prepare_split(args.data_root, args.category, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    if args.stats_output.exists():
        stats = json.loads(args.stats_output.read_text(encoding="utf-8"))
    else:
        stats = {}
    stats[args.category] = split["stats"]
    args.stats_output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(split["stats"], indent=2))


if __name__ == "__main__":
    main()
