"""Generate SYNTHETIC test frames for the addendum section 5 matrix from MVTec pill images.

Every frame is synthetic (tiles of MVTec images on a background matched to MVTec). Each case folder
gets a reference.json describing the item scale so the app's item finder uses the right reference.
Images come from `calib` / `val_good` only (never `eval`) except I3, which tiles `eval` images so the
per-item verdicts can be compared with the single-image eval replay.

Usage: python tools/make_test_frames.py --data-root ml/data --split ml/splits/pill.json \
          --calib-scores results/calibration/calib_scores_pill_bank4096.csv \
          --thresholds results/calibration/thresholds_pill_bank4096.json --capsule capsule.tar.xz --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

TILE = 400
BG_BGR = (26, 15, 15)  # MVTec pill background median RGB (15, 15, 26), stored BGR


def tile(path: Path) -> np.ndarray:
    return cv2.resize(cv2.imread(str(path)), (TILE, TILE), interpolation=cv2.INTER_AREA)


def canvas(w: int, h: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.empty((h, w, 3), np.uint8)
    c[:] = BG_BGR
    noise = rng.normal(0, 2, c.shape)
    return np.clip(c + noise, 0, 255).astype(np.uint8)


def paste(c: np.ndarray, t: np.ndarray, x: int, y: int) -> None:
    """Paste a tile; where two tiles overlap keep the brighter pixel so items can touch."""
    h, w = t.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, c.shape[1]), min(y + h, c.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = c[y0:y1, x0:x1]
    part = t[y0 - y:y1 - y, x0 - x:x1 - x]
    np.copyto(region, np.maximum(region, part))


def grid(paths: list[Path], cols: int, rows: int) -> np.ndarray:
    c = canvas(cols * TILE, rows * TILE)
    for i, p in enumerate(paths):
        paste(c, tile(p), (i % cols) * TILE, (i // cols) * TILE)
    return c


def write_case(out: Path, case: str, frames: dict[str, np.ndarray], ref_area: float, mean_rgb: list[float]) -> None:
    d = out / case
    d.mkdir(parents=True, exist_ok=True)
    sizes = {img.shape[:2] for img in frames.values()}
    assert len(sizes) == 1, f"{case}: all frames in a folder must share the canvas size"
    for name, img in frames.items():
        cv2.imwrite(str(d / name), img)
    h, w = sizes.pop()
    area_fraction = ref_area * TILE * TILE / (w * h)
    (d / "reference.json").write_text(json.dumps({"area_fraction": area_fraction, "aspect": 1.46, "mean_rgb": mean_rgb}), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--calib-scores", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True)
    ap.add_argument("--capsule", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    split = json.loads(a.split.read_text())
    root = a.data_root / split["category"]
    th = json.loads(a.thresholds.read_text())
    ref_area = th["quality_and_finder"]["target_item_area_fraction"]
    rows = list(csv.DictReader(a.calib_scores.open()))
    goods = sorted((r for r in rows if r["label"] == "good"), key=lambda r: float(r["score"]))
    defects = sorted((r for r in rows if r["label"] != "good"), key=lambda r: -float(r["score"]))
    good_paths = [root / r["image"] for r in goods[:16]]  # lowest-scoring good images
    top_defect = root / defects[0]["image"]
    band = [r for r in rows if r["label"] != "good" and th["t_pass"] <= float(r["score"]) < th["t_reject"]]

    # Mean pill colour (RGB) inside the item mask of the good tiles, for the colour check of the reference.
    cols = []
    for p in good_paths[:8]:
        t = tile(p)
        g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cols.append(cv2.mean(t, m)[:3][::-1])
    mean_rgb = [round(float(v), 1) for v in np.mean(cols, axis=0)]

    out = a.out
    two = [good_paths[0], good_paths[1], top_defect, good_paths[2]]
    write_case(out, "F1_empty", {"frame.png": canvas(1200, 1200)}, ref_area, mean_rgb)
    write_case(out, "F2_blur", {"frame.png": cv2.GaussianBlur(grid(two, 2, 2), (0, 0), 8)}, ref_area, mean_rgb)
    base = grid(two, 2, 2)
    dark = np.clip(base.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)
    bright = np.clip(base.astype(np.float32) + 130, 0, 255).astype(np.uint8)
    write_case(out, "F3_exposure", {"dark.png": dark, "bright.png": bright}, ref_area, mean_rgb)
    glare = base.copy()
    cv2.ellipse(glare, (400, 400), (200, 120), 0, 0, 360, (255, 255, 255), -1)
    write_case(out, "F4_glare", {"frame.png": glare}, ref_area, mean_rgb)

    single = lambda p: (lambda c: (paste(c, tile(p), 400, 400), c)[1])(canvas(1200, 1200))  # noqa: E731
    write_case(out, "I1_good", {"frame.png": single(good_paths[0])}, ref_area, mean_rgb)
    write_case(out, "I2_defect", {"frame.png": single(top_defect)}, ref_area, mean_rgb)

    eval_paths = [root / r for r in split["eval"]]
    rng = np.random.default_rng(7)
    pick = [eval_paths[i] for i in rng.choice(len(eval_paths), 13, replace=False)]
    tray2 = canvas(1200, 1200)
    for i, p in enumerate(pick[:4]):
        paste(tray2, tile(p), 200 + (i % 2) * TILE, 200 + (i // 2) * TILE)
    write_case(out, "I3_trays", {"tray2x2.png": tray2, "tray3x3.png": grid(pick[4:13], 3, 3)}, ref_area, mean_rgb)
    (out / "I3_trays" / "tiles.json").write_text(json.dumps({
        "tray2x2.png": [str(p.relative_to(root)).replace("\\", "/") for p in pick[:4]],
        "tray3x3.png": [str(p.relative_to(root)).replace("\\", "/") for p in pick[4:13]],
    }, indent=1))

    touching = canvas(1200, 1200)
    paste(touching, tile(good_paths[3]), 150, 400)
    paste(touching, tile(good_paths[4]), 150 + int(TILE * 0.80), 400)
    write_case(out, "I4_touching", {"frame.png": touching}, ref_area, mean_rgb)

    partial = canvas(1200, 1200)
    paste(partial, tile(good_paths[5]), 100, 400)
    paste(partial, tile(good_paths[6]), 1200 - TILE // 2, 400)
    write_case(out, "I5_partial", {"frame.png": partial}, ref_area, mean_rgb)

    foreign = canvas(1200, 1200)
    paste(foreign, tile(good_paths[7]), 0, 0)
    cv2.ellipse(foreign, (600, 800), (173, 118), 0, 0, 360, (40, 200, 230), -1)  # yellow pill-sized object
    cv2.rectangle(foreign, (850, 150), (1150, 220), (140, 140, 140), -1)         # grey bar (wrong aspect)
    write_case(out, "I6_foreign", {"frame.png": foreign}, ref_area, mean_rgb)

    dark_obj = canvas(1200, 1200)
    paste(dark_obj, tile(good_paths[7]), 0, 0)
    cv2.ellipse(dark_obj, (600, 800), (173, 118), 0, 0, 360, (200, 80, 20), -1)  # dark blue: known limitation
    write_case(out, "I6b_dark_foreign", {"frame.png": dark_obj}, ref_area, mean_rgb)

    write_case(out, "I7_too_many", {"frame.png": grid(good_paths[:16], 4, 4)}, ref_area, mean_rgb)

    with tarfile.open(a.capsule) as tf:
        member = tf.getmember("capsule/train/good/000.png")
        data = np.frombuffer(tf.extractfile(member).read(), np.uint8)
    capsule = cv2.resize(cv2.imdecode(data, cv2.IMREAD_COLOR), (TILE, TILE), interpolation=cv2.INTER_AREA)
    cap = canvas(1200, 1200)
    paste(cap, capsule, 400, 400)
    write_case(out, "I8_wrong_product", {"frame.png": cap}, ref_area, mean_rgb)

    mixed = canvas(1200, 1200)
    for i, (x, y) in enumerate([(0, 0), (400, 0), (800, 0)]):
        paste(mixed, tile(good_paths[8 + i]), x, y)
    paste(mixed, tile(good_paths[11]), 100, 600)
    paste(mixed, tile(good_paths[12]), 100 + int(TILE * 0.80), 600)
    write_case(out, "I9_mixed_cluster", {"frame.png": mixed}, ref_area, mean_rgb)

    write_case(out, "I10_near_threshold", {f"band_{i}.png": single(root / r["image"]) for i, r in enumerate(band[:4])}, ref_area, mean_rgb)
    (out / "I10_near_threshold" / "sources.json").write_text(json.dumps([{"image": r["image"], "score_torch": r["score"]} for r in band[:4]], indent=1))
    print(f"wrote synthetic frames to {out}; mean pill RGB {mean_rgb}; near-threshold calib images: {len(band)}")


if __name__ == "__main__":
    main()
