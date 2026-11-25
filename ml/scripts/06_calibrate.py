"""Calibrate PatchCore-lite thresholds from val_good + calib only (plan 6.4, addendum 2.1/2.2).

Writes thresholds_<category>_bank<N>.json and per-image score CSVs. Never reads `eval`.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.patchcore_lite import PatchCoreLite

TO_TENSOR = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def load_model(checkpoint: Path, device: torch.device) -> PatchCoreLite:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model = PatchCoreLite(bank=state["bank"])
    model.load_state_dict(state)
    return model.to(device).eval()


def defect_class(relative: str) -> str:
    parts = Path(relative).parts
    return parts[1] if parts[0] == "test" else "good"


def mask_for(root: Path, relative: str) -> np.ndarray:
    cls = defect_class(relative)
    if cls == "good":
        return np.zeros((256, 256), dtype=bool)
    path = root / "ground_truth" / cls / (Path(relative).stem + "_mask.png")
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((256, 256), Image.NEAREST)) > 127


@torch.no_grad()
def score_all(model, root: Path, relatives: list[str], device) -> tuple[np.ndarray, np.ndarray]:
    maps, scores = [], []
    for relative in relatives:
        with Image.open(root / relative) as image:
            tensor = TO_TENSOR(image.convert("RGB")).unsqueeze(0).to(device)
        anomaly, score = model(tensor)
        maps.append(anomaly[0, 0].cpu().numpy())
        scores.append(float(score[0]))
    return np.stack(maps), np.asarray(scores)


def best_pixel_threshold(maps: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
    values, truth = maps.ravel(), masks.ravel()
    best_t, best_f1 = float(values.max()), -1.0
    for t in np.quantile(values, np.linspace(0.80, 0.9995, 200)):
        pred = values >= t
        tp = np.count_nonzero(pred & truth)
        fp = np.count_nonzero(pred & ~truth)
        fn = np.count_nonzero(~pred & truth)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def f1_optimal(scores: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_f1 = float(scores.max()), -1.0
    for t in np.unique(scores):
        pred = scores >= t
        tp = np.count_nonzero(pred & labels)
        f1 = 2 * tp / max(2 * tp + np.count_nonzero(pred & ~labels) + np.count_nonzero(~pred & labels), 1)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def image_statistics(root: Path, relatives: list[str]) -> dict:
    """Item fraction, background tone, luma and blur statistics on 256x256 training-domain images."""
    side_fracs, area_fracs, bg, lumas, clipped, laps, laps_blur = [], [], [], [], [], [], []
    for relative in relatives:
        bgr = cv2.resize(cv2.imread(str(root / relative)), (256, 256), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
        if np.mean(border > 0) > 0.5:
            binary = 255 - binary
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            w, h = stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT]
            side_fracs.append(max(w, h) / 256.0)
            area_fracs.append(stats[largest, cv2.CC_STAT_AREA] / 256.0**2)
        bg.append(np.median(bgr[binary == 0], axis=0)[::-1])  # RGB
        lumas.append(float(gray.mean()))
        clipped.append(float(np.mean(gray >= 250)))
        laps.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        laps_blur.append(float(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 3.0), cv2.CV_64F).var()))
    laps, laps_blur = np.asarray(laps), np.asarray(laps_blur)
    # Blur threshold: geometric midpoint between the sharpest blurred and the least sharp original.
    blur_threshold = float(np.sqrt(laps.min() * laps_blur.max())) if laps.min() > laps_blur.max() else float(np.percentile(laps, 1))
    return {
        "target_item_side_fraction": float(np.median(side_fracs)),
        "target_item_area_fraction": float(np.median(area_fracs)),
        "background_rgb_median": [round(float(v), 1) for v in np.median(np.stack(bg), axis=0)],
        "luma_mean_range": [float(np.min(lumas)), float(np.max(lumas))],
        "clipped_fraction_max": float(np.max(clipped)),
        "laplacian_var_sharp_min": float(laps.min()),
        "laplacian_var_blur_sigma3_max": float(laps_blur.max()),
        "blur_threshold": blur_threshold,
        "n_images": len(relatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    root = args.data_root / split["category"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = image_statistics(root, split["train_good"])
    print("image statistics:", json.dumps(stats, indent=1))
    calib = split["calib"]
    calib_labels = np.asarray([defect_class(r) != "good" for r in calib])
    calib_masks = np.stack([mask_for(root, r) for r in calib])
    gt_area = calib_masks.reshape(len(calib), -1).mean(axis=1)

    summary = []
    for checkpoint in args.checkpoints:
        model = load_model(checkpoint, device)
        bank = int(model.bank.shape[0])
        _, val_scores = score_all(model, root, split["val_good"], device)
        calib_maps, calib_scores = score_all(model, root, calib, device)

        good_scores = np.concatenate([val_scores, calib_scores[~calib_labels]])
        defect_scores = calib_scores[calib_labels]
        t_reject = float(np.quantile(good_scores, 0.95, method="higher"))  # <=5% false alarms
        t_pass = float(np.quantile(defect_scores, 0.05, method="lower"))   # <=5% misses
        reinspect_enabled = t_pass < t_reject
        if not reinspect_enabled:
            t_pass = t_reject = f1_optimal(calib_scores, calib_labels)

        t_pix, pixel_f1 = best_pixel_threshold(calib_maps, calib_masks)
        area_pct = (calib_maps >= t_pix).reshape(len(calib), -1).mean(axis=1)
        defect_area = area_pct[calib_labels]
        sev_cuts = [float(np.quantile(defect_area, 1 / 3)), float(np.quantile(defect_area, 2 / 3))]
        gt_cuts = [float(np.quantile(gt_area[calib_labels], 1 / 3)), float(np.quantile(gt_area[calib_labels], 2 / 3))]
        # Confidence scale: median absolute distance of calib scores to the nearest threshold.
        dist = np.minimum(np.abs(calib_scores - t_pass), np.abs(calib_scores - t_reject))
        confidence_scale = float(max(np.median(dist), 1e-3))
        # Sensitivity step: a quarter of the gap between good and defective medians.
        sensitivity_step = float(abs(np.median(defect_scores) - np.median(good_scores)) / 4)

        result = {
            "category": split["category"],
            "split_seed": split["seed"],
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": checkpoint.name,
            "bank_size": bank,
            "t_pass": t_pass,
            "t_reject": t_reject,
            "reinspect_enabled": bool(reinspect_enabled),
            "t_pix": t_pix,
            "pixel_f1_calib": pixel_f1,
            "sev_cuts_area_pct": sev_cuts,
            "gt_mask_area_tertiles": gt_cuts,
            "confidence_scale": confidence_scale,
            "sensitivity_step": sensitivity_step,
            "calib_image_auroc": float(roc_auc_score(calib_labels, calib_scores)),
            "n_val_good": len(split["val_good"]),
            "n_calib": len(calib),
            "quality_and_finder": stats,
        }
        out = args.output_dir / f"thresholds_{split['category']}_bank{bank}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (args.output_dir / f"calib_scores_{split['category']}_bank{bank}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "set", "label", "score", "area_pct"])
            for r, s in zip(split["val_good"], val_scores):
                writer.writerow([r, "val_good", "good", f"{s:.5f}", ""])
            for r, s, a in zip(calib, calib_scores, area_pct):
                writer.writerow([r, "calib", defect_class(r), f"{s:.5f}", f"{a:.5f}"])
        summary.append((bank, result["calib_image_auroc"], t_pass, t_reject))
        print(f"bank {bank}: calib AUROC {result['calib_image_auroc']:.4f} t_pass {t_pass:.3f} t_reject {t_reject:.3f} re-inspect {reinspect_enabled}")
    print(summary)


if __name__ == "__main__":
    main()
