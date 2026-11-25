"""Teach PatchCore-lite a new product from a few phone photos (good samples only for training).

Usage (repository root):
  python ml/scripts/07_new_product.py --photos paracetamol_data --name paracetamol --out ml/checkpoints/paracetamol

Steps: find each tablet exactly like the app (Otsu + morphology, moments angle, square crop with the item's long
side at 86.7% of the crop, padded with the background colour, INTER_AREA to 256) -> label per tablet (LABELS
below, left to right) -> augment good training views -> coreset memory bank -> calibrate on held-out good views
and on defect tablets -> ONNX + config JSON in the app's format.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.patchcore_lite import PatchCoreLite, greedy_coreset  # noqa: E402

TARGET_SIDE = 0.8671875  # same crop rule as the pill model (MVTec median)
# Labels per photo (sorted by filename), one per tablet from left to right.
# good = scored face of the intact tablet; ink / half / crack = defective; blur = excluded (quality gate case).
LABELS = {0: ["good"], 1: ["blur"], 2: ["good"], 3: ["good"], 4: ["ink"], 5: ["ink"], 6: ["ink"], 7: ["ink"],
          8: ["half", "half"], 9: ["half", "half"], 10: ["half", "half"], 11: ["good", "crack"], 12: ["good", "crack"],
          13: ["good", "ink"], 14: ["good", "ink"], 15: ["good", "ink"]}
HELD_OUT_GOOD = {3, 14}  # photos whose good tablet is used for calibration only, never for training


def find_blobs(rgb):
    h, w = rgb.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    work = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
    if (border > 0).mean() > 0.5:
        b = 255 - b
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    b = cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < work.shape[0] * work.shape[1] * 0.002:
            continue
        x, y, ww, hh = st[i, :4]
        if x <= 1 or y <= 1 or x + ww >= work.shape[1] - 1 or y + hh >= work.shape[0] - 1:
            continue
        mask = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        m = cv2.moments(c)
        angle = math.degrees(0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
        out.append(dict(cx=cen[i][0] / scale, cy=cen[i][1] / scale, long=max(rw, rh) / scale, angle=angle,
                        area=st[i, cv2.CC_STAT_AREA] / (work.shape[0] * work.shape[1]), aspect=max(rw, rh) / max(min(rw, rh), 1)))
    return sorted(out, key=lambda o: o["cx"])


def crop(rgb, blob, bg, angle_jitter=0.0, scale_jitter=1.0):
    side = blob["long"] / TARGET_SIDE * scale_jitter
    half = int(math.ceil(side * 0.75))
    padded = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_CONSTANT, value=bg)
    cx, cy = int(round(blob["cx"])), int(round(blob["cy"]))
    roi = padded[cy:cy + 2 * half, cx:cx + 2 * half]
    angle = (blob["angle"] if abs(blob["angle"]) > 2 else 0.0) + angle_jitter
    if angle:
        mtx = cv2.getRotationMatrix2D((half, half), angle, 1.0)
        roi = cv2.warpAffine(roi, mtx, (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    s = max(8, int(round(side)))
    start = half - s // 2
    return cv2.resize(roi[start:start + s, start:start + s], (256, 256), interpolation=cv2.INTER_AREA)


def augment(rgb, blob, bg, rng, n):
    outs = []
    for i in range(n):
        c = crop(rgb, blob, bg, angle_jitter=rng.uniform(-8, 8), scale_jitter=rng.uniform(0.94, 1.06))
        if i % 4 == 1:
            c = c[:, ::-1]
        elif i % 4 == 2:
            c = c[::-1, :]
        elif i % 4 == 3:
            c = c[::-1, ::-1]
        c = np.clip(c.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        if rng.random() < 0.4:
            c = cv2.GaussianBlur(c, (0, 0), rng.uniform(0.5, 1.3))
        outs.append(np.ascontiguousarray(c))
    return outs


def to_tensor(crops):
    return torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank", type=int, default=2048)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    files = sorted(p for p in a.photos.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files]
    # background colour: median of pixels outside the tablets (dark fixture)
    bg = tuple(int(v) for v in np.median(np.concatenate([im[::8, ::8].reshape(-1, 3) for im in imgs]), axis=0))
    train, held_good, defects, items, skipped = [], [], [], [], []
    for idx, (p, im) in enumerate(zip(files, imgs)):
        blobs, labels = find_blobs(im), LABELS.get(idx, [])
        if len(blobs) != len(labels):
            skipped.append((p.name, len(blobs), len(labels)))
            continue
        for blob, lab in zip(blobs, labels):
            base = crop(im, blob, bg)
            items.append(dict(photo=p.name, label=lab, area=blob["area"], aspect=blob["aspect"],
                              lapvar=float(cv2.Laplacian(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())))
            cv2.imwrite(str(a.out / f"crop_{idx:02d}_{lab}_{len(items)}.png"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
            if lab == "good" and idx in HELD_OUT_GOOD:
                held_good += [base] + augment(im, blob, bg, rng, 15)
            elif lab == "good":
                train += [base] + augment(im, blob, bg, rng, 40)
            elif lab != "blur":
                defects += [base] + augment(im, blob, bg, rng, 3)
    print(f"background {bg}; train {len(train)} crops, held-out good {len(held_good)}, defect {len(defects)}; skipped {skipped}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreLite().to(dev).eval()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(train), 16):
            e = model.embed(to_tensor(train[i:i + 16]).to(dev))
            feats.append(e.flatten(2).transpose(1, 2).reshape(-1, 128))
    feats = torch.cat(feats)
    if feats.shape[0] > 120_000:
        feats = feats[torch.randperm(feats.shape[0], generator=torch.Generator().manual_seed(0))[:120_000].to(feats.device)]
    bank = greedy_coreset(feats, a.bank, seed=42).detach().cpu()
    model = PatchCoreLite(bank=bank).eval()

    def score(crops):
        maps, scores = [], []
        with torch.inference_mode():
            for i in range(0, len(crops), 16):
                m, s = model(to_tensor(crops[i:i + 16]))
                maps.append(m.squeeze(1).numpy()); scores.append(s.numpy().reshape(-1))
        return np.concatenate(maps), np.concatenate(scores)

    gm, gs = score(held_good)
    dm, ds = score(defects)
    # Thresholds (documented in docs/DECISIONS.md): PASS below the highest held-out good score, REJECT at or above the
    # lowest defect score; between them RE-INSPECT. If they overlap, both are set to the midpoint (no RE-INSPECT band).
    t_pass, t_reject = float(gs.max()), float(ds.min())
    if t_pass >= t_reject:
        t_pass = t_reject = float((np.percentile(gs, 95) + np.percentile(ds, 5)) / 2)
    t_pix = float(np.percentile(gm, 99.9))
    area = (dm >= t_pix).reshape(len(dm), -1).mean(axis=1)
    cuts = [float(np.percentile(area, 33.3)), float(np.percentile(area, 66.7))]
    conf = float(np.median(np.abs(np.concatenate([gs, ds]) - np.clip(np.concatenate([gs, ds]), t_pass, t_reject))) or 0.5)
    good_lap = [it["lapvar"] for it in items if it["label"] == "good"]
    blur_lap = [it["lapvar"] for it in items if it["label"] == "blur"]
    blur_thr = float(min(good_lap) * 0.5) if good_lap else 8.456
    good_area = [it["area"] for it in items if it["label"] == "good"]
    print(f"good held-out scores {gs.min():.2f}..{gs.max():.2f}; defect scores {ds.min():.2f}..{ds.max():.2f}")
    print(f"t_pass {t_pass:.3f} t_reject {t_reject:.3f} t_pix {t_pix:.3f} cuts {cuts} blur {blur_thr:.1f} (good lap {min(good_lap):.1f}, blur photo lap {blur_lap})")

    ckpt = a.out / f"patchcore_{a.name}_bank_{a.bank}.pt"
    torch.save({"state_dict": model.state_dict(), "bank_size": a.bank, "train_image_count": len(train)}, ckpt)
    onnx_path = a.out / "model.onnx"
    torch.onnx.export(model, (torch.zeros(1, 3, 256, 256),), str(onnx_path), input_names=["image"],
                      output_names=["anomaly_map", "score"], opset_version=18, dynamo=False)
    base_cfg = json.loads(Path("results/calibration/thresholds_pill_bank4096.json").read_text())
    cfg = dict(base_cfg)
    cfg.update(category=a.name, bank_size=a.bank, checkpoint=ckpt.name, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               t_pass=t_pass, t_reject=t_reject, reinspect_enabled=t_pass < t_reject, t_pix=t_pix, sev_cuts_area_pct=cuts,
               confidence_scale=max(conf, 0.1), n_val_good=len(gs), n_calib=len(ds),
               calibration_note="few-shot product from phone photos; thresholds from held-out good views and defect tablets",
               score_ranges={"good_heldout": [float(gs.min()), float(gs.max())], "defect": [float(ds.min()), float(ds.max())]})
    q = dict(base_cfg["quality_and_finder"])
    q.update(target_item_area_fraction=float(np.median(good_area)), background_rgb_median=list(map(float, bg)), blur_threshold=blur_thr, n_images=len(files))
    cfg["quality_and_finder"] = q
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    Path("results/products").mkdir(parents=True, exist_ok=True)
    Path(f"results/products/{a.name}_calibration.json").write_text(json.dumps(
        {"items": items, "skipped": skipped, "good_heldout_scores": gs.tolist(), "defect_scores": ds.tolist(),
         "t_pass": t_pass, "t_reject": t_reject, "t_pix": t_pix}, indent=1))
    print("wrote", onnx_path, a.out / "config.json")


if __name__ == "__main__":
    main()
