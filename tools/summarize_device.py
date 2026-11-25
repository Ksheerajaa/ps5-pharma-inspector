"""Summarise on-device benchmark CSVs pulled from the phone (results/device/) into results/device_summary.json.

Usage: python tools/summarize_device.py --device-dir results/device --torch-parity results/parity_pill_bank4096_onnx.csv
- bench_vision_bank*.csv: per-iteration latency (200 timed iterations after warm-up)
- bench_fps_*.csv: frames processed in 60 s by the full pipeline (finder + crops + vision)
- golden_device_*.csv: device scores for golden images (whole image -> 256, OpenCV INTER_AREA) vs
  PyTorch scores from the export parity check (PIL bilinear-antialias resize, as in training)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round((len(s) - 1) * q)))]


def read_csv(p: Path) -> tuple[str, list[dict]]:
    lines = p.read_text(encoding="utf-8").splitlines()
    header = lines[0] if lines and lines[0].startswith("#") else ""
    body = lines[1:] if header else lines
    return header.lstrip("# "), list(csv.DictReader(body))


def latest(d: Path, pattern: str) -> Path | None:
    files = sorted(d.glob(pattern))
    return files[-1] if files else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-dir", type=Path, default=Path("results/device"))
    ap.add_argument("--torch-parity", type=Path, default=Path("results/parity_pill_bank4096_onnx.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/device_summary.json"))
    a = ap.parse_args()
    out: dict = {"vision_latency_ms": {}}

    for bank in (256, 1024, 4096):
        p = latest(a.device_dir, f"bench_vision_bank{bank}_*.csv")
        if not p:
            continue
        header, rows = read_csv(p)
        out["device"] = header
        stats = {}
        for col in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
            v = [float(r[col]) for r in rows]
            stats[col] = {"median": statistics.median(v), "p95": pct(v, 0.95), "mean": round(statistics.mean(v), 2), "n": len(v)}
        out["vision_latency_ms"][f"bank{bank}"] = {"file": p.name, **stats}

    p = latest(a.device_dir, "bench_fps_*.csv")
    if p:
        _, rows = read_csv(p)
        r = rows[0]
        secs = float(r["seconds"])
        out["fps_full_pipeline"] = {"file": p.name, "frames": int(r["frames"]), "inspected_items": int(r["items"]), "seconds": secs,
                                    "frames_per_s": round(int(r["frames"]) / secs, 2), "items_per_min": round(int(r["items"]) * 60 / secs, 1),
                                    "note": "replay frames (first 20 non-golden replay images), single-item frames; camera not involved"}

    p = latest(a.device_dir, "golden_device_*.csv")
    if p and a.torch_parity.exists():
        _, dev = read_csv(p)
        torch = {r["input"].split("/test/")[-1]: float(r["score_torch"]) for r in csv.DictReader(a.torch_parity.open(encoding="utf-8"))}
        pairs = []
        for r in dev:
            t = torch.get(r["file"])
            if t is not None:
                d = float(r["score_device"])
                pairs.append({"file": r["file"], "score_torch": t, "score_device": d, "rel_diff": round(abs(d - t) / abs(t), 4)})
        rd = [x["rel_diff"] for x in pairs]
        out["golden_parity"] = {"file": p.name, "n": len(pairs), "rel_diff_median": statistics.median(rd) if rd else None,
                                "rel_diff_max": max(rd) if rd else None, "pairs": pairs,
                                "note": "difference includes resize (OpenCV INTER_AREA on device vs PIL bilinear-antialias in PyTorch) and fp32 runtime differences"}

    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "golden_parity"}, indent=1))
    if "golden_parity" in out:
        g = out["golden_parity"]
        print("golden:", g["n"], "median rel diff", g["rel_diff_median"], "max", g["rel_diff_max"])
        for x in g["pairs"]:
            print(" ", x)


if __name__ == "__main__":
    main()
