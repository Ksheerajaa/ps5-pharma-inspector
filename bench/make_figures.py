"""Generate the report's charts and background images from files under results/.

Usage (repository root): python bench/make_figures.py
Writes PNGs to report/figures/. Palette validated with the dataviz validator (PASS #6FA3E8, RE-INSPECT #1F4E9E,
REJECT #E8431A: CVD and normal-vision separation pass; light blue carries direct labels for contrast).
"""
from __future__ import annotations

import csv
import glob
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

R, OUT, FONTS = Path("results"), Path("report/figures"), Path("report/fonts")
PASS, CHECK, REJECT = "#6FA3E8", "#1F4E9E", "#E8431A"
INK, MUTED, GRID = "#1b2430", "#5a6a80", "#d5deea"

for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "Arimo", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


def device_csv(p: Path):
    lines = p.read_text(encoding="utf-8").splitlines()
    return list(csv.DictReader(io.StringIO("\n".join(lines[1:] if lines[0].startswith("#") else lines))))


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=MUTED)


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def verdicts():
    d = json.loads((R / "detection_eval_device.json").read_text())["verdicts"]
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    rows = [("Good tablets", d["good"]), ("Defective tablets", d["defective"])]
    for y, (name, v) in enumerate(rows):
        n = sum(v.values()); left = 0
        for k, c in (("PASS", PASS), ("RE-INSPECT", CHECK), ("REJECT", REJECT)):
            w = 100 * v.get(k, 0) / n
            if w <= 0:
                continue
            ax.barh(y, w - 0.4, left=left + 0.2, color=c, height=0.55)
            if w > 6:
                ax.text(left + w / 2, y, f"{v[k]}", ha="center", va="center", color="white" if c != PASS else INK, fontsize=8.5, fontweight="bold")
            left += w
    ax.set_yticks([0, 1], [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("share of tablets (%)"); ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (PASS, CHECK, REJECT)], labels=["PASS", "RE-INSPECT", "REJECT"],
              ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8)
    title(ax, "Verdicts on the held-out split", "counts inside the bars · run on the phone")
    save(fig, "fig_verdicts.png")


def per_class():
    src = json.loads((R / "detection_eval_device.json").read_text())["source"]
    per = defaultdict(Counter)
    for r in device_csv(R / "device" / src):
        if r["file"].startswith("eval/") and r["item_status"] == "OK" and r["file"].split("/")[1] != "good":
            per[r["file"].split("/")[1]][r["verdict"] != "PASS"] += 1
    items = sorted(per.items(), key=lambda kv: kv[1][True] / sum(kv[1].values()))
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for y, (cls, c) in enumerate(items):
        n = sum(c.values()); rate = 100 * c[True] / n
        ax.barh(y, rate, color=CHECK, height=0.6)
        ax.text(rate + 1, y, f"{c[True]}/{n}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(items)), [k.replace("_", " ") for k, _ in items]); ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("defective tablets flagged (%)"); ax.grid(axis="y", visible=False)
    title(ax, "Recall by defect class")
    save(fig, "fig_per_class.png")


def latency():
    s = json.loads((R / "device_summary.json").read_text())["vision_latency_ms"]
    banks = list(s)
    inf = [s[b]["inference_ms"]["median"] for b in banks]; post = [s[b]["postprocess_ms"]["median"] for b in banks]
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    y = np.arange(len(banks))
    ax.barh(y, [i - 0.6 for i in inf], color=CHECK, height=0.55, label="model inference")
    ax.barh(y, [p - 0.6 for p in post], left=inf, color=PASS, height=0.55, label="analysis and heatmap")
    for i, b in enumerate(banks):
        ax.text(inf[i] / 2, i, f"{inf[i]:.0f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(inf[i] + post[i] + 2, i, f"{inf[i] + post[i]:.0f} ms", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [f"bank {b.replace('bank', '')}" for b in banks]); ax.invert_yaxis(); ax.set_xlim(0, 115)
    ax.set_xlabel("median time per tablet (ms)"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=8)
    title(ax, "Vision time per tablet on the phone", "200 timed runs per memory bank")
    save(fig, "fig_latency.png")


def frames():
    src = json.loads((R / "detection_eval_device.json").read_text())["source"]
    by = defaultdict(list)
    for r in device_csv(R / "device" / src):
        by[r["file"]].append(r)
    pts = [(len([r for r in by[f] if r["item_status"] == "OK"]), float(by[f][0]["frame_ms"])) for f in
           ("testcases/I1_good/frame.png", "testcases/I3_trays/tray2x2.png", "testcases/I3_trays/tray3x3.png")]
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.bar([str(n) for n, _ in pts], [ms for _, ms in pts], color=CHECK, width=0.55)
    for i, (n, ms) in enumerate(pts):
        ax.text(i, ms + 15, f"{ms:.0f} ms", ha="center", fontsize=8, color=INK)
    ax.set_xlabel("tablets on the tray"); ax.set_ylim(0, 1000); ax.grid(axis="x", visible=False)
    ax.axhline(1000, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    title(ax, "Whole-tray decision time", "1 s line for reference")
    save(fig, "fig_frames.png")


def advisory():
    a = json.loads((R / "advisory_summary.json").read_text())
    v1, v2 = a["v1_prompt_seed2026"], a["v2_prompt_seed2027"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1))
    for ax, vals, lab, unit, t in (
        (axes[0], [v1["format_valid"], v2["format_valid"]], "valid answers (of 50)", "", "Answers that pass validation"),
        (axes[1], [v1["total_ms"]["p95"] / 1000, v2["total_ms"]["p95"] / 1000], "slowest 5% of answers (s)", " s", "Answer time, 95th percentile"),
    ):
        ax.bar(["first prompt", "final prompt"], vals, color=[PASS, CHECK], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02 + 0.2, f"{v:.0f}{unit}" if unit == "" else f"{v:.1f}{unit}", ha="center", fontsize=8.5, color=INK)
        ax.set_ylabel(lab); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(vals) * 1.25)
        title(ax, t)
    fig.tight_layout(w_pad=3)
    save(fig, "fig_advisory.png")


def longrun():
    rows = device_csv(Path(sorted(glob.glob(str(R / "device" / "longrun_*.csv")))[-1]))
    t = np.array([float(r["elapsed_s"]) for r in rows]) / 60
    fr = np.array([int(r["frames"]) for r in rows])
    rate = np.diff(fr) / np.diff(t * 60)
    temp = [float(r["battery_temp_c"]) for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 3.2), sharex=True)
    a1.plot(t[1:], rate, color=CHECK, lw=2)
    a1.set_ylabel("frames per s"); a1.set_ylim(0, max(rate) * 1.25)
    title(a1, "15-minute continuous run", "vision and Llama together, phone on charge")
    a2.plot(t, temp, color=REJECT, lw=2)
    a2.set_ylabel("battery °C"); a2.set_xlabel("minutes"); a2.set_ylim(min(temp) - 2, max(temp) + 2)
    th = next((x for x, r in zip(t, rows) if int(r["thermal_status"]) >= 3), None)
    if th is not None:
        for a in (a1, a2):
            a.axvline(th, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        a2.text(th + 0.15, min(temp), "Android thermal status 3", fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout()
    save(fig, "fig_longrun.png")


def backgrounds():
    """Gradients as images: CSS gradients are tiled in Chrome PDFs and show seams in some viewers."""
    h, w = 900, 2480
    top, mid = np.array([79, 143, 232]), np.array([169, 203, 245])
    y = np.linspace(0, 1, h)[:, None]
    col = np.where(y < 0.55, top + (mid - top) * (y / 0.55), mid + (255 - mid) * ((y - 0.55) / 0.45))
    plt.imsave(OUT / "bg_cover.png", np.repeat(col[:, None, :], w, axis=1).reshape(h, w, 3).astype(np.uint8))
    h, w = 300, 600
    a, b = np.array([11, 42, 91]), np.array([47, 111, 209])
    yy, xx = np.mgrid[0:h, 0:w]
    k = ((xx / w) * 0.6 + (yy / h) * 0.4)[..., None]
    plt.imsave(OUT / "bg_stat.png", (a + (b - a) * k).astype(np.uint8))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (verdicts, per_class, latency, frames, advisory, longrun, backgrounds):
        fn()
