"""Generate every results table in report/report.md from files under results/.

The report contains marker pairs `<!-- BEGIN:name -->` ... `<!-- END:name -->`; this script replaces the text
between them. No number in those tables is typed by hand.

Usage (repository root): python bench/make_tables.py [--report report/report.md]
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

R = Path("results")
DEV = R / "device"


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def device_csv(p: Path) -> tuple[str, list[dict]]:
    lines = p.read_text(encoding="utf-8").splitlines()
    header = lines[0].lstrip("# ") if lines[0].startswith("#") else ""
    return header, list(csv.DictReader(io.StringIO("\n".join(lines[1 if header else 0:]))))


def latest(pattern: str) -> Path:
    files = sorted(glob.glob(str(DEV / pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return Path(files[-1])


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f3(x: float) -> str:
    return f"{x:.3f}"


def t_dataset() -> str:
    d = load_json(R / "dataset_stats.json")["pill"]
    rows = [[c, v["total"], v["calib"], v["eval"]] for c, v in sorted(d["test_by_class"].items())]
    head = f"train_good {d['train_good']} (model fitting), val_good {d['val_good']} (calibration); MVTec test split into calib {d['calib']} / eval {d['eval']} (seed 42, stratified).\n\n"
    return head + table(["MVTec test class", "total", "calib", "eval"], rows)


def t_calibration() -> str:
    rows = []
    for bank in (256, 1024, 4096):
        t = load_json(R / "calibration" / f"thresholds_pill_bank{bank}.json")
        rows.append([bank, f3(t["calib_image_auroc"]), f3(t["t_pass"]), f3(t["t_reject"]), f3(t["t_pix"]), f3(t["pixel_f1_calib"]),
                     " / ".join(f"{100 * c:.2f}%" for c in t["sev_cuts_area_pct"])])
    return table(["bank", "calib image AUROC", "t_pass (P5 defective)", "t_reject (P95 good)", "t_pix", "pixel F1 (calib)", "severity area cuts"], rows)


def replay_rows() -> list[dict]:
    src = load_json(R / "detection_eval_device.json")["source"]
    return device_csv(DEV / src)[1]


def t_detection() -> str:
    d = load_json(R / "detection_eval_device.json")
    rows = [["images (defective / good)", f"{d['n_images']} ({d['n_defective']} / {d['n_good']})"],
            ["precision", f3(d["precision"])], ["recall", f3(d["recall"])], ["F1", f3(d["f1"])],
            ["image AUROC", f3(d["image_auroc"])], ["miss rate", f3(d["miss_rate"])], ["false-alarm rate", f3(d["false_alarm_rate"])],
            ["TP / FP / FN / TN", f"{d['tp']} / {d['fp']} / {d['fn']} / {d['tn']}"],
            ["not inspected", len(d["not_inspected"])]]
    v = d["verdicts"]
    vt = table(["truth", "PASS", "RE-INSPECT", "REJECT"], [[t] + [v.get(t, {}).get(k, 0) for k in ("PASS", "RE-INSPECT", "REJECT")] for t in ("good", "defective")])
    return ("Held-out evaluation split, run once on the phone (memory bank 4096). A tablet counts as flagged when its verdict is "
            "RE-INSPECT or REJECT. Source: `results/detection_eval_device.json`.\n\n" + table(["metric", "value"], rows) + "\n\nVerdicts:\n\n" + vt)


def t_per_class() -> str:
    per = defaultdict(Counter)
    for r in replay_rows():
        if not r["file"].startswith("eval/") or r["item_status"] != "OK":
            continue
        cls = r["file"].split("/")[1]
        per[cls][r["verdict"]] += 1
    rows = []
    for cls, c in sorted(per.items()):
        n = sum(c.values())
        flagged = c["RE-INSPECT"] + c["REJECT"]
        rate = ("flagged " if cls != "good" else "false alarms ") + f"{flagged}/{n}"
        rows.append([cls, n, c["PASS"], c["RE-INSPECT"], c["REJECT"], rate])
    return table(["eval class", "n", "PASS", "RE-INSPECT", "REJECT", "rate"], rows)


def t_vision_latency() -> str:
    s = load_json(R / "device_summary.json")
    rows = []
    for bank, v in s["vision_latency_ms"].items():
        rows.append([bank.replace("bank", ""), v["inference_ms"]["n"], f"{v['inference_ms']['median']:.0f} / {v['inference_ms']['p95']:.0f}",
                     f"{v['postprocess_ms']['median']:.0f} / {v['postprocess_ms']['p95']:.0f}", f"{v['total_ms']['median']:.0f} / {v['total_ms']['p95']:.0f}"])
    return "Nothing Phone (3a), ONNX Runtime 1.23.2 on CPU (4 threads), 200 timed runs after 20 warm-up runs.\n\n" + table(
        ["bank", "n", "inference ms (median / p95)", "post-processing ms (median / p95)", "total per item ms (median / p95)"], rows)


def t_frame_latency() -> str:
    by = defaultdict(list)
    for r in replay_rows():
        by[r["file"]].append(r)
    rows = []
    for name, f in (("1 item", "testcases/I1_good/frame.png"), ("4 items", "testcases/I3_trays/tray2x2.png"), ("9 items", "testcases/I3_trays/tray3x3.png")):
        rs = [r for r in by[f] if r["item_status"] == "OK"]
        vis = [float(r["vision_ms"]) for r in rs]
        rows.append([name, len(rs), by[f][0]["frame_ms"], f"{sum(vis):.0f}", f"{float(by[f][0]['frame_ms']) / max(len(rs), 1):.0f}"])
    ev = load_json(R / "detection_eval_device.json")
    return ("Single replay frame each (n = 1 per row), bank 4096. Frame time covers gate, item finder, crops, and vision plus analysis per item; it excludes image decoding.\n\n"
            + table(["frame", "items inspected", "frame ms", "sum of vision ms", "frame ms per item"], rows)
            + f"\n\nAcross the 86 single-item eval frames: vision median {ev['vision_ms_median']:.0f} ms (p95 {ev['vision_ms_p95']:.0f}), frame median {ev['frame_ms_median']:.0f} ms (p95 {ev['frame_ms_p95']:.0f}).")


def t_throughput() -> str:
    s = load_json(R / "device_summary.json")
    fps = s["fps_full_pipeline"]
    g = s["golden_parity"]
    rows = [["full pipeline, replay, 60 s", f"{fps['frames_per_s']} frames/s", f"{fps['items_per_min']} items/min", fps["note"]],
            ["golden parity (device vs PyTorch score)", f"n = {g['n']}", f"median rel. diff {100 * g['rel_diff_median']:.1f}%, max {100 * g['rel_diff_max']:.1f}%", g["note"]]]
    return table(["measurement", "value", "derived", "note"], rows)


def t_advisory() -> str:
    a = load_json(R / "advisory_summary.json")
    rows = []
    for k, v in a.items():
        mr = v.get("manual_review", {})
        rows.append([k.replace("_", " "), v["n"], f"{v['format_valid']}/{v['n']}",
                     f"{mr['factually_consistent']}/{v['n']}" if mr else "not reviewed",
                     f"{v['ttft_ms']['median'] / 1000:.2f} / {v['ttft_ms']['p95'] / 1000:.2f}",
                     f"{v['decode_tok_s']['median']:.1f}", f"{v['tokens']['median']:.0f}",
                     f"{v['total_ms']['median'] / 1000:.2f} / {v['total_ms']['p95'] / 1000:.2f} / {v['total_ms']['max'] / 1000:.1f}",
                     f"{v['pss_mb']['median']:.0f}"])
    head = re.search(r"load_ms=(\d+); pss_before_mb=(\d+); pss_loaded_mb=(\d+)", a["v2_prompt_seed2027"]["header"])
    note = f"\n\nModel load {int(head.group(1)) / 1000:.1f} s; app PSS {head.group(2)} MB before loading Llama, {head.group(3)} MB after loading." if head else ""
    return table(["run", "n", "format-valid", "manually consistent", "TTFT s (median / p95)", "decode tok/s (median)", "tokens (median)",
                  "total s (median / p95 / max)", "PSS MB (median)"], rows) + note


def t_longrun() -> str:
    p = latest("longrun_*.csv")
    _, rows = device_csv(p)
    first, last = rows[0], rows[-1]

    def rate(a, b):
        return (int(b["frames"]) - int(a["frames"])) / (float(b["elapsed_s"]) - float(a["elapsed_s"]))
    early = rate(rows[0], rows[min(4, len(rows) - 1)])
    late = rate(rows[-5], rows[-1])
    th3 = next((r["elapsed_s"] for r in rows if int(r["thermal_status"]) >= 3), "never")
    return table(["measurement", "value"], [
        ["duration / frames", f"{last['elapsed_s']} s / {last['frames']}"],
        ["throughput first 2 min / last 2 min", f"{early:.1f} / {late:.1f} frames/s"],
        ["battery temperature start / end", f"{first['battery_temp_c']} / {last['battery_temp_c']} C"],
        ["max Android thermal status (first time >= 3)", f"{max(int(r['thermal_status']) for r in rows)} ({th3} s)"],
        ["PSS max", f"{max(float(r['pss_mb']) for r in rows):.0f} MB"],
        ["advisory queue depth (end)", last["queue_depth"]],
        ["battery", f"{first['battery_pct']}% -> {last['battery_pct']}% (charging: power not measured)"],
    ]) + f"\n\nSource: `results/device/{p.name}`."


def t_testcases() -> str:
    rows = list(csv.DictReader((R / "testcases.csv").open(encoding="utf-8")))
    passed = sum(r["result"] == "PASS" for r in rows)
    body = [[r["case_id"], r["scenario"], r["expected"], r["observed"][:110].replace("|", "/"), r["result"]] for r in rows]
    return f"All frames are synthetic (MVTec tiles on a matched background). {passed}/{len(rows)} pass.\n\n" + table(["id", "scenario", "expected", "observed", "result"], body)


TABLES = {
    "dataset": t_dataset, "calibration": t_calibration, "detection": t_detection, "per_class": t_per_class,
    "vision_latency": t_vision_latency, "frame_latency": t_frame_latency, "throughput": t_throughput,
    "advisory": t_advisory, "longrun": t_longrun, "testcases": t_testcases,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=Path("report/report.md"))
    a = ap.parse_args()
    text = a.report.read_text(encoding="utf-8")
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            print("marker missing:", name)
            continue
        text = pat.sub(lambda m: m.group(1) + "\n" + fn() + "\n" + m.group(2), text)
        print("filled", name)
    a.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
