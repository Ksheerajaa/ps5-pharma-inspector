"""Score an on-device replay CSV: detection metrics on `eval` and the synthetic test-case matrix.

Usage: python tools/eval_device.py --replay results/device/replay_*.csv --out results
Writes results/detection_eval_device.json, results/verdicts_eval_device.csv and results/testcases.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--torch-scores", type=Path, nargs="*", default=[])
    a = ap.parse_args()

    rows = list(csv.DictReader(a.replay.open(encoding="utf-8")))
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)

    # ---- detection on eval (one item per MVTec image) ----
    labels, scores, flagged, verdict_table = [], [], [], defaultdict(lambda: defaultdict(int))
    vision_ms, frame_ms, problems = [], [], []
    for f, rs in sorted(by_file.items()):
        if not f.startswith("eval/"):
            continue
        cls = f.split("/")[1]
        truth = cls != "good"
        ok = [r for r in rs if r["item_status"] == "OK" and r["score"]]
        if len(ok) != 1:
            problems.append((f, [(r["frame_status"], r["item_status"], r["item_reason"]) for r in rs]))
            verdict = "NOT_INSPECTED"
            score = float("inf")  # treated as flagged: never a silent PASS
        else:
            verdict = ok[0]["verdict"]
            score = float(ok[0]["score"])
            vision_ms.append(float(ok[0]["vision_ms"]))
            frame_ms.append(float(ok[0]["frame_ms"]))
        labels.append(truth)
        scores.append(score)
        flagged.append(verdict != "PASS")
        verdict_table["defective" if truth else "good"][verdict] += 1

    tp = sum(l and p for l, p in zip(labels, flagged))
    fp = sum((not l) and p for l, p in zip(labels, flagged))
    fn = sum(l and not p for l, p in zip(labels, flagged))
    tn = sum((not l) and not p for l, p in zip(labels, flagged))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    finite = [s if s != float("inf") else 1e9 for s in scores]

    def med(v):
        s = sorted(v)
        return s[len(s) // 2] if s else None

    def p95(v):
        s = sorted(v)
        return s[int((len(s) - 1) * 0.95)] if s else None

    det = {
        "source": a.replay.name,
        "set": "eval (touched once for this configuration)",
        "operating_point": "flagged = verdict RE-INSPECT or REJECT (score >= t_pass); not-inspected counted as flagged",
        "n_images": len(labels), "n_defective": sum(labels), "n_good": len(labels) - sum(labels),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "miss_rate": fn / max(tp + fn, 1), "false_alarm_rate": fp / max(fp + tn, 1),
        "image_auroc": roc_auc_score(labels, finite),
        "vision_ms_median": med(vision_ms), "vision_ms_p95": p95(vision_ms),
        "frame_ms_median": med(frame_ms), "frame_ms_p95": p95(frame_ms),
        "not_inspected": problems,
        "verdicts": {k: dict(v) for k, v in verdict_table.items()},
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "detection_eval_device.json").write_text(json.dumps(det, indent=2), encoding="utf-8")
    with (a.out / "verdicts_eval_device.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["truth", "PASS", "RE-INSPECT", "REJECT", "NOT_INSPECTED"])
        for t in ("good", "defective"):
            w.writerow([t] + [verdict_table[t].get(v, 0) for v in ("PASS", "RE-INSPECT", "REJECT", "NOT_INSPECTED")])
    print(json.dumps({k: v for k, v in det.items() if k != "not_inspected"}, indent=1))
    print("not inspected:", problems)

    # ---- per-class recall on eval ----
    # ---- synthetic test cases ----
    single_verdict = {f[len("eval/"):]: [r for r in rs if r["verdict"]][0]["verdict"] for f, rs in by_file.items()
                      if f.startswith("eval/") and any(r["verdict"] for r in rs)}
    tc_dir = None
    results = []

    def case(cid, scenario, file, expected, check):
        rs = by_file.get(file)
        if rs is None:
            results.append([cid, scenario, file, expected, "frame not replayed", "NOT RUN"])
            return
        observed, passed = check(rs)
        results.append([cid, scenario, file, expected, observed, "PASS" if passed else "FAIL"])

    def frame_status(rs):
        return rs[0]["frame_status"], rs[0]["frame_reason"]

    def statuses(rs):
        return [r["item_status"] for r in rs if r["item_status"]]

    def summary(rs):
        st, reason = frame_status(rs)
        if st != "OK":
            return f"{st} ({reason})"
        return "; ".join(f"#{r['item_idx']} {r['item_status']}{' ' + r['verdict'] if r['verdict'] else ''}" for r in rs)

    t = "testcases/"
    case("F1", "Empty tray", t + "F1_empty/frame.png", "NO_ITEMS", lambda rs: (summary(rs), frame_status(rs)[0] == "NO_ITEMS"))
    case("F2", "Blurred frame", t + "F2_blur/frame.png", "BAD_IMAGE (blur)", lambda rs: (summary(rs), frame_status(rs) == ("BAD_IMAGE", "blur")))
    case("F3a", "Too dark", t + "F3_exposure/dark.png", "BAD_IMAGE (dark)", lambda rs: (summary(rs), frame_status(rs) == ("BAD_IMAGE", "dark")))
    case("F3b", "Too bright", t + "F3_exposure/bright.png", "BAD_IMAGE (bright)", lambda rs: (summary(rs), frame_status(rs) == ("BAD_IMAGE", "bright")))
    case("F4", "Glare patch", t + "F4_glare/frame.png", "BAD_IMAGE (glare)", lambda rs: (summary(rs), frame_status(rs) == ("BAD_IMAGE", "glare")))
    case("I1", "One good item", t + "I1_good/frame.png", "1 item, OK, PASS", lambda rs: (summary(rs), statuses(rs) == ["OK"] and rs[0]["verdict"] == "PASS"))
    case("I2", "One defective item", t + "I2_defect/frame.png", "1 item, RE-INSPECT or REJECT",
         lambda rs: (summary(rs), statuses(rs) == ["OK"] and rs[0]["verdict"] in ("RE-INSPECT", "REJECT")))

    tiles_path = Path(a.replay).parent / "tiles.json"
    for name, n in (("tray2x2.png", 4), ("tray3x3.png", 9)):
        def check_tray(rs, n=n, name=name):
            ok = [r for r in rs if r["item_status"] == "OK"]
            obs = summary(rs)
            agree = None
            if tiles_path.exists():
                tiles = json.loads(tiles_path.read_text())[name]
                # tiles are stored relative to the category root ("test/<class>/<file>"); eval rows are "<class>/<file>"
                pairs = [(r["verdict"], single_verdict.get(src.removeprefix("test/"))) for r, src in zip(sorted(ok, key=lambda r: int(r["item_idx"])), tiles)]
                agree = sum(x == y for x, y in pairs)
                obs += f" | agreement with single-image eval verdicts {agree}/{len(pairs)}"
            return obs, len(ok) == n
        case("I3", f"Mixed tray {name[4:7]}", t + "I3_trays/" + name, f"{n} items inspected; verdicts compared with single-image runs", check_tray)
    case("I4", "Touching items", t + "I4_touching/frame.png", "CLUSTER, not inspected",
         lambda rs: (summary(rs), statuses(rs) == ["CLUSTER"]))
    case("I5", "Item cut by frame edge", t + "I5_partial/frame.png", "PARTIAL for the cut item; other item inspected",
         lambda rs: (summary(rs), sorted(statuses(rs)) == ["OK", "PARTIAL"]))
    case("I6", "Foreign objects", t + "I6_foreign/frame.png", "UNKNOWN_OBJECT for both objects, never PASS",
         lambda rs: (summary(rs), statuses(rs).count("UNKNOWN_OBJECT") == 2 and all(r["verdict"] != "PASS" for r in rs if r["item_status"] != "OK")))
    case("I6b", "Dark foreign object on dark fixture (known limitation)", t + "I6b_dark_foreign/frame.png", "UNKNOWN_OBJECT",
         lambda rs: (summary(rs), "UNKNOWN_OBJECT" in statuses(rs)))
    case("I7", "More than 12 items", t + "I7_too_many/frame.png", "12 inspected, 4 TOO_MANY",
         lambda rs: (summary(rs)[:120] + "...", statuses(rs).count("OK") == 12 and statuses(rs).count("TOO_MANY") == 4))
    case("I8", "Wrong product (capsule)", t + "I8_wrong_product/frame.png", "UNKNOWN_OBJECT",
         lambda rs: (summary(rs), statuses(rs) == ["UNKNOWN_OBJECT"]))
    case("I9", "Mixed tray with cluster", t + "I9_mixed_cluster/frame.png", "3 inspected + 1 CLUSTER needing attention",
         lambda rs: (summary(rs), sorted(statuses(rs)) == ["CLUSTER", "OK", "OK", "OK"]))
    for i in range(4):
        case(f"I10.{i}", "Near-threshold calib defect", t + f"I10_near_threshold/band_{i}.png", "RE-INSPECT (calib score in band)",
             lambda rs: (summary(rs) + f" score {rs[0]['score']}", rs[0]["verdict"] == "RE-INSPECT"))

    with (a.out / "testcases.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "scenario", "input (synthetic)", "expected", "observed", "result"])
        w.writerows(results)
    for r in results:
        print(r[0], r[5], "|", r[4][:150])


if __name__ == "__main__":
    main()
